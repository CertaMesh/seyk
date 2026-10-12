//! Finding searchable resources under the paths a caller passes in.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use globset::{Glob, GlobSet, GlobSetBuilder};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::segment::TextFormat;
use crate::{Error, Modality, Result};

/// Directory names skipped unless `default_excludes` is turned off.
pub const DEFAULT_EXCLUDED_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
];

const MARKDOWN_EXT: &[&str] = &["md", "markdown", "mdown", "mkd"];
const PLAIN_EXT: &[&str] = &["txt", "text", "rst", "org", "adoc", "tex"];
const CODE_EXT: &[&str] = &[
    "rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "kt", "swift", "c", "h", "cc", "cpp",
    "hpp", "cs", "rb", "php", "sh", "bash", "zsh", "lua", "sql", "toml", "yaml", "yml", "scala",
    "ex", "exs", "hs", "ml",
];
const IMAGE_EXT: &[&str] = &["png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff"];

/// What kind of resource a file is, which decides how it is segmented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "type", content = "format")]
pub enum ResourceKind {
    Pdf,
    Text(TextFormat),
    Image,
    Audio,
}

impl ResourceKind {
    pub fn modality(self) -> Modality {
        match self {
            ResourceKind::Pdf => Modality::Document,
            ResourceKind::Text(_) => Modality::Text,
            ResourceKind::Image => Modality::Image,
            ResourceKind::Audio => Modality::Audio,
        }
    }

    /// Classify by extension. Unknown extensions return `None`.
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        let ext = ext.as_str();
        if ext == "pdf" {
            Some(Self::Pdf)
        } else if MARKDOWN_EXT.contains(&ext) {
            Some(Self::Text(TextFormat::Markdown))
        } else if PLAIN_EXT.contains(&ext) {
            Some(Self::Text(TextFormat::Plain))
        } else if CODE_EXT.contains(&ext) {
            Some(Self::Text(TextFormat::Code))
        } else if ["wav", "flac", "mp3", "m4a"].contains(&ext) {
            Some(Self::Audio)
        } else if IMAGE_EXT.contains(&ext) {
            Some(Self::Image)
        } else {
            None
        }
    }
}

/// Which files to consider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryOptions {
    /// If non-empty, only files matching one of these globs are searched.
    /// A pattern without `/` matches the file name, otherwise the path relative to the root.
    pub include: Vec<String>,
    /// Files or directories matching any of these globs are skipped.
    pub exclude: Vec<String>,
    /// Skip [`DEFAULT_EXCLUDED_DIRS`].
    pub default_excludes: bool,
    /// Visit files and directories whose name starts with `.`.
    pub include_hidden: bool,
    /// Skip files larger than this.
    pub max_file_bytes: u64,
    /// Audio-only policy; engine configuration overrides request discovery values.
    pub max_audio_duration_secs: u64,
}

impl Default for DiscoveryOptions {
    fn default() -> Self {
        Self {
            include: Vec::new(),
            exclude: Vec::new(),
            default_excludes: true,
            include_hidden: false,
            max_file_bytes: 32 * 1024 * 1024,
            max_audio_duration_secs: 7200,
        }
    }
}

/// A file found during discovery, not yet read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredResource {
    /// Absolute, canonical path.
    pub path: PathBuf,
    pub kind: ResourceKind,
    pub size_bytes: u64,
    pub modified_ns: Option<i64>,
}

struct Matcher {
    by_name: GlobSet,
    by_path: GlobSet,
    empty: bool,
}

impl Matcher {
    fn new(patterns: &[String]) -> Result<Self> {
        let mut by_name = GlobSetBuilder::new();
        let mut by_path = GlobSetBuilder::new();
        for p in patterns {
            if p.contains('/') {
                by_path.add(Glob::new(p.trim_start_matches("./"))?);
            } else {
                by_name.add(Glob::new(p)?);
            }
        }
        Ok(Self {
            by_name: by_name.build()?,
            by_path: by_path.build()?,
            empty: patterns.is_empty(),
        })
    }

    fn is_match(&self, rel: &Path) -> bool {
        rel.file_name().is_some_and(|n| self.by_name.is_match(n)) || self.by_path.is_match(rel)
    }
}

/// Walk `roots` and return every searchable file, sorted by path, without duplicates.
///
/// Files passed explicitly are always considered, even when hidden. Files with
/// an unknown extension are treated as plain text only when passed explicitly.
pub fn discover(roots: &[PathBuf], opts: &DiscoveryOptions) -> Result<Vec<DiscoveredResource>> {
    discover_with_policy(roots, opts, |_| true)
}

/// An additional fixed policy may only narrow discovery, including explicit roots.
pub(crate) fn discover_with_policy(
    roots: &[PathBuf],
    opts: &DiscoveryOptions,
    allowed: impl Fn(&Path) -> bool,
) -> Result<Vec<DiscoveredResource>> {
    let include = Matcher::new(&opts.include)?;
    let exclude = Matcher::new(&opts.exclude)?;
    let mut out = Vec::new();

    for root in roots {
        if !allowed(root) {
            continue;
        }
        let root = root.canonicalize().map_err(|e| Error::io(root, e))?;
        if !allowed(&root) {
            continue;
        }
        let meta = std::fs::metadata(&root).map_err(|e| Error::io(&root, e))?;
        if meta.is_file() {
            let kind =
                ResourceKind::from_path(&root).unwrap_or(ResourceKind::Text(TextFormat::Plain));
            if let Some(r) = to_resource(&root, kind, &meta, opts) {
                out.push(r);
            }
            continue;
        }

        let walker = WalkDir::new(&root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                if !allowed(e.path()) {
                    return false;
                }
                if e.depth() == 0 {
                    return true;
                }
                let name = e.file_name().to_string_lossy();
                if !opts.include_hidden && name.starts_with('.') {
                    return false;
                }
                if e.file_type().is_dir()
                    && opts.default_excludes
                    && DEFAULT_EXCLUDED_DIRS.contains(&name.as_ref())
                {
                    return false;
                }
                let rel = e.path().strip_prefix(&root).unwrap_or(e.path());
                !exclude.is_match(rel)
            });

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                // Unreadable directories are skipped rather than failing the whole search.
                Err(_) => continue,
            };
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            let rel = path.strip_prefix(&root).unwrap_or(path);
            if !include.empty && !include.is_match(rel) {
                continue;
            }
            let Some(kind) = ResourceKind::from_path(path) else {
                continue;
            };
            let Ok(meta) = entry.metadata() else { continue };
            if let Some(r) = to_resource(path, kind, &meta, opts) {
                out.push(r);
            }
        }
    }

    out.sort_by(|a, b| a.path.cmp(&b.path));
    out.dedup_by(|a, b| a.path == b.path);
    Ok(out)
}

fn to_resource(
    path: &Path,
    kind: ResourceKind,
    meta: &std::fs::Metadata,
    opts: &DiscoveryOptions,
) -> Option<DiscoveredResource> {
    let limit = if kind == ResourceKind::Audio {
        audio_byte_limit(opts.max_audio_duration_secs).ok()?
    } else {
        opts.max_file_bytes
    };
    if meta.len() > limit {
        return None;
    }
    Some(DiscoveredResource {
        path: path.to_path_buf(),
        kind,
        size_bytes: meta.len(),
        modified_ns: modified_ns(meta),
    })
}

pub(crate) fn modified_ns(meta: &std::fs::Metadata) -> Option<i64> {
    let t = meta.modified().ok()?;
    let d = t.duration_since(UNIX_EPOCH).ok()?;
    i64::try_from(d.as_nanos()).ok()
}

/// Largest supported PCM rate/channels/depth plus 64 MiB container slack.
pub fn audio_byte_limit(seconds: u64) -> crate::Result<u64> {
    if seconds == 0 {
        return Err(crate::Error::Config(
            "max audio duration must be positive".into(),
        ));
    }
    seconds
        .checked_mul(192_000 * 8 * 4)
        .and_then(|n| n.checked_add(64 * 1024 * 1024))
        .ok_or_else(|| crate::Error::Config("max audio duration exceeds byte budget range".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    #[cfg_attr(
        windows,
        ignore = "Solo #4348: discovery fixture assumes slash-separated paths"
    )]
    fn walks_and_filters() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("notes/deep")).unwrap();
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
        fs::create_dir_all(root.join("drafts")).unwrap();
        fs::write(root.join("notes/a.md"), "a").unwrap();
        fs::write(root.join("notes/deep/b.txt"), "b").unwrap();
        fs::write(root.join("notes/c.bin"), "c").unwrap();
        fs::write(root.join(".git/config.txt"), "x").unwrap();
        fs::write(root.join("node_modules/pkg/readme.md"), "x").unwrap();
        fs::write(root.join("drafts/old.md"), "x").unwrap();
        fs::write(root.join(".hidden.md"), "x").unwrap();

        let names = |opts: &DiscoveryOptions| -> Vec<String> {
            discover(&[root.to_path_buf()], opts)
                .unwrap()
                .into_iter()
                .map(|r| {
                    r.path
                        .strip_prefix(root.canonicalize().unwrap())
                        .unwrap()
                        .display()
                        .to_string()
                })
                .collect()
        };

        assert_eq!(
            names(&DiscoveryOptions::default()),
            ["drafts/old.md", "notes/a.md", "notes/deep/b.txt"]
        );

        let opts = DiscoveryOptions {
            exclude: vec!["drafts".into()],
            ..Default::default()
        };
        assert_eq!(names(&opts), ["notes/a.md", "notes/deep/b.txt"]);

        let opts = DiscoveryOptions {
            include: vec!["*.md".into()],
            ..Default::default()
        };
        assert_eq!(names(&opts), ["drafts/old.md", "notes/a.md"]);

        let opts = DiscoveryOptions {
            include: vec!["notes/**".into()],
            ..Default::default()
        };
        assert_eq!(names(&opts), ["notes/a.md", "notes/deep/b.txt"]);
    }

    #[test]
    fn explicit_files_are_always_considered() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("notes.log");
        fs::write(&f, "hello").unwrap();
        let found = discover(&[f], &DiscoveryOptions::default()).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, ResourceKind::Text(TextFormat::Plain));
    }
}
