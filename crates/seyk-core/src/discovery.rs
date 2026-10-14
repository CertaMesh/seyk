//! Finding searchable resources under the paths a caller passes in.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use caseless::Caseless;
use unicode_normalization::UnicodeNormalization;

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

/// Secret names are blocked independently of all discovery options. File rules
/// apply to the final component (assumed to be a file); password-store directory
/// rules apply throughout. Use `ensure_allowed_path` for filesystem paths where
/// the final component may be a directory. This is not content inspection.
/// Keep the list here; all filesystem entry points use this matcher.
pub fn is_secret_path(path: &Path) -> bool {
    secret_path(path, false)
}

fn secret_path(path: &Path, is_dir: bool) -> bool {
    const SUFFIXES: &[&str] = &[
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".jks",
        ".keystore",
        ".kdbx",
        ".kdb",
        ".1pif",
        ".agilekeychain",
        ".opvault",
        ".env",
        ".keychain",
        ".keychain-db",
    ];
    const NAMES: &[&str] = &[
        ".netrc",
        "netrc",
        "_netrc",
        ".pgpass",
        "pgpass",
        "pgpass.conf",
        "credentials",
        "credentials.json",
    ];
    const SSH_KEYS: &[&str] = &["id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_xmss"];
    let mut previous = String::new();
    let mut components = path.components().peekable();
    while let Some(component) = components.next() {
        let std::path::Component::Normal(name) = component else {
            continue;
        };
        // Unicode compatibility caseless normalization, including full folds
        // (e.g. long s, Kelvin sign and sharp s). A conservative superset of
        // case-insensitive filesystem equivalence, identical on all platforms.
        let name = name.to_string_lossy();
        let name: String = if name.is_ascii() {
            name.to_ascii_lowercase()
        } else {
            name.nfd()
                .default_case_fold()
                .nfkd()
                .default_case_fold()
                .nfkd()
                .collect()
        };
        if name.ends_with(".opvault") || name.ends_with(".agilekeychain") {
            return true;
        }
        let file = components.peek().is_none() && !is_dir;
        if file
            && (SUFFIXES.iter().any(|suffix| name.ends_with(*suffix))
                || NAMES.contains(&name.as_str())
                || name == ".env"
                || name.starts_with(".env.")
                || (previous == ".docker" && name == "config.json")
                || (!name.ends_with(".pub")
                    && SSH_KEYS.iter().any(|key| {
                        name == *key
                            || name.strip_prefix(*key).is_some_and(|rest| {
                                rest.starts_with('.')
                                    || rest.starts_with('_')
                                    || rest.starts_with('-')
                            })
                    })))
        {
            return true;
        }
        previous = name;
    }
    false
}

/// Check every supplied component and symlink target before opening a path.
/// Missing paths are still checked (watcher removal events need this); ordinary
/// I/O remains the caller's responsibility. Symlink loops fail after 40 hops.
/// This name policy is not an atomic open or a filesystem sandbox.
pub fn ensure_allowed_path(path: &Path) -> Result<()> {
    let absolute = unresolved_absolute_path(path)?;
    let mut resolver = PolicyResolver {
        original: path,
        hops: 0,
        aliases: std::collections::HashMap::new(),
    };
    let state = resolver.walk(ResolvedPath::default(), &absolute)?;
    // No symlinks remain here. Also check the filesystem's stored spelling.
    if let Ok(real) = state.path.canonicalize() {
        resolver.check(&real, state.is_dir())?;
    }
    Ok(())
}

#[derive(Clone, Default)]
struct ResolvedPath {
    path: PathBuf,
    directories: Vec<bool>,
}

impl ResolvedPath {
    fn is_dir(&self) -> bool {
        self.directories.last().copied().unwrap_or(true)
    }
}

struct PolicyResolver<'a> {
    original: &'a Path,
    hops: usize,
    // Retain every directory alias for this traversal, even after an absolute
    // target or .. leaves and re-enters that directory. Canonical keys below
    // only inspect prefixes whose symlink hops we have already checked.
    aliases: std::collections::HashMap<PathBuf, Vec<std::ffi::OsString>>,
}

impl PolicyResolver<'_> {
    fn check(&self, path: &Path, is_dir: bool) -> Result<()> {
        if secret_path(path, is_dir) {
            return Err(Error::SecretFile {
                path: self.original.to_path_buf(),
            });
        }
        Ok(())
    }

    fn check_name(
        &self,
        parent: &ResolvedPath,
        name: &std::ffi::OsStr,
        is_dir: bool,
    ) -> Result<()> {
        self.check(&parent.path.join(name), is_dir)?;
        if !self.aliases.is_empty() {
            let key = parent
                .path
                .canonicalize()
                .unwrap_or_else(|_| parent.path.clone());
            if let Some(names) = self.aliases.get(&key) {
                for alias in names {
                    self.check(&Path::new(alias).join(name), is_dir)?;
                }
            }
        }
        Ok(())
    }

    fn walk(&mut self, mut state: ResolvedPath, path: &Path) -> Result<ResolvedPath> {
        use std::path::Component;
        let mut components = path.components().peekable();
        while let Some(component) = components.next() {
            match component {
                Component::Prefix(_) | Component::RootDir => {
                    state.path.push(component);
                    state.directories.clear();
                }
                Component::CurDir => {}
                Component::ParentDir => {
                    if !state.is_dir() {
                        return Err(Error::io(
                            self.original,
                            std::io::Error::new(
                                std::io::ErrorKind::NotADirectory,
                                "path component is not a directory",
                            ),
                        ));
                    }
                    if state.path.pop() {
                        state.directories.pop();
                    }
                }
                Component::Normal(name) => {
                    // Intrinsically sensitive directories are refused even if
                    // a later .. or absolute link target would erase the name.
                    self.check_name(&state, name, true)?;
                    let next = state.path.join(name);
                    let metadata = match std::fs::symlink_metadata(&next) {
                        Ok(metadata) => Some(metadata),
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                        Err(e) => return Err(Error::io(self.original, e)),
                    };
                    if metadata
                        .as_ref()
                        .is_some_and(|m| m.file_type().is_symlink())
                    {
                        self.hops += 1;
                        if self.hops > 40 {
                            return Err(Error::io(
                                self.original,
                                std::io::Error::other("too many symbolic link hops (maximum 40)"),
                            ));
                        }
                        let target =
                            std::fs::read_link(&next).map_err(|e| Error::io(self.original, e))?;
                        let resolved = self.walk(state.clone(), &target)?;
                        self.check_name(&state, name, resolved.is_dir())?;
                        if resolved.is_dir() {
                            let key = resolved
                                .path
                                .canonicalize()
                                .unwrap_or_else(|_| resolved.path.clone());
                            self.aliases
                                .entry(key)
                                .or_default()
                                .push(name.to_os_string());
                        }
                        state = resolved;
                    } else {
                        let is_dir = metadata
                            .as_ref()
                            .map_or(components.peek().is_some(), |m| m.is_dir());
                        self.check_name(&state, name, is_dir)?;
                        state.path = next;
                        state.directories.push(is_dir);
                    }
                }
            }
        }
        Ok(state)
    }
}

// Unlike std::path::absolute on Windows, do not collapse caller-supplied ..
// before the resolver can inspect the components and apply symlink semantics.
fn unresolved_absolute_path(path: &Path) -> Result<PathBuf> {
    if path.as_os_str().is_empty() {
        return Err(Error::io(
            path,
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty path"),
        ));
    }
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    #[cfg(windows)]
    if let Some(std::path::Component::Prefix(prefix)) = path.components().next() {
        // Resolve only the drive's working directory for a drive-relative path
        // such as C:notes/../file; keep all supplied normal/parent components.
        let base =
            std::path::absolute(Path::new(prefix.as_os_str())).map_err(|e| Error::io(path, e))?;
        return Ok(base.join(path.components().skip(1).collect::<PathBuf>()));
    }
    std::env::current_dir()
        .map(|base| base.join(path))
        .map_err(|e| Error::io(path, e))
}

/// Make a logical root absolute without erasing symlink names used by policy.
/// Resolve `..` with filesystem semantics before a file-URI codec can collapse
/// it lexically. Keep the logical parent when it denotes that same directory.
pub fn absolute_logical_path(path: &Path) -> Result<PathBuf> {
    ensure_allowed_path(path)?;
    logical_absolute_path(path)
}

fn logical_absolute_path(path: &Path) -> Result<PathBuf> {
    let absolute = unresolved_absolute_path(path)?;
    let mut logical = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::ParentDir => {
                let real_parent = logical
                    .join("..")
                    .canonicalize()
                    .map_err(|e| Error::io(path, e))?;
                if logical
                    .parent()
                    .and_then(|p| p.canonicalize().ok())
                    .as_ref()
                    == Some(&real_parent)
                {
                    logical.pop();
                } else {
                    logical = real_parent;
                }
            }
            std::path::Component::CurDir => {}
            component => logical.push(component),
        }
    }
    Ok(logical)
}

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

/// Which files to consider. Secret paths are always denied, regardless of these options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryOptions {
    /// If non-empty, only files matching one of these globs are searched.
    /// A pattern without `/` matches the file name, otherwise the path relative to the root.
    pub include: Vec<String>,
    /// Files or directories matching any of these globs are skipped.
    pub exclude: Vec<String>,
    /// Skip [`DEFAULT_EXCLUDED_DIRS`]. Disabling this does not admit secret paths.
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
/// Files passed explicitly are considered even when hidden, except secret paths,
/// which return [`Error::SecretFile`]. No discovery option overrides this policy.
/// Files with an unknown extension are treated as plain text only when passed explicitly.
pub fn discover(roots: &[PathBuf], opts: &DiscoveryOptions) -> Result<Vec<DiscoveredResource>> {
    let include = Matcher::new(&opts.include)?;
    let exclude = Matcher::new(&opts.exclude)?;
    let mut out = Vec::new();

    for root in roots {
        let logical_root = absolute_logical_path(root)?;
        let root = root.canonicalize().map_err(|e| Error::io(root, e))?;
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
                let logical = logical_root.join(e.path().strip_prefix(&root).unwrap_or(e.path()));
                if ensure_allowed_path(e.path()).is_err() || ensure_allowed_path(&logical).is_err()
                {
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

    #[test]
    fn absolute_policy_path_keeps_parent_components() {
        let relative = Path::new("vault.opvault/../safe.txt");
        assert!(
            unresolved_absolute_path(relative)
                .unwrap()
                .ends_with(relative)
        );
        assert!(unresolved_absolute_path(Path::new("")).is_err());
        #[cfg(windows)]
        for input in [
            r"C:\vault.opvault\..\safe.txt",
            r"\\?\C:\vault.opvault\..\safe.txt",
        ] {
            assert_eq!(
                unresolved_absolute_path(Path::new(input)).unwrap(),
                Path::new(input)
            );
        }
    }

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
