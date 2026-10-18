//! Finding searchable resources under the paths a caller passes in.

use std::path::{Path, PathBuf};

use caseless::Caseless;
use unicode_normalization::UnicodeNormalization;

use globset::{Glob, GlobSet, GlobSetBuilder};
use serde::{Deserialize, Serialize};

use crate::segment::TextFormat;
use crate::{Error, Modality, Result};

/// A policy-checked root with separate logical and canonical identities.
///
/// Use the logical path for URIs, display and discovery (which must retain
/// directory aliases), and the resolved path for authorization. This captures
/// a path identity, not a pinned descriptor; callers must revalidate before
/// using a stored root to authorize access.
#[derive(Debug, Clone)]
pub struct PolicyRoot {
    logical: PathBuf,
    resolved: PathBuf,
}

impl PolicyRoot {
    /// Validate the name policy once and capture an existing root's identities.
    /// Tilde expansion, if desired, remains the caller's responsibility.
    pub fn new(path: &Path) -> Result<Self> {
        ensure_allowed_path(path)?;
        let logical = logical_absolute_path(path)?;
        // Resolve the supplied spelling once, preserving filesystem handling
        // of a file with a trailing slash or `/.` on each platform.
        let resolved = path.canonicalize().map_err(|e| Error::io(path, e))?;
        Ok(Self { logical, resolved })
    }

    pub fn uri_path(&self) -> &Path {
        &self.logical
    }

    pub fn authorize_path(&self) -> &Path {
        &self.resolved
    }

    /// Map a resolved descendant back to the spelling used for name policy.
    /// Reject parent traversal in the relative part instead of emitting an escape.
    pub fn logical_child(&self, resolved: &Path) -> Option<PathBuf> {
        logical_child(&self.logical, &self.resolved, resolved)
    }

    /// Recheck name policy and report whether the captured identity is current.
    /// A symlink chain can introduce a denied alias without changing its target.
    pub fn revalidate(&self) -> std::result::Result<bool, RootRevalidationError> {
        ensure_allowed_path(&self.logical).map_err(RootRevalidationError::Policy)?;
        self.logical
            .canonicalize()
            .map(|current| current == self.resolved)
            .map_err(RootRevalidationError::Identity)
    }

    /// Resolve live for callers whose existing behavior follows root retargets.
    /// This observation does not recheck name policy and cannot be used as a
    /// policy-checked root. Check the resulting logical child before authorizing it.
    ///
    /// ```compile_fail
    /// use seyk_core::discovery::PolicyRoot;
    /// fn unchecked_root(root: &PolicyRoot) -> PolicyRoot {
    ///     root.current().unwrap()
    /// }
    /// ```
    pub fn current(&self) -> std::io::Result<LiveRoot<'_>> {
        Ok(LiveRoot {
            logical: &self.logical,
            resolved: self.logical.canonicalize()?,
        })
    }
}

/// Revalidation preserves the distinction between a name-policy refusal and
/// an I/O failure while checking the captured identity.
#[derive(Debug, thiserror::Error)]
pub enum RootRevalidationError {
    #[error(transparent)]
    Policy(Error),
    #[error(transparent)]
    Identity(std::io::Error),
}

/// A live path observation, not a policy-checked authorization root.
#[derive(Debug)]
pub struct LiveRoot<'a> {
    logical: &'a Path,
    resolved: PathBuf,
}

impl LiveRoot<'_> {
    pub fn uri_path(&self) -> &Path {
        self.logical
    }

    /// The live resolved path; name policy must still be checked by the caller.
    pub fn authorize_path(&self) -> &Path {
        &self.resolved
    }

    pub fn logical_child(&self, resolved: &Path) -> Option<PathBuf> {
        logical_child(self.logical, &self.resolved, resolved)
    }
}

fn logical_child(logical: &Path, root: &Path, resolved: &Path) -> Option<PathBuf> {
    let relative = resolved.strip_prefix(root).ok()?;
    relative
        .components()
        .all(|component| matches!(component, std::path::Component::Normal(_)))
        .then(|| logical.join(relative))
}

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
    resolve_allowed_path(path).map(|_| ())
}

struct CheckedPath {
    path: PathBuf,
    aliases: Vec<std::ffi::OsString>,
}

fn resolve_allowed_path(path: &Path) -> Result<CheckedPath> {
    let absolute = unresolved_absolute_path(path)?;
    let mut resolver = PolicyResolver {
        original: path,
        hops: 0,
        aliases: std::collections::HashMap::new(),
    };
    let state = resolver.walk(ResolvedPath::default(), &absolute)?;
    // No symlinks remain here. Also check the filesystem's stored spelling.
    let real = state.path.canonicalize().unwrap_or(state.path.clone());
    resolver.check(&real, state.is_dir())?;
    let aliases = resolver.aliases.remove(&real).unwrap_or_default();
    Ok(CheckedPath {
        path: real,
        aliases,
    })
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

/// Match request globs without requiring a missing resource to exist. Excluding
/// a directory also excludes its descendants, just as discovery prunes the walk.
pub(crate) fn matches_scope_filters(
    path: &Path,
    root: &Path,
    opts: &DiscoveryOptions,
) -> Result<bool> {
    Ok(ScopeFilters::new(opts)?.matches(path, root))
}

/// Compile once for a cached-identity query; excludes prune ancestors too.
pub(crate) struct ScopeFilters {
    include: Matcher,
    exclude: Matcher,
}

impl ScopeFilters {
    pub(crate) fn new(opts: &DiscoveryOptions) -> Result<Self> {
        Ok(Self {
            include: Matcher::new(&opts.include)?,
            exclude: Matcher::new(&opts.exclude)?,
        })
    }

    pub(crate) fn matches(&self, path: &Path, root: &Path) -> bool {
        let Ok(relative) = path.strip_prefix(root) else {
            return false;
        };
        if relative
            .ancestors()
            .filter(|p| !p.as_os_str().is_empty())
            .any(|p| self.exclude.is_match(p))
        {
            return false;
        }
        self.include.empty || self.include.is_match(relative)
    }
}

/// Walk `roots` and return every searchable file, sorted by path, without duplicates.
///
/// Files passed explicitly are considered even when hidden, except secret paths,
/// which return [`Error::SecretFile`]. No discovery option overrides this policy.
/// Files with an unknown extension are treated as plain text only when passed explicitly.
pub fn discover(roots: &[PathBuf], opts: &DiscoveryOptions) -> Result<Vec<DiscoveredResource>> {
    Ok(discover_with_policy(roots, opts, |_| true)?.resources)
}

/// Discovered resources plus the canonical roots the walk actually used.
pub(crate) struct WalkedDiscovery {
    pub(crate) resources: Vec<DiscoveredResource>,
    /// Each walked root as resolved once before its walk, in root order; roots
    /// the policy skipped are absent. Retargeting a root alias during the walk
    /// cannot change these, so every resource above lies under one of them.
    pub(crate) walked_roots: Vec<PathBuf>,
}

/// An additional fixed policy may only narrow discovery, including explicit roots.
pub(crate) fn discover_with_policy(
    roots: &[PathBuf],
    opts: &DiscoveryOptions,
    allowed: impl Fn(&Path) -> bool,
) -> Result<WalkedDiscovery> {
    let include = Matcher::new(&opts.include)?;
    let exclude = Matcher::new(&opts.exclude)?;
    let mut out = Vec::new();
    let mut walked_roots = Vec::new();

    for root in roots {
        // Resolve all root hops once, retaining intermediate directory names
        // (notably a .docker alias) that canonicalization would erase.
        let checked = resolve_allowed_path(root)?;
        let logical_root = logical_absolute_path(root)?;
        if !allowed(root) {
            continue;
        }
        let root = checked.path;
        if !allowed(&root) {
            continue;
        }
        walked_roots.push(root.clone());
        #[cfg(all(feature = "test-support", debug_assertions))]
        test_hooks::root_resolved(&logical_root);
        let filters = WalkFilters {
            root: &root,
            logical_root: &logical_root,
            aliases: &checked.aliases,
            opts,
            include: &include,
            exclude: &exclude,
            allowed: &allowed,
        };
        #[cfg(unix)]
        walk_root_pinned(&filters, &mut out)?;
        #[cfg(not(unix))]
        walk_root_by_path(&filters, &mut out)?;
    }

    out.sort_by(|a, b| a.path.cmp(&b.path));
    out.dedup_by(|a, b| a.path == b.path);
    Ok(WalkedDiscovery {
        resources: out,
        walked_roots,
    })
}

/// One resolved root and every filter its walk applies.
struct WalkFilters<'a, F> {
    root: &'a Path,
    logical_root: &'a Path,
    aliases: &'a [std::ffi::OsString],
    opts: &'a DiscoveryOptions,
    include: &'a Matcher,
    exclude: &'a Matcher,
    allowed: &'a F,
}

impl<F: Fn(&Path) -> bool> WalkFilters<'_, F> {
    /// Whether to visit an entry and, for a directory, descend into it.
    /// Links below the root are never followed, so stored entry names need
    /// only the name policy. Do not resolve every ancestor again.
    fn admit(&self, path: &Path, depth: usize, is_dir: bool, is_symlink: bool) -> bool {
        if is_symlink {
            return false;
        }
        let rel = path.strip_prefix(self.root).unwrap_or(path);
        if secret_path(path, is_dir)
            || secret_path(&self.logical_root.join(rel), is_dir)
            || (depth == 1
                && self
                    .aliases
                    .iter()
                    .any(|alias| secret_path(&Path::new(alias).join(rel), is_dir)))
            || !(self.allowed)(path)
        {
            return false;
        }
        if depth == 0 {
            return true;
        }
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !self.opts.include_hidden && name.starts_with('.') {
            return false;
        }
        if is_dir && self.opts.default_excludes && DEFAULT_EXCLUDED_DIRS.contains(&name.as_ref()) {
            return false;
        }
        !self.exclude.is_match(rel)
    }

    /// An admitted regular file, if the include globs, kind and size allow it.
    fn resource(&self, path: &Path, len: u64, modified: Option<i64>) -> Option<DiscoveredResource> {
        let rel = path.strip_prefix(self.root).unwrap_or(path);
        if !self.include.empty && !self.include.is_match(rel) {
            return None;
        }
        to_resource(
            path,
            ResourceKind::from_path(path)?,
            len,
            modified,
            self.opts,
        )
    }

    /// A root passed as a file is considered whatever its extension.
    fn explicit_file(&self, len: u64, modified: Option<i64>) -> Option<DiscoveredResource> {
        let kind =
            ResourceKind::from_path(self.root).unwrap_or(ResourceKind::Text(TextFormat::Plain));
        to_resource(self.root, kind, len, modified, self.opts)
    }
}

/// Ancestor directory handles a walk keeps open. Deeper levels are reopened
/// from the nearest kept one when the walk returns to them.
#[cfg(unix)]
const MAX_OPEN_ANCESTORS: usize = 64;

#[cfg(unix)]
struct Frame {
    path: PathBuf,
    name: std::ffi::OsString,
    depth: usize,
    identity: (u64, u64),
    dir: Option<crate::pinned_dir::PinnedDir>,
    entries: std::vec::IntoIter<crate::pinned_dir::Entry>,
}

/// Walk from one directory handle opened on the resolved root. Every child
/// is opened or statted relative to its parent's handle without following a
/// link, so swapping the root or an ancestor for a link mid-walk cannot list
/// anything outside the inode tree resolved here.
#[cfg(unix)]
fn walk_root_pinned<F: Fn(&Path) -> bool>(
    filters: &WalkFilters<'_, F>,
    out: &mut Vec<DiscoveredResource>,
) -> Result<()> {
    use crate::pinned_dir::{EntryType, PinnedDir, entry_type, modified_ns};
    let root = filters.root;
    let changed = || -> Result<()> {
        Err(Error::ChangedWhileReading {
            resource: crate::ResourceId::from_path(root)?,
        })
    };
    // The resolved root has no link component; finding one now means it changed.
    let link = |e: &std::io::Error| e.kind() == std::io::ErrorKind::InvalidInput;
    let parent = match root.file_name() {
        None => None,
        Some(_) => match PinnedDir::open_parent(root) {
            Ok(parent) => Some(parent),
            Err(e) if link(&e) => return changed(),
            Err(e) => return Err(Error::io(root, e)),
        },
    };
    #[cfg(all(feature = "test-support", debug_assertions))]
    test_hooks::root_opening(filters.logical_root);
    // Open the root first and take its identity from the handle; never stat it
    // first and compare. On Linux a stat does not trigger an automount (autofs,
    // systemd automount) but the open does, so the two would name different
    // inodes and every walk of an idle automounted root would fail. O_NOFOLLOW
    // on the open and the type check on the handle keep the root link-free.
    let opened = match &parent {
        None => PinnedDir::open_absolute(root, true),
        Some((parent, name)) => parent.open_dir(name, true),
    };
    let dir = match opened {
        Ok(dir) => dir,
        Err(e) if link(&e) => return changed(),
        // Not a directory, or one that cannot be opened: decide by a no-follow
        // stat. A root file has no automount and is never opened here.
        Err(open_error) => {
            let stat = match &parent {
                None => return Err(Error::io(root, open_error)),
                Some((parent, name)) => parent.stat(name),
            };
            let stat = match stat {
                Ok(stat) => stat,
                Err(e) if link(&e) => return changed(),
                Err(e) => return Err(Error::io(root, e)),
            };
            return match entry_type(&stat) {
                EntryType::File => {
                    out.extend(filters.explicit_file(stat.st_size as u64, modified_ns(&stat)));
                    Ok(())
                }
                EntryType::Symlink => changed(),
                // Like an unreadable directory, a root that cannot be listed
                // yields nothing.
                EntryType::Dir | EntryType::Other => Ok(()),
            };
        }
    };
    #[cfg(all(feature = "test-support", debug_assertions))]
    test_hooks::root_pinned(filters.logical_root);
    if !filters.admit(root, 0, true, false) {
        return Ok(());
    }
    let Ok(entries) = dir.entries() else {
        return Ok(());
    };
    let mut stack = vec![Frame {
        path: root.to_path_buf(),
        name: std::ffi::OsString::new(),
        depth: 0,
        identity: dir.identity(),
        dir: Some(dir),
        entries: entries.into_iter(),
    }];
    while let Some(top) = stack.len().checked_sub(1) {
        if stack[top].dir.is_none() && !reopen(&mut stack) {
            continue;
        }
        let frame = &mut stack[top];
        let Some(entry) = frame.entries.next() else {
            stack.pop();
            continue;
        };
        let parent = frame.dir.as_ref().expect("the walked directory is open");
        let path = frame.path.join(&entry.name);
        let depth = frame.depth + 1;
        let kind = match entry.kind {
            Some(kind) => kind,
            None => match parent.stat(&entry.name) {
                Ok(stat) => entry_type(&stat),
                Err(_) => continue,
            },
        };
        if !filters.admit(
            &path,
            depth,
            kind == EntryType::Dir,
            kind == EntryType::Symlink,
        ) {
            continue;
        }
        match kind {
            EntryType::File => {
                let Ok(stat) = parent.stat(&entry.name) else {
                    continue;
                };
                if entry_type(&stat) == EntryType::File {
                    out.extend(filters.resource(&path, stat.st_size as u64, modified_ns(&stat)));
                }
            }
            EntryType::Dir => {
                // Unreadable or replaced directories are skipped rather than
                // failing the whole search.
                let Ok(child) = parent.open_dir(&entry.name, true) else {
                    continue;
                };
                let Ok(entries) = child.entries() else {
                    continue;
                };
                if top >= MAX_OPEN_ANCESTORS {
                    stack[top].dir = None;
                }
                stack.push(Frame {
                    path,
                    name: entry.name,
                    depth,
                    identity: child.identity(),
                    dir: Some(child),
                    entries: entries.into_iter(),
                });
            }
            EntryType::Symlink | EntryType::Other => {}
        }
    }
    Ok(())
}

/// Reopen the top frame's directory from the nearest open ancestor, by name
/// and without following links. If any level is no longer the directory
/// first walked, drop it and everything below: that subtree is skipped, like
/// an unreadable directory. Returns whether the top frame is open again.
#[cfg(unix)]
fn reopen(stack: &mut Vec<Frame>) -> bool {
    let top = stack.len() - 1;
    let open = stack
        .iter()
        .rposition(|f| f.dir.is_some())
        .expect("the root frame stays open");
    let mut current: Option<crate::pinned_dir::PinnedDir> = None;
    for level in open + 1..=top {
        let base = current
            .as_ref()
            .or(stack[open].dir.as_ref())
            .expect("an open base directory");
        match base.open_dir(&stack[level].name, false) {
            Ok(dir) if dir.identity() == stack[level].identity => current = Some(dir),
            _ => {
                stack.truncate(level);
                return false;
            }
        }
    }
    stack[top].dir = current;
    true
}

/// Path-based walk: the Windows implementation, and the parity reference for
/// the pinned walk in tests. It does not pin the walk to the resolved root.
#[cfg(any(not(unix), test))]
fn walk_root_by_path<F: Fn(&Path) -> bool>(
    filters: &WalkFilters<'_, F>,
    out: &mut Vec<DiscoveredResource>,
) -> Result<()> {
    use walkdir::WalkDir;
    let root = filters.root;
    let meta = std::fs::metadata(root).map_err(|e| Error::io(root, e))?;
    if meta.is_file() {
        out.extend(filters.explicit_file(meta.len(), modified_ns(&meta)));
        return Ok(());
    }
    let walker = WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            filters.admit(
                e.path(),
                e.depth(),
                e.file_type().is_dir(),
                e.file_type().is_symlink(),
            )
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
        let Ok(meta) = entry.metadata() else { continue };
        out.extend(filters.resource(entry.path(), meta.len(), modified_ns(&meta)));
    }
    Ok(())
}

/// Deterministic seam for races between root resolution and the walk.
#[cfg(all(feature = "test-support", debug_assertions))]
pub mod test_hooks {
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    type Hook = Box<dyn FnOnce() + Send>;
    static ROOT_RESOLVED: Mutex<Vec<(PathBuf, Hook)>> = Mutex::new(Vec::new());
    static ROOT_OPENING: Mutex<Vec<(PathBuf, Hook)>> = Mutex::new(Vec::new());
    static ROOT_PINNED: Mutex<Vec<(PathBuf, Hook)>> = Mutex::new(Vec::new());

    /// Run `hook` once, after discovery has resolved the logical `root` and
    /// before it walks the resolved directory. Keyed by root so parallel tests
    /// on other paths never trigger it.
    pub fn after_root_resolved(root: PathBuf, hook: impl FnOnce() + Send + 'static) {
        ROOT_RESOLVED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((root, Box::new(hook)));
    }

    /// Run `hook` once, after discovery has opened the parent of the directory
    /// resolved for the logical `root` and just before it opens that directory.
    pub fn before_root_opened(root: PathBuf, hook: impl FnOnce() + Send + 'static) {
        ROOT_OPENING
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((root, Box::new(hook)));
    }

    /// Run `hook` once, after discovery has opened the handle it walks from
    /// for the logical `root` and before it lists any entry.
    pub fn after_root_pinned(root: PathBuf, hook: impl FnOnce() + Send + 'static) {
        ROOT_PINNED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((root, Box::new(hook)));
    }

    pub(super) fn root_resolved(root: &Path) {
        run(&ROOT_RESOLVED, root);
    }

    #[cfg(unix)]
    pub(super) fn root_opening(root: &Path) {
        run(&ROOT_OPENING, root);
    }

    #[cfg(unix)]
    pub(super) fn root_pinned(root: &Path) {
        run(&ROOT_PINNED, root);
    }

    fn run(registry: &Mutex<Vec<(PathBuf, Hook)>>, root: &Path) {
        let hook = {
            let mut hooks = registry.lock().unwrap_or_else(|e| e.into_inner());
            hooks
                .iter()
                .position(|(r, _)| r == root)
                .map(|i| hooks.remove(i).1)
        };
        if let Some(hook) = hook {
            hook();
        }
    }
}

fn to_resource(
    path: &Path,
    kind: ResourceKind,
    len: u64,
    modified_ns: Option<i64>,
    opts: &DiscoveryOptions,
) -> Option<DiscoveredResource> {
    let limit = if kind == ResourceKind::Audio {
        audio_byte_limit(opts.max_audio_duration_secs).ok()?
    } else {
        opts.max_file_bytes
    };
    if len > limit {
        return None;
    }
    Some(DiscoveredResource {
        path: path.to_path_buf(),
        kind,
        size_bytes: len,
        modified_ns,
    })
}

#[cfg(any(not(unix), test))]
pub(crate) fn modified_ns(meta: &std::fs::Metadata) -> Option<i64> {
    let t = meta.modified().ok()?;
    let d = t.duration_since(std::time::UNIX_EPOCH).ok()?;
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

        let names = |opts: &DiscoveryOptions| -> Vec<PathBuf> {
            discover(&[root.to_path_buf()], opts)
                .unwrap()
                .into_iter()
                .map(|r| {
                    r.path
                        .strip_prefix(root.canonicalize().unwrap())
                        .unwrap()
                        .to_path_buf()
                })
                .collect()
        };

        assert_eq!(
            names(&DiscoveryOptions::default()),
            ["drafts/old.md", "notes/a.md", "notes/deep/b.txt"].map(PathBuf::from)
        );

        let opts = DiscoveryOptions {
            exclude: vec!["drafts".into()],
            ..Default::default()
        };
        assert_eq!(
            names(&opts),
            ["notes/a.md", "notes/deep/b.txt"].map(PathBuf::from)
        );

        let opts = DiscoveryOptions {
            include: vec!["*.md".into()],
            ..Default::default()
        };
        assert_eq!(
            names(&opts),
            ["drafts/old.md", "notes/a.md"].map(PathBuf::from)
        );

        let opts = DiscoveryOptions {
            include: vec!["notes/**".into()],
            ..Default::default()
        };
        assert_eq!(
            names(&opts),
            ["notes/a.md", "notes/deep/b.txt"].map(PathBuf::from)
        );
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

    /// Run both walkers over one root exactly as `discover_with_policy` does.
    #[cfg(unix)]
    fn both_walks(
        root: &Path,
        opts: &DiscoveryOptions,
        allowed: impl Fn(&Path) -> bool,
    ) -> (Vec<DiscoveredResource>, Vec<DiscoveredResource>) {
        let checked = resolve_allowed_path(root).unwrap();
        let logical_root = logical_absolute_path(root).unwrap();
        let (include, exclude) = (
            Matcher::new(&opts.include).unwrap(),
            Matcher::new(&opts.exclude).unwrap(),
        );
        let filters = WalkFilters {
            root: &checked.path,
            logical_root: &logical_root,
            aliases: &checked.aliases,
            opts,
            include: &include,
            exclude: &exclude,
            allowed: &allowed,
        };
        let (mut pinned, mut by_path) = (Vec::new(), Vec::new());
        walk_root_pinned(&filters, &mut pinned).unwrap();
        walk_root_by_path(&filters, &mut by_path).unwrap();
        pinned.sort_by(|a, b| a.path.cmp(&b.path));
        by_path.sort_by(|a, b| a.path.cmp(&b.path));
        (pinned, by_path)
    }

    #[cfg(unix)]
    #[test]
    fn pinned_walk_matches_the_path_walk() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let root = base.join("root");
        for d in [
            "sub/deep",
            ".hiddendir",
            "target",
            "node_modules",
            "excluded",
            "empty",
            "real-docker",
            ".git",
        ] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        for (f, body) in [
            ("a.md", "alpha"),
            ("b.txt", "beta"),
            ("c.rs", "fn c() {}"),
            ("unknown.xyz", "no kind"),
            ("big.txt", "0123456789abcdef"),
            (".hidden.md", "hidden"),
            (".hiddendir/x.md", "hidden dir"),
            ("target/x.rs", "excluded by default"),
            ("node_modules/y.js", "excluded by default"),
            ("sub/deep/d.md", "deep"),
            ("sub/secret.pem", "secret"),
            ("sub/id_rsa", "secret"),
            ("sub/.env", "secret"),
            ("excluded/e.md", "glob excluded"),
            ("real-docker/config.json", "docker"),
            ("real-docker/notes.md", "docker notes"),
            (".git/HEAD.txt", "vcs"),
            ("song.mp3", "not really audio"),
        ] {
            std::fs::write(root.join(f), body).unwrap();
        }
        symlink(root.join("a.md"), root.join("link.md")).unwrap();
        symlink(root.join("sub"), root.join("linkdir")).unwrap();
        symlink(root.join("real-docker"), base.join(".docker")).unwrap();
        let mut deep = root.join("tall");
        // Stay below PATH_MAX (1024 on macOS) for the path-based reference.
        for _ in 0..150 {
            deep = deep.join("l");
        }
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(deep.join("bottom.md"), "bottom").unwrap();
        std::fs::write(root.join("tall/l/l/mid.md"), "mid").unwrap();

        let small = DiscoveryOptions {
            max_file_bytes: 10,
            ..Default::default()
        };
        let everything = DiscoveryOptions {
            include_hidden: true,
            default_excludes: false,
            ..Default::default()
        };
        let globbed = DiscoveryOptions {
            include: vec!["*.md".into(), "sub/**".into()],
            exclude: vec!["excluded".into(), "tall/l/l/l/l".into()],
            ..Default::default()
        };
        let denied = root.join("sub");
        type Allowed = Box<dyn Fn(&Path) -> bool>;
        let cases: Vec<(PathBuf, DiscoveryOptions, Allowed)> = vec![
            (
                root.clone(),
                DiscoveryOptions::default(),
                Box::new(|_| true),
            ),
            (root.clone(), small, Box::new(|_| true)),
            (root.clone(), everything.clone(), Box::new(|_| true)),
            (root.clone(), globbed, Box::new(|_| true)),
            (
                root.clone(),
                everything.clone(),
                Box::new(move |p: &Path| !p.starts_with(&denied)),
            ),
            (base.join(".docker"), everything, Box::new(|_| true)),
            (
                root.join("a.md"),
                DiscoveryOptions::default(),
                Box::new(|_| true),
            ),
            (
                root.join("unknown.xyz"),
                DiscoveryOptions::default(),
                Box::new(|_| true),
            ),
        ];
        for (i, (path, opts, allowed)) in cases.into_iter().enumerate() {
            let (pinned, by_path) = both_walks(&path, &opts, allowed);
            assert!(!pinned.is_empty(), "case {i} found nothing");
            assert_eq!(pinned, by_path, "case {i}");
        }
        let (pinned, _) = both_walks(&root, &DiscoveryOptions::default(), |_| true);
        let names: Vec<_> = pinned
            .iter()
            .map(|r| r.path.strip_prefix(&root).unwrap().to_path_buf())
            .collect();
        assert!(names.contains(&deep.strip_prefix(&root).unwrap().join("bottom.md")));
        assert!(
            !names
                .iter()
                .any(|n| n.starts_with("linkdir") || n == Path::new("link.md"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn deep_pinned_walk_matches_the_path_walk() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let mut deep = root.clone();
        for _ in 0..300 {
            deep = deep.join("d");
            std::fs::create_dir(&deep).unwrap();
            std::fs::write(deep.join("n.md"), "note").unwrap();
        }
        let (pinned, by_path) = both_walks(&root, &DiscoveryOptions::default(), |_| true);
        assert_eq!(pinned.len(), 300);
        assert_eq!(pinned, by_path);
    }
}
