//! Resource access. Filesystem is the only supported production provider.

#[cfg(not(unix))]
use crate::discovery::modified_ns;
use crate::discovery::{
    DiscoveryOptions, PolicyRoot, ResourceKind, discover_with_policy, ensure_allowed_path,
};
use crate::{Error, ResourceId, Result};
use serde::{Deserialize, Serialize};
use std::fs::{File, Metadata};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// Opaque change token. Equal known tokens let the warm pass skip reading.
/// Filesystem tokens are cheap change hints, not proof of unchanged bytes.
/// Unknown tokens force a reread and hash check.
///
/// Unstable: may change before 1.0; filesystem is the only supported provider.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResourceVersion(Option<String>);

impl ResourceVersion {
    pub fn known(token: impl Into<String>) -> Self {
        Self(Some(token.into()))
    }
    pub fn unknown() -> Self {
        Self(None)
    }
    pub fn matches(&self, other: &Self) -> bool {
        self.0.is_some() && self == other
    }
    pub fn is_known(&self) -> bool {
        self.0.is_some()
    }
    /// Build `fs:{size}:{mtime}` as a cheap change hint, not a content proof.
    /// An equal-size rewrite with a restored mtime can skip the warm-pass read,
    /// leaving stale vectors and keyword rows. Extraction still checks the
    /// content hash and refuses changed text.
    pub(crate) fn file(size: u64, modified: Option<i64>) -> Self {
        modified
            .map(|t| Self::known(format!("fs:{size}:{t}")))
            .unwrap_or_default()
    }
}

/// Scope for enumeration and deletion checks. Filters retain filesystem semantics.
/// Roots identify a resource or hierarchical URI/path descendants for deletion.
/// DiscoveryOptions remains a filesystem-specific boundary leak.
///
/// Unstable: may change before 1.0; filesystem is the only supported provider.
#[derive(Debug, Clone)]
pub struct ResourceScope {
    pub roots: Vec<ResourceId>,
    pub discovery: DiscoveryOptions,
}

impl ResourceScope {
    /// Check hierarchical root boundaries without filesystem access.
    pub(crate) fn contains(&self, id: &ResourceId) -> bool {
        self.roots.iter().any(|root| {
            if let (Some(path), Some(root)) = (id.to_file_path(), root.to_file_path()) {
                path.starts_with(root)
            } else {
                id == root
                    || id
                        .as_str()
                        .strip_prefix(root.as_str())
                        .is_some_and(|rest| root.as_str().ends_with('/') || rest.starts_with('/'))
            }
        })
    }

    pub(crate) fn from_paths(
        paths: &[PathBuf],
        mut roots: Vec<ResourceId>,
        discovery: DiscoveryOptions,
    ) -> Result<Self> {
        for root in &roots {
            if let Some(path) = root.to_file_path() {
                ensure_allowed_path(&path)?;
            }
        }
        for p in paths {
            // Retain the caller's directory names (e.g. a .docker symlink).
            // Discovery checks logical children before emitting canonical IDs.
            let root = PolicyRoot::new(p)?;
            roots.push(ResourceId::from_path(root.uri_path())?);
        }
        Ok(Self { roots, discovery })
    }
}

/// A resource available for indexing, without loading its contents.
/// `kind` currently comes from filesystem discovery; ResourceKind/TextFormat
/// should move into shared content types at the crate split.
///
/// Unstable: may change before 1.0; filesystem is the only supported provider.
#[derive(Debug, Clone)]
pub struct ProvidedResource {
    pub id: ResourceId,
    pub kind: ResourceKind,
    pub size_bytes: u64,
    pub modified_ns: Option<i64>,
    pub version: ResourceVersion,
    /// Title prefix supplied to the embedder (the file stem for local files).
    pub title: Option<String>,
}

impl ProvidedResource {
    /// Hash every descriptor field that influences preparation, independently of bytes.
    pub(crate) fn fingerprint(&self) -> String {
        let encoded = serde_json::to_vec(&(self.kind, &self.title)).expect("descriptor serializes");
        blake3::hash(&encoded).to_hex().to_string()
    }
}

/// Seekable encoded source; verification is the audio reader's responsibility.
pub trait ReadSeek: std::io::Read + std::io::Seek + Send + Sync {}
impl<T: std::io::Read + std::io::Seek + Send + Sync> ReadSeek for T {}

/// Enumerate, read and check freshness without coupling the engine to local files.
///
/// Unstable: may change before 1.0; filesystem is the only supported provider.
pub trait ResourceProvider: Send + Sync {
    /// Seekable source for bounded audio reads; no whole-file fallback.
    fn open(&self, id: &ResourceId) -> Result<Box<dyn ReadSeek>> {
        Err(Error::UnsupportedAudio {
            resource: id.clone(),
            reason: "provider does not support seekable audio reads",
        })
    }
    /// Return resources in deterministic order, without duplicate identities.
    fn list(&self, scope: &ResourceScope) -> Result<Vec<ProvidedResource>>;
    /// `list`, plus the resolved root identity each scope root was walked under
    /// when the provider can prove it. The default proves nothing.
    fn list_walked(&self, scope: &ResourceScope) -> Result<ResourceListing> {
        Ok(ResourceListing {
            resources: self.list(scope)?,
            walked_roots: None,
        })
    }
    /// Opt in to authorizing indexed identities without discovering uncached descendants.
    /// Other providers retain their normal listing behavior in cached-only searches.
    fn supports_cached_listing(&self) -> bool {
        false
    }
    /// Current eligibility for supplied cached identities, including scope and policy.
    /// Called only after `supports_cached_listing` opts in; no content may be read here.
    fn list_cached(
        &self,
        scope: &ResourceScope,
        _ids: &[ResourceId],
    ) -> Result<Vec<ProvidedResource>> {
        self.list(scope)
    }
    fn read(&self, id: &ResourceId) -> Result<Vec<u8>>;
    /// Return a change token consistent with the token supplied by `list`.
    /// Equal known tokens let the warm pass skip reading when preparation is
    /// unchanged. Filesystem size and mtime are cheap change hints, not proof
    /// of unchanged bytes: an equal-size rewrite with a restored mtime can
    /// leave stale vectors and keyword rows. Extraction still checks the
    /// content hash and refuses changed text. Use unknown tokens in both
    /// `list` and `version` to force a reread and hash check.
    fn version(&self, id: &ResourceId) -> Result<ResourceVersion>;
    /// Resolve deletion roots once per pass, before filtering cached identities.
    /// An error retains entries and is reported as skipped work by the engine.
    fn deletion_scope(&self, scope: &ResourceScope) -> Result<ResourceScope> {
        Ok(scope.clone())
    }
    /// True only for confirmed deletion inside scope. Errors and exclusions
    /// must not count as deletion. Pass the scope returned by `deletion_scope`.
    /// The conservative default retains entries.
    fn is_deleted(&self, _id: &ResourceId, _scope: &ResourceScope) -> Result<bool> {
        Ok(false)
    }
}

/// A provider listing and, when known, the roots that listing actually walked.
#[derive(Debug, Clone, Default)]
pub struct ResourceListing {
    pub resources: Vec<ProvidedResource>,
    /// Resolved once per walked scope root, before its walk, in scope order.
    /// `None` when the provider cannot attribute its listing to resolved roots.
    pub walked_roots: Option<Vec<ResourceId>>,
}

/// Local files, using the same discovery filters and no-follow walk as `discover`.
///
/// On Unix, `read`, `open` and `version` reach a resource from "/" one
/// component at a time without following any symbolic link, so swapping an
/// ancestor for a link cannot redirect them outside the tree that discovery
/// walked. Resource ids must therefore be canonical paths, as discovery and
/// `SearchEngine` produce. An id with a link component (an explicit symlink,
/// or an alias such as /tmp or /var on macOS) fails closed with an error that
/// says so; canonicalize such a path before building its id. On Windows these
/// methods remain path-based and are not pinned.
#[derive(Debug, Default)]
pub struct FsProvider;

fn file_path(id: &ResourceId) -> Result<PathBuf> {
    id.to_file_path().ok_or_else(|| {
        Error::Config(format!(
            "filesystem provider cannot access {} resources",
            id.scheme()
        ))
    })
}

impl FsProvider {
    pub(crate) fn open_verified_handle(id: &ResourceId) -> Result<File> {
        let path = file_path(id)?;
        ensure_allowed_path(&path)?;
        open_local_content(&path)
    }
}

impl ResourceProvider for FsProvider {
    fn open(&self, id: &ResourceId) -> Result<Box<dyn ReadSeek>> {
        let path = file_path(id)?;
        ensure_allowed_path(&path)?;
        Ok(Box::new(open_local_content(&path)?))
    }
    fn list(&self, scope: &ResourceScope) -> Result<Vec<ProvidedResource>> {
        Ok(self.list_walked(scope)?.resources)
    }
    fn list_walked(&self, scope: &ResourceScope) -> Result<ResourceListing> {
        let paths = scope
            .roots
            .iter()
            .map(file_path)
            .collect::<Result<Vec<_>>>()?;
        let walked = discover_with_policy(&paths, &scope.discovery, |_| true)?;
        let resources = walked
            .resources
            .into_iter()
            .map(|r| {
                Ok(ProvidedResource {
                    id: ResourceId::from_path(&r.path)?,
                    kind: r.kind,
                    size_bytes: r.size_bytes,
                    modified_ns: r.modified_ns,
                    version: ResourceVersion::file(r.size_bytes, r.modified_ns),
                    title: r.path.file_stem().map(|s| s.to_string_lossy().into_owned()),
                })
            })
            .collect::<Result<_>>()?;
        Ok(ResourceListing {
            resources,
            walked_roots: Some(
                walked
                    .walked_roots
                    .iter()
                    .map(ResourceId::from_path)
                    .collect::<Result<_>>()?,
            ),
        })
    }
    fn read(&self, id: &ResourceId) -> Result<Vec<u8>> {
        let path = file_path(id)?;
        ensure_allowed_path(&path)?;
        let mut file = open_local_content(&path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| Error::io(path, e))?;
        Ok(bytes)
    }
    fn version(&self, id: &ResourceId) -> Result<ResourceVersion> {
        let path = file_path(id)?;
        ensure_allowed_path(&path)?;
        local_version(&path)
    }
    fn deletion_scope(&self, scope: &ResourceScope) -> Result<ResourceScope> {
        let roots = scope
            .roots
            .iter()
            .map(|root| {
                let path = file_path(root)?;
                let real = path.canonicalize().map_err(|e| Error::io(&path, e))?;
                // Both roots and children are compared through the file-URI codec,
                // normalizing Windows verbatim prefixes without statting a missing child.
                ResourceId::from_path(real)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ResourceScope {
            roots,
            discovery: scope.discovery.clone(),
        })
    }
    fn is_deleted(&self, id: &ResourceId, scope: &ResourceScope) -> Result<bool> {
        let Some(path) = id.to_file_path() else {
            return Ok(false);
        };
        if !scope.contains(id) {
            return Ok(false);
        }
        match std::fs::metadata(&path) {
            Ok(_) => Ok(false),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(true),
            Err(e) => Err(Error::io(path, e)),
        }
    }
}

fn placeholder_marker(path: &Path) -> bool {
    path.components().any(|part| {
        let bytes = part.as_os_str().as_encoded_bytes();
        bytes[bytes.len().saturating_sub(7)..].eq_ignore_ascii_case(b".icloud")
    })
}

fn unavailable_content() -> io::Error {
    io::Error::new(
        io::ErrorKind::PermissionDenied,
        "unmaterialized local content is unavailable; choose a fully local copy",
    )
}

#[cfg(target_os = "macos")]
pub(crate) fn ensure_materialized_flags(flags: u32) -> io::Result<()> {
    // Darwin sys/stat.h: read-only synthetic SF_DATALESS, not a settable user flag.
    const SF_DATALESS: u32 = 0x4000_0000;
    if flags & SF_DATALESS != 0 {
        Err(unavailable_content())
    } else {
        Ok(())
    }
}

fn ensure_content_metadata(metadata: &Metadata) -> io::Result<()> {
    if !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "content source is not a regular file",
        ));
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::darwin::fs::MetadataExt;
        ensure_materialized_flags(metadata.st_flags())?;
    }
    Ok(())
}

/// Same token as discovery: size and modification time of the file itself.
#[cfg(unix)]
fn local_version(path: &Path) -> Result<ResourceVersion> {
    use crate::pinned_dir::{EntryType, PinnedDir, entry_type, link_component, modified_ns};
    let stat = PinnedDir::open_parent(path)
        .and_then(|(parent, name)| parent.stat(name))
        .map_err(|e| Error::io(path, e))?;
    if entry_type(&stat) == EntryType::Symlink {
        return Err(Error::io(path, link_component()));
    }
    Ok(ResourceVersion::file(
        stat.st_size as u64,
        modified_ns(&stat),
    ))
}

#[cfg(not(unix))]
fn local_version(path: &Path) -> Result<ResourceVersion> {
    let meta = std::fs::metadata(path).map_err(|e| Error::io(path, e))?;
    Ok(ResourceVersion::file(meta.len(), modified_ns(&meta)))
}

/// Check again for each content open; discovery/version metadata is never permission to read.
/// This rejects observed placeholders before open and checks the actual descriptor afterward.
/// A provider can still change materialization between check and open: this is not atomic
/// protection against every hydration race. The path is opened from "/" without following
/// any link (see [`FsProvider`]), so it cannot leave the canonical tree it names.
#[cfg(unix)]
fn open_local_content(path: &Path) -> Result<File> {
    use crate::pinned_dir::{EntryType, PinnedDir, entry_type, identity, link_component};
    use std::os::unix::fs::MetadataExt;
    if placeholder_marker(path) {
        return Err(Error::io(path, unavailable_content()));
    }
    let (parent, name) = PinnedDir::open_parent(path).map_err(|e| Error::io(path, e))?;
    let before = parent.stat(name).map_err(|e| Error::io(path, e))?;
    match entry_type(&before) {
        EntryType::File => {
            crate::pinned_dir::materialized(&before).map_err(|e| Error::io(path, e))?
        }
        EntryType::Symlink => return Err(Error::io(path, link_component())),
        _ => {
            return Err(Error::io(
                path,
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "content source is not a regular file",
                ),
            ));
        }
    }
    // O_NONBLOCK: a substitution with a FIFO must not block before validation.
    let file = parent.open_file(name).map_err(|e| Error::io(path, e))?;
    let actual = file.metadata().map_err(|e| Error::io(path, e))?;
    ensure_content_metadata(&actual).map_err(|e| Error::io(path, e))?;
    if identity(&before) != (actual.dev(), actual.ino()) {
        return Err(Error::ChangedWhileReading {
            resource: ResourceId::from_path(path)?,
        });
    }
    Ok(file)
}

/// Path-based on Windows: ordinary explicit symlinks are followed.
#[cfg(not(unix))]
fn open_local_content(path: &Path) -> Result<File> {
    if placeholder_marker(path) {
        return Err(Error::io(path, unavailable_content()));
    }
    let resolved = path.canonicalize().map_err(|e| Error::io(path, e))?;
    if placeholder_marker(&resolved) {
        return Err(Error::io(path, unavailable_content()));
    }
    let before = std::fs::metadata(path).map_err(|e| Error::io(path, e))?;
    ensure_content_metadata(&before).map_err(|e| Error::io(path, e))?;
    let file = File::open(path).map_err(|e| Error::io(path, e))?;
    let actual = file.metadata().map_err(|e| Error::io(path, e))?;
    ensure_content_metadata(&actual).map_err(|e| Error::io(path, e))?;
    Ok(file)
}

#[cfg(all(test, target_os = "macos"))]
mod materialization_tests {
    use super::*;
    #[test]
    fn darwin_dataless_flag_is_denied_without_manufacturing_a_cloud_file() {
        assert!(ensure_materialized_flags(0x4000_0000).is_err());
        assert!(ensure_materialized_flags(0x4000_8000).is_err());
        assert!(ensure_materialized_flags(0).is_ok());
        assert!(ensure_materialized_flags(0x8000).is_ok());
    }
}
