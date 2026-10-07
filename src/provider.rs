//! Resource access. Filesystem is the only supported production provider.

use crate::discovery::{DiscoveryOptions, ResourceKind, discover, modified_ns};
use crate::{Error, ResourceId, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Opaque freshness token. Unknown tokens never establish freshness.
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
    pub(crate) fn file(size: u64, modified: Option<i64>) -> Self {
        modified
            .map(|t| Self::known(format!("fs:{size}:{t}")))
            .unwrap_or_default()
    }
}

/// Scope for enumeration and deletion checks. Filters retain filesystem semantics.
///
/// Unstable: may change before 1.0; filesystem is the only supported provider.
#[derive(Debug, Clone)]
pub struct ResourceScope {
    pub roots: Vec<ResourceId>,
    pub discovery: DiscoveryOptions,
}

impl ResourceScope {
    pub(crate) fn from_paths(
        paths: &[PathBuf],
        mut roots: Vec<ResourceId>,
        discovery: DiscoveryOptions,
    ) -> Result<Self> {
        for p in paths {
            let absolute = p.canonicalize().map_err(|e| Error::io(p, e))?;
            roots.push(ResourceId::from_path(absolute)?);
        }
        Ok(Self { roots, discovery })
    }
}

/// A resource available for indexing, without loading its contents.
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

/// Enumerate, read and check freshness without coupling the engine to local files.
///
/// Unstable: may change before 1.0; filesystem is the only supported provider.
pub trait ResourceProvider: Send + Sync {
    /// Return resources in deterministic order, without duplicate identities.
    fn list(&self, scope: &ResourceScope) -> Result<Vec<ProvidedResource>>;
    fn read(&self, id: &ResourceId) -> Result<Vec<u8>>;
    /// A known token must change when the resource bytes change. Use unknown
    /// when freshness cannot be established; the engine will reread and hash.
    fn version(&self, id: &ResourceId) -> Result<ResourceVersion>;
    /// True only for confirmed deletion inside scope. Errors and exclusions
    /// must not count as deletion. The conservative default retains entries.
    fn is_deleted(&self, _id: &ResourceId, _scope: &ResourceScope) -> Result<bool> {
        Ok(false)
    }
}

/// Local files, using the same discovery filters and no-follow walk as `discover`.
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

impl ResourceProvider for FsProvider {
    fn list(&self, scope: &ResourceScope) -> Result<Vec<ProvidedResource>> {
        let paths = scope
            .roots
            .iter()
            .map(file_path)
            .collect::<Result<Vec<_>>>()?;
        discover(&paths, &scope.discovery)?
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
            .collect()
    }
    fn read(&self, id: &ResourceId) -> Result<Vec<u8>> {
        let path = file_path(id)?;
        std::fs::read(&path).map_err(|e| Error::io(path, e))
    }
    fn version(&self, id: &ResourceId) -> Result<ResourceVersion> {
        let path = file_path(id)?;
        let meta = std::fs::metadata(&path).map_err(|e| Error::io(path, e))?;
        Ok(ResourceVersion::file(meta.len(), modified_ns(&meta)))
    }
    fn is_deleted(&self, id: &ResourceId, scope: &ResourceScope) -> Result<bool> {
        let Some(path) = id.to_file_path() else {
            return Ok(false);
        };
        let mut inside = false;
        for root in &scope.roots {
            let root = file_path(root)?;
            let real = root.canonicalize().map_err(|e| Error::io(&root, e))?;
            // The URI codec maps Windows VerbatimDisk and Disk paths to the
            // same decoded representation, without statting the missing child.
            inside |= path.starts_with(file_path(&ResourceId::from_path(real)?)?);
        }
        if !inside {
            return Ok(false);
        }
        match std::fs::metadata(&path) {
            Ok(_) => Ok(false),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(true),
            Err(e) => Err(Error::io(path, e)),
        }
    }
}
