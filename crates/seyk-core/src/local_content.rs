//! Opt-in policy for automatic local-content collections on Unix.
//!
//! The same fixed path policy narrows discovery and every content access. Roots
//! are pinned directories; reads never follow symlinks. This is not an atomic
//! cloud-state or namespace snapshot: a directory can be renamed while traversed,
//! and a file can become dataless between the precheck and content open.

use crate::discovery::{
    ResourceKind, ScopeFilters, audio_byte_limit, discover_with_policy, ensure_allowed_path,
    matches_scope_filters,
};
use crate::provider::ReadSeek;
use crate::{
    Error, ProvidedResource, ResourceId, ResourceListing, ResourceProvider, ResourceScope,
    ResourceVersion, Result,
};
use std::collections::HashMap;
use std::ffi::CString;
use std::fs::File;
use std::io::{self, Read};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Advertised by the opt-in MCP server. Incompatible policy changes require a new version.
pub const LOCAL_CONTENT_POLICY_VERSION: u32 = 1;

const FORBIDDEN: &[&[u8]] = &[
    b"library",
    b"cache",
    b"caches",
    b".cache",
    b".build",
    b"build",
    b"dist",
    b"target",
    b"node_modules",
    b"__pycache__",
    b".git",
    b".hg",
    b".svn",
    b".venv",
];

fn policy_allows(path: &Path) -> bool {
    let parts: Vec<_> = path
        .components()
        .filter_map(|c| match c {
            Component::Normal(p) => Some(p.as_bytes()),
            _ => None,
        })
        .collect();
    if path
        .components()
        .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_)))
    {
        return false;
    }
    if parts.iter().any(|p| {
        p.starts_with(b".")
            || FORBIDDEN.iter().any(|n| p.eq_ignore_ascii_case(n))
            || p[p.len().saturating_sub(7)..].eq_ignore_ascii_case(b".icloud")
    }) {
        return false;
    }
    // Narrow custom Rust artifact layout, normalized for this semantic policy.
    // Ordinary targets/data remains eligible.
    for (i, part) in parts.iter().enumerate() {
        if (part.eq_ignore_ascii_case(b"debug") || part.eq_ignore_ascii_case(b"release"))
            && i > 0
            && i + 1 < parts.len()
            && parts[i.saturating_sub(3)..i]
                .iter()
                .any(|p| p.eq_ignore_ascii_case(b"target") || p.eq_ignore_ascii_case(b"targets"))
            && [
                b"incremental".as_slice(),
                b".fingerprint",
                b"build",
                b"deps",
                b"examples",
            ]
            .iter()
            .any(|n| parts[i + 1].eq_ignore_ascii_case(n))
        {
            return false;
        }
    }
    true
}

#[derive(Debug)]
struct Root {
    path: PathBuf,
    directory: File,
    identity: (u64, u64),
}

/// Fixed, opt-in content provider. Ordinary `FsProvider` behavior is unchanged.
/// Explicit roots must already identify real directories without any symlink
/// component (including POSIX aliases such as /tmp on macOS).
#[derive(Debug)]
pub struct LocalContentProvider {
    roots: Vec<Root>,
    // Keep the first verified directory identity for each requested deletion
    // scope. Never adopt its replacement during this provider's lifetime.
    deletion_roots: Mutex<HashMap<PathBuf, Arc<Root>>>,
}

fn denied(path: &Path) -> Error {
    Error::io(
        path,
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "path is outside the fixed local-content policy",
        ),
    )
}
fn c_name(name: &std::ffi::OsStr) -> io::Result<CString> {
    CString::new(name.as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid filesystem name"))
}
fn stat_at(directory: &File, name: &std::ffi::OsStr) -> io::Result<libc::stat> {
    let name = c_name(name)?;
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: live directory fd, NUL-terminated name, writable stat storage.
    if unsafe {
        libc::fstatat(
            directory.as_raw_fd(),
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: successful fstatat initialized the complete output.
    Ok(unsafe { stat.assume_init() })
}
fn descriptor_stat(file: &File) -> io::Result<libc::stat> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: live fd and writable output; success initializes the output.
    if unsafe { libc::fstat(file.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { stat.assume_init() })
}
#[allow(clippy::unnecessary_cast)] // stat integer types vary between Unix targets.
fn identity(stat: &libc::stat) -> (u64, u64) {
    (stat.st_dev as u64, stat.st_ino as u64)
}
fn materialized(stat: &libc::stat) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    crate::provider::ensure_materialized_flags(stat.st_flags)?;
    #[cfg(not(target_os = "macos"))]
    let _ = stat;
    Ok(())
}
fn ensure_kind(stat: &libc::stat, directory: bool) -> io::Result<()> {
    let expected = if directory {
        libc::S_IFDIR
    } else {
        libc::S_IFREG
    };
    if stat.st_mode & libc::S_IFMT != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "local-content source has an unsupported type",
        ));
    }
    materialized(stat)
}
fn open_at(directory: &File, name: &std::ffi::OsStr, is_dir: bool) -> io::Result<File> {
    let before = stat_at(directory, name)?;
    ensure_kind(&before, is_dir)?;
    let name = c_name(name)?;
    let flags = libc::O_RDONLY
        | libc::O_NOFOLLOW
        | libc::O_CLOEXEC
        | libc::O_NONBLOCK
        | if is_dir { libc::O_DIRECTORY } else { 0 };
    // SAFETY: live directory fd and NUL-terminated name; no creation mode needed.
    let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: successful openat returned a new fd owned only by this File.
    let file = unsafe { File::from_raw_fd(fd) };
    let after = descriptor_stat(&file)?;
    ensure_kind(&after, is_dir)?;
    if identity(&before) != identity(&after) {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "local-content path changed while opening",
        ));
    }
    Ok(file)
}
fn open_directory(path: &Path) -> io::Result<File> {
    if !path.is_absolute()
        || path
            .components()
            .any(|c| !matches!(c, Component::RootDir | Component::Normal(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "local-content root must be an absolute normalized directory",
        ));
    }
    let mut directory = File::open("/")?;
    for c in path.components() {
        if let Component::Normal(name) = c {
            directory = open_at(&directory, name, true)?;
        }
    }
    Ok(directory)
}
fn version(stat: &libc::stat) -> ResourceVersion {
    ResourceVersion::known(format!(
        "local:{}:{}:{}:{}:{}:{}:{}",
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime,
        stat.st_mtime_nsec,
        stat.st_ctime,
        stat.st_ctime_nsec
    ))
}

// libc time_t and the nanosecond field can independently be i32 or i64.
// Normalize both before arithmetic; unknown timestamps do not exclude a file.
fn timestamp_ns(seconds: impl Into<i64>, nanos: impl Into<i64>) -> Option<i64> {
    seconds
        .into()
        .checked_mul(1_000_000_000)?
        .checked_add(nanos.into())
}

impl LocalContentProvider {
    pub fn new(paths: &[PathBuf]) -> Result<Self> {
        if paths.is_empty() {
            return Err(Error::Config(
                "local-content requires explicit directory roots".into(),
            ));
        }
        let mut roots = Vec::new();
        for path in paths {
            ensure_allowed_path(path)?;
            if !policy_allows(path) {
                return Err(denied(path));
            }
            let directory = open_directory(path).map_err(|e| Error::io(path, e))?;
            let identity = identity(&descriptor_stat(&directory).map_err(|e| Error::io(path, e))?);
            roots.push(Root {
                path: path.clone(),
                directory,
                identity,
            });
        }
        Ok(Self {
            roots,
            deletion_roots: Mutex::new(HashMap::new()),
        })
    }
    /// Exact, non-canonicalized scope names validated when the provider was created.
    pub fn roots(&self) -> Vec<PathBuf> {
        self.roots.iter().map(|r| r.path.clone()).collect()
    }
    /// Metadata-only authorization for original MCP path inputs, before any
    /// caller canonicalization could erase a symlink from the request.
    pub fn validate_path(&self, path: &Path) -> Result<()> {
        let stat = self.stat(path)?;
        match stat.st_mode & libc::S_IFMT {
            libc::S_IFDIR | libc::S_IFREG => Ok(()),
            _ => Err(denied(path)),
        }
    }
    fn root_for(&self, path: &Path) -> Result<&Root> {
        if !policy_allows(path) {
            return Err(denied(path));
        }
        let root = self
            .roots
            .iter()
            .filter(|r| path.starts_with(&r.path))
            .max_by_key(|r| r.path.components().count())
            .ok_or_else(|| denied(path))?;
        let relative = path.strip_prefix(&root.path).map_err(|_| denied(path))?;
        if relative.components().any(|c| match c {
            Component::Normal(n) => n.as_bytes().starts_with(b"."),
            _ => true,
        }) {
            return Err(denied(path));
        }
        Ok(root)
    }
    fn validate_root(&self, root: &Root) -> Result<()> {
        let current = open_directory(&root.path).map_err(|e| Error::io(&root.path, e))?;
        if identity(&descriptor_stat(&current).map_err(|e| Error::io(&root.path, e))?)
            != root.identity
        {
            return Err(Error::ChangedWhileReading {
                resource: ResourceId::from_path(&root.path)?,
            });
        }
        Ok(())
    }
    fn parent(&self, path: &Path) -> Result<(File, std::ffi::OsString)> {
        let root = self.root_for(path)?;
        self.validate_root(root)?;
        let relative = path.strip_prefix(&root.path).map_err(|_| denied(path))?;
        let mut parts = relative.components().peekable();
        let mut directory = root.directory.try_clone().map_err(|e| Error::io(path, e))?;
        while let Some(Component::Normal(name)) = parts.next() {
            if parts.peek().is_none() {
                return Ok((directory, name.to_owned()));
            }
            directory = open_at(&directory, name, true).map_err(|e| Error::io(path, e))?;
        }
        Err(denied(path))
    }
    fn stat(&self, path: &Path) -> Result<libc::stat> {
        ensure_allowed_path(path)?;
        let root = self.root_for(path)?;
        if path == root.path {
            self.validate_root(root)?;
            return descriptor_stat(&root.directory).map_err(|e| Error::io(path, e));
        }
        let (directory, name) = self.parent(path)?;
        let stat = stat_at(&directory, &name).map_err(|e| Error::io(path, e))?;
        if stat.st_mode & libc::S_IFMT == libc::S_IFLNK {
            return Err(denied(path));
        }
        materialized(&stat).map_err(|e| Error::io(path, e))?;
        Ok(stat)
    }
    fn content(&self, path: &Path) -> Result<File> {
        ensure_allowed_path(path)?;
        let (directory, name) = self.parent(path)?;
        open_at(&directory, &name, false).map_err(|e| Error::io(path, e))
    }
    fn directory(&self, path: &Path) -> Result<File> {
        ensure_allowed_path(path)?;
        let root = self.root_for(path)?;
        if path == root.path {
            self.validate_root(root)?;
            return root.directory.try_clone().map_err(|e| Error::io(path, e));
        }
        let (directory, name) = self.parent(path)?;
        open_at(&directory, &name, true).map_err(|e| Error::io(path, e))
    }
    fn validate_deletion_root(&self, root: &Root) -> Result<()> {
        let current = self.directory(&root.path)?;
        if identity(&descriptor_stat(&current).map_err(|e| Error::io(&root.path, e))?)
            != root.identity
        {
            return Err(Error::ChangedWhileReading {
                resource: ResourceId::from_path(&root.path)?,
            });
        }
        Ok(())
    }
}
impl ResourceProvider for LocalContentProvider {
    fn supports_cached_listing(&self) -> bool {
        true
    }

    fn list_cached(
        &self,
        scope: &ResourceScope,
        ids: &[ResourceId],
    ) -> Result<Vec<ProvidedResource>> {
        let filters = ScopeFilters::new(&scope.discovery)?;
        let mut roots = Vec::new();
        for id in &scope.roots {
            let path = id
                .to_file_path()
                .ok_or_else(|| Error::Config("local-content requires file resources".into()))?;
            // Verify the requested scope through the configured root's pinned
            // identity, even for an empty cache. No descendant enumeration.
            let stat = self.stat(&path)?;
            let file = stat.st_mode & libc::S_IFMT == libc::S_IFREG;
            ensure_kind(&stat, !file).map_err(|e| Error::io(&path, e))?;
            roots.push((path, file));
        }
        let mut resources = Vec::new();
        for id in ids {
            let Some(path) = id.to_file_path() else {
                continue;
            };
            if !policy_allows(&path) || ensure_allowed_path(&path).is_err() {
                continue;
            }
            let explicit = roots.iter().any(|(root, file)| *file && path == *root);
            if !explicit
                && !roots
                    .iter()
                    .any(|(root, file)| !file && filters.matches(&path, root))
            {
                continue;
            }
            // Explicit files retain discovery's unknown-extension and filter
            // semantics; the fixed local policy cannot be disabled by either.
            let kind = match ResourceKind::from_path(&path) {
                Some(kind) => kind,
                None if explicit => ResourceKind::Text(crate::segment::TextFormat::Plain),
                None => continue,
            };
            let Ok(stat) = self.stat(&path) else { continue };
            if ensure_kind(&stat, false).is_err() {
                continue;
            }
            let limit = if kind == ResourceKind::Audio {
                let Ok(limit) = audio_byte_limit(scope.discovery.max_audio_duration_secs) else {
                    continue;
                };
                limit
            } else {
                scope.discovery.max_file_bytes
            };
            if stat.st_size < 0 || stat.st_size as u64 > limit {
                continue;
            }
            resources.push((
                path.clone(),
                ProvidedResource {
                    id: id.clone(),
                    kind,
                    size_bytes: stat.st_size as u64,
                    modified_ns: timestamp_ns(stat.st_mtime, stat.st_mtime_nsec),
                    version: version(&stat),
                    title: path.file_stem().map(|s| s.to_string_lossy().into_owned()),
                },
            ));
        }
        resources.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.id.as_str().cmp(b.1.id.as_str()))
        });
        resources.dedup_by(|a, b| a.0 == b.0);
        Ok(resources
            .into_iter()
            .map(|(_, resource)| resource)
            .collect())
    }

    fn list(&self, scope: &ResourceScope) -> Result<Vec<ProvidedResource>> {
        Ok(self.list_walked(scope)?.resources)
    }

    fn list_walked(&self, scope: &ResourceScope) -> Result<ResourceListing> {
        let paths = scope
            .roots
            .iter()
            .map(|id| {
                id.to_file_path()
                    .ok_or_else(|| Error::Config("local-content requires file resources".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        for path in &paths {
            self.stat(path)?;
        }
        let walked = discover_with_policy(&paths, &scope.discovery, |p| self.stat(p).is_ok())?;
        let resources = walked
            .resources
            .into_iter()
            .filter_map(|r| {
                // Discovery is metadata-only and path-based. Revalidate through the
                // pinned scope; only the confined content method can return bytes.
                let stat = self.stat(&r.path).ok()?;
                ensure_kind(&stat, false).ok()?;
                Some(Ok(ProvidedResource {
                    id: ResourceId::from_path(&r.path).ok()?,
                    kind: r.kind,
                    size_bytes: stat.st_size as u64,
                    modified_ns: timestamp_ns(stat.st_mtime, stat.st_mtime_nsec),
                    version: version(&stat),
                    title: r.path.file_stem().map(|s| s.to_string_lossy().into_owned()),
                }))
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
        let path = id
            .to_file_path()
            .ok_or_else(|| Error::Config("local-content requires file resources".into()))?;
        let mut file = self.content(&path)?;
        let before = descriptor_stat(&file).map_err(|e| Error::io(&path, e))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| Error::io(&path, e))?;
        let after = descriptor_stat(&file).map_err(|e| Error::io(&path, e))?;
        ensure_kind(&after, false).map_err(|e| Error::io(&path, e))?;
        if !version(&before).matches(&version(&after)) {
            return Err(Error::ChangedWhileReading {
                resource: id.clone(),
            });
        }
        Ok(bytes)
    }
    fn open(&self, id: &ResourceId) -> Result<Box<dyn ReadSeek>> {
        let path = id
            .to_file_path()
            .ok_or_else(|| Error::Config("local-content requires file resources".into()))?;
        Ok(Box::new(self.content(&path)?))
    }
    fn version(&self, id: &ResourceId) -> Result<ResourceVersion> {
        let path = id
            .to_file_path()
            .ok_or_else(|| Error::Config("local-content requires file resources".into()))?;
        let stat = self.stat(&path)?;
        ensure_kind(&stat, false).map_err(|e| Error::io(&path, e))?;
        Ok(version(&stat))
    }
    fn deletion_scope(&self, scope: &ResourceScope) -> Result<ResourceScope> {
        let mut verified = Vec::new();
        for id in &scope.roots {
            let path = id
                .to_file_path()
                .ok_or_else(|| Error::Config("local-content requires file resources".into()))?;
            let directory = self.directory(&path)?;
            let identity = identity(&descriptor_stat(&directory).map_err(|e| Error::io(&path, e))?);
            verified.push(Root {
                path,
                directory,
                identity,
            });
        }
        let mut anchors = self
            .deletion_roots
            .lock()
            .map_err(|_| Error::Config("deletion scope lock poisoned".into()))?;
        for root in verified {
            if let Some(pinned) = anchors.get(&root.path) {
                if root.identity != pinned.identity {
                    return Err(Error::ChangedWhileReading {
                        resource: ResourceId::from_path(&root.path)?,
                    });
                }
            } else {
                anchors.insert(root.path.clone(), Arc::new(root));
            }
        }
        Ok(scope.clone())
    }
    fn is_deleted(&self, id: &ResourceId, scope: &ResourceScope) -> Result<bool> {
        let Some(path) = id.to_file_path() else {
            return Ok(false);
        };
        if !scope.contains(id) {
            return Ok(false);
        }
        let Ok(configured_root) = self.root_for(&path) else {
            return Ok(false);
        };
        let mut eligible = Vec::new();
        for id in &scope.roots {
            let Some(root) = id.to_file_path() else {
                continue;
            };
            if path.starts_with(&root) && matches_scope_filters(&path, &root, &scope.discovery)? {
                eligible.push(root);
            }
        }
        let Some(root) = eligible.into_iter().max_by_key(|r| r.components().count()) else {
            return Ok(false);
        };
        let pinned = self
            .deletion_roots
            .lock()
            .map_err(|_| Error::Config("deletion scope lock poisoned".into()))?
            .get(&root)
            .cloned()
            .ok_or_else(|| Error::Config("deletion scope has not been verified".into()))?;
        self.validate_root(configured_root)?;
        self.validate_deletion_root(&pinned)?;
        let mut directory = pinned
            .directory
            .try_clone()
            .map_err(|e| Error::io(&path, e))?;
        let relative = path.strip_prefix(&root).map_err(|_| denied(&path))?;
        let mut parts = relative.components().peekable();
        while let Some(Component::Normal(name)) = parts.next() {
            match stat_at(&directory, name) {
                Ok(_) if parts.peek().is_none() => return Ok(false),
                Ok(_) => {
                    directory = open_at(&directory, name, true).map_err(|e| Error::io(&path, e))?
                }
                Err(e) if e.raw_os_error() == Some(libc::ENOENT) => {
                    // A missing child never authorizes deletion if either the
                    // configured root or requested scope changed during the walk.
                    self.validate_root(configured_root)?;
                    self.validate_deletion_root(&pinned)?;
                    return Ok(true);
                }
                Err(e) => return Err(Error::io(&path, e)),
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod timestamp_tests {
    use super::timestamp_ns;
    #[test]
    fn native_i32_and_i64_timestamp_fields_produce_i64_nanoseconds() {
        let expected: Option<i64> = Some(2_000_000_007);
        assert_eq!(timestamp_ns(2_i32, 7_i32), expected);
        assert_eq!(timestamp_ns(2_i64, 7_i32), expected);
        assert_eq!(timestamp_ns(2_i64, 7_i64), expected);
        assert_eq!(
            timestamp_ns(i32::MAX, 7_i32),
            Some(2_147_483_647_000_000_007)
        );
        assert_eq!(timestamp_ns(i64::MAX, 0_i32), None);
    }
}
