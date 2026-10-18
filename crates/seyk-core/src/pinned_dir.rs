//! Descriptor-relative, no-follow filesystem access on Unix.
//!
//! A directory handle names one inode. Children are opened relative to it and
//! never through a symbolic link, so an ancestor swapped to a link cannot
//! redirect a walk or a read outside the inode tree the handle was opened on.
//! This is not a namespace snapshot: entries can still be renamed while walked.

use std::ffi::{CStr, CString, OsStr, OsString};
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Component, Path};

/// Directory open flags when the handle only anchors further `*at` calls.
/// Search permission suffices, as for path-based access through an ancestor.
#[cfg(any(target_os = "linux", target_os = "android"))]
const SEARCH: libc::c_int = libc::O_PATH | libc::O_DIRECTORY;
#[cfg(target_vendor = "apple")]
const SEARCH: libc::c_int = libc::O_SEARCH;
#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
const SEARCH: libc::c_int = libc::O_RDONLY | libc::O_DIRECTORY;

pub(crate) fn c_name(name: &OsStr) -> io::Result<CString> {
    CString::new(name.as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid filesystem name"))
}
pub(crate) fn stat_at(directory: &File, name: &OsStr) -> io::Result<libc::stat> {
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
pub(crate) fn descriptor_stat(file: &File) -> io::Result<libc::stat> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: live fd and writable output; success initializes the output.
    if unsafe { libc::fstat(file.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: successful fstat initialized the complete output.
    Ok(unsafe { stat.assume_init() })
}
#[allow(clippy::unnecessary_cast)] // stat integer types vary between Unix targets.
pub(crate) fn identity(stat: &libc::stat) -> (u64, u64) {
    (stat.st_dev as u64, stat.st_ino as u64)
}
pub(crate) fn materialized(stat: &libc::stat) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    crate::provider::ensure_materialized_flags(stat.st_flags)?;
    #[cfg(not(target_os = "macos"))]
    let _ = stat;
    Ok(())
}
pub(crate) fn ensure_kind(stat: &libc::stat, directory: bool) -> io::Result<()> {
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
pub(crate) fn open_at(directory: &File, name: &OsStr, is_dir: bool) -> io::Result<File> {
    let before = stat_at(directory, name)?;
    ensure_kind(&before, is_dir)?;
    #[cfg(test)]
    local_content_tests::opening(directory, name);
    let file = open_raw(
        directory,
        name,
        libc::O_RDONLY | libc::O_NONBLOCK | if is_dir { libc::O_DIRECTORY } else { 0 },
    )?;
    let after = descriptor_stat(&file)?;
    ensure_kind(&after, is_dir)?;
    // A directory open may trigger a Linux automount that stat leaves idle.
    // Adopt the opened directory's identity; callers pin it for their lifetime.
    // Regular files must still be the exact inode checked before opening.
    if !is_dir && identity(&before) != identity(&after) {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "local-content path changed while opening",
        ));
    }
    Ok(file)
}
pub(crate) fn open_directory(path: &Path) -> io::Result<File> {
    if !normalized_absolute(path) {
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

fn normalized_absolute(path: &Path) -> bool {
    path.is_absolute()
        && path
            .components()
            .all(|c| matches!(c, Component::RootDir | Component::Normal(_)))
}

/// Always O_NOFOLLOW and O_CLOEXEC; the caller validates what was opened.
fn open_raw(directory: &File, name: &OsStr, flags: libc::c_int) -> io::Result<File> {
    let name = c_name(name)?;
    // SAFETY: live directory fd and NUL-terminated name; no creation mode needed.
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            flags | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        let error = io::Error::last_os_error();
        return Err(if error.raw_os_error() == Some(libc::ELOOP) {
            link_component()
        } else {
            error
        });
    }
    // SAFETY: successful openat returned a new fd owned only by this File.
    Ok(unsafe { File::from_raw_fd(fd) })
}

pub(crate) fn link_component() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "path has a symbolic link component; local content is opened only through its canonical path",
    )
}

/// What an entry is, without following a link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntryType {
    Dir,
    File,
    Symlink,
    Other,
}

pub(crate) fn entry_type(stat: &libc::stat) -> EntryType {
    match stat.st_mode & libc::S_IFMT {
        libc::S_IFDIR => EntryType::Dir,
        libc::S_IFREG => EntryType::File,
        libc::S_IFLNK => EntryType::Symlink,
        _ => EntryType::Other,
    }
}

/// Same value as `discovery::modified_ns` on `std::fs::Metadata`: whole
/// nanoseconds since the epoch, `None` before it or beyond i64.
#[allow(clippy::unnecessary_cast)] // time_t and the nanosecond field vary between targets.
pub(crate) fn modified_ns(stat: &libc::stat) -> Option<i64> {
    let (seconds, nanos) = (stat.st_mtime as i64, stat.st_mtime_nsec as i64);
    if seconds < 0 || !(0..1_000_000_000).contains(&nanos) {
        return None;
    }
    i64::try_from(i128::from(seconds) * 1_000_000_000 + i128::from(nanos)).ok()
}

/// A directory opened without following links, with the inode it named.
#[derive(Debug)]
pub(crate) struct PinnedDir {
    file: File,
    identity: (u64, u64),
}

pub(crate) struct Entry {
    pub(crate) name: OsString,
    /// `None` when the filesystem does not report a type; stat to find out.
    pub(crate) kind: Option<EntryType>,
}

impl PinnedDir {
    /// Open an absolute, normalized directory path from "/", one no-follow
    /// component at a time. `readable` also allows listing its entries.
    pub(crate) fn open_absolute(path: &Path, readable: bool) -> io::Result<Self> {
        match parent_and_name(path)? {
            None => Self::new(File::open("/")?),
            Some((parent, name)) => parent.open_dir(name, readable),
        }
    }

    /// Open the parent of an absolute, normalized path the same way, for
    /// search only, and return it with the final component.
    pub(crate) fn open_parent(path: &Path) -> io::Result<(Self, &OsStr)> {
        parent_and_name(path)?.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "path has no final component")
        })
    }

    fn new(file: File) -> io::Result<Self> {
        let stat = descriptor_stat(&file)?;
        if entry_type(&stat) != EntryType::Dir {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                "path component is not a directory",
            ));
        }
        Ok(Self {
            file,
            identity: identity(&stat),
        })
    }

    pub(crate) fn identity(&self) -> (u64, u64) {
        self.identity
    }

    /// No-follow stat of a child.
    pub(crate) fn stat(&self, name: &OsStr) -> io::Result<libc::stat> {
        stat_at(&self.file, name)
    }

    /// A child directory; a link (on Linux, O_PATH opens the link itself) or
    /// any other type is refused after the open.
    pub(crate) fn open_dir(&self, name: &OsStr, readable: bool) -> io::Result<Self> {
        let flags = if readable {
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NONBLOCK
        } else {
            SEARCH
        };
        let file = open_raw(&self.file, name, flags).map_err(|error| {
            // With O_DIRECTORY, some systems report a link as ENOTDIR, not ELOOP.
            match stat_at(&self.file, name) {
                Ok(stat) if entry_type(&stat) == EntryType::Symlink => link_component(),
                _ => error,
            }
        })?;
        let stat = descriptor_stat(&file)?;
        match entry_type(&stat) {
            EntryType::Dir => Ok(Self {
                file,
                identity: identity(&stat),
            }),
            EntryType::Symlink => Err(link_component()),
            _ => Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                "path component is not a directory",
            )),
        }
    }

    /// A child opened for reading, never through a link and without blocking
    /// on a FIFO. The caller checks what the descriptor refers to.
    pub(crate) fn open_file(&self, name: &OsStr) -> io::Result<File> {
        open_raw(&self.file, name, libc::O_RDONLY | libc::O_NONBLOCK)
    }

    /// Every entry except "." and "..", read in one pass. Needs a readable handle.
    pub(crate) fn entries(&self) -> io::Result<Vec<Entry>> {
        // fdopendir owns its descriptor; give it a duplicate and rewind, since
        // duplicates share one directory offset.
        let fd = self.file.try_clone()?.into_raw_fd();
        // SAFETY: fd is an open directory descriptor that we own.
        let dir = unsafe { libc::fdopendir(fd) };
        if dir.is_null() {
            let error = io::Error::last_os_error();
            // SAFETY: fdopendir failed, so fd is still ours to close.
            unsafe { libc::close(fd) };
            return Err(error);
        }
        // SAFETY: dir is a live DIR stream.
        unsafe { libc::rewinddir(dir) };
        let mut entries = Vec::new();
        let result = loop {
            clear_errno();
            // SAFETY: dir is a live DIR stream; the returned entry stays valid
            // until the next readdir call on it, and we copy what we need.
            let entry = unsafe { libc::readdir(dir) };
            if entry.is_null() {
                let error = io::Error::last_os_error();
                break match error.raw_os_error() {
                    Some(0) | None => Ok(()),
                    Some(_) => Err(error),
                };
            }
            // Records are variable-length (d_reclen can be shorter than
            // size_of::<dirent>()), so never make a reference to the whole
            // entry; read fields through raw pointers, as std does.
            // SAFETY: entry is live until the next readdir call and d_name is
            // NUL-terminated within the record.
            let name = unsafe {
                CStr::from_ptr(std::ptr::addr_of!((*entry).d_name).cast::<libc::c_char>())
            }
            .to_bytes();
            if name == b"." || name == b".." {
                continue;
            }
            entries.push(Entry {
                name: OsString::from_vec(name.to_vec()),
                // SAFETY: entry is the live record readdir just returned.
                kind: unsafe { d_type(entry) },
            });
        };
        // SAFETY: dir is live and closed exactly once; this also closes fd.
        unsafe { libc::closedir(dir) };
        result.map(|()| entries)
    }
}

fn parent_and_name(path: &Path) -> io::Result<Option<(PinnedDir, &OsStr)>> {
    if !normalized_absolute(path) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "path must be absolute and normalized",
        ));
    }
    let mut names: Vec<&OsStr> = path
        .components()
        .filter_map(|c| match c {
            Component::Normal(name) => Some(name),
            _ => None,
        })
        .collect();
    let Some(last) = names.pop() else {
        return Ok(None);
    };
    let mut directory = PinnedDir::new(File::open("/")?)?;
    for name in names {
        directory = directory.open_dir(name, false)?;
    }
    Ok(Some((directory, last)))
}

/// # Safety
/// `entry` must be a live record returned by readdir.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
))]
unsafe fn d_type(entry: *const libc::dirent) -> Option<EntryType> {
    // SAFETY: per the contract, d_type lies within the live record.
    match unsafe { std::ptr::addr_of!((*entry).d_type).read() } {
        libc::DT_DIR => Some(EntryType::Dir),
        libc::DT_REG => Some(EntryType::File),
        libc::DT_LNK => Some(EntryType::Symlink),
        libc::DT_UNKNOWN => None,
        _ => Some(EntryType::Other),
    }
}
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
)))]
unsafe fn d_type(_: *const libc::dirent) -> Option<EntryType> {
    None
}

// clear_errno needs this thread's errno location. Without it a stale errno
// would make every end of directory look like an error and skip every
// directory, so an unknown Unix target must not compile.
#[cfg(not(any(
    target_os = "linux",
    target_os = "redox",
    target_os = "android",
    target_os = "netbsd",
    target_os = "openbsd",
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "dragonfly"
)))]
compile_error!("pinned_dir: no errno location known for this Unix target");

/// readdir signals errors only through errno, so clear it first.
fn clear_errno() {
    // SAFETY: each returns this thread's errno location.
    #[cfg(any(target_os = "linux", target_os = "redox"))]
    unsafe {
        *libc::__errno_location() = 0
    };
    #[cfg(any(target_os = "android", target_os = "netbsd", target_os = "openbsd"))]
    unsafe {
        *libc::__errno() = 0
    };
    #[cfg(any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "dragonfly"
    ))]
    unsafe {
        *libc::__error() = 0
    };
}

#[cfg(test)]
#[path = "pinned_dir/local_content_tests.rs"]
mod local_content_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    fn canonical_temp() -> (tempfile::TempDir, std::path::PathBuf) {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        (temp, root)
    }

    #[test]
    fn modified_ns_matches_std_metadata() {
        let (_temp, root) = canonical_temp();
        let path = root.join("a.txt");
        std::fs::write(&path, b"x").unwrap();
        let (parent, name) = PinnedDir::open_parent(&path).unwrap();
        let stat = parent.stat(name).unwrap();
        let meta = std::fs::symlink_metadata(&path).unwrap();
        assert_eq!(modified_ns(&stat), crate::discovery::modified_ns(&meta));
        assert_eq!(stat.st_size as u64, meta.len());
        assert_eq!(identity(&stat), (meta.dev(), meta.ino()));
    }

    #[test]
    fn entries_list_names_and_types_without_dot_entries() {
        let (_temp, root) = canonical_temp();
        std::fs::create_dir(root.join("d")).unwrap();
        std::fs::write(root.join("f"), b"").unwrap();
        std::os::unix::fs::symlink(root.join("f"), root.join("l")).unwrap();
        let dir = PinnedDir::open_absolute(&root, true).unwrap();
        let mut seen: Vec<_> = dir
            .entries()
            .unwrap()
            .into_iter()
            .map(|e| {
                let kind = e
                    .kind
                    .unwrap_or_else(|| entry_type(&dir.stat(&e.name).unwrap()));
                (e.name, kind)
            })
            .collect();
        seen.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            seen,
            [
                ("d".into(), EntryType::Dir),
                ("f".into(), EntryType::File),
                ("l".into(), EntryType::Symlink)
            ]
        );
        // A second listing of the same handle starts from the beginning.
        assert_eq!(dir.entries().unwrap().len(), 3);
    }

    #[test]
    fn link_components_are_refused_with_a_clear_error() {
        let (_temp, root) = canonical_temp();
        std::fs::create_dir(root.join("real")).unwrap();
        std::fs::write(root.join("real/f.txt"), b"x").unwrap();
        std::os::unix::fs::symlink(root.join("real"), root.join("alias")).unwrap();
        for readable in [false, true] {
            let error = PinnedDir::open_absolute(&root.join("alias"), readable).unwrap_err();
            assert!(error.to_string().contains("symbolic link"), "{error}");
        }
        let error = PinnedDir::open_parent(&root.join("alias/f.txt")).unwrap_err();
        assert!(error.to_string().contains("symbolic link"), "{error}");
        assert!(PinnedDir::open_parent(&root.join("real/../real/f.txt")).is_err());
        assert!(PinnedDir::open_parent(Path::new("relative/f.txt")).is_err());
    }

    #[test]
    fn search_only_ancestors_need_no_read_permission() {
        let (_temp, root) = canonical_temp();
        let locked = root.join("locked");
        std::fs::create_dir(&locked).unwrap();
        std::fs::write(locked.join("f.txt"), b"x").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o311)).unwrap();
        let opened = PinnedDir::open_parent(&locked.join("f.txt"))
            .and_then(|(parent, name)| parent.open_file(name));
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(opened.is_ok(), "{opened:?}");
    }
}
