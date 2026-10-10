//! Publish immutable model files without ever writing through a shared hard link.
//!
//! Reuse reads metadata and a bounded copy-verification stamp, never weights.
//! Malformed stamps or changed stats cause safe republishing. No stamp field
//! selects a path, and no existing target or stamp is ever opened for writing.
//! A genuine copy receipt records that size and BLAKE3 matched the immutable
//! source at publication; reuse establishes only that the recorded stats match.
//! It is not authenticated and does not prove current content against forged
//! receipts or corruption preserving those stats. User-owned 0700 model/cache
//! directories and 0600 receipts are the trust boundary. Anyone able to forge a
//! receipt there can also replace the weights. Same-user malware is outside
//! Seyk's threat model (todo 4334); the receipt detects accidental stale/partial
//! publications and stat-changing corruption, not that attacker.
use super::{Result, io_error};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::{Duration, SystemTime},
};

const TEMP_PREFIX: &str = ".seyk-materialize-";
const MAX_SWEEP_ENTRIES: usize = 256;
const MAX_SWEEP_DIRECTORIES: usize = 1024;
const STALE_AGE: Duration = Duration::from_secs(24 * 60 * 60);

#[cfg(test)]
std::thread_local! {
    static DIGEST_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn temp_prefix() -> String {
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{TEMP_PREFIX}{created}-")
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Stat {
    len: u64,
    mtime_ns: u128,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(windows)]
    identity: Vec<u8>,
}
impl Stat {
    fn read(path: &Path, meta: &fs::Metadata) -> Result<Self> {
        let mtime_ns = meta
            .modified()
            .map_err(|e| io_error(path, e))?
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| invalid(path, e.to_string()))?
            .as_nanos();
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        #[cfg(windows)]
        let identity = {
            use std::hash::{Hash, Hasher};
            // Record the exact volume-serial/file-index key bytes, not a lossy
            // hash. A changed key representation invalidates the stamp.
            #[derive(Default)]
            struct KeyBytes(Vec<u8>);
            impl Hasher for KeyBytes {
                fn finish(&self) -> u64 {
                    0
                }
                fn write(&mut self, bytes: &[u8]) {
                    self.0.extend_from_slice(bytes);
                }
            }
            let handle = same_file::Handle::from_path(path).map_err(|e| io_error(path, e))?;
            let mut key = KeyBytes::default();
            handle.hash(&mut key);
            key.0
        };
        Ok(Self {
            len: meta.len(),
            mtime_ns,
            #[cfg(unix)]
            dev: meta.dev(),
            #[cfg(unix)]
            ino: meta.ino(),
            #[cfg(windows)]
            identity,
        })
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stamp {
    version: u8,
    source: Stat,
    target: Stat,
    digest: String,
}
fn stamp_path(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_owned();
    name.push(".seyk-verified.json");
    name.into()
}
fn invalid(path: &Path, message: impl Into<String>) -> super::Error {
    io_error(
        path,
        io::Error::new(io::ErrorKind::InvalidData, message.into()),
    )
}
fn digest(path: &Path) -> Result<blake3::Hash> {
    #[cfg(test)]
    DIGEST_CALLS.with(|calls| calls.set(calls.get() + 1));
    let mut file = fs::File::open(path).map_err(|e| io_error(path, e))?;
    let mut hasher = blake3::Hasher::new();
    hasher
        .update_reader(&mut file)
        .map_err(|e| io_error(path, e))?;
    Ok(hasher.finalize())
}
fn matches(source: &Path, target: &Path) -> Result<bool> {
    let source_meta = fs::metadata(source).map_err(|e| io_error(source, e))?;
    let target_meta = match fs::symlink_metadata(target) {
        Ok(meta) => meta,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(io_error(target, e)),
    };
    if !source_meta.is_file() || !target_meta.is_file() || target_meta.len() != source_meta.len() {
        return Ok(false);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if source_meta.dev() == target_meta.dev() && source_meta.ino() == target_meta.ino() {
            return Ok(true);
        }
    }
    #[cfg(windows)]
    if same_file::is_same_file(source, target).map_err(|e| io_error(target, e))? {
        return Ok(true);
    }
    let stamp_path = stamp_path(target);
    // Symlink, oversized, unreadable and malformed stamps are cache misses.
    let stamp = (|| -> Option<Stamp> {
        let meta = fs::symlink_metadata(&stamp_path).ok()?;
        if !meta.is_file() || meta.len() > 4096 {
            return None;
        }
        let mut bytes = Vec::new();
        fs::File::open(&stamp_path)
            .ok()?
            .take(4097)
            .read_to_end(&mut bytes)
            .ok()?;
        serde_json::from_slice(&bytes).ok()
    })();
    let Some(stamp) = stamp else { return Ok(false) };
    Ok(stamp.version == 1
        && blake3::Hash::from_hex(&stamp.digest).is_ok()
        && stamp.source == Stat::read(source, &source_meta)?
        && stamp.target == Stat::read(target, &target_meta)?)
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Point {
    SourceDigest,
    CopyChunk,
    Verify,
    StampWrite,
    StampRename,
    TargetRename,
}
// Model-free fault/counting seams; production uses the filesystem directly.
trait Operations {
    fn link(&self, source: &Path, target: &Path) -> io::Result<()> {
        fs::hard_link(source, target)
    }
    fn digest(&self, path: &Path) -> Result<blake3::Hash> {
        digest(path)
    }
    fn persist(&self, staged: tempfile::TempPath, target: &Path) -> io::Result<()> {
        staged.persist(target).map_err(|e| e.error)
    }
    fn check(&self, _point: Point) -> io::Result<()> {
        Ok(())
    }
}
struct Filesystem;
impl Operations for Filesystem {}
fn check(ops: &impl Operations, point: Point, path: &Path) -> Result<()> {
    ops.check(point).map_err(|e| io_error(path, e))
}
fn copy_required(error: &io::Error) -> bool {
    if error.kind() == io::ErrorKind::CrossesDevices {
        return true;
    }
    #[cfg(unix)]
    {
        error.raw_os_error() == Some(1)
    } // EPERM: filesystem disallows links.
    #[cfg(not(unix))]
    {
        error.kind() == io::ErrorKind::PermissionDenied
    }
}
fn sweep_stale_once(parent: &Path) {
    static SWEPT: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    let mut directories = SWEPT
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if directories.len() >= MAX_SWEEP_DIRECTORIES || !directories.insert(parent.to_owned()) {
        return;
    }
    drop(directories);
    // Best effort, once per process/directory, with bounded work and tracking.
    // A later process can clean entries beyond this process's scan budget.
    let _ = sweep_stale(parent);
}
fn sweep_stale(parent: &Path) -> Result<()> {
    for entry in fs::read_dir(parent)
        .map_err(|e| io_error(parent, e))?
        .take(MAX_SWEEP_ENTRIES)
    {
        let entry = entry.map_err(|e| io_error(parent, e))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let created = name
            .strip_prefix(TEMP_PREFIX)
            .and_then(|tail| tail.split_once('-'))
            .and_then(|(seconds, _)| seconds.parse::<u64>().ok())
            .and_then(|seconds| SystemTime::UNIX_EPOCH.checked_add(Duration::from_secs(seconds)));
        // Unlink only our old staging entries, without following symlinks.
        // Recent concurrent publications are left alone. Windows may deny
        // deletion of an open file, so cleanup is best effort.
        // Hard links inherit the source mtime: use creation time in the name
        // so a newly staged link to old weights cannot be swept as stale.
        if let Some(created) = created
            && SystemTime::now()
                .duration_since(created)
                .is_ok_and(|age| age > STALE_AGE)
        {
            let _ = fs::remove_file(entry.path());
        }
    }
    Ok(())
}
pub(super) fn create_private_dir(path: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path).map_err(|e| io_error(path, e))?;
    if !fs::symlink_metadata(path)
        .map_err(|e| io_error(path, e))?
        .is_dir()
    {
        return Err(invalid(
            path,
            "model cache path must be a directory, not a symlink",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Tighten pre-existing materialization directories too. Set permissions
        // on the directory handle, never on model hard links or HF sources.
        fs::File::open(path)
            .and_then(|dir| dir.set_permissions(fs::Permissions::from_mode(0o700)))
            .map_err(|e| io_error(path, e))?;
    }
    Ok(())
}
fn staging_file(parent: &Path) -> Result<tempfile::NamedTempFile> {
    let staged = tempfile::Builder::new()
        .prefix(&temp_prefix())
        .tempfile_in(parent)
        .map_err(|e| io_error(parent, e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        staged
            .as_file()
            .set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|e| io_error(staged.path(), e))?;
    }
    Ok(staged)
}
pub(super) fn publish(source: &Path, target: &Path) -> Result<()> {
    publish_with(source, target, || {}, &Filesystem)
}
#[cfg(test)]
fn publish_after_check(source: &Path, target: &Path, before_publish: impl FnOnce()) -> Result<()> {
    publish_with(source, target, before_publish, &Filesystem)
}
fn publish_with(
    source: &Path,
    target: &Path,
    before_publish: impl FnOnce(),
    ops: &impl Operations,
) -> Result<()> {
    let parent = target.parent().ok_or_else(|| {
        io_error(
            target,
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "model destination has no parent directory",
            ),
        )
    })?;
    sweep_stale_once(parent);
    if matches(source, target)? {
        return Ok(());
    }
    before_publish();
    let link_error = match ops.link(source, target) {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };
    // A competitor may have installed a correct entry after our cache miss.
    // Recheck before replacing its shared inode or hitting Windows AccessDenied
    // when ORT already holds the correct file open.
    if link_error.kind() == io::ErrorKind::AlreadyExists {
        if matches(source, target)? {
            return Ok(());
        }
        let staged = tempfile::Builder::new()
            .prefix(&temp_prefix())
            .make_in(parent, |path| ops.link(source, path));
        match staged {
            Ok(staged) => {
                check(ops, Point::TargetRename, target)?;
                // POSIX rename of two links to the same inode succeeds without
                // removing the old name. Keep the guard alive to unlink that
                // staging name even when a competing repair won the race.
                fs::rename(staged.path(), target).map_err(|e| io_error(target, e))?;
                return Ok(());
            }
            Err(e) if copy_required(&e) => {}
            Err(e) => return Err(io_error(target, e)),
        }
    } else if !copy_required(&link_error) {
        return Err(io_error(target, link_error));
    }
    publish_copy(source, target, parent, ops)
}
fn publish_copy(source: &Path, target: &Path, parent: &Path, ops: &impl Operations) -> Result<()> {
    let source_stat = Stat::read(
        source,
        &fs::metadata(source).map_err(|e| io_error(source, e))?,
    )?;
    check(ops, Point::SourceDigest, source)?;
    let expected_hash = ops.digest(source)?;
    let mut staged = staging_file(parent)?;
    let mut input = fs::File::open(source).map_err(|e| io_error(source, e))?;
    let mut hash = blake3::Hasher::new();
    let mut copied = 0;
    let mut buffer = [0; 64 * 1024];
    loop {
        let n = input.read(&mut buffer).map_err(|e| io_error(source, e))?;
        if n == 0 {
            break;
        }
        staged
            .write_all(&buffer[..n])
            .map_err(|e| io_error(staged.path(), e))?;
        hash.update(&buffer[..n]);
        copied += n as u64;
        check(ops, Point::CopyChunk, staged.path())?;
    }
    staged.flush().map_err(|e| io_error(staged.path(), e))?;
    check(ops, Point::Verify, staged.path())?;
    if copied != source_stat.len
        || hash.finalize() != expected_hash
        || Stat::read(
            source,
            &fs::metadata(source).map_err(|e| io_error(source, e))?,
        )? != source_stat
    {
        return Err(invalid(source, "model file changed while materializing"));
    }
    // Windows may finalize last-write time only when the writer closes. Capture
    // metadata after closing it so a verified copy's first reuse is stat-only.
    let staged = staged.into_temp_path();
    let staged_meta = fs::metadata(&staged).map_err(|e| io_error(&*staged, e))?;
    if staged_meta.len() != source_stat.len {
        return Err(invalid(
            source,
            "staged model length changed while materializing",
        ));
    }
    let stamp = Stamp {
        version: 1,
        source: source_stat,
        target: Stat::read(&staged, &staged_meta)?,
        digest: expected_hash.to_hex().to_string(),
    };
    let mut staged_stamp = staging_file(parent)?;
    check(ops, Point::StampWrite, staged_stamp.path())?;
    serde_json::to_writer(&mut staged_stamp, &stamp)
        .map_err(|e| invalid(staged_stamp.path(), e.to_string()))?;
    staged_stamp
        .flush()
        .map_err(|e| io_error(staged_stamp.path(), e))?;
    // Publish stamp first. A crash, error or race between these renames makes
    // the stats disagree, causing safe republishing on the next load.
    check(ops, Point::StampRename, target)?;
    ops.persist(staged_stamp.into_temp_path(), &stamp_path(target))
        .map_err(|e| io_error(target, e))?;
    check(ops, Point::TargetRename, target)?;
    ops.persist(staged, target)
        .map_err(|e| io_error(target, e))?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Default)]
    struct TestOps {
        copy_only: bool,
        digests: AtomicUsize,
        chunks: AtomicUsize,
        fail: Option<Point>,
    }
    impl Operations for TestOps {
        fn link(&self, source: &Path, target: &Path) -> io::Result<()> {
            if self.copy_only {
                Err(io::ErrorKind::CrossesDevices.into())
            } else {
                fs::hard_link(source, target)
            }
        }
        fn digest(&self, path: &Path) -> Result<blake3::Hash> {
            self.digests.fetch_add(1, Ordering::SeqCst);
            digest(path)
        }
        fn check(&self, point: Point) -> io::Result<()> {
            if point == Point::CopyChunk {
                self.chunks.fetch_add(1, Ordering::SeqCst);
            }
            if self.fail == Some(point) {
                Err(io::Error::other("injected materializer failure"))
            } else {
                Ok(())
            }
        }
    }
    fn no_staging_files(dir: &Path) {
        assert!(fs::read_dir(dir).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(TEMP_PREFIX)
        }));
    }

    #[test]
    fn linked_and_verified_copy_reuse_never_read_weights() {
        for copy_only in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let source = dir.path().join("source");
            let target = dir.path().join("weights");
            fs::write(&source, vec![42; 150_000]).unwrap();
            let ops = TestOps {
                copy_only,
                ..Default::default()
            };
            publish_with(&source, &target, || {}, &ops).unwrap();
            assert_eq!(fs::read(&target).unwrap(), fs::read(&source).unwrap());
            if !copy_only {
                assert_shared_inode(&source, &target);
            }
            let digests = ops.digests.load(Ordering::SeqCst);
            let chunks = ops.chunks.load(Ordering::SeqCst);
            let all_digests = DIGEST_CALLS.with(|calls| calls.get());
            assert_eq!(digests, usize::from(copy_only));
            for _ in 0..50 {
                publish_with(&source, &target, || {}, &ops).unwrap();
            }
            assert_eq!(ops.digests.load(Ordering::SeqCst), digests);
            assert_eq!(ops.chunks.load(Ordering::SeqCst), chunks);
            assert_eq!(DIGEST_CALLS.with(|calls| calls.get()), all_digests);
            no_staging_files(dir.path());
        }
    }

    #[test]
    fn changed_stats_and_untrusted_stamps_trigger_safe_republication() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let target = dir.path().join("weights");
        let old = dir.path().join("old-source");
        fs::write(&source, b"correct").unwrap();
        let ops = TestOps {
            copy_only: true,
            ..Default::default()
        };
        publish_with(&source, &target, || {}, &ops).unwrap();
        let stale_stamp = fs::read(stamp_path(&target)).unwrap();
        // Replace the target with a same-size hard link to another immutable
        // fixture. A stamp miss must never write through that link.
        fs::write(&old, b"corrupt").unwrap();
        fs::remove_file(&target).unwrap();
        fs::hard_link(&old, &target).unwrap();
        assert!(!matches(&source, &target).unwrap());
        publish_with(&source, &target, || {}, &ops).unwrap();
        assert_eq!(fs::read(&old).unwrap(), b"corrupt");
        assert_eq!(fs::read(&target).unwrap(), b"correct");
        // A source mtime change invalidates reuse even if its length is stable.
        fs::OpenOptions::new()
            .write(true)
            .open(&source)
            .unwrap()
            .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(100))
            .unwrap();
        assert!(!matches(&source, &target).unwrap());
        publish_with(&source, &target, || {}, &ops).unwrap();
        for stamp in [stale_stamp, b"not json".to_vec(), vec![b'x'; 4097]] {
            fs::remove_file(stamp_path(&target)).unwrap();
            fs::write(stamp_path(&target), stamp).unwrap();
            assert!(!matches(&source, &target).unwrap());
            publish_with(&source, &target, || {}, &ops).unwrap();
            assert_eq!(fs::read(&source).unwrap(), b"correct");
            assert_eq!(fs::read(&target).unwrap(), b"correct");
        }
        no_staging_files(dir.path());
    }

    #[test]
    fn copy_failures_preserve_source_and_old_target_and_clean_staging() {
        for fail in [
            Point::SourceDigest,
            Point::CopyChunk,
            Point::Verify,
            Point::StampWrite,
            Point::StampRename,
            Point::TargetRename,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let source = dir.path().join("source");
            let old = dir.path().join("old-source");
            let target = dir.path().join("weights");
            let bytes = vec![42; 150_000];
            fs::write(&source, &bytes).unwrap();
            fs::write(&old, b"old correct destination").unwrap();
            fs::hard_link(&old, &target).unwrap();
            let ops = TestOps {
                copy_only: true,
                fail: Some(fail),
                ..Default::default()
            };
            assert!(matches!(
                publish_with(&source, &target, || {}, &ops),
                Err(super::super::Error::Io { .. })
            ));
            assert_eq!(fs::read(&source).unwrap(), bytes, "{fail:?}");
            assert_eq!(
                fs::read(&target).unwrap(),
                b"old correct destination",
                "{fail:?}"
            );
            assert_eq!(
                fs::read(&old).unwrap(),
                b"old correct destination",
                "{fail:?}"
            );
            no_staging_files(dir.path());
        }
    }

    struct RejectPersist {
        target: PathBuf,
        attempts: AtomicUsize,
    }
    impl Operations for RejectPersist {
        fn link(&self, _: &Path, _: &Path) -> io::Result<()> {
            Err(io::ErrorKind::CrossesDevices.into())
        }
        fn persist(&self, staged: tempfile::TempPath, target: &Path) -> io::Result<()> {
            if target == self.target {
                self.attempts.fetch_add(1, Ordering::SeqCst);
                // Fail at the publication operation, after verification and
                // stamp publication. The owned staging guard must clean up.
                Err(io::ErrorKind::PermissionDenied.into())
            } else {
                Filesystem.persist(staged, target)
            }
        }
    }
    fn rejected_copy_preserves_files(symlink_target: bool) {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let old = dir.path().join("old-source");
        let target = dir.path().join("weights");
        let bytes = vec![42; 150_000];
        fs::write(&source, &bytes).unwrap();
        fs::write(&old, b"old correct destination").unwrap();
        if symlink_target {
            #[cfg(unix)]
            std::os::unix::fs::symlink(&old, &target).unwrap();
            #[cfg(not(unix))]
            unreachable!("symlink rejection test is Unix-only");
        } else {
            fs::hard_link(&old, &target).unwrap();
        }
        let source_hash = digest(&source).unwrap();
        let old_hash = digest(&old).unwrap();
        let ops = RejectPersist {
            target: target.clone(),
            attempts: AtomicUsize::new(0),
        };
        let error = publish_with(&source, &target, || {}, &ops).unwrap_err();
        assert!(matches!(error, super::super::Error::Io { source, .. }
            if source.kind() == io::ErrorKind::PermissionDenied));
        assert_eq!(ops.attempts.load(Ordering::SeqCst), 1);
        assert_eq!(fs::read(&source).unwrap(), bytes);
        assert_eq!(digest(&source).unwrap(), source_hash);
        assert_eq!(fs::read(&target).unwrap(), b"old correct destination");
        assert_eq!(fs::read(&old).unwrap(), b"old correct destination");
        assert_eq!(digest(&old).unwrap(), old_hash);
        if symlink_target {
            assert!(
                fs::symlink_metadata(&target)
                    .unwrap()
                    .file_type()
                    .is_symlink()
            );
            assert_eq!(fs::read_link(&target).unwrap(), old);
        } else {
            assert_shared_inode(&old, &target);
        }
        no_staging_files(dir.path());
    }
    #[test]
    fn failed_copy_persist_preserves_existing_hard_link_and_cleans_staging() {
        rejected_copy_preserves_files(false);
    }
    #[cfg(unix)]
    #[test]
    fn rejected_symlink_replacement_preserves_link_and_source_and_cleans_staging() {
        rejected_copy_preserves_files(true);
    }

    #[test]
    fn failed_link_repair_keeps_existing_target_and_cleans_staging() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let target = dir.path().join("weights");
        fs::write(&source, b"correct").unwrap();
        fs::write(&target, b"old").unwrap();
        let ops = TestOps {
            fail: Some(Point::TargetRename),
            ..Default::default()
        };
        assert!(publish_with(&source, &target, || {}, &ops).is_err());
        assert_eq!(fs::read(&source).unwrap(), b"correct");
        assert_eq!(fs::read(&target).unwrap(), b"old");
        no_staging_files(dir.path());
        // Real rename failure, not only an injected preflight failure.
        fs::remove_file(&target).unwrap();
        fs::create_dir(&target).unwrap();
        assert!(publish(&source, &target).is_err());
        assert!(target.is_dir());
        assert_eq!(fs::read(&source).unwrap(), b"correct");
        no_staging_files(dir.path());
    }

    #[test]
    fn digest_mismatch_is_an_integrity_io_error_and_never_publishes() {
        struct WrongDigest;
        impl Operations for WrongDigest {
            fn link(&self, _: &Path, _: &Path) -> io::Result<()> {
                Err(io::ErrorKind::CrossesDevices.into())
            }
            fn digest(&self, _: &Path) -> Result<blake3::Hash> {
                Ok(blake3::hash(b"a different verified source"))
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let target = dir.path().join("weights");
        fs::write(&source, b"correct").unwrap();
        fs::write(&target, b"old").unwrap();
        let error = publish_with(&source, &target, || {}, &WrongDigest).unwrap_err();
        assert!(matches!(error, super::super::Error::Io { source, .. }
            if source.kind() == io::ErrorKind::InvalidData));
        assert_eq!(fs::read(&source).unwrap(), b"correct");
        assert_eq!(fs::read(&target).unwrap(), b"old");
        no_staging_files(dir.path());
    }

    #[test]
    fn unexpected_link_errors_do_not_fall_back_to_copy() {
        struct BadLink;
        impl Operations for BadLink {
            fn link(&self, _: &Path, _: &Path) -> io::Result<()> {
                Err(io::Error::other("unexpected link failure"))
            }
            fn digest(&self, _: &Path) -> Result<blake3::Hash> {
                panic!("unexpected link errors must not hash or copy")
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let target = dir.path().join("weights");
        fs::write(&source, b"correct").unwrap();
        assert!(publish_with(&source, &target, || {}, &BadLink).is_err());
        assert!(!target.exists());
        assert_eq!(fs::read(&source).unwrap(), b"correct");
        no_staging_files(dir.path());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_destinations_and_stamps_are_never_followed_for_writes() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let old = dir.path().join("old-source");
        let target = dir.path().join("weights");
        fs::write(&source, b"correct").unwrap();
        fs::write(&old, b"corrupt").unwrap();
        for copy_only in [false, true] {
            if target.exists() {
                fs::remove_file(&target).unwrap();
            }
            symlink(&old, &target).unwrap();
            symlink(&old, stamp_path(&target)).unwrap();
            let ops = TestOps {
                copy_only,
                ..Default::default()
            };
            assert!(!matches(&source, &target).unwrap());
            publish_with(&source, &target, || {}, &ops).unwrap();
            assert_eq!(fs::read(&old).unwrap(), b"corrupt");
            assert_eq!(fs::read(&source).unwrap(), b"correct");
            assert_eq!(fs::read(&target).unwrap(), b"correct");
            fs::remove_file(stamp_path(&target)).unwrap();
            no_staging_files(dir.path());
        }
    }

    #[cfg(unix)]
    #[test]
    fn model_directories_and_copy_receipts_are_private_without_chmodding_sources() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("models/key");
        create_private_dir(&cache).unwrap();
        for path in [&cache, &cache.parent().unwrap().to_path_buf()] {
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
        fs::set_permissions(&cache, fs::Permissions::from_mode(0o755)).unwrap();
        create_private_dir(&cache).unwrap();
        assert_eq!(
            fs::metadata(&cache).unwrap().permissions().mode() & 0o777,
            0o700
        );
        let source = dir.path().join("source");
        fs::write(&source, b"correct").unwrap();
        fs::set_permissions(&source, fs::Permissions::from_mode(0o644)).unwrap();
        let target = cache.join("copy");
        let ops = TestOps {
            copy_only: true,
            ..Default::default()
        };
        publish_with(&source, &target, || {}, &ops).unwrap();
        for path in [&target, &stamp_path(&target)] {
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let linked = cache.join("linked");
        publish(&source, &linked).unwrap();
        assert_shared_inode(&source, &linked);
        assert_eq!(
            fs::metadata(&source).unwrap().permissions().mode() & 0o777,
            0o644
        );
        assert_eq!(fs::read(&source).unwrap(), b"correct");
    }
    #[test]
    fn reuse_sweeps_abandoned_staging_once_without_reading_weights() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let target = dir.path().join("weights");
        fs::write(&source, b"correct").unwrap();
        fs::hard_link(&source, &target).unwrap();
        let old = dir.path().join(format!("{TEMP_PREFIX}1-abandoned"));
        let recent = dir.path().join(format!("{}active", temp_prefix()));
        fs::write(&old, b"old staging bytes").unwrap();
        fs::write(&recent, b"active staging bytes").unwrap();
        let before = DIGEST_CALLS.with(|calls| calls.get());
        publish(&source, &target).unwrap();
        assert!(!old.exists());
        assert!(recent.exists());
        fs::write(&old, b"newly abandoned").unwrap();
        publish(&source, &target).unwrap();
        assert!(old.exists(), "reuse must not repeatedly scan the directory");
        assert_eq!(DIGEST_CALLS.with(|calls| calls.get()), before);
        assert_eq!(fs::read(&source).unwrap(), b"correct");
        assert_shared_inode(&source, &target);
    }
    #[test]
    fn stale_sweep_has_a_bounded_entry_budget() {
        let dir = tempfile::tempdir().unwrap();
        for index in 0..MAX_SWEEP_ENTRIES + 8 {
            fs::write(
                dir.path().join(format!("{TEMP_PREFIX}1-{index}")),
                b"staging",
            )
            .unwrap();
        }
        sweep_stale(dir.path()).unwrap();
        let remaining = fs::read_dir(dir.path()).unwrap().count();
        assert!(remaining >= 8, "scan must visit at most its entry budget");
        assert!(remaining < MAX_SWEEP_ENTRIES + 8);
    }

    #[test]
    fn stale_sweep_uses_creation_names_not_inherited_link_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let target = dir.path().join("weights");
        fs::write(&source, b"correct").unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&source)
            .unwrap()
            .set_modified(SystemTime::UNIX_EPOCH)
            .unwrap();
        let old = dir.path().join(format!("{TEMP_PREFIX}1-abandoned"));
        let recent = dir.path().join(format!("{}active", temp_prefix()));
        let unrelated = dir.path().join("unrelated");
        fs::hard_link(&source, &old).unwrap();
        fs::hard_link(&source, &recent).unwrap();
        fs::write(&unrelated, b"unrelated").unwrap();
        publish(&source, &target).unwrap();
        assert!(!old.exists());
        assert!(recent.exists());
        assert_eq!(fs::read(&source).unwrap(), b"correct");
        assert!(unrelated.exists());
    }

    fn assert_shared_inode(source: &Path, target: &Path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let source = fs::metadata(source).unwrap();
            let target = fs::metadata(target).unwrap();
            assert_eq!((source.dev(), source.ino()), (target.dev(), target.ino()));
        }
        #[cfg(windows)]
        assert!(same_file::is_same_file(source, target).unwrap());
    }

    #[test]
    fn competing_materializers_preserve_source_with_existing_hard_link() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let target = dir.path().join("weights");
        let bytes = b"complete disposable model weights";
        fs::write(&source, bytes).unwrap();
        let hash = digest(&source).unwrap();
        let checked = Barrier::new(3);
        let release = Barrier::new(3);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        publish_after_check(&source, &target, || {
                            checked.wait();
                            release.wait();
                        })
                        .unwrap();
                    })
                })
                .collect();
            checked.wait();
            // Both writers saw a miss. Install the shared inode before either
            // hard_link attempt, deterministically triggering AlreadyExists.
            fs::hard_link(&source, &target).unwrap();
            release.wait();
            for worker in workers {
                worker.join().unwrap();
            }
        });
        assert_eq!(fs::read(&source).unwrap(), bytes);
        assert_eq!(digest(&source).unwrap(), hash);
        assert_eq!(fs::read(&target).unwrap(), bytes);
        assert_eq!(digest(&target).unwrap(), hash);
        assert_shared_inode(&source, &target);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
    }

    #[test]
    fn concurrent_repairs_replace_same_size_corrupt_hard_link() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let stale_source = dir.path().join("stale-source");
        let target = dir.path().join("weights");
        fs::write(&source, b"correct weights").unwrap();
        fs::write(&stale_source, b"corrupt weights").unwrap();
        fs::hard_link(&stale_source, &target).unwrap();
        let hash = digest(&source).unwrap();
        let stale_hash = digest(&stale_source).unwrap();
        let checked = Barrier::new(2);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        publish_after_check(&source, &target, || {
                            checked.wait();
                        })
                        .unwrap();
                    })
                })
                .collect();
            for worker in workers {
                worker.join().unwrap();
            }
        });
        assert_eq!(fs::read(&source).unwrap(), b"correct weights");
        assert_eq!(digest(&source).unwrap(), hash);
        assert_eq!(fs::read(&target).unwrap(), b"correct weights");
        assert_eq!(digest(&target).unwrap(), hash);
        assert_eq!(fs::read(&stale_source).unwrap(), b"corrupt weights");
        assert_eq!(digest(&stale_source).unwrap(), stale_hash);
        assert_shared_inode(&source, &target);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 3);
    }

    #[test]
    fn reuse_checks_content_or_inode_and_repairs_partial_files() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let target = dir.path().join("weights");
        fs::write(&source, b"complete").unwrap();
        fs::hard_link(&source, &target).unwrap();
        assert!(matches(&source, &target).unwrap());
        publish(&source, &target).unwrap();
        fs::remove_file(&target).unwrap();
        for bytes in [b"corrupt!".as_slice(), b"partial".as_slice()] {
            if target.exists() {
                fs::remove_file(&target).unwrap();
            }
            fs::write(&target, bytes).unwrap();
            assert!(!matches(&source, &target).unwrap());
            publish(&source, &target).unwrap();
            assert_eq!(fs::read(&target).unwrap(), b"complete");
            assert_eq!(fs::read(&source).unwrap(), b"complete");
        }
        assert!(matches(&source, &target).unwrap());
        publish(&source, &target).unwrap();
    }
}
