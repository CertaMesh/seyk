//! Publish immutable model files without ever writing through a shared hard link.
use super::{Error, Result, io_error};
use std::{fs, path::Path};

pub(super) fn publish(source: &Path, target: &Path) -> Result<()> {
    publish_after_check(source, target, || {})
}

fn digest(path: &Path) -> Result<blake3::Hash> {
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
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(io_error(target, e)),
    };
    if !target_meta.is_file() || target_meta.len() != source_meta.len() {
        return Ok(false);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if source_meta.dev() == target_meta.dev() && source_meta.ino() == target_meta.ino() {
            return Ok(true);
        }
    }
    Ok(digest(source)? == digest(target)?)
}

// The hook lets tests force competing publishers past the same cache miss.
fn publish_after_check(source: &Path, target: &Path, before_publish: impl FnOnce()) -> Result<()> {
    if matches(source, target)? {
        return Ok(());
    }
    before_publish();
    // hard_link has no-clobber semantics. In particular, AlreadyExists must
    // NEVER fall back to copying onto target: it may now share source's inode.
    if fs::hard_link(source, target).is_ok() {
        return Ok(());
    }
    let expected_size = fs::metadata(source).map_err(|e| io_error(source, e))?.len();
    let expected_hash = digest(source)?;
    let parent = target
        .parent()
        .ok_or_else(|| Error::Config("model destination has no parent directory".into()))?;
    let mut staged = tempfile::NamedTempFile::new_in(parent).map_err(|e| io_error(parent, e))?;
    let mut input = fs::File::open(source).map_err(|e| io_error(source, e))?;
    let copied =
        std::io::copy(&mut input, staged.as_file_mut()).map_err(|e| io_error(staged.path(), e))?;
    if copied != expected_size || digest(staged.path())? != expected_hash {
        return Err(Error::Config(format!(
            "model file changed while materializing {}",
            source.display()
        )));
    }
    // Same-directory persist atomically replaces the directory entry, including
    // an existing hard link or symlink, without opening that entry for writing.
    staged
        .persist(target)
        .map_err(|e| io_error(target, e.error))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

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
