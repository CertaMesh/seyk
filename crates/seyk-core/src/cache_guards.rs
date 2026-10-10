//! Best-effort backup and sync safeguards; neither substitutes for encryption.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::{Error, Result};

const TAG: &str = "Signature: 8a477f597d28d172789f06886806bc55\n# Seyk cache: disposable derived data.\n# https://bford.info/cachedir/\n";
static SYNC_WARNED: AtomicBool = AtomicBool::new(false);
static TAG_ID: AtomicU64 = AtomicU64::new(0);

pub(crate) fn prepare(dir: &Path) -> Result<()> {
    use std::io::Write;
    let tag = dir.join("CACHEDIR.TAG");
    let temporary = dir.join(format!(
        ".CACHEDIR.TAG-{}-{}",
        std::process::id(),
        TAG_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    // Publish a complete tag atomically without replacing an existing file.
    // Concurrent cache opens must never observe a partially written signature.
    let mut file = options
        .open(&temporary)
        .map_err(|e| Error::io(&temporary, e))?;
    let result = (|| {
        file.write_all(TAG.as_bytes())
            .map_err(|e| Error::io(&temporary, e))?;
        match std::fs::hard_link(&temporary, &tag) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if !std::fs::symlink_metadata(&tag)
                    .map_err(|e| Error::io(&tag, e))?
                    .is_file()
                {
                    return Err(Error::Config(format!(
                        "cache tag is not a regular file: {}",
                        tag.display()
                    )));
                }
                let contents = std::fs::read(&tag).map_err(|e| Error::io(&tag, e))?;
                if !contents.starts_with(b"Signature: 8a477f597d28d172789f06886806bc55") {
                    return Err(Error::Config(format!(
                        "invalid cache tag: {}",
                        tag.display()
                    )));
                }
                Ok(())
            }
            Err(e) => Err(Error::io(&tag, e)),
        }
    })();
    drop(file);
    let _ = std::fs::remove_file(&temporary);
    result?;
    #[cfg(target_os = "macos")]
    if !exclude_from_backup(dir) {
        eprintln!(
            "warning: could not exclude cache from Time Machine: {}",
            dir.display()
        );
    }
    if in_synced_folder(dir) && !SYNC_WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "warning: Seyk cache is inside a known synced folder; sensitive cache data may leave this machine: {}",
            dir.display()
        );
    }
    Ok(())
}

/// Set the directory's own exclusion, even when an ancestor already excludes it.
#[cfg(target_os = "macos")]
fn exclude_from_backup(dir: &Path) -> bool {
    use std::ffi::c_void;
    use std::os::unix::ffi::OsStrExt;

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        static kCFURLIsExcludedFromBackupKey: *const c_void;
        static kCFBooleanTrue: *const c_void;
        fn CFURLCreateFromFileSystemRepresentation(
            allocator: *const c_void,
            bytes: *const u8,
            length: isize,
            is_directory: u8,
        ) -> *const c_void;
        fn CFURLSetResourcePropertyForKey(
            url: *const c_void,
            key: *const c_void,
            value: *const c_void,
            error: *mut *const c_void,
        ) -> u8;
        fn CFRelease(value: *const c_void);
    }
    let Ok(dir) = std::fs::canonicalize(dir) else {
        return false;
    };
    let bytes = dir.as_os_str().as_bytes();
    let Ok(length) = isize::try_from(bytes.len()) else {
        return false;
    };
    // SAFETY: Core Foundation copies the live path bytes, and the returned URL
    // has one owned reference. The key/Boolean are framework-owned constants.
    // A null error pointer requests no error object. Release the URL exactly once.
    unsafe {
        let url =
            CFURLCreateFromFileSystemRepresentation(std::ptr::null(), bytes.as_ptr(), length, 1);
        if url.is_null() {
            return false;
        }
        let success = CFURLSetResourcePropertyForKey(
            url,
            kCFURLIsExcludedFromBackupKey,
            kCFBooleanTrue,
            std::ptr::null_mut(),
        ) != 0;
        CFRelease(url);
        success
    }
}

pub(crate) fn backup_excluded(dir: &Path) -> Option<bool> {
    #[cfg(target_os = "macos")]
    {
        let dir = std::fs::canonicalize(dir).ok()?;
        let output = std::process::Command::new("/usr/bin/tmutil")
            .arg("isexcluded")
            .arg(dir)
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let output = String::from_utf8(output.stdout).ok()?;
        if output.starts_with("[Excluded]") {
            Some(true)
        } else if output.starts_with("[Included]") {
            Some(false)
        } else {
            None
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = dir;
        None
    }
}

// Resolve symlinked ancestors, including when the final cache file doesn't exist.
fn resolved(path: &Path) -> Option<PathBuf> {
    let absolute = std::path::absolute(path).ok()?;
    let mut ancestor = absolute.as_path();
    let mut suffix = Vec::new();
    loop {
        if let Ok(mut base) = std::fs::canonicalize(ancestor) {
            for part in suffix.iter().rev() {
                base.push(part);
            }
            // Missing suffixes may contain `..`; compare normalized components.
            let mut normalized = PathBuf::new();
            for part in base.components() {
                match part {
                    std::path::Component::ParentDir => {
                        normalized.pop();
                    }
                    std::path::Component::CurDir => {}
                    _ => normalized.push(part.as_os_str()),
                }
            }
            return Some(normalized);
        }
        suffix.push(ancestor.file_name()?.to_os_string());
        ancestor = ancestor.parent()?;
    }
}

pub(crate) fn in_synced_folder(path: &Path) -> bool {
    detected(path, dirs::home_dir().as_deref())
}

fn detected(path: &Path, home: Option<&Path>) -> bool {
    let Some(path) = resolved(path) else {
        return false;
    };
    if path.ancestors().any(|p| p.join(".stfolder").is_dir()) {
        return true;
    }
    let mut roots = Vec::new();
    if let Some(home) = home {
        for suffix in [
            "Library/Mobile Documents",
            "Dropbox",
            "OneDrive",
            "Google Drive",
            "GoogleDrive",
            "Sync",
        ] {
            roots.push(home.join(suffix));
        }
        let cloud = home.join("Library/CloudStorage");
        if let Ok(entries) = std::fs::read_dir(cloud) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if ["Dropbox", "OneDrive", "GoogleDrive"]
                    .iter()
                    .any(|prefix| name.starts_with(prefix))
                {
                    roots.push(entry.path());
                }
            }
        }
        for config in [
            home.join(".dropbox/info.json"),
            home.join("AppData/Local/Dropbox/info.json"),
            home.join("AppData/Roaming/Dropbox/info.json"),
        ] {
            if let Ok(bytes) = std::fs::read(config)
                && let Ok(serde_json::Value::Object(accounts)) = serde_json::from_slice(&bytes)
            {
                roots.extend(
                    accounts
                        .values()
                        .filter_map(|account| account.get("path")?.as_str().map(PathBuf::from)),
                );
            }
        }
    }
    for key in ["OneDrive", "OneDriveCommercial", "OneDriveConsumer"] {
        if let Some(root) = std::env::var_os(key) {
            roots.push(PathBuf::from(root));
        }
    }
    roots
        .iter()
        .filter_map(|root| resolved(root))
        .any(|root| path.starts_with(root))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_are_standard_and_idempotent_and_exclusion_is_verified() {
        let temp = tempfile::tempdir().unwrap();
        prepare(temp.path()).unwrap();
        prepare(temp.path()).unwrap();
        assert_eq!(
            std::fs::read_to_string(temp.path().join("CACHEDIR.TAG")).unwrap(),
            TAG
        );
        #[cfg(target_os = "macos")]
        {
            assert_eq!(backup_excluded(temp.path()), Some(true));
            let destination = tempfile::tempdir().unwrap();
            let moved = destination.path().join("moved-cache");
            std::fs::rename(temp.path(), &moved).unwrap();
            assert_eq!(backup_excluded(&moved), Some(true));
        }
        #[cfg(not(target_os = "macos"))]
        assert_eq!(backup_excluded(temp.path()), None);
    }

    #[test]
    fn concurrent_preparation_publishes_only_complete_tags() {
        let temp = tempfile::tempdir().unwrap();
        prepare(temp.path()).unwrap();
        std::fs::remove_file(temp.path().join("CACHEDIR.TAG")).unwrap();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let dir = temp.path();
                scope.spawn(move || prepare(dir).unwrap());
            }
        });
        assert_eq!(
            std::fs::read_to_string(temp.path().join("CACHEDIR.TAG")).unwrap(),
            TAG
        );
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
    }

    #[test]
    fn synced_warning_once_per_process() {
        const CHILD: &str = "SEYK_GUARDS_WARNING_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let temp = tempfile::tempdir().unwrap();
            std::fs::create_dir(temp.path().join(".stfolder")).unwrap();
            prepare(temp.path()).unwrap();
            prepare(temp.path()).unwrap();
            return;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "cache_guards::tests::synced_warning_once_per_process",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stderr)
                .matches("inside a known synced folder")
                .count(),
            1
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlink_tag_is_rejected_without_modifying_target() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target");
        std::fs::write(&target, TAG).unwrap();
        std::os::unix::fs::symlink(&target, temp.path().join("CACHEDIR.TAG")).unwrap();
        assert!(prepare(temp.path()).is_err());
        assert_eq!(std::fs::read_to_string(target).unwrap(), TAG);
    }

    #[test]
    fn existing_unrelated_tag_is_never_overwritten() {
        let temp = tempfile::tempdir().unwrap();
        let tag = temp.path().join("CACHEDIR.TAG");
        std::fs::write(&tag, "other data").unwrap();
        assert!(prepare(temp.path()).is_err());
        assert_eq!(std::fs::read_to_string(tag).unwrap(), "other data");
    }

    #[test]
    fn sync_detection_covers_provider_roots_config_and_markers() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path();
        for suffix in [
            "Library/Mobile Documents/com~apple~CloudDocs",
            "Dropbox",
            "OneDrive",
            "Google Drive",
            "Library/CloudStorage/GoogleDrive-account",
            "Library/CloudStorage/OneDrive-account",
        ] {
            let root = home.join(suffix);
            std::fs::create_dir_all(&root).unwrap();
            assert!(
                detected(&root.join("cache/cache.sqlite3"), Some(home)),
                "{suffix}"
            );
        }
        assert!(!detected(
            &home.join("Dropbox-other/cache.sqlite3"),
            Some(home)
        ));
        assert!(!detected(
            &home.join("Library/Caches/seyk/cache.sqlite3"),
            Some(home)
        ));
        let custom = home.join("custom-dropbox");
        std::fs::create_dir_all(home.join(".dropbox")).unwrap();
        std::fs::write(
            home.join(".dropbox/info.json"),
            serde_json::json!({"personal": {"path": custom}}).to_string(),
        )
        .unwrap();
        assert!(detected(&custom.join("cache.sqlite3"), Some(home)));
        let syncthing = home.join("custom-sync");
        std::fs::create_dir_all(syncthing.join(".stfolder")).unwrap();
        assert!(detected(&syncthing.join("nested/cache.sqlite3"), None));
        #[cfg(unix)]
        {
            let alias = home.join("alias");
            std::os::unix::fs::symlink(home.join("Dropbox"), &alias).unwrap();
            assert!(detected(&alias.join("cache.sqlite3"), Some(home)));
            assert!(!detected(
                &home.join("Dropbox/../local/cache.sqlite3"),
                Some(home)
            ));
        }
    }
}
