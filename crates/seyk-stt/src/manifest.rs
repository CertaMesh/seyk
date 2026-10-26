//! Checked asset set. No network is used by verification or inference.
use serde::Deserialize;
use seyk_core::{Error, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
};

pub const REPO: &str = "Xenova/whisper-tiny";
pub const REVISION: &str = "5332fcc35e32a33b86612b9a57a89be7906102b1";
pub const MANIFEST_SHA256: &str =
    "3cefb2a90f9cee1983933dcaee20a25689c8b35a3c9bedfd94ae3c38ffb1a254";
pub const MANIFEST: &str = include_str!("../assets/manifest.json");
#[derive(Deserialize)]
pub struct Asset {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Deserialize)]
pub(crate) struct Manifest {
    pub files: Vec<Asset>,
    pub graphs: Vec<Graph>,
    pub policy: Policy,
}
impl Manifest {
    pub fn graph(&self, file: &str) -> (&Asset, &Graph) {
        let graph = self
            .graphs
            .iter()
            .find(|graph| graph.file == file)
            .expect("compiled manifest graph");
        let asset = self
            .files
            .iter()
            .find(|asset| asset.path == graph.file)
            .expect("compiled manifest graph asset");
        (asset, graph)
    }
}
#[derive(Deserialize)]
pub(crate) struct Graph {
    pub file: String,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}
#[derive(Deserialize)]
pub(crate) struct Port {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<serde_json::Value>,
}
#[derive(Deserialize)]
pub(crate) struct Policy {
    pub suppress_tokens: Vec<usize>,
    pub lang_to_id: std::collections::BTreeMap<String, usize>,
}
pub(crate) fn manifest() -> &'static Manifest {
    static VALUE: OnceLock<Manifest> = OnceLock::new();
    VALUE.get_or_init(|| serde_json::from_str(MANIFEST).expect("compiled manifest"))
}
pub fn assets() -> &'static [Asset] {
    &manifest().files
}
pub fn manifest_sha256() -> String {
    format!("{:x}", Sha256::digest(MANIFEST.as_bytes()))
}
pub(crate) fn model_error(reason: &'static str) -> Error {
    Error::SttModel { reason }
}
pub fn default_directory() -> Result<PathBuf> {
    Ok(dirs::cache_dir()
        .ok_or_else(|| model_error("cache directory unavailable"))?
        .join("seyk")
        .join("stt")
        .join(manifest_sha256()))
}
// Inspect targets before open; O_NONBLOCK also closes the Unix FIFO-swap race.
// Symlinks are admitted only when they resolve to regular files.
fn regular_file(path: &Path) -> Result<File> {
    if !fs::metadata(path).is_ok_and(|m| m.is_file()) {
        return Err(model_error(
            "asset missing, unreadable or not a regular file",
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|_| model_error("file unreadable"))?;
    if !file.metadata().is_ok_and(|m| m.is_file()) {
        return Err(model_error("asset is not a regular file"));
    }
    Ok(file)
}

pub(crate) fn read_asset(dir: &Path, asset: &Asset) -> Result<Vec<u8>> {
    let file = regular_file(&dir.join(&asset.path))?;
    if file
        .metadata()
        .map_err(|_| model_error("file unreadable"))?
        .len()
        != asset.bytes
    {
        return Err(model_error("file length mismatch"));
    }
    let mut bytes = Vec::new();
    file.take(asset.bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| model_error("file unreadable"))?;
    if bytes.len() as u64 != asset.bytes || format!("{:x}", Sha256::digest(&bytes)) != asset.sha256
    {
        return Err(model_error("checksum mismatch"));
    }
    Ok(bytes)
}
fn marker_matches(path: &Path) -> bool {
    let Ok(file) = regular_file(path) else {
        return false;
    };
    if !file.metadata().is_ok_and(|m| m.is_file() && m.len() == 64) {
        return false;
    }
    let mut bytes = Vec::with_capacity(65);
    file.take(65).read_to_end(&mut bytes).is_ok() && bytes == manifest_sha256().as_bytes()
}

// Only the pinned asset tree and our installation marker are admitted. File
// symlinks are safe because sessions consume the exact verified bytes in memory;
// directory symlinks and extra files are rejected rather than followed.
pub(crate) fn verify_layout(dir: &Path) -> Result<()> {
    fn visit(root: &Path, relative: &Path) -> Result<()> {
        for entry in fs::read_dir(root.join(relative))
            .map_err(|_| model_error("directory missing or unreadable"))?
        {
            let entry = entry.map_err(|_| model_error("directory unreadable"))?;
            let path = relative.join(entry.file_name());
            let kind = entry
                .file_type()
                .map_err(|_| model_error("file unreadable"))?;
            if kind.is_dir() && path == Path::new("onnx") {
                visit(root, &path)?;
            } else if path == Path::new("manifest.sha256") {
                if !marker_matches(&entry.path()) {
                    return Err(model_error("manifest marker mismatch"));
                }
            } else if !assets().iter().any(|asset| path == Path::new(&asset.path)) || kind.is_dir()
            {
                return Err(model_error("unexpected asset"));
            } else if !fs::metadata(entry.path()).is_ok_and(|m| m.is_file()) {
                return Err(model_error("asset is not a regular file"));
            }
        }
        Ok(())
    }
    visit(dir, Path::new(""))
}

/// Offline full-manifest verification, also used for caller-supplied directories.
pub fn verify_directory(dir: &Path) -> Result<()> {
    verify_layout(dir)?;
    for asset in assets() {
        verify_asset(dir, asset)?;
    }
    Ok(())
}

fn verify_asset(dir: &Path, asset: &Asset) -> Result<()> {
    let file = regular_file(&dir.join(&asset.path))?;
    let mut reader = file.take(asset.bytes + 1);
    let mut hash = Sha256::new();
    let mut count = 0u64;
    let mut block = [0u8; 64 * 1024];
    loop {
        let n = reader
            .read(&mut block)
            .map_err(|_| model_error("file unreadable"))?;
        if n == 0 {
            break;
        }
        count += n as u64;
        hash.update(&block[..n]);
    }
    if count != asset.bytes || format!("{:x}", hash.finalize()) != asset.sha256 {
        return Err(model_error("file length or checksum mismatch"));
    }
    Ok(())
}

#[cfg(unix)]
fn set_asset_permissions(file: &File) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    file.set_permissions(fs::Permissions::from_mode(0o644))
        .map_err(|_| model_error("cannot set asset permissions"))
}

/// Stage a caller's explicitly downloaded files, then atomically publish one
/// complete verified directory. Never used by inference. Existing unmanaged
/// directories are preserved. A managed corrupt installation can be repaired.
pub fn install_with(
    destination: &Path,
    mut fetch: impl FnMut(&Asset) -> Result<PathBuf>,
) -> Result<PathBuf> {
    let parent = destination
        .parent()
        .ok_or_else(|| model_error("invalid installation path"))?;
    fs::create_dir_all(parent).map_err(|_| model_error("cannot create installation parent"))?;
    let lock_path = parent.join(".stt-install.lock");
    // Keep one stable inode. OS locks are released even when the process is killed.
    let _lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|source| Error::Io {
            path: lock_path.clone(),
            source,
        })?;
    _lock.try_lock().map_err(|error| Error::Io {
        path: lock_path,
        source: match error {
            std::fs::TryLockError::WouldBlock => std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "another STT install is running",
            ),
            std::fs::TryLockError::Error(source) => source,
        },
    })?;
    if destination.exists() && verify_directory(destination).is_ok() {
        return Ok(destination.into());
    }
    let stamp = manifest_sha256();
    if destination.exists() && !marker_matches(&destination.join("manifest.sha256")) {
        return Err(model_error(
            "destination is unmanaged; preserve it and choose an empty directory",
        ));
    }
    let stage = tempfile::Builder::new()
        .prefix(".stt-stage-")
        .tempdir_in(parent)
        .map_err(|_| model_error("cannot stage installation"))?;
    for asset in assets() {
        let source = if verify_asset(destination, asset).is_ok() {
            destination.join(&asset.path)
        } else {
            fetch(asset)?
        };
        let target = stage.path().join(&asset.path);
        fs::create_dir_all(target.parent().expect("asset parent"))
            .map_err(|_| model_error("cannot stage asset"))?;
        let mut input = regular_file(&source)?.take(asset.bytes + 1);
        let mut output = File::create(target).map_err(|_| model_error("cannot stage asset"))?;
        let n = std::io::copy(&mut input, &mut output)
            .map_err(|_| model_error("cannot stage asset"))?;
        if n != asset.bytes {
            return Err(model_error("download length mismatch"));
        }
        #[cfg(unix)]
        set_asset_permissions(&output)?;
        output
            .sync_all()
            .map_err(|_| model_error("cannot sync asset"))?;
    }
    verify_directory(stage.path())?;
    let mut marker = File::create(stage.path().join("manifest.sha256"))
        .map_err(|_| model_error("cannot write manifest marker"))?;
    marker
        .write_all(stamp.as_bytes())
        .map_err(|_| model_error("cannot write manifest marker"))?;
    #[cfg(unix)]
    set_asset_permissions(&marker)?;
    marker
        .sync_all()
        .map_err(|_| model_error("cannot sync manifest marker"))?;
    drop(marker);
    let backup = tempfile::Builder::new()
        .prefix(".stt-old-")
        .tempdir_in(parent)
        .map_err(|_| model_error("cannot prepare replacement"))?;
    let old = backup.path().join("model");
    if destination.exists() {
        fs::rename(destination, &old).map_err(|_| model_error("cannot replace installation"))?;
    }
    if fs::rename(stage.path(), destination).is_err() {
        if old.exists() && fs::rename(&old, destination).is_err() {
            // Preserve recovery data if restoration fails, rather than deleting it.
            let _ = backup.keep();
        }
        return Err(model_error("cannot publish installation"));
    }
    Ok(destination.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_graph_resolves_to_exactly_one_asset() {
        let manifest = manifest();
        for graph in &manifest.graphs {
            assert_eq!(
                manifest
                    .files
                    .iter()
                    .filter(|asset| asset.path == graph.file)
                    .count(),
                1,
                "{}",
                graph.file
            );
            assert_eq!(
                manifest
                    .graphs
                    .iter()
                    .filter(|other| other.file == graph.file)
                    .count(),
                1,
                "{}",
                graph.file
            );
            assert_eq!(manifest.graph(&graph.file).0.path, graph.file);
        }
    }

    #[test]
    fn graph_lookup_is_independent_of_manifest_array_order() {
        let mut shuffled: Manifest = serde_json::from_str(MANIFEST).unwrap();
        shuffled.files.reverse();
        shuffled.graphs.rotate_left(1);
        for file in [
            "onnx/encoder_model.onnx",
            "onnx/decoder_model.onnx",
            "onnx/decoder_with_past_model.onnx",
        ] {
            let (expected, expected_graph) = manifest().graph(file);
            let (asset, graph) = shuffled.graph(file);
            assert_eq!(asset.path, file);
            assert_eq!(graph.file, file);
            assert_eq!(asset.bytes, expected.bytes);
            assert_eq!(asset.sha256, expected.sha256);
            for (ports, expected_ports) in [
                (&graph.inputs, &expected_graph.inputs),
                (&graph.outputs, &expected_graph.outputs),
            ] {
                assert_eq!(ports.len(), expected_ports.len());
                for (port, expected) in ports.iter().zip(expected_ports) {
                    assert_eq!(port.name, expected.name);
                    assert_eq!(port.dtype, expected.dtype);
                    assert_eq!(port.shape, expected.shape);
                }
            }
        }
    }

    #[test]
    fn stt_install_lock_names_path_and_recovers_from_stale_file() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join(".stt-install.lock");
        fs::write(&lock_path, b"legacy interrupted install").unwrap();
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        lock.try_lock().unwrap();
        let error = install_with(&dir.path().join("model"), |_| {
            panic!("must not fetch while locked")
        })
        .unwrap_err();
        assert!(error.to_string().contains(&lock_path.display().to_string()));
        assert!(error.to_string().contains("another STT install is running"));
        drop(lock);
        let error = install_with(&dir.path().join("model"), |_| {
            Err(model_error("fake fetch reached"))
        })
        .unwrap_err();
        assert!(error.to_string().contains("fake fetch reached"));
    }

    #[test]
    fn stt_install_lock_releases_after_killed_owner() {
        use std::io::{BufRead, BufReader};
        use std::process::{Command, Stdio};
        const CHILD: &str = "SEYK_STT_INSTALL_LOCK_CHILD";
        if let Some(dir) = std::env::var_os(CHILD) {
            let _ = install_with(&PathBuf::from(dir).join("model"), |_| {
                println!("STT_LOCK_READY");
                std::io::stdout().flush().unwrap();
                loop {
                    std::thread::park();
                }
            });
            panic!("child failed to acquire the installation lock");
        }
        let dir = tempfile::tempdir().unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "manifest::tests::stt_install_lock_releases_after_killed_owner",
                "--nocapture",
            ])
            .env(CHILD, dir.path())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        loop {
            line.clear();
            assert_ne!(
                output.read_line(&mut line).unwrap(),
                0,
                "child exited before acquiring lock"
            );
            if line.trim() == "STT_LOCK_READY" {
                break;
            }
        }
        child.kill().unwrap();
        child.wait().unwrap();
        let error = install_with(&dir.path().join("model"), |_| {
            Err(model_error("fake fetch reached"))
        })
        .unwrap_err();
        assert!(error.to_string().contains("fake fetch reached"));
    }

    #[test]
    fn missing_corrupt_and_equal_length_tamper_are_typed() {
        let dir = tempfile::tempdir().unwrap();
        let asset = Asset {
            path: "test".into(),
            bytes: 4,
            sha256: format!("{:x}", Sha256::digest(b"good")),
        };
        assert!(matches!(
            read_asset(dir.path(), &asset),
            Err(Error::SttModel { .. })
        ));
        for bytes in [b"short".as_slice(), b"evil".as_slice()] {
            fs::write(dir.path().join("test"), bytes).unwrap();
            let e = read_asset(dir.path(), &asset).unwrap_err();
            assert!(matches!(e, Error::SttModel { .. }));
            assert!(e.to_string().contains("seyk setup audio"));
        }
        fs::write(dir.path().join("test"), b"good").unwrap();
        assert_eq!(read_asset(dir.path(), &asset).unwrap(), b"good");
    }
    #[test]
    fn failed_download_never_publishes_or_overwrites_unmanaged_data() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("model");
        assert!(install_with(&destination, |_| Err(model_error("test failure"))).is_err());
        assert!(!destination.exists());
        fs::create_dir(&destination).unwrap();
        fs::write(destination.join("keep"), b"unrelated").unwrap();
        assert!(install_with(&destination, |_| panic!("must not download")).is_err());
        assert_eq!(fs::read(destination.join("keep")).unwrap(), b"unrelated");
    }
}

#[cfg(all(test, unix))]
mod special_file_tests {
    use super::*;
    #[test]
    fn fifo_and_symlink_to_fifo_fail_promptly_without_a_writer() {
        use std::{
            os::unix::{ffi::OsStrExt, fs::symlink},
            time::Duration,
        };
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let dir = tempfile::tempdir().unwrap();
            let fifo = dir.path().join("fifo");
            let name = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
            symlink(&fifo, dir.path().join("link")).unwrap();
            for name in ["fifo", "link"] {
                let asset = Asset {
                    path: name.into(),
                    bytes: 4,
                    sha256: String::new(),
                };
                assert!(matches!(
                    read_asset(dir.path(), &asset),
                    Err(Error::SttModel { .. })
                ));
                assert!(!marker_matches(&dir.path().join(name)));
                assert!(matches!(
                    regular_file(&dir.path().join(name)),
                    Err(Error::SttModel { .. })
                ));
            }
            let model = dir.path().join("model");
            fs::create_dir(&model).unwrap();
            symlink(&fifo, model.join("config.json")).unwrap();
            assert!(matches!(
                verify_directory(&model),
                Err(Error::SttModel { .. })
            ));
            fs::remove_file(model.join("config.json")).unwrap();
            symlink(&fifo, model.join("manifest.sha256")).unwrap();
            assert!(matches!(
                verify_directory(&model),
                Err(Error::SttModel { .. })
            ));
            assert!(matches!(
                install_with(&model, |_| panic!("must reject before fetch")),
                Err(Error::SttModel { .. })
            ));
            send.send(()).unwrap();
        });
        // Regression must fail, rather than hanging the test runner on a FIFO open.
        receive
            .recv_timeout(Duration::from_secs(2))
            .expect("special-file rejection blocked");
    }
}
