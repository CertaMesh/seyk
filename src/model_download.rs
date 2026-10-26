use crate::{Error, Result};
use seyk_embeddinggemma::{GemmaConfig, ModelSource};
use std::path::{Path, PathBuf};

/// Download the files `config` needs into the Hugging Face cache.
///
/// Provisioning functions use the network only for public model file requests.
pub fn download(config: &GemmaConfig) -> Result<PathBuf> {
    let ModelSource::HuggingFace { repo, revision } = &config.source else {
        return Err(Error::Config(
            "model source is a local directory; nothing to download".into(),
        ));
    };
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .map_err(|e| Error::Embedding(format!("cannot start download: {e}")))?;
    download_files(revision, &config.files(), |revision, file| {
        api.repo(hf_hub::Repo::with_revision(
            repo.clone(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ))
        .get(file)
        .map_err(|e| Error::Embedding(format!("downloading {file}: {e}")))
    })
}

/// Pinned q8 assets, from the HF LFS metadata at ONNX_REVISION.
#[cfg(feature = "audio")]
pub struct AudioAsset {
    pub path: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

#[cfg(feature = "audio")]
pub fn audio_assets() -> &'static [AudioAsset] {
    &[
        AudioAsset {
            path: seyk_embeddinggemma::audio::FILES[0],
            bytes: 249330,
            sha256: "04a9a9094ba76fb169e4c69be45a4b621580188654d6be59eab860eadcd42d3c",
        },
        AudioAsset {
            path: seyk_embeddinggemma::audio::FILES[1],
            bytes: 340058624,
            sha256: "aa6361d898e1f1d6303f8dd2b5fabf4fd3629e15fd709e9cb06ac5cb9416a030",
        },
    ]
}

#[cfg(feature = "audio")]
pub fn audio_directory() -> PathBuf {
    hf_hub::Cache::from_env()
        .path()
        .join(format!(
            "models--{}",
            seyk_embeddinggemma::ONNX_REPO.replace('/', "--")
        ))
        .join("snapshots")
        .join(seyk_embeddinggemma::ONNX_REVISION)
}

#[cfg(feature = "audio")]
fn verify_audio_asset(path: &Path, asset: &AudioAsset) -> Result<()> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let invalid = || {
        Error::Embedding(format!(
            "audio asset {} missing or checksum mismatch; run seyk setup audio",
            asset.path
        ))
    };
    if !std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() == asset.bytes) {
        return Err(invalid());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path).map_err(|_| invalid())?;
    if !file.metadata().is_ok_and(|m| m.is_file()) {
        return Err(invalid());
    }
    let mut reader = file.take(asset.bytes + 1);
    let mut hash = Sha256::new();
    let mut count = 0;
    let mut block = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut block).map_err(|_| invalid())?;
        if n == 0 {
            break;
        }
        count += n as u64;
        hash.update(&block[..n]);
    }
    if count != asset.bytes || format!("{:x}", hash.finalize()) != asset.sha256 {
        return Err(invalid());
    }
    Ok(())
}

/// Verify pinned audio weights offline, without creating an inference session.
#[cfg(feature = "audio")]
pub fn verify_audio(dir: &Path) -> Result<()> {
    for asset in audio_assets() {
        verify_audio_asset(&dir.join(asset.path), asset)?;
    }
    Ok(())
}

/// Explicitly provision pinned q8 audio files; never called by inference.
#[cfg(feature = "audio")]
pub fn download_audio() -> Result<PathBuf> {
    let dir = audio_directory();
    install_audio_with(&dir, audio_assets(), |asset| {
        let api = hf_hub::api::sync::ApiBuilder::from_env()
            .with_progress(true)
            .build()
            .map_err(|e| Error::Embedding(format!("cannot start download: {e}")))?;
        api.repo(hf_hub::Repo::with_revision(
            seyk_embeddinggemma::ONNX_REPO.into(),
            hf_hub::RepoType::Model,
            seyk_embeddinggemma::ONNX_REVISION.into(),
        ))
        .download(asset.path)
        .map_err(|e| Error::Embedding(format!("downloading {}: {e}", asset.path)))
    })
}

#[cfg(feature = "audio")]
fn install_audio_with(
    dir: &Path,
    assets: &[AudioAsset],
    mut fetch: impl FnMut(&AudioAsset) -> Result<PathBuf>,
) -> Result<PathBuf> {
    for asset in assets {
        if verify_audio_asset(&dir.join(asset.path), asset).is_ok() {
            continue;
        }
        let path = fetch(asset)?;
        if path != dir.join(asset.path) {
            return Err(Error::Embedding(
                "audio download returned a different snapshot".into(),
            ));
        }
        verify_audio_asset(&path, asset)?;
    }
    Ok(dir.into())
}

fn download_files(
    revision: &str,
    files: &[String],
    mut get: impl FnMut(&str, &str) -> Result<PathBuf>,
) -> Result<PathBuf> {
    // Resolve once through the first file, retaining hf-hub's cache-first
    // behaviour (an info() request would require a network connection).
    let first = files.first().expect("at least one file");
    let dir = snapshot_root(&get(revision, first)?, first);
    let commit = dir.file_name().and_then(|s| s.to_str()).ok_or_else(|| {
        Error::Embedding(format!("invalid model snapshot path: {}", dir.display()))
    })?;
    for file in &files[1..] {
        // Python-populated snapshots need not have refs/<sha>, which hf-hub's
        // get() requires. Reuse files in the selected snapshot directly.
        if dir.join(file).exists() {
            continue;
        }
        let path = get(commit, file)?;
        if snapshot_root(&path, file) != dir {
            return Err(Error::Embedding(format!(
                "downloading {file}: expected snapshot {commit}, got {}",
                path.display()
            )));
        }
    }
    Ok(dir)
}

fn snapshot_root(file: &Path, rel: &str) -> PathBuf {
    let depth = Path::new(rel).components().count();
    let mut p = file.to_path_buf();
    for _ in 0..depth {
        p.pop();
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "audio")]
    #[test]
    fn audio_install_verifies_repairs_and_skips_without_network() {
        let dir = tempfile::tempdir().unwrap();
        let assets = [AudioAsset {
            path: "audio.bin",
            bytes: 4,
            sha256: "770e607624d689265ca6c44884d0807d9b054d23c473c106c72be9de08b7376c",
        }];
        let mut requests = 0;
        let mut install = || {
            install_audio_with(dir.path(), &assets, |asset| {
                requests += 1;
                let path = dir.path().join(asset.path);
                std::fs::write(&path, b"good").unwrap();
                Ok(path)
            })
        };
        install().unwrap();
        install().unwrap();
        std::fs::write(dir.path().join("audio.bin"), b"evil").unwrap();
        install().unwrap();
        assert_eq!(requests, 2);
        assert_eq!(
            std::fs::read(dir.path().join("audio.bin")).unwrap(),
            b"good"
        );
        install_audio_with(dir.path(), &assets, |_| panic!("must skip verified file")).unwrap();
    }

    #[cfg(feature = "audio")]
    #[test]
    fn corrupt_audio_download_fails_verification() {
        let dir = tempfile::tempdir().unwrap();
        let assets = [AudioAsset {
            path: "audio.bin",
            bytes: 4,
            sha256: "invalid",
        }];
        let error = install_audio_with(dir.path(), &assets, |asset| {
            let path = dir.path().join(asset.path);
            std::fs::write(&path, b"evil").unwrap();
            Ok(path)
        })
        .unwrap_err();
        assert!(error.to_string().contains("run seyk setup audio"));
    }

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn files() -> Vec<String> {
        let mut config = GemmaConfig::default();
        config.vision = true;
        config.files()
    }

    fn write_file(dir: &Path, file: &str) -> PathBuf {
        let path = dir.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"synthetic model file").unwrap();
        path
    }

    #[test]
    fn moving_alias_is_only_used_for_the_first_file() {
        let cache = tempfile::tempdir().unwrap();
        let files = files();
        let mut alias = A;
        let mut requests = Vec::new();
        let dir = download_files("main", &files, |revision, file| {
            requests.push((revision.to_string(), file.to_string()));
            let commit = if revision == "main" { alias } else { revision };
            let path = write_file(&cache.path().join("snapshots").join(commit), file);
            alias = B;
            Ok(path)
        })
        .unwrap();
        assert_eq!(dir, cache.path().join("snapshots").join(A));
        assert_eq!(requests[0], ("main".into(), files[0].clone()));
        assert_eq!(requests.len(), files.len());
        for ((revision, file), expected) in requests[1..].iter().zip(&files[1..]) {
            assert_eq!(revision, A);
            assert_eq!(file, expected);
        }
        assert!(files.iter().all(|file| dir.join(file).exists()));
    }

    #[test]
    fn cached_alias_snapshot_needs_no_network_or_sha_ref() {
        let cache = tempfile::tempdir().unwrap();
        let repo = "test/synthetic";
        let files = files();
        let dir = cache
            .path()
            .join("models--test--synthetic/snapshots")
            .join(A);
        for file in &files {
            write_file(&dir, file);
        }
        hf_hub::Cache::new(cache.path().to_path_buf())
            .model(repo.into())
            .create_ref(A)
            .unwrap();
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache.path().to_path_buf())
            .with_endpoint("http://127.0.0.1:0".into())
            .with_token(None)
            .with_progress(false)
            .build()
            .unwrap();
        let downloaded = download_files("main", &files, |revision, file| {
            api.repo(hf_hub::Repo::with_revision(
                repo.into(),
                hf_hub::RepoType::Model,
                revision.into(),
            ))
            .get(file)
            .map_err(|e| Error::Embedding(e.to_string()))
        })
        .unwrap();
        assert_eq!(downloaded, dir);
        assert!(
            !cache
                .path()
                .join("models--test--synthetic/refs")
                .join(A)
                .exists()
        );
    }

    /// A loopback-only HF endpoint, optionally redirecting within the fixture.
    struct HttpFixture {
        endpoint: String,
        requests: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        worker: Option<std::thread::JoinHandle<()>>,
    }

    impl HttpFixture {
        fn new() -> Self {
            Self::with_redirect(false)
        }

        fn with_redirect(redirect: bool) -> Self {
            use std::io::{BufRead, BufReader, Write};
            use std::sync::{
                Arc, Mutex,
                atomic::{AtomicBool, Ordering},
            };
            use std::time::Duration;

            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            listener.set_nonblocking(true).unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let (recorded, stopping) = (requests.clone(), stop.clone());
            let worker = std::thread::spawn(move || {
                while !stopping.load(Ordering::SeqCst) {
                    let (mut stream, _) = match listener.accept() {
                        Ok(connection) => connection,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(1));
                            continue;
                        }
                        Err(e) => panic!("fixture accept: {e}"),
                    };
                    stream.set_nonblocking(false).unwrap();
                    stream
                        .set_read_timeout(Some(Duration::from_secs(5)))
                        .unwrap();
                    stream
                        .set_write_timeout(Some(Duration::from_secs(5)))
                        .unwrap();
                    let mut reader = BufReader::new(&stream);
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let path = line.split_whitespace().nth(1).unwrap().to_string();
                    recorded.lock().unwrap().push(path.clone());
                    let mut metadata = false;
                    loop {
                        line.clear();
                        assert_ne!(reader.read_line(&mut line).unwrap(), 0);
                        if line == "\r\n" {
                            break;
                        }
                        metadata |= line.trim().eq_ignore_ascii_case("range: bytes=0-0");
                    }
                    let (commit, payload) = if redirect {
                        if path == "/test/synthetic/resolve/main/config.json" {
                            write!(stream,
                                "HTTP/1.1 302 Found\r\nLocation: /resolved/config.json\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            ).unwrap();
                            continue;
                        }
                        if path != "/resolved/config.json" {
                            write!(stream, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
                            continue;
                        }
                        (A, "{}".to_string())
                    } else {
                        // Remote main has advanced to B. A is still downloadable.
                        let (revision, file) = path
                            .strip_prefix("/test/synthetic/resolve/")
                            .unwrap()
                            .split_once('/')
                            .unwrap();
                        let commit = if revision == "main" { B } else { revision };
                        assert!(commit == A || commit == B);
                        (commit, format!("synthetic {commit} {file}"))
                    };
                    let body = if metadata {
                        &payload.as_bytes()[..1]
                    } else {
                        payload.as_bytes()
                    };
                    let end = body.len() - 1;
                    write!(stream,
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes 0-{end}/{}\r\nx-repo-commit: {commit}\r\nETag: \"{}\"\r\nConnection: close\r\n\r\n",
                        body.len(), payload.len(), blake3::hash(payload.as_bytes())
                    ).unwrap();
                    stream.write_all(body).unwrap();
                }
            });
            Self {
                endpoint,
                requests,
                stop,
                worker: Some(worker),
            }
        }
    }

    impl Drop for HttpFixture {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
            let result = self.worker.take().unwrap().join();
            if !std::thread::panicking() {
                result.unwrap();
            }
        }
    }

    // Run real HTTP clients without changing the parallel test process environment.
    fn ran_proxy_free_child(test_name: &str) -> bool {
        const CHILD: &str = "SEYK_HF_FIXTURE_CHILD";
        const PROXY_VARS: [&str; 6] = [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ];
        if std::env::var(CHILD).ok().as_deref() != Some(test_name) {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            child
                .args([
                    "--exact",
                    &format!("model_download::tests::{test_name}"),
                    "--nocapture",
                ])
                .env(CHILD, test_name);
            // Command-local removal is race-free and independent of the ureq version.
            for var in PROXY_VARS {
                child.env_remove(var);
            }
            let output = child.output().unwrap();
            assert!(
                output.status.success(),
                "fixture child failed: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return true;
        }
        for var in PROXY_VARS {
            assert!(
                std::env::var_os(var).is_none(),
                "proxy variable inherited: {var}"
            );
        }
        false
    }

    #[test]
    fn uncached_hub_download_follows_redirects_and_populates_cache() {
        if ran_proxy_free_child("uncached_hub_download_follows_redirects_and_populates_cache") {
            return;
        }
        let cache = tempfile::tempdir().unwrap();
        let fixture = HttpFixture::with_redirect(true);
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache.path().to_path_buf())
            .with_endpoint(fixture.endpoint.clone())
            .with_token(None)
            .with_progress(false)
            .with_retries(0)
            .build()
            .unwrap();
        let repo = api.model("test/synthetic".into());
        let path = repo.get("config.json").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"{}");
        assert_eq!(path.parent().unwrap().file_name().unwrap(), A);
        let request_count = {
            let requests = fixture.requests.lock().unwrap();
            assert!(
                requests
                    .iter()
                    .any(|r| r == "/test/synthetic/resolve/main/config.json")
            );
            assert!(requests.iter().any(|r| r == "/resolved/config.json"));
            assert!(requests.iter().all(|r| matches!(
                r.as_str(),
                "/test/synthetic/resolve/main/config.json" | "/resolved/config.json"
            )));
            requests.len()
        };
        assert_eq!(repo.get("config.json").unwrap(), path);
        assert_eq!(fixture.requests.lock().unwrap().len(), request_count);
    }

    #[test]
    fn partial_old_cache_downloads_missing_files_from_pinned_snapshot() {
        if ran_proxy_free_child("partial_old_cache_downloads_missing_files_from_pinned_snapshot") {
            return;
        }
        let cache = tempfile::tempdir().unwrap();
        let repo = "test/synthetic";
        let files = files();
        assert_eq!(files[0], "tokenizer.json");
        let repo_dir = cache.path().join("models--test--synthetic");
        let dir = repo_dir.join("snapshots").join(A);
        let tokenizer = write_file(&dir, &files[0]);
        hf_hub::Cache::new(cache.path().to_path_buf())
            .model(repo.into())
            .create_ref(A)
            .unwrap();
        assert!(!repo_dir.join("refs").join(A).exists());
        assert!(files[1..].iter().all(|file| !dir.join(file).exists()));
        let fixture = HttpFixture::new();
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache.path().to_path_buf())
            .with_endpoint(fixture.endpoint.clone())
            .with_token(None)
            .with_progress(false)
            .with_retries(0)
            .build()
            .unwrap();
        let downloaded = download_files("main", &files, |revision, file| {
            api.repo(hf_hub::Repo::with_revision(
                repo.into(),
                hf_hub::RepoType::Model,
                revision.into(),
            ))
            .get(file)
            .map_err(|e| Error::Embedding(e.to_string()))
        })
        .unwrap();
        assert_eq!(downloaded, dir);
        assert_eq!(std::fs::read(tokenizer).unwrap(), b"synthetic model file");
        assert_eq!(
            std::fs::read_to_string(repo_dir.join("refs/main")).unwrap(),
            A
        );
        assert!(!repo_dir.join("snapshots").join(B).exists());
        let requests = fixture.requests.lock().unwrap();
        assert!(!requests.is_empty());
        for request in requests.iter() {
            assert!(
                request.starts_with(&format!("/test/synthetic/resolve/{A}/")),
                "{request}"
            );
        }
        for file in &files[1..] {
            let path = format!("/test/synthetic/resolve/{A}/{file}");
            assert!(requests.contains(&path), "missing request for {file}");
            assert_eq!(
                std::fs::read_to_string(dir.join(file)).unwrap(),
                format!("synthetic {A} {file}")
            );
        }
        assert!(!requests.iter().any(|r| r.ends_with("tokenizer.json")));
    }

    #[test]
    fn missing_pinned_file_returns_download_error() {
        let cache = tempfile::tempdir().unwrap();
        let error = download_files("main", &files(), |revision, file| {
            if revision == "main" {
                Ok(write_file(&cache.path().join("snapshots").join(A), file))
            } else {
                assert_eq!(revision, A);
                Err(Error::Embedding("missing pinned file".into()))
            }
        })
        .unwrap_err();
        assert!(error.to_string().contains("missing pinned file"));
    }

    #[test]
    fn rejects_a_file_returned_from_another_snapshot() {
        let cache = tempfile::tempdir().unwrap();
        let error = download_files("main", &files(), |revision, file| {
            let commit = if revision == "main" { A } else { B };
            Ok(write_file(
                &cache.path().join("snapshots").join(commit),
                file,
            ))
        })
        .unwrap_err();
        assert!(error.to_string().contains("expected snapshot"));
    }
}
