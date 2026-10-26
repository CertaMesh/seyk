//! Gemma 4 audio preprocessing and the optional pinned q8 encoder.
use super::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::io::Read;
use std::sync::OnceLock;

/// Version of the reference feature and token policy.
pub const FEATURE_POLICY: &str = "gemma4-audio-v1;16k;magnitude;hann-periodic320;hop160;fft512;mel128-htk;ln+0.001;pad-left160;frame321;audio-begin-end;no-prefix";
/// Only these q8 files are provisioned for audio (independent of text precision).
pub const FILES: [&str; 2] = [
    "onnx/audio_encoder_quantized.onnx",
    "onnx/audio_encoder_quantized.onnx_data",
];

pub(super) struct AudioEncoder {
    snapshot: OnceLock<AudioSnapshot>,
    identity_init: Mutex<()>,
    dir: PathBuf,
    session: Mutex<Option<Session>>,
    #[cfg(test)]
    hash_passes: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    session_loads: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    bytes_read: std::sync::atomic::AtomicUsize,
}
impl AudioEncoder {
    pub fn new(dir: &Path) -> Self {
        Self {
            snapshot: OnceLock::new(),
            identity_init: Mutex::new(()),
            dir: dir.into(),
            session: Mutex::new(None),
            #[cfg(test)]
            hash_passes: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            session_loads: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            bytes_read: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// The first successful audio use pins both identity and inference bytes.
    pub fn identity(&self) -> Result<&str> {
        Ok(&self.snapshot()?.identity)
    }

    fn snapshot(&self) -> Result<&AudioSnapshot> {
        self.prepare_with(|_, _| Ok(()))
    }

    /// Errors publish nothing; the next request can retry the complete capture.
    fn prepare_with(
        &self,
        mut check: impl FnMut(&Path, CapturePoint) -> Result<()>,
    ) -> Result<&AudioSnapshot> {
        if let Some(snapshot) = self.snapshot.get() {
            return Ok(snapshot);
        }
        let _guard = self.identity_init.lock().map_err(ort_err)?;
        if let Some(snapshot) = self.snapshot.get() {
            return Ok(snapshot);
        }
        #[cfg(test)]
        self.hash_passes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut hash = blake3::Hasher::new();
        hash.update(FEATURE_POLICY.as_bytes());
        let mut assets = [Vec::new(), Vec::new()];
        for (name, bytes) in FILES.into_iter().zip(&mut assets) {
            let path = self.dir.join(name);
            *bytes = capture_asset(&path, &mut check)?;
            #[cfg(test)]
            self.bytes_read
                .fetch_add(bytes.len(), std::sync::atomic::Ordering::Relaxed);
            hash.update(name.as_bytes());
            hash.update(bytes);
        }
        let [graph, external] = assets;
        let snapshot = AudioSnapshot {
            identity: hash.finalize().to_hex().to_string(),
            graph,
            external,
        };
        Ok(self.snapshot.get_or_init(|| snapshot))
    }

    fn with_session<T>(
        &self,
        threads: Option<usize>,
        run: impl FnOnce(&mut Session) -> Result<T>,
    ) -> Result<T> {
        let snapshot = self.snapshot()?;
        let mut state = self.session.lock().map_err(ort_err)?;
        if state.is_none() {
            *state = Some(snapshot.session(threads)?);
            #[cfg(test)]
            self.session_loads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        run(state.as_mut().expect("initialized"))
    }
}

struct AudioSnapshot {
    identity: String,
    graph: Vec<u8>,
    external: Vec<u8>,
}
impl AudioSnapshot {
    fn session(&self, threads: Option<usize>) -> Result<Session> {
        let mut builder = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?;
        if let Some(threads) = threads {
            builder = builder.with_intra_threads(threads).map_err(ort_err)?;
        }
        // rc.13 owns this Cow in the builder, not in the returned Session.
        // Keep the snapshot for retry, and the builder alive through commit.
        // The q8 snapshot retains ~325 MiB; this clone adds ~325 MiB transiently.
        builder = builder
            .with_external_initializer_file_in_memory(
                "audio_encoder_quantized.onnx_data",
                std::borrow::Cow::Owned(self.external.clone()),
            )
            .map_err(ort_err)?;
        builder.commit_from_memory(&self.graph).map_err(ort_err)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CapturePoint {
    Opened,
    Read,
}

fn capture_asset(
    path: &Path,
    check: &mut impl FnMut(&Path, CapturePoint) -> Result<()>,
) -> Result<Vec<u8>> {
    let mut file = regular_audio_file(path)?;
    let before = file.metadata().map_err(|error| io_error(path, error))?;
    let changed = || Error::Embedding(format!("audio encoder asset changed: {}", path.display()));
    let capacity = usize::try_from(before.len())
        .ok()
        .and_then(|len| len.checked_add(1))
        .ok_or_else(|| Error::Embedding("audio encoder asset is too large".into()))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| Error::Embedding(format!("cannot allocate audio snapshot: {error}")))?;
    check(path, CapturePoint::Opened)?;
    // Read at most the original length plus one sentinel byte, even if a
    // concurrent writer keeps appending. Never publish a partial capture.
    (&mut file)
        .take(capacity as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| io_error(path, error))?;
    check(path, CapturePoint::Read)?;
    let after = file.metadata().map_err(|error| io_error(path, error))?;
    if bytes.len() as u64 != before.len()
        || before.len() != after.len()
        || before.modified().map_err(|error| io_error(path, error))?
            != after.modified().map_err(|error| io_error(path, error))?
    {
        return Err(changed());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if (
            before.dev(),
            before.ino(),
            before.ctime(),
            before.ctime_nsec(),
        ) != (after.dev(), after.ino(), after.ctime(), after.ctime_nsec())
        {
            return Err(changed());
        }
    }
    Ok(bytes)
}

fn regular_audio_file(path: &Path) -> Result<std::fs::File> {
    let map_error = |error: std::io::Error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            Error::ModelNotFound(
                "audio encoder is missing; run seyk setup audio, then retry".into(),
            )
        } else {
            io_error(path, error)
        }
    };
    let not_regular = || {
        Error::ModelNotFound(format!(
            "audio encoder asset is not a regular file: {}",
            path.display()
        ))
    };
    // Follow symlinks, but reject special targets before opening on all platforms.
    if !std::fs::metadata(path).map_err(map_error)?.is_file() {
        return Err(not_regular());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // A regular target may be replaced by a FIFO after the metadata check.
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path).map_err(map_error)?;
    // Check the opened handle (fstat/S_ISREG on Unix), not the path again.
    if !file.metadata().map_err(map_error)?.is_file() {
        return Err(not_regular());
    }
    Ok(file)
}

pub struct Features {
    pub values: Vec<f32>,
    pub mask: Vec<bool>,
}

/// Cached Gemma4AudioFeatureExtractor: 128 HTK mel filters, magnitude (not power),
/// periodic Hann, semicausal zero padding, 321-sample analysis window.
pub fn preprocess(samples: &[f32]) -> Result<Features> {
    if samples.len() < 161 || samples.len() > 160000 || samples.iter().any(|x| !x.is_finite()) {
        return Err(Error::Config(
            "audio requires 161..160000 finite mono samples".into(),
        ));
    }
    let padded = samples.len().div_ceil(128) * 128;
    let frames = (padded + 160).saturating_sub(321) / 160 + 1;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(512);
    let window: Vec<f32> = (0..320)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 320.0).cos()) as f32)
        .collect();
    let max_mel = 2595.0_f64 * (1.0_f64 + 8000.0 / 700.0).log10();
    let freqs: Vec<f64> = (0..130)
        .map(|i| 700.0 * (10.0_f64.powf(max_mel * i as f64 / 129.0 / 2595.0) - 1.0))
        .collect();
    let mut filters = vec![0.0_f64; 257 * 128];
    for k in 0..257 {
        for m in 0..128 {
            let hz = k as f64 * 8000.0 / 256.0;
            filters[k * 128 + m] = ((hz - freqs[m]) / (freqs[m + 1] - freqs[m]))
                .min((freqs[m + 2] - hz) / (freqs[m + 2] - freqs[m + 1]))
                .max(0.0);
        }
    }
    let mut values = Vec::with_capacity(frames * 128);
    let mut mask = Vec::with_capacity(frames);
    for frame in 0..frames {
        let valid = frame * 160 + 160 < samples.len();
        mask.push(valid);
        let mut buf = vec![Complex::new(0.0, 0.0); 512];
        for i in 0..320 {
            let pos = (frame * 160 + i) as isize - 160;
            if pos >= 0 && (pos as usize) < samples.len() {
                buf[i].re = (samples[pos as usize] * window[i]) as f64;
            }
        }
        fft.process(&mut buf);
        let mag: Vec<f64> = buf[..257].iter().map(|x| x.norm()).collect();
        for m in 0..128 {
            let mel: f64 = (0..257).map(|k| mag[k] * filters[k * 128 + m]).sum();
            values.push(if valid {
                (mel + 0.001).ln() as f32
            } else {
                0.0
            });
        }
    }
    Ok(Features { values, mask })
}

impl EmbeddingGemma2 {
    pub fn embed_audio(&self, samples: &[f32], sample_rate: u32) -> Result<Embedding> {
        if sample_rate != 16000 {
            return Err(Error::Config("audio requires 16000 Hz".into()));
        }
        let pre = preprocess(samples)?;
        self.embed_audio_features(pre)
    }
    fn embed_audio_features(&self, pre: Features) -> Result<Embedding> {
        let audio = self.audio.as_ref().ok_or(Error::UnsupportedModality {
            modality: Modality::Audio,
        })?;
        let mut mask = pre.mask.clone();
        for _ in 0..2 {
            mask = mask.iter().step_by(2).copied().collect();
        }
        let tokens = mask.iter().filter(|x| **x).count();
        let features = audio.with_session(self.config.threads, |s| {
            let out = s.run(ort::inputs![
                "input_features" => Tensor::from_array(([1usize,pre.mask.len(),128],pre.values)).map_err(ort_err)?,
                "input_features_mask" => Tensor::from_array(([1usize,pre.mask.len()],pre.mask)).map_err(ort_err)?,
            ]).map_err(ort_err)?;
            let (shape, data) = out["audio_features"]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            if shape.as_ref() != [tokens as i64, 512] {
                return Err(Error::Embedding(format!(
                    "audio shape mismatch {shape:?} vs {tokens}"
                )));
            }
            Ok(data.to_vec())
        })?;
        let ids = self
            .template_token_ids(&format!("<|audio>{}<audio|>", "<|audio|>".repeat(tokens)))?
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        if ids.iter().filter(|x| **x == 258881).count() != tokens
            || !ids.contains(&256000)
            || !ids.contains(&258883)
        {
            return Err(Error::Embedding(
                "audio token layout truncated or invalid".into(),
            ));
        }
        Ok(self
            .run_text(&[&ids], None, Some((features, tokens)))?
            .remove(0))
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;
    use std::sync::atomic::Ordering;

    const GOLDEN: &str = "60fe96207a5119529b8f6d86a53e2fac20fdd06d8b66e9a24dae0a61e8fd04b6";

    fn files(dir: &Path) {
        std::fs::create_dir_all(dir.join("onnx")).unwrap();
        std::fs::write(dir.join(FILES[0]), b"graph-v1").unwrap();
        std::fs::write(dir.join(FILES[1]), b"weights-v1").unwrap();
    }

    #[test]
    fn construction_is_read_free_and_digest_matches_pre_lazy_golden() {
        let dir = tempfile::tempdir().unwrap();
        files(dir.path());
        let encoder = AudioEncoder::new(dir.path());
        assert_eq!(encoder.hash_passes.load(Ordering::Relaxed), 0);
        assert_eq!(encoder.bytes_read.load(Ordering::Relaxed), 0);
        // Captured with the pre-change constructor's streaming recipe.
        assert_eq!(encoder.identity().unwrap(), GOLDEN);
        assert_eq!(encoder.bytes_read.load(Ordering::Relaxed), 18);
        assert_eq!(encoder.identity().unwrap(), GOLDEN);
        assert_eq!(encoder.hash_passes.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn failures_do_not_publish_and_can_be_retried() {
        let dir = tempfile::tempdir().unwrap();
        let encoder = AudioEncoder::new(dir.path());
        assert!(matches!(encoder.identity(), Err(Error::ModelNotFound(_))));
        assert!(encoder.snapshot.get().is_none());
        files(dir.path());
        std::fs::remove_file(dir.path().join(FILES[1])).unwrap();
        assert!(encoder.identity().is_err());
        assert!(encoder.snapshot.get().is_none());
        // A readable graph followed by an unreadable asset must not publish a
        // partial digest. A directory must be rejected before reading.
        std::fs::create_dir(dir.path().join(FILES[1])).unwrap();
        assert!(encoder.identity().is_err());
        assert!(encoder.snapshot.get().is_none());
        std::fs::remove_dir(dir.path().join(FILES[1])).unwrap();
        std::fs::write(dir.path().join(FILES[1]), b"weights-v1").unwrap();
        assert_eq!(encoder.identity().unwrap(), GOLDEN);
        assert_eq!(encoder.hash_passes.load(Ordering::Relaxed), 4);
    }

    #[cfg(unix)]
    fn bounded(test: impl FnOnce() + Send + 'static) {
        let (send, receive) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            test();
            send.send(()).unwrap();
        });
        receive
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("special-file identity preparation must finish within two seconds");
        worker.join().unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn fifo_without_writer_fails_promptly_and_regular_replacement_retries() {
        use std::os::unix::{ffi::OsStrExt, fs::symlink};
        bounded(|| {
            for name in FILES {
                for through_symlink in [false, true] {
                    let dir = tempfile::tempdir().unwrap();
                    files(dir.path());
                    let path = dir.path().join(name);
                    std::fs::remove_file(&path).unwrap();
                    let fifo = if through_symlink {
                        dir.path().join("fifo")
                    } else {
                        path.clone()
                    };
                    let c_path = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
                    // No writer is ever opened, so a blocking open would hang.
                    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
                    if through_symlink {
                        symlink(&fifo, &path).unwrap();
                    }
                    let encoder = AudioEncoder::new(dir.path());
                    assert!(
                        matches!(encoder.identity(), Err(Error::ModelNotFound(message)) if message.contains("not a regular file"))
                    );
                    assert!(encoder.snapshot.get().is_none());
                    std::fs::remove_file(&fifo).unwrap();
                    let bytes: &[u8] = if name == FILES[0] {
                        b"graph-v1"
                    } else {
                        b"weights-v1"
                    };
                    std::fs::write(&fifo, bytes).unwrap();
                    assert_eq!(encoder.identity().unwrap(), GOLDEN);
                    assert_eq!(encoder.hash_passes.load(Ordering::Relaxed), 2);
                }
            }
        });
    }

    #[test]
    #[cfg(unix)]
    fn device_targets_fail_promptly_without_publishing_identity() {
        bounded(|| {
            for name in FILES {
                let dir = tempfile::tempdir().unwrap();
                files(dir.path());
                let path = dir.path().join(name);
                std::fs::remove_file(&path).unwrap();
                std::os::unix::fs::symlink("/dev/null", &path).unwrap();
                let encoder = AudioEncoder::new(dir.path());
                assert!(
                    matches!(encoder.identity(), Err(Error::ModelNotFound(message)) if message.contains("not a regular file"))
                );
                assert!(encoder.snapshot.get().is_none());
            }
        });
    }

    #[test]
    fn concurrent_identity_requests_publish_one_complete_hash() {
        let dir = tempfile::tempdir().unwrap();
        files(dir.path());
        let encoder = AudioEncoder::new(dir.path());
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        assert_eq!(encoder.identity().unwrap(), GOLDEN);
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
        });
        assert_eq!(encoder.hash_passes.load(Ordering::Relaxed), 1);
        assert_eq!(encoder.bytes_read.load(Ordering::Relaxed), 18);
    }

    #[test]
    fn equal_size_tamper_with_restored_metadata_changes_new_identity() {
        for name in FILES {
            let dir = tempfile::tempdir().unwrap();
            files(dir.path());
            let original = AudioEncoder::new(dir.path());
            assert_eq!(original.identity().unwrap(), GOLDEN);
            let path = dir.path().join(name);
            let metadata = std::fs::metadata(&path).unwrap();
            let mut bytes = std::fs::read(&path).unwrap();
            bytes[0] ^= 1;
            let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            use std::io::Write;
            file.write_all(&bytes).unwrap();
            file.set_times(std::fs::FileTimes::new().set_modified(metadata.modified().unwrap()))
                .unwrap();
            let after = file.metadata().unwrap();
            assert_eq!(metadata.len(), after.len());
            assert_eq!(metadata.modified().unwrap(), after.modified().unwrap());
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                assert_eq!(metadata.ino(), after.ino());
            }
            assert_ne!(AudioEncoder::new(dir.path()).identity().unwrap(), GOLDEN);
            assert_eq!(original.identity().unwrap(), GOLDEN);
            let snapshot = original.snapshot().unwrap();
            assert_eq!(snapshot.graph, b"graph-v1");
            assert_eq!(snapshot.external, b"weights-v1");
        }
    }

    #[test]
    #[ignore = "requires pinned local text weights; never downloads"]
    fn text_build_and_embedding_read_zero_audio_bytes() {
        let model = EmbeddingGemma2::builder()
            .vision(false)
            .audio(true)
            .threads(1)
            .build()
            .unwrap();
        let audio = model.audio.as_ref().unwrap();
        assert_eq!(model.input_identity(Modality::Text).unwrap(), "");
        assert!(model.supports(Modality::Audio));
        assert!(model.embed_query("synthetic text-only query").is_ok());
        assert!(
            model
                .embed_text_document(None, "synthetic text-only document")
                .is_ok()
        );
        assert_eq!(audio.hash_passes.load(Ordering::Relaxed), 0);
        assert_eq!(audio.bytes_read.load(Ordering::Relaxed), 0);
        assert!(audio.snapshot.get().is_none());
    }
}

#[cfg(test)]
mod snapshot_tests;
