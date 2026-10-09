//! Public/synthetic audio only. One invocation per file/mode gives a fresh RSS peak.
#[cfg(all(feature = "audio", feature = "onnx", unix))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use seyk::{
        Embedder, Embedding, EmbeddingRequest, Modality, ModelIdentity, ResourceId, SearchEngine,
        audio,
    };
    use std::{
        io::Seek,
        sync::{Arc, Mutex},
        time::Instant,
    };
    fn load() -> [f64; 3] {
        let mut l = [0.0; 3];
        unsafe {
            assert_eq!(libc::getloadavg(l.as_mut_ptr(), 3), 3);
        };
        l
    }
    fn rss() -> u64 {
        let mut r = std::mem::MaybeUninit::<libc::rusage>::uninit();
        unsafe {
            assert_eq!(libc::getrusage(libc::RUSAGE_SELF, r.as_mut_ptr()), 0);
            let n = r.assume_init().ru_maxrss as u64;
            if cfg!(target_os = "macos") {
                n
            } else {
                n * 1024
            }
        }
    }
    const LIMIT_SECS: u64 = 7200;
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 || !matches!(args[1].as_str(), "prepare" | "index") {
        return Err("usage: bench_audio <prepare|index> <audio-file> <duration-seconds>".into());
    }
    let mode = &args[1];
    let path = std::fs::canonicalize(&args[2])?;
    let duration: f64 = args[3]
        .parse()
        .map_err(|_| "usage: bench_audio <prepare|index> <audio-file> <duration-seconds>")?;
    if !duration.is_finite() || duration <= 0.0 {
        return Err(
            "usage: bench_audio <prepare|index> <audio-file> <positive-duration-seconds>".into(),
        );
    }
    let id = ResourceId::from_path(&path)?;
    let before = load();
    let started = Instant::now();
    if mode == "prepare" {
        let mut source = audio::VerifiedReader::new(
            Box::new(std::fs::File::open(&path)?),
            &id,
            seyk::discovery::audio_byte_limit(LIMIT_SECS)?,
        )?;
        let hash = source.stats();
        std::io::copy(&mut source, &mut std::io::sink())?;
        let verify = source.stats();
        source.rewind()?;
        let decode = Instant::now();
        let mut windows = 0;
        let result = audio::windows_verified(source, &id, LIMIT_SECS, |_, _| {
            windows += 1;
            Ok(())
        });
        let decode = decode.elapsed().as_secs_f64();
        println!(
            "{}",
            serde_json::json!({"mode":mode,"file":path.file_name(),"duration_s":duration,"bytes":hash.hash_bytes,"hash_s":hash.hash_time.as_secs_f64(),"hash_MB_s":hash.hash_bytes as f64/1e6/hash.hash_time.as_secs_f64(),"verify_s":verify.verify_time.as_secs_f64(),"verify_MB_s":verify.verified_bytes as f64/1e6/verify.verify_time.as_secs_f64(),"decode_and_verify_s":decode,"windows":windows,"windows_s":windows as f64/decode,"seconds_per_audio_minute":decode*60.0/duration,"peak_rss_bytes":rss(),"load_before":before,"load_after":load(),"error":result.err().map(|e|e.to_string())})
        );
    } else {
        struct Timed {
            model: seyk::embedder::gemma::GemmaEmbedder,
            timing: Arc<Mutex<(f64, usize)>>,
        }
        impl Embedder for Timed {
            fn identity(&self) -> &ModelIdentity {
                self.model.identity()
            }
            fn input_identity(&self, m: Modality) -> &str {
                self.model.input_identity(m)
            }
            fn supports(&self, m: Modality) -> bool {
                self.model.supports(m)
            }
            fn embed_batch(
                &self,
                r: &[EmbeddingRequest<'_>],
            ) -> seyk_embed::Result<Vec<Embedding>> {
                let t = Instant::now();
                let result = self.model.embed_batch(r);
                let mut s = self.timing.lock().unwrap();
                s.0 += t.elapsed().as_secs_f64();
                s.1 += r.len();
                result
            }
        }
        let timing = Arc::new(Mutex::new((0.0, 0)));
        let model = seyk::embedder::gemma::EmbeddingGemma2::builder()
            .dimension(256)
            .vision(false)
            .audio(true)
            .threads(1)
            .build()?;
        let engine = SearchEngine::builder()
            .max_audio_duration_secs(LIMIT_SECS)
            .embedder(Timed {
                model,
                timing: timing.clone(),
            })
            .build()?;
        let t = Instant::now();
        let report = engine.index().paths([&path]).execute()?;
        let elapsed = t.elapsed().as_secs_f64();
        let (embed, n) = *timing.lock().unwrap();
        println!(
            "{}",
            serde_json::json!({"mode":mode,"file":path.file_name(),"duration_s":duration,"index_s":elapsed,"embedding_s":embed,"embedded_windows":n,"embedding_windows_s":n as f64/embed,"index_windows_s":n as f64/elapsed,"seconds_per_audio_minute":elapsed*60.0/duration,"peak_rss_bytes":rss(),"load_before":before,"load_after":load(),"total_with_load_s":started.elapsed().as_secs_f64(),"skipped":report.sync.skipped})
        );
        if duration <= LIMIT_SECS as f64 && !report.sync.skipped.is_empty() {
            return Err("unexpected benchmark rejection".into());
        }
        if duration > LIMIT_SECS as f64 {
            assert_eq!(n, 0);
            assert_eq!(report.sync.skipped.len(), 1);
        }
    }
    Ok(())
}
#[cfg(not(all(feature = "audio", feature = "onnx", unix)))]
fn main() {
    eprintln!("requires audio + onnx, and Unix RSS/load counters");
}
