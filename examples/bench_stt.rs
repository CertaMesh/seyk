//! Public synthetic audio only. Run each duration in a fresh process for RSS.
#[cfg(all(feature = "audio-stt", unix))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use seyk::{
        FsProvider, Modality, ResourceId, ResourceLocator, ResourceMetadata, audio::AudioExtractor,
    };
    use std::time::Instant;
    fn load() -> [f64; 3] {
        let mut value = [0.0; 3];
        unsafe {
            assert_eq!(libc::getloadavg(value.as_mut_ptr(), 3), 3);
        }
        value
    }
    fn rss() -> u64 {
        let mut value = std::mem::MaybeUninit::<libc::rusage>::uninit();
        unsafe {
            assert_eq!(libc::getrusage(libc::RUSAGE_SELF, value.as_mut_ptr()), 0);
            let value = value.assume_init().ru_maxrss as u64;
            if cfg!(target_os = "macos") {
                value
            } else {
                value * 1024
            }
        }
    }
    let args: Vec<_> = std::env::args().collect();
    let usage =
        "usage: bench_stt <model-directory> <public-wav> <10|30> [start-seconds [warm-runs]]";
    if !(4..=6).contains(&args.len()) || !matches!(args[3].as_str(), "10" | "30") {
        return Err(usage.into());
    }
    let seconds: u64 = args[3].parse()?;
    let start_seconds: u64 = args.get(4).map(|v| v.parse()).transpose()?.unwrap_or(0);
    let runs: usize = args.get(5).map(|v| v.parse()).transpose()?.unwrap_or(20);
    if start_seconds > 7200 - seconds || (runs > 0 && runs < 20) {
        return Err(usage.into());
    }
    let path = std::fs::canonicalize(&args[2])?;
    let source = ResourceId::from_path(&path)?;
    let file = std::fs::File::open(&path)?;
    let size = file.metadata()?.len();
    let mut hash = blake3::Hasher::new();
    hash.update_reader(file)?;
    let metadata = ResourceMetadata {
        media_type: Some("audio/wav".into()),
        modality: Modality::Audio,
        size_bytes: size,
        modified_ns: None,
        content_hash: hash.finalize().to_hex().to_string(),
    };
    let locator = ResourceLocator::Time {
        start_ms: start_seconds * 1000,
        end_ms: (start_seconds + seconds) * 1000,
    };
    let before = load();
    let start = Instant::now();
    let whisper = seyk::stt::Whisper::load(&args[1])?;
    let cold = start.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({"phase":"cold_load","seconds":cold,"load_before":before,"load_after":load(),"peak_rss_bytes":rss()})
    );
    let extractor = AudioExtractor::new(whisper);
    let start = Instant::now();
    let before = load();
    let first = extractor.extract(&FsProvider, &source, &locator, &metadata, 7200)?;
    assert!(!first.text.is_empty());
    drop(first);
    println!(
        "{}",
        serde_json::json!({"phase":"first_extract","start_seconds":start_seconds,"audio_seconds":seconds,"source_bytes":size,"seconds":start.elapsed().as_secs_f64(),"load_before":before,"load_after":load(),"peak_rss_bytes":rss()})
    );
    if runs == 0 {
        return Ok(());
    }
    let before = load();
    let mut times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let result = extractor.extract(&FsProvider, &source, &locator, &metadata, 7200)?;
        times.push(start.elapsed().as_secs_f64());
        assert!(!result.text.is_empty());
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"phase":"warm_extract","audio_seconds":seconds,"n":times.len(),"p50_seconds":times[runs.div_ceil(2)-1],"p95_seconds":times[(runs*95).div_ceil(100)-1],"seconds_per_audio_minute_p50":times[runs.div_ceil(2)-1]*60.0/seconds as f64,"peak_rss_bytes":rss(),"load_before":before,"load_after":load()})
    );
    Ok(())
}
#[cfg(not(all(feature = "audio-stt", unix)))]
fn main() {
    eprintln!("bench_stt requires audio-stt and Unix RSS/load counters");
}
