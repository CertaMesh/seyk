//! Compare model construction and the first search without downloading weights.
//! Run with `cargo run --release --example bench_audio_identity --features audio`.
use seyk::embedder::gemma::EmbeddingGemma2;
use seyk::{SearchEngine, SearchMode};
use std::sync::Arc;
use std::time::Instant;

#[cfg(unix)]
fn load() -> [f64; 3] {
    let mut values = [0.0; 3];
    // SAFETY: values has space for all three load averages.
    unsafe {
        libc::getloadavg(values.as_mut_ptr(), 3);
    }
    values
}

#[cfg(not(unix))]
fn load() -> [f64; 3] {
    [f64::NAN; 3]
}

fn main() {
    let dir = tempfile::tempdir().unwrap();
    let text = dir.path().join("text.txt");
    std::fs::write(
        &text,
        "A synthetic document about finding information locally.",
    )
    .unwrap();
    let wave = dir.path().join("signal.wav");
    let mut bytes = Vec::new();
    bytes.extend(b"RIFF");
    bytes.extend(32036u32.to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(16000u32.to_le_bytes());
    bytes.extend(32000u32.to_le_bytes());
    bytes.extend(2u16.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend(32000u32.to_le_bytes());
    for i in 0..16000 {
        let sample = ((i as f64 * 440.0 * std::f64::consts::TAU / 16000.0).sin() * 8000.0) as i16;
        bytes.extend(sample.to_le_bytes());
    }
    std::fs::write(&wave, bytes).unwrap();
    for (name, path) in [("text", text), ("audio", wave)] {
        let initial_load = load();
        let start = Instant::now();
        let model = EmbeddingGemma2::builder()
            .vision(false)
            .audio(true)
            .threads(1)
            .build()
            .unwrap();
        let engine = SearchEngine::builder()
            .embedder(Arc::new(model))
            .build()
            .unwrap();
        let search_load = load();
        let search_start = Instant::now();
        let result = engine
            .search("finding information")
            .paths([path])
            .mode(SearchMode::Semantic)
            .execute()
            .unwrap();
        assert!(
            result.stats.sync.skipped.is_empty(),
            "{:?}",
            result.stats.sync.skipped
        );
        assert!(!result.hits.is_empty());
        println!(
            "{name} first_search={:.6}s load={search_load:?}; construction_plus_search={:.6}s load={initial_load:?}",
            search_start.elapsed().as_secs_f64(),
            start.elapsed().as_secs_f64()
        );
    }
}
