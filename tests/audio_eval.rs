//! Frozen synthetic evaluation through the production engine, local weights only.
#![cfg(all(feature = "audio", feature = "onnx"))]
use serde::Deserialize;
use seyk::{ResourceLocator, SearchEngine, SearchMode};
use seyk_embeddinggemma::EmbeddingGemma2;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};
#[derive(Deserialize)]
struct Recording {
    id: usize,
    path: PathBuf,
    format: String,
    duration_ms: u64,
}
#[derive(Deserialize)]
struct Query {
    query: String,
    recording: usize,
    start_ms: f64,
    end_ms: f64,
    boundary: bool,
}
#[derive(Deserialize)]
struct Corpus {
    dataset: String,
    scripts_sha256: String,
    recordings: Vec<Recording>,
    queries: Vec<Query>,
}
#[test]
#[ignore = "requires generated synthetic set in SEYK_AUDIO_MANIFEST and pinned local q8 audio weights"]
fn retrieval_gate() {
    let path = std::env::var("SEYK_AUDIO_MANIFEST").expect("run scripts/audio/generate.py first");
    let mut corpus: Corpus = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for recording in &mut corpus.recordings {
        recording.path = std::fs::canonicalize(&recording.path).unwrap();
    }
    assert!(corpus.queries.len() >= 40);
    let paths: Vec<_> = corpus.recordings.iter().map(|r| r.path.clone()).collect();
    let mut reports = Vec::new();
    for dimension in [256, 768] {
        let model = EmbeddingGemma2::builder()
            .dimension(dimension)
            .vision(false)
            .audio(true)
            .threads(1)
            .build()
            .unwrap();
        let engine = SearchEngine::builder().embedder(model).build().unwrap();
        let start = Instant::now();
        let stats = engine.index().paths(&paths).execute().unwrap();
        assert!(stats.sync.skipped.is_empty(), "{:?}", stats.sync.skipped);
        let seconds = start.elapsed().as_secs_f64();
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for query in &corpus.queries {
            let results = engine
                .search(&query.query)
                .paths(&paths)
                .sync(false)
                .mode(SearchMode::Semantic)
                .limit(100)
                .execute()
                .unwrap();
            let recording = corpus
                .recordings
                .iter()
                .find(|r| r.id == query.recording)
                .unwrap();
            assert!(results.hits.iter().all(|h| h.preview.is_none()));
            let rank = results
                .hits
                .iter()
                .position(|hit| {
                    if hit.resource.to_file_path().as_ref() != Some(&recording.path) {
                        return false;
                    }
                    let ResourceLocator::Time { start_ms, end_ms } = hit.locator else {
                        return false;
                    };
                    // Evidence must contain the whole annotated phrase, not merely overlap it.
                    start_ms as f64 <= query.start_ms && end_ms as f64 >= query.end_ms
                })
                .map_or(usize::MAX, |i| i + 1);
            groups.entry("all".into()).or_default().push(rank);
            groups
                .entry(recording.format.clone())
                .or_default()
                .push(rank);
            if query.boundary {
                groups.entry("boundary".into()).or_default().push(rank);
            }
        }
        let metrics: BTreeMap<_,_> = groups.iter().map(|(group,ranks)| (group.clone(), serde_json::json!({"queries":ranks.len(),"recall_at_5":ranks.iter().filter(|r| **r<=5).count() as f64/ranks.len() as f64,"mrr_at_20":ranks.iter().map(|r| if *r<=20 {1.0/ *r as f64}else{0.0}).sum::<f64>()/ranks.len() as f64,"ranks":ranks}))).collect();
        let recall = metrics["all"]["recall_at_5"].as_f64().unwrap();
        let report = serde_json::json!({"dataset":corpus.dataset,"scripts_sha256":corpus.scripts_sha256,"dimension":dimension,"seconds":seconds,"seconds_per_audio_minute":seconds*60000.0/corpus.recordings.iter().map(|r|r.duration_ms).sum::<u64>() as f64,"metrics":metrics});
        println!("AUDIO_EVAL {report}");
        reports.push(report);
        if let Ok(path) = std::env::var("SEYK_AUDIO_REPORT") {
            std::fs::write(path, serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
        }
        assert!(recall >= 0.8, "{dimension}-dim gate {recall}");
    }
}

#[test]
#[ignore = "largest allowed synthetic file; local q8 weights; run alone for RSS measurement"]
fn largest_allowed_file() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("largest.wav");
    let bytes = 32u32 * 1024 * 1024;
    let samples = (bytes - 44) / 2;
    let mut file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    file.write_all(b"RIFF").unwrap();
    file.write_all(&(bytes - 8).to_le_bytes()).unwrap();
    file.write_all(b"WAVEfmt ").unwrap();
    file.write_all(&16u32.to_le_bytes()).unwrap();
    file.write_all(&1u16.to_le_bytes()).unwrap();
    file.write_all(&1u16.to_le_bytes()).unwrap();
    file.write_all(&16000u32.to_le_bytes()).unwrap();
    file.write_all(&32000u32.to_le_bytes()).unwrap();
    file.write_all(&2u16.to_le_bytes()).unwrap();
    file.write_all(&16u16.to_le_bytes()).unwrap();
    file.write_all(b"data").unwrap();
    file.write_all(&(bytes - 44).to_le_bytes()).unwrap();
    for i in 0..samples {
        file.write_all(&((i.wrapping_mul(7919) % 32767) as i16).to_le_bytes())
            .unwrap();
    }
    file.flush().unwrap();
    drop(file);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), u64::from(bytes));
    let model = EmbeddingGemma2::builder()
        .dimension(256)
        .vision(false)
        .audio(true)
        .threads(1)
        .build()
        .unwrap();
    let engine = SearchEngine::builder().embedder(model).build().unwrap();
    let start = Instant::now();
    let stats = engine.index().paths([&path]).execute().unwrap().sync;
    assert!(stats.skipped.is_empty(), "{:?}", stats.skipped);
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "AUDIO_LARGEST bytes={bytes} duration_seconds={} seconds={seconds} seconds_per_audio_minute={} segments_embedded={}",
        samples as f64 / 16000.0,
        seconds * 960000.0 / samples as f64,
        stats.segments_embedded
    );
}
