//! End-to-end synthetic sync workload, compatible with the pre-fix API.
mod common;

use common::FakeEmbedder;
use seyk_core::segment::{TextSegmenter, TextSegmenterConfig};
use seyk_core::{Dimension, SearchEngine, SqliteCache};
use std::{fs, time::Instant};

#[test]
#[ignore = "synthetic 4000-resource before/after sync measurement; no models or user content"]
fn measure_4000_changed_resources_sync() {
    let temp = tempfile::tempdir().unwrap();
    let notes = temp.path().join("notes");
    fs::create_dir(&notes).unwrap();
    let words = (0..31)
        .map(|i| format!("syntheticword{i:02}"))
        .collect::<Vec<_>>()
        .join(" ");
    for i in 0..4000 {
        fs::write(
            notes.join(format!("note{i:04}.txt")),
            format!("old{i:04} {words}"),
        )
        .unwrap();
    }
    let engine = SearchEngine::builder()
        .embedder(FakeEmbedder::new(Dimension::D768, "sync-cost"))
        .cache(SqliteCache::open(temp.path().join("cache.sqlite3")).unwrap())
        .segmenter(TextSegmenter::new(TextSegmenterConfig {
            min_chars: 0,
            target_chars: 4096,
            max_chars: 8192,
            overlap_chars: 0,
        }))
        .build()
        .unwrap();
    assert_eq!(
        engine
            .index()
            .paths([&notes])
            .execute()
            .unwrap()
            .sync
            .resources_updated,
        4000
    );
    for i in 0..4000 {
        fs::write(
            notes.join(format!("note{i:04}.txt")),
            format!("changed{i:04} {words}"),
        )
        .unwrap();
    }
    let label = std::env::var("SEYK_SYNC_MEASUREMENT_LABEL").unwrap_or_else(|_| "current".into());
    eprintln!(
        "SYNC_COST {label}: starting sync of 4000 changed resources / 4000 segments / 32 body tokens each / 4000 768-f32 vectors; 128000 body-token occurrences; sources in temporary directory"
    );
    let start = Instant::now();
    let report = engine.index().paths([&notes]).execute().unwrap();
    let elapsed = start.elapsed();
    assert_eq!(report.sync.resources_updated, 4000);
    assert_eq!(report.sync.segments_embedded, 4000);
    assert_eq!(report.resources_removed, 0);
    eprintln!(
        "SYNC_COST {label}: 4000 changed resources / 4000 segments / 32 body tokens each / 4000 768-f32 vectors: {elapsed:?}"
    );
}
