#![cfg(not(feature = "pdf"))]
mod common;
use seyk_core::{Dimension, SearchEngine};

#[test]
fn pdf_is_discovered_but_reports_the_disabled_feature() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("file.PDF");
    std::fs::write(&path, b"not raw UTF-8 text to index").unwrap();
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "no-pdf"))
        .build()
        .unwrap();
    let result = engine.search("text").paths([&path]).execute().unwrap();
    assert_eq!(result.stats.sync.resources_discovered, 1);
    assert_eq!(result.stats.sync.resources_updated, 0);
    assert_eq!(result.stats.sync.skipped.len(), 1);
    assert_eq!(
        result.stats.sync.skipped[0].reason,
        "PDF support is not compiled in (feature `pdf`)"
    );
    assert!(result.hits.is_empty());
}
