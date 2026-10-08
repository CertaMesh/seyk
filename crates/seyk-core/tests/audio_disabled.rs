#![cfg(not(feature = "audio"))]
mod common;
use seyk_core::{Dimension, SearchEngine};
#[test]
fn default_core_discovers_audio_but_does_not_decode_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("track.wav");
    std::fs::write(&path, b"no decoder should touch this").unwrap();
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "no-audio"))
        .build()
        .unwrap();
    let stats = engine.index().paths([path]).execute().unwrap().sync;
    assert_eq!(stats.resources_updated, 0);
    assert_eq!(stats.skipped.len(), 1);
    assert!(stats.skipped[0].reason.contains("audio"));
}
