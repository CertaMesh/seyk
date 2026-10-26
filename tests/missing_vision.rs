#![cfg(all(feature = "cli", feature = "image"))]
mod common;
#[cfg(feature = "pdf-render")]
#[path = "../crates/seyk-core/tests/support/pdf_fixture.rs"]
mod fixture;

use seyk::embedder::gemma::{self, GemmaConfig, ModelSource};
use seyk::{Dimension, SearchEngine, SearchMode, SqliteCache};
use std::sync::Arc;

#[test]
fn missing_vision_profile_preserves_text_vectors_and_reports_skipped_media() {
    let temp = tempfile::tempdir().unwrap();
    let model_dir = temp.path().join("model");
    let docs = temp.path().join("docs");
    std::fs::create_dir_all(model_dir.join("onnx")).unwrap();
    std::fs::create_dir(&docs).unwrap();
    let mut requested = GemmaConfig::default();
    requested.source = ModelSource::Directory(model_dir.clone());
    requested.dimension = Dimension::D128;
    requested.vision = true;
    let mut text_only = requested.clone();
    text_only.vision = false;
    for file in text_only.files() {
        std::fs::write(model_dir.join(file), b"synthetic weights").unwrap();
    }
    std::fs::write(docs.join("note.txt"), "quartz lighthouse").unwrap();
    std::fs::write(docs.join("photo.png"), b"never decoded").unwrap();
    #[cfg(feature = "pdf-render")]
    std::fs::write(
        docs.join("scan.pdf"),
        fixture::pdf(&[""], fixture::Kind::EmptyPage),
    )
    .unwrap();
    let (effective, reason) = gemma::runtime_config(&requested).unwrap();
    assert!(!effective.vision);
    let reason = reason.unwrap();
    let identity = gemma::model_identity(&requested).unwrap();
    assert_eq!(identity, gemma::model_identity(&text_only).unwrap());
    let cache_path = temp.path().join("cache.sqlite");
    let fake = Arc::new(
        common::FakeEmbedder::with_identity(identity.clone())
            .with_missing_vision(Some(reason.clone())),
    );
    let engine = SearchEngine::builder()
        .embedder(fake.clone())
        .cache(SqliteCache::open(&cache_path).unwrap())
        .build()
        .unwrap();
    let profile_id = engine.profile_id().to_string();
    let result = engine
        .search("quartz")
        .mode(SearchMode::Semantic)
        .paths([&docs])
        .execute()
        .unwrap();
    assert!(result.degraded.is_none());
    assert_eq!(result.hits.len(), 1);
    assert!(result.hits[0].signals.semantic.is_some());
    assert!(fake.resource_count() > 0);
    assert_eq!(result.stats.sync.skipped.len(), 1);
    assert_eq!(result.stats.sync.skipped[0].reason, reason);
    assert!(reason.contains("onnx/vision_encoder_quantized.onnx_data"));
    assert!(reason.contains("seyk model download"));
    #[cfg(feature = "pdf-render")]
    {
        assert_eq!(result.stats.sync.page_diagnostics.len(), 1);
        assert_eq!(result.stats.sync.page_diagnostics[0].reason, reason);
        assert_eq!(result.stats.sync.page_diagnostics[0].page, 1);
    }
    drop(engine);
    let fake = Arc::new(common::FakeEmbedder::with_identity(identity));
    let engine = SearchEngine::builder()
        .embedder(fake.clone())
        .cache(SqliteCache::open(&cache_path).unwrap())
        .build()
        .unwrap();
    assert_eq!(engine.profile_id(), profile_id);
    engine
        .search("quartz")
        .mode(SearchMode::Semantic)
        .paths([&docs])
        .execute()
        .unwrap();
    assert_eq!(
        fake.resource_count(),
        0,
        "text-only reuses unchanged text vectors"
    );
    std::fs::write(docs.join("note.txt"), "quartz changed text").unwrap();
    engine
        .search("quartz")
        .mode(SearchMode::Semantic)
        .paths([&docs])
        .execute()
        .unwrap();
    assert!(fake.resource_count() > 0, "changed inputs must embed again");

    let profile = |text_only: bool| {
        let mut cmd = assert_cmd::Command::cargo_bin("seyk").unwrap();
        cmd.args(["model", "profile", "--json", "--model"])
            .arg(&model_dir)
            .args(["--dimension", "128"]);
        if text_only {
            cmd.arg("--text-only");
        }
        serde_json::from_slice::<serde_json::Value>(&cmd.assert().success().get_output().stdout)
            .unwrap()
    };
    let implicit = profile(false);
    let explicit = profile(true);
    assert_eq!(implicit["profile_id"], profile_id);
    assert_eq!(implicit["profile"], explicit["profile"]);
    assert_eq!(implicit["model_present"], false);
    assert_eq!(explicit["model_present"], true);
    assert_eq!(
        implicit["missing_files"],
        serde_json::json!([
            "onnx/vision_encoder_quantized.onnx",
            "onnx/vision_encoder_quantized.onnx_data"
        ])
    );
}
