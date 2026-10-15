//! Local model regressions for literal tokenizer control strings; never downloads.
#![cfg(feature = "cli")]

use serde_json::Value;
use seyk::{
    Embedder, Embedding, EmbeddingRequest, Modality, ModelIdentity, SearchEngine, SqliteCache,
};
use seyk_embeddinggemma::EmbeddingGemma2;
use std::{path::Path, process::Command};

fn literals() -> String {
    let fixture: Value = serde_json::from_str(include_str!(
        "../crates/seyk-embeddinggemma/tests/fixtures/special_tokens.json"
    ))
    .unwrap();
    fixture["added_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["content"].as_str().unwrap())
        .collect::<Vec<_>>()
        .join(" ")
}

fn cli(args: &[&str], path: &Path, cache: &Path) -> Value {
    let out = Command::new(env!("CARGO_BIN_EXE_seyk"))
        .args(args)
        .arg(path)
        .args(["--threads", "2", "--text-only", "--cache-path"])
        .arg(cache)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
#[ignore = "requires local q8 text weights (SEYK_MODEL or HF_HOME); no downloads"]
fn literal_special_tokens_index_and_are_findable() {
    let dir = tempfile::tempdir().unwrap();
    let corpus = dir.path().join("corpus");
    std::fs::create_dir(&corpus).unwrap();
    let path = corpus.join("literal-tokens.txt");
    let text = format!(
        "Gemma placeholder documentation: <start_of_image> {}.\n",
        literals()
    );
    std::fs::write(&path, &text).unwrap();
    std::fs::write(
        corpus.join("bread-recipe.txt"),
        "To bake bread, mix flour, water, yeast, and salt. Knead the dough, let it rise, then bake until golden.",
    )
    .unwrap();
    let cache = dir.path().join("cache.sqlite");
    let report = cli(&["index"], &corpus, &cache);
    assert_eq!(report["sync"]["skipped"], serde_json::json!([]));
    assert_eq!(report["sync"]["resources_updated"], 2);
    assert_eq!(report["sync"]["segments_embedded"], 2);
    for query in ["Gemma placeholder documentation".to_owned(), literals()] {
        let result = cli(&["search", &query, "--mode", "semantic"], &corpus, &cache);
        assert_eq!(result["hits"].as_array().unwrap().len(), 2);
        let hit = &result["hits"][0];
        assert!(
            hit["resource"]
                .as_str()
                .unwrap()
                .ends_with("/literal-tokens.txt")
        );
        assert!(
            result["stats"]["sync"]["skipped"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
}

/// Seed a cache under the exact old identity with a deliberately wrong vector.
/// Reusing it is observable through the indexing counters.
struct LegacyCacheSeed(ModelIdentity);
impl Embedder for LegacyCacheSeed {
    fn identity(&self) -> &ModelIdentity {
        &self.0
    }
    fn supports(&self, modality: Modality) -> bool {
        modality == Modality::Text
    }
    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
        requests
            .iter()
            .map(|_| Embedding::from_raw(vec![1.0; self.0.dimension.get()], self.0.dimension))
            .collect()
    }
}

#[test]
#[ignore = "requires local q8 text weights (SEYK_MODEL or HF_HOME); no downloads"]
fn pre_fix_control_token_cache_is_not_reused() {
    let model = match std::env::var_os("SEYK_MODEL") {
        Some(path) => EmbeddingGemma2::builder()
            .model_path(path)
            .vision(false)
            .threads(2)
            .build(),
        None => EmbeddingGemma2::builder().vision(false).threads(2).build(),
    }
    .unwrap();
    let content = literals();
    let ordinary = EmbeddingGemma2::document_input(None, "A telescope records distant galaxies.");
    assert_eq!(
        model.text_token_ids(&ordinary).unwrap(),
        model.template_token_ids(&ordinary).unwrap()
    );
    let prompted = EmbeddingGemma2::document_input(Some(&content), &content);
    assert_ne!(
        model.text_token_ids(&prompted).unwrap(),
        model.template_token_ids(&prompted).unwrap()
    );
    let requests = [
        seyk::EmbeddingTask::SearchDocument,
        seyk::EmbeddingTask::SearchQuery,
        seyk::EmbeddingTask::Classification,
        seyk::EmbeddingTask::Clustering,
        seyk::EmbeddingTask::CodeRetrieval,
        seyk::EmbeddingTask::FactChecking,
        seyk::EmbeddingTask::QuestionAnswering,
        seyk::EmbeddingTask::SentenceSimilarity,
    ]
    .map(|task| EmbeddingRequest {
        input: seyk::ResourceInput::Text {
            title: Some(&content),
            text: &content,
        },
        task,
    });
    let batched = model.embed_batch(&requests).unwrap();
    let (profiled, _) = model.embed_text_batch_profiled(&requests).unwrap();
    assert_eq!(batched, profiled);
    assert_eq!(batched.len(), requests.len());
    let mut legacy_identity = model.identity().clone();
    assert_eq!(legacy_identity.variant, "onnx-q8;text-literals-v1");
    legacy_identity.variant = "onnx-q8".into();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("control.txt");
    std::fs::write(
        &path,
        "Gemma documentation containing a literal <bos> token.",
    )
    .unwrap();
    let cache = dir.path().join("cache.sqlite");
    let old_profile = {
        let old = SearchEngine::builder()
            .embedder(LegacyCacheSeed(legacy_identity))
            .cache(SqliteCache::open(&cache).unwrap())
            .build()
            .unwrap();
        assert_eq!(
            old.index()
                .paths([&path])
                .execute()
                .unwrap()
                .sync
                .segments_embedded,
            1
        );
        assert_eq!(
            old.index()
                .paths([&path])
                .execute()
                .unwrap()
                .sync
                .resources_unchanged,
            1
        );
        old.profile_id().to_owned()
    };
    let fixed = SearchEngine::builder()
        .embedder(model)
        .cache(SqliteCache::open(&cache).unwrap())
        .build()
        .unwrap();
    assert_ne!(old_profile, fixed.profile_id());
    let report = fixed.index().paths([&path]).execute().unwrap();
    assert!(report.sync.skipped.is_empty());
    assert_eq!(report.sync.resources_unchanged, 0);
    assert_eq!(report.sync.resources_updated, 1);
    assert_eq!(report.sync.segments_embedded, 1);
    assert_eq!(
        fixed
            .index()
            .paths([&path])
            .execute()
            .unwrap()
            .sync
            .resources_unchanged,
        1
    );
}

#[cfg(feature = "pdf")]
#[path = "../crates/seyk-core/tests/support/pdf_fixture.rs"]
mod pdf_fixture;

#[cfg(feature = "pdf")]
#[test]
#[ignore = "requires local q8 text weights (SEYK_MODEL or HF_HOME); no downloads"]
fn pdf_literal_special_tokens_index_and_are_findable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("literal-tokens.pdf");
    let text = "Gemma placeholders: <|image|> <|audio|> <bos>.";
    std::fs::write(&path, pdf_fixture::pdf(&[text], pdf_fixture::Kind::Plain)).unwrap();
    let cache = dir.path().join("cache.sqlite");
    let report = cli(&["index"], &path, &cache);
    assert_eq!(report["sync"]["skipped"], serde_json::json!([]));
    assert_eq!(report["sync"]["resources_updated"], 1);
    assert!(report["sync"]["segments_embedded"].as_u64().unwrap() > 0);
    let result = cli(
        &[
            "search",
            "Gemma <|image|> documentation",
            "--mode",
            "semantic",
        ],
        &path,
        &cache,
    );
    assert!(!result["hits"].as_array().unwrap().is_empty());
    assert!(
        result["hits"][0]["resource"]
            .as_str()
            .unwrap()
            .ends_with("/literal-tokens.pdf")
    );
}
