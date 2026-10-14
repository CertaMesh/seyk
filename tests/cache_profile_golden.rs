//! Cache identity constructed through the actual backend and engine.
#![cfg(feature = "onnx")]

use seyk::SearchEngine;
use seyk_embeddinggemma::EmbeddingGemma2;

#[test]
#[ignore = "needs the pinned q8 EmbeddingGemma 2 weights in the local HF cache"]
fn real_gemma_profile_changes_only_for_literal_token_policy() {
    let mut golden: serde_json::Value = serde_json::from_str(include_str!(
        "../crates/seyk-core/tests/fixtures/pre_split_preparation.json"
    ))
    .unwrap();
    let model = EmbeddingGemma2::builder()
        .dimension(128)
        .vision(false)
        .threads(1)
        .build()
        .unwrap();
    let engine = SearchEngine::builder().embedder(model).build().unwrap();
    golden["profile"]["model"]["variant"] = "onnx-q8;text-literals-v1".into();
    assert_eq!(
        serde_json::to_value(engine.profile()).unwrap(),
        golden["profile"]
    );
    assert_ne!(engine.profile_id(), golden["profile_id"].as_str().unwrap());
}

#[test]
fn model_output_normalization_keeps_pre_split_vector_bytes() {
    let golden: serde_json::Value = serde_json::from_str(include_str!(
        "../crates/seyk-core/tests/fixtures/pre_split_preparation.json"
    ))
    .unwrap();
    for dimension in seyk::Dimension::ALL {
        let raw = (0..768).map(|i| (i % 17) as f32 - 8.0).collect();
        let embedding = seyk::Embedding::from_raw(raw, dimension).unwrap();
        let bytes: Vec<_> = embedding
            .as_slice()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            blake3::hash(&bytes).to_hex().as_str(),
            golden["model_output_hashes"][dimension.to_string()]
                .as_str()
                .unwrap()
        );
    }
}
