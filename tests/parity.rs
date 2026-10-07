//! The Rust/ONNX path must match the official sentence-transformers output.
//!
//! Reference data: tests/fixtures/reference_text.json, produced by
//! scripts/reference_text_embeddings.py (float32, prompts `SearchQuery` / `Document`).

mod model_support;

use gemma_search::embedder::gemma::{GemmaEmbedder, Precision};
use gemma_search::{Dimension, Embedder};
use serde::Deserialize;

#[derive(Deserialize)]
struct Item {
    text: String,
    title: Option<String>,
    input: String,
    ids: Vec<u32>,
    emb: Vec<f32>,
}

#[derive(Deserialize)]
struct Reference {
    queries: Vec<Item>,
    docs: Vec<Item>,
}

fn reference() -> Reference {
    serde_json::from_str(include_str!("fixtures/reference_text.json")).unwrap()
}

fn min_cosine() -> f32 {
    match model_support::precision() {
        Precision::Fp32 => 0.999,
        // The export reports worst-case cosine 0.9997 for q8 against fp32.
        Precision::Q8 => 0.995,
    }
}

fn truncated(v: &[f32], dim: usize) -> Vec<f32> {
    let mut t = v[..dim].to_vec();
    let n = t.iter().map(|x| x * x).sum::<f32>().sqrt();
    t.iter_mut().for_each(|x| *x /= n);
    t
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
#[ignore = "needs EmbeddingGemma 2 weights"]
fn prompts_and_token_ids_match_reference() {
    let e = model_support::embedder(Dimension::D768);
    let r = reference();
    for q in &r.queries {
        assert_eq!(GemmaEmbedder::query_input(&q.text), q.input);
        assert_eq!(e.token_ids(&q.input).unwrap(), q.ids, "query {:?}", q.text);
    }
    for d in &r.docs {
        assert_eq!(
            GemmaEmbedder::document_input(d.title.as_deref(), &d.text),
            d.input
        );
        assert_eq!(e.token_ids(&d.input).unwrap(), d.ids, "doc {:?}", d.text);
    }
}

#[test]
#[ignore = "needs EmbeddingGemma 2 weights"]
fn embeddings_match_reference_at_every_dimension() {
    let r = reference();
    for dim in Dimension::ALL {
        let e = model_support::embedder(dim);
        for q in &r.queries {
            let got = e.embed_query(&q.text).unwrap();
            let c = cos(got.as_slice(), &truncated(&q.emb, dim.get()));
            assert!(c >= min_cosine(), "query {:?} at {dim}: cosine {c}", q.text);
        }
        for d in &r.docs {
            let got = e.embed_text_document(d.title.as_deref(), &d.text).unwrap();
            let c = cos(got.as_slice(), &truncated(&d.emb, dim.get()));
            assert!(c >= min_cosine(), "doc {:?} at {dim}: cosine {c}", d.text);
        }
    }
}

#[test]
#[ignore = "needs EmbeddingGemma 2 weights"]
fn batching_does_not_change_vectors() {
    let e = model_support::embedder(Dimension::D256);
    let r = reference();
    let inputs: Vec<_> = r
        .docs
        .iter()
        .map(|d| gemma_search::ResourceInput::Text {
            title: d.title.as_deref(),
            text: &d.text,
        })
        .collect();
    let batched = e.embed_resources(&inputs).unwrap();
    for (d, b) in r.docs.iter().zip(&batched) {
        let single = e.embed_text_document(d.title.as_deref(), &d.text).unwrap();
        assert!(
            single.cosine(b) > 0.9999,
            "padding changed the vector for {:?}",
            d.text
        );
    }
}
