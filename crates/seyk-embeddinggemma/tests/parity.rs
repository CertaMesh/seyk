//! The Rust/ONNX path must match the official sentence-transformers output.
//!
//! Reference data: tests/fixtures/reference_text.json, produced by
//! scripts/reference_text_embeddings.py (float32, prompts `SearchQuery` / `Document`).

mod model_support;

use serde::Deserialize;
use seyk_embed::{Dimension, Embedder};
use seyk_embeddinggemma::{GemmaEmbedder, Precision};

const MIN_BATCHING_COSINE: f32 = 0.9999;

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
        precision => panic!("no parity threshold for {precision:?}"),
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
        assert_eq!(
            e.text_token_ids(&q.input).unwrap(),
            q.ids,
            "query {:?}",
            q.text
        );
    }
    for d in &r.docs {
        assert_eq!(
            GemmaEmbedder::document_input(d.title.as_deref(), &d.text),
            d.input
        );
        assert_eq!(
            e.text_token_ids(&d.input).unwrap(),
            d.ids,
            "doc {:?}",
            d.text
        );
    }
}

#[test]
#[ignore = "needs EmbeddingGemma 2 weights"]
fn embeddings_match_reference_at_every_dimension() {
    let r = reference();
    let mut worst = 1.0f32;
    for dim in Dimension::ALL {
        let e = model_support::embedder(dim);
        for q in &r.queries {
            let got = e.embed_query(&q.text).unwrap();
            let c = cos(got.as_slice(), &truncated(&q.emb, dim.get()));
            worst = worst.min(c);
            assert!(c >= min_cosine(), "query {:?} at {dim}: cosine {c}", q.text);
        }
        for d in &r.docs {
            let got = e.embed_text_document(d.title.as_deref(), &d.text).unwrap();
            let c = cos(got.as_slice(), &truncated(&d.emb, dim.get()));
            worst = worst.min(c);
            assert!(c >= min_cosine(), "doc {:?} at {dim}: cosine {c}", d.text);
        }
    }
    eprintln!(
        "worst cosine vs reference ({:?}): {worst:.5}",
        model_support::precision()
    );
}

#[test]
#[ignore = "needs EmbeddingGemma 2 weights"]
fn batching_does_not_change_vectors() {
    let e = model_support::embedder(Dimension::D256);
    let r = reference();
    let inputs: Vec<_> = r
        .docs
        .iter()
        .map(|d| seyk_embed::ResourceInput::Text {
            title: d.title.as_deref(),
            text: &d.text,
        })
        .collect();
    let batched = e.embed_resources(&inputs).unwrap();
    for (d, b) in r.docs.iter().zip(&batched) {
        let single = e.embed_text_document(d.title.as_deref(), &d.text).unwrap();
        assert!(
            single.cosine(b) > MIN_BATCHING_COSINE,
            "padding changed the vector for {:?}",
            d.text
        );
    }
}

/// Check budget-induced shape changes against the existing same-model batching tolerance.
#[test]
#[ignore = "needs EmbeddingGemma 2 weights and memory-measurement approval"]
fn memory_budget_batching_parity() {
    use seyk_embeddinggemma::{BatchBudget, CpuAllocator};
    let r = reference();
    let long = "the ".repeat(1100);
    let medium = "the ".repeat(700);
    let mut inputs: Vec<_> = r
        .docs
        .iter()
        .map(|d| seyk_embed::ResourceInput::Text {
            title: d.title.as_deref(),
            text: &d.text,
        })
        .collect();
    inputs.extend([
        seyk_embed::ResourceInput::Text {
            title: None,
            text: &long,
        },
        seyk_embed::ResourceInput::Text {
            title: None,
            text: &medium,
        },
    ]);
    let mut failures = Vec::new();
    for dimension in Dimension::ALL {
        let mut base_config = model_support::config(dimension);
        base_config.threads = Some(1);
        base_config.batch_budget = None;
        let baseline = GemmaEmbedder::load(base_config.clone()).unwrap();
        let max_length = inputs
            .iter()
            .map(|input| match input {
                seyk_embed::ResourceInput::Text { title, text } => baseline
                    .text_token_ids(&GemmaEmbedder::document_input(*title, text))
                    .unwrap()
                    .len(),
                _ => unreachable!(),
            })
            .max()
            .unwrap();
        assert!(
            inputs.len().min(base_config.batch_size) * max_length * max_length
                > BatchBudget::DEFAULT.attention_cells,
            "fixture must force subdivision under the default budget"
        );
        let expected_docs = baseline.embed_resources(&inputs).unwrap();
        let expected_queries: Vec<_> = r
            .queries
            .iter()
            .map(|q| baseline.embed_query(&q.text).unwrap())
            .collect();
        drop(baseline);
        for (sessions, policy) in [
            (1, CpuAllocator::Default),
            (1, CpuAllocator::NoArena),
            (1, CpuAllocator::ArenaShrink),
            (2, CpuAllocator::ArenaShrink),
        ] {
            let mut config = base_config.clone();
            config.sessions = sessions;
            config.batch_budget = Some(BatchBudget::DEFAULT);
            config.cpu_allocator = policy;
            let candidate = GemmaEmbedder::load(config).unwrap();
            let actual_docs = candidate.embed_resources(&inputs).unwrap();
            let actual_queries: Vec<_> = r
                .queries
                .iter()
                .map(|q| candidate.embed_query(&q.text).unwrap())
                .collect();
            assert_eq!(actual_docs.len(), expected_docs.len());
            assert_eq!(actual_queries.len(), expected_queries.len());
            let mut max_abs_delta = 0.0_f64;
            let mut min_cosine = f32::INFINITY;
            let mut outside_tolerance = 0;
            for (a, b) in expected_docs
                .iter()
                .chain(&expected_queries)
                .zip(actual_docs.iter().chain(&actual_queries))
            {
                let cosine = a.cosine(b);
                if cosine.is_nan() || cosine < min_cosine {
                    min_cosine = cosine;
                }
                if cosine.is_nan() || cosine <= MIN_BATCHING_COSINE {
                    outside_tolerance += 1;
                }
                for (&x, &y) in a.as_slice().iter().zip(b.as_slice()) {
                    max_abs_delta = max_abs_delta.max((f64::from(x) - f64::from(y)).abs());
                }
            }
            if outside_tolerance > 0 {
                failures.push(format!(
                    "dimension={} sessions={sessions} allocator={} outside_tolerance={outside_tolerance} max_abs_delta={max_abs_delta:e} min_cosine={min_cosine:.9}",
                    dimension.get(),
                    policy.name()
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "memory budget exceeded batching tolerance (cosine > {MIN_BATCHING_COSINE}):\n{}",
        failures.join("\n")
    );
}
