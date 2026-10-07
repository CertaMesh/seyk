//! Loading the real EmbeddingGemma 2 model in tests.
//!
//! Model-backed tests are `#[ignore]`d so `cargo test` works without weights.
//! Run them with `cargo test -- --ignored` after `gemma-search model download`.
//! `GEMMA_SEARCH_MODEL` may point at a local model directory and
//! `GEMMA_SEARCH_TEST_PRECISION` selects `q8` (default) or `fp32`.

#![allow(dead_code)]

use std::path::PathBuf;

use gemma_search::Dimension;
use gemma_search::embedder::gemma::{GemmaConfig, GemmaEmbedder, ModelSource, Precision};

pub fn precision() -> Precision {
    std::env::var("GEMMA_SEARCH_TEST_PRECISION")
        .ok()
        .map(|p| {
            p.parse()
                .expect("GEMMA_SEARCH_TEST_PRECISION is q8 or fp32")
        })
        .unwrap_or_default()
}

pub fn config(dimension: Dimension) -> GemmaConfig {
    let source = std::env::var_os("GEMMA_SEARCH_MODEL")
        .map(|p| ModelSource::Directory(PathBuf::from(p)))
        .unwrap_or_default();
    GemmaConfig {
        source,
        dimension,
        precision: precision(),
        vision: false,
        ..Default::default()
    }
}

pub fn embedder(dimension: Dimension) -> GemmaEmbedder {
    GemmaEmbedder::load(config(dimension)).unwrap_or_else(|e| {
        panic!("cannot load EmbeddingGemma 2 (run `gemma-search model download` or set GEMMA_SEARCH_MODEL): {e}")
    })
}
