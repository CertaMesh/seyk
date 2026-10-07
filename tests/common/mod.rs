#![allow(dead_code)] // each test binary uses a different subset

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use seyk::{Dimension, Embedder, Embedding, Modality, ModelIdentity, ResourceInput, Result};

/// Deterministic text-only embedder used by integration tests.
pub struct FakeEmbedder {
    identity: ModelIdentity,
    embedded_resources: Arc<AtomicUsize>,
}

impl FakeEmbedder {
    pub fn new(dimension: Dimension, revision: &str) -> Self {
        Self {
            identity: ModelIdentity {
                model_id: "test/fake".into(),
                revision: revision.into(),
                variant: "hashed-bow".into(),
                dimension,
                normalization: "l2".into(),
                query_prompt: "query".into(),
                document_prompt: "document".into(),
                max_tokens: 512,
            },
            embedded_resources: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn resource_count(&self) -> usize {
        self.embedded_resources.load(Ordering::SeqCst)
    }

    pub fn reset_resource_count(&self) {
        self.embedded_resources.store(0, Ordering::SeqCst);
    }

    fn embed(&self, text: &str) -> Result<Embedding> {
        let mut vector = vec![0.0; self.identity.dimension.get()];
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
        {
            let hash = blake3::hash(word.to_ascii_lowercase().as_bytes());
            let bytes = hash.as_bytes();
            let index = u16::from_le_bytes([bytes[0], bytes[1]]) as usize % vector.len();
            vector[index] += if bytes[2] & 1 == 0 { 1.0 } else { -1.0 };
        }
        Embedding::from_raw(vector, self.identity.dimension)
    }
}

impl Embedder for FakeEmbedder {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn supports(&self, modality: Modality) -> bool {
        modality == Modality::Text
    }

    fn embed_query(&self, query: &str) -> Result<Embedding> {
        self.embed(query)
    }

    fn embed_resources(&self, inputs: &[ResourceInput<'_>]) -> Result<Vec<Embedding>> {
        self.embedded_resources
            .fetch_add(inputs.len(), Ordering::SeqCst);
        inputs
            .iter()
            .map(|input| match input {
                ResourceInput::Text { title, text } => {
                    self.embed(&format!("{} {text}", title.unwrap_or_default()))
                }
                _ => unreachable!("fake embedder only receives text"),
            })
            .collect()
    }
}
