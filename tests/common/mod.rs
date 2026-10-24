#![allow(dead_code)] // each test binary uses a different subset

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use seyk::{Dimension, Embedder, Embedding, Modality, ModelIdentity, ResourceInput};

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

    fn embed(&self, text: &str) -> seyk_embed::Result<Embedding> {
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

    fn embed_query(&self, query: &str) -> seyk_embed::Result<Embedding> {
        self.embed(query)
    }

    fn embed_resources(&self, inputs: &[ResourceInput<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
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

    fn embed_batch(
        &self,
        requests: &[seyk_embed::EmbeddingRequest<'_>],
    ) -> seyk_embed::Result<Vec<Embedding>> {
        if requests
            .iter()
            .all(|r| r.task == seyk_embed::EmbeddingTask::SearchDocument)
        {
            let inputs: Vec<_> = requests.iter().map(|r| r.input.clone()).collect();
            return self.embed_resources(&inputs);
        }
        requests
            .iter()
            .map(|r| match &r.input {
                ResourceInput::Text { text, .. }
                    if r.task == seyk_embed::EmbeddingTask::SearchQuery =>
                {
                    self.embed_query(text)
                }
                _ => self
                    .embed_resources(std::slice::from_ref(&r.input))
                    .map(|mut v| v.remove(0)),
            })
            .collect()
    }
}

/// Check decoded fields so JSON escaping cannot hide a leaked Windows path.
pub fn assert_json_excludes(value: &serde_json::Value, private: &str) {
    match value {
        serde_json::Value::String(text) => assert!(!text.contains(private)),
        serde_json::Value::Array(values) => {
            for value in values {
                assert_json_excludes(value, private);
            }
        }
        serde_json::Value::Object(fields) => {
            for (key, value) in fields {
                assert!(!key.contains(private));
                assert_json_excludes(value, private);
            }
        }
        _ => {}
    }
}
