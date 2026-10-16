//! Identity/capabilities for a vector-free lexical engine. Inference is an error.
use crate::{Dimension, Embedder, Embedding, EmbeddingRequest, Modality, ModelIdentity};

pub(crate) struct LexicalIdentity(ModelIdentity);

impl Default for LexicalIdentity {
    fn default() -> Self {
        Self(ModelIdentity {
            model_id: "seyk/lexical-only".into(),
            revision: "1".into(),
            variant: "no-vectors".into(),
            dimension: Dimension::D128,
            normalization: String::new(),
            query_prompt: String::new(),
            document_prompt: String::new(),
            max_tokens: 0,
        })
    }
}

impl Embedder for LexicalIdentity {
    fn identity(&self) -> &ModelIdentity {
        &self.0
    }
    fn supports(&self, modality: Modality) -> bool {
        modality == Modality::Text
    }
    fn embed_batch(&self, _: &[EmbeddingRequest<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
        Err(seyk_embed::Error::Embedding(
            "lexical engine cannot produce vectors".into(),
        ))
    }
}
