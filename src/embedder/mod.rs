//! The embedding backend abstraction.
//!
//! The search engine only talks to [`Embedder`]. EmbeddingGemma 2 is one
//! implementation; another backend can be plugged in without touching search code.

#[cfg(feature = "onnx")]
pub mod gemma;

use serde::{Deserialize, Serialize};

use crate::{Dimension, Embedding, Modality, Result};

/// Something to embed as a searchable resource (the "document" side of retrieval).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ResourceInput<'a> {
    /// A text segment. `title` is a heading or file name, when one is known.
    Text {
        title: Option<&'a str>,
        text: &'a str,
    },
    /// An encoded image file (PNG, JPEG, ...).
    Image { bytes: &'a [u8] },
    /// Mono PCM samples. Reserved: no shipped embedder accepts audio yet.
    Audio {
        samples: &'a [f32],
        sample_rate: u32,
    },
    /// Decoded frames with their timestamps. Reserved: no shipped embedder accepts video yet.
    Video { frames: &'a [VideoFrame<'a>] },
}

impl ResourceInput<'_> {
    pub fn modality(&self) -> Modality {
        match self {
            ResourceInput::Text { .. } => Modality::Text,
            ResourceInput::Image { .. } => Modality::Image,
            ResourceInput::Audio { .. } => Modality::Audio,
            ResourceInput::Video { .. } => Modality::Video,
        }
    }
}

/// One decoded video frame (RGB8, row-major).
#[derive(Debug, Clone)]
pub struct VideoFrame<'a> {
    pub timestamp_ms: u64,
    pub width: u32,
    pub height: u32,
    pub rgb: &'a [u8],
}

/// Everything that determines what vector an embedder produces for a given input.
///
/// Persisted alongside cached vectors. If any field changes, cached vectors
/// from the old profile are never used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// For example `google/embeddinggemma-2`.
    pub model_id: String,
    /// Exact weights revision (a commit hash when known).
    pub revision: String,
    /// Runtime and precision, for example `onnx-q8`.
    pub variant: String,
    pub dimension: Dimension,
    /// How vectors are normalised, for example `l2-after-truncation`.
    pub normalization: String,
    pub query_prompt: String,
    pub document_prompt: String,
    /// Inputs longer than this many tokens are truncated.
    pub max_tokens: usize,
}

/// A backend that maps queries and resources into one shared vector space.
pub trait Embedder: Send + Sync {
    fn identity(&self) -> &ModelIdentity;

    fn dimension(&self) -> Dimension {
        self.identity().dimension
    }

    /// Whether [`Embedder::embed_resources`] accepts inputs of this modality.
    fn supports(&self, modality: Modality) -> bool;

    /// Embed a natural-language search query (the "query" side of retrieval).
    fn embed_query(&self, query: &str) -> Result<Embedding>;

    /// Embed searchable resources. Returns one embedding per input, in order.
    ///
    /// Implementations should batch internally; callers may pass many inputs.
    fn embed_resources(&self, inputs: &[ResourceInput<'_>]) -> Result<Vec<Embedding>>;

    fn embed_text_document(&self, title: Option<&str>, text: &str) -> Result<Embedding> {
        let mut out = self.embed_resources(&[ResourceInput::Text { title, text }])?;
        out.pop()
            .ok_or_else(|| crate::Error::Embedding("embedder returned no vectors".into()))
    }
}

impl<E: Embedder + ?Sized> Embedder for Box<E> {
    fn identity(&self) -> &ModelIdentity {
        (**self).identity()
    }
    fn supports(&self, modality: Modality) -> bool {
        (**self).supports(modality)
    }
    fn embed_query(&self, query: &str) -> Result<Embedding> {
        (**self).embed_query(query)
    }
    fn embed_resources(&self, inputs: &[ResourceInput<'_>]) -> Result<Vec<Embedding>> {
        (**self).embed_resources(inputs)
    }
}

impl<E: Embedder + ?Sized> Embedder for std::sync::Arc<E> {
    fn identity(&self) -> &ModelIdentity {
        (**self).identity()
    }
    fn supports(&self, modality: Modality) -> bool {
        (**self).supports(modality)
    }
    fn embed_query(&self, query: &str) -> Result<Embedding> {
        (**self).embed_query(query)
    }
    fn embed_resources(&self, inputs: &[ResourceInput<'_>]) -> Result<Vec<Embedding>> {
        (**self).embed_resources(inputs)
    }
}
