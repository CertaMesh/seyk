use serde::{Deserialize, Serialize};

use crate::{Dimension, Embedding, EmbeddingTask, Modality, Result};

/// A borrowed input to embed. The request selects its task independently.
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

/// One borrowed input and its embedding purpose.
#[derive(Debug, Clone)]
pub struct EmbeddingRequest<'a> {
    pub input: ResourceInput<'a>,
    pub task: EmbeddingTask,
}

/// Synchronous, task-aware local embedding interface.
/// Successful batches return one result per request, in input order.
/// Async callers can use their executor's blocking pool.
pub trait Embedder: Send + Sync {
    fn identity(&self) -> &ModelIdentity;
    fn dimension(&self) -> Dimension {
        self.identity().dimension
    }
    fn supports(&self, modality: Modality) -> bool;
    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> Result<Vec<Embedding>>;

    fn embed(&self, input: ResourceInput<'_>, task: EmbeddingTask) -> Result<Embedding> {
        let mut out = self.embed_batch(&[EmbeddingRequest { input, task }])?;
        if out.len() != 1 {
            return Err(crate::Error::Embedding(
                "embedder returned the wrong number of vectors".into(),
            ));
        }
        Ok(out.remove(0))
    }
    fn embed_text(&self, text: &str, task: EmbeddingTask) -> Result<Embedding> {
        self.embed(ResourceInput::Text { title: None, text }, task)
    }
    /// Retrieval convenience adapter.
    fn embed_query(&self, query: &str) -> Result<Embedding> {
        self.embed_text(query, EmbeddingTask::SearchQuery)
    }
    /// Retrieval convenience adapter. Text titles are preserved.
    fn embed_resources(&self, inputs: &[ResourceInput<'_>]) -> Result<Vec<Embedding>> {
        let requests: Vec<_> = inputs
            .iter()
            .cloned()
            .map(|input| EmbeddingRequest {
                input,
                task: EmbeddingTask::SearchDocument,
            })
            .collect();
        self.embed_batch(&requests)
    }
    fn embed_text_document(&self, title: Option<&str>, text: &str) -> Result<Embedding> {
        self.embed(
            ResourceInput::Text { title, text },
            EmbeddingTask::SearchDocument,
        )
    }
}

impl<E: Embedder + ?Sized> Embedder for Box<E> {
    fn identity(&self) -> &ModelIdentity {
        (**self).identity()
    }
    fn supports(&self, modality: Modality) -> bool {
        (**self).supports(modality)
    }
    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> Result<Vec<Embedding>> {
        (**self).embed_batch(requests)
    }
}
impl<E: Embedder + ?Sized> Embedder for std::sync::Arc<E> {
    fn identity(&self) -> &ModelIdentity {
        (**self).identity()
    }
    fn supports(&self, modality: Modality) -> bool {
        (**self).supports(modality)
    }
    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> Result<Vec<Embedding>> {
        (**self).embed_batch(requests)
    }
}
