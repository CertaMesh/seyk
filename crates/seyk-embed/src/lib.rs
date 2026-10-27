//! Lightweight task-aware embedding interfaces, independent of search and runtimes.
pub mod diagnostics;
mod embedder;
mod embedding;
mod error;
pub mod quant;
pub use embedder::{Embedder, EmbeddingRequest, ModelIdentity, ResourceInput, VideoFrame};
pub use embedding::{Dimension, Embedding, cosine_similarity, dot_product, normalize};
pub use error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The kind of content a resource holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Modality {
    /// Container metadata, never an embedding input.
    Document,
    Text,
    Image,
    Audio,
    Video,
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Modality::Document => "document",
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
            Modality::Video => "video",
        })
    }
}

/// The purpose of an embedding. Each backend documents its supported mapping.
/// These are embedding tasks, never instructions to generate answers or summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EmbeddingTask {
    SearchQuery,
    SearchDocument,
    Classification,
    Clustering,
    CodeRetrieval,
    FactChecking,
    QuestionAnswering,
    SentenceSimilarity,
}
