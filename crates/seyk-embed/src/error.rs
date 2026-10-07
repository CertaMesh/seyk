use std::path::PathBuf;

/// Embedding errors, with no search or inference-runtime types.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("embedding backend error: {0}")]
    Embedding(String),
    #[error("model files not found: {0}")]
    ModelNotFound(String),
    #[error("{modality} resources are not supported by this embedder")]
    UnsupportedModality { modality: crate::Modality },
    #[error("task {task:?} is not supported for {modality}")]
    UnsupportedTask {
        task: crate::EmbeddingTask,
        modality: crate::Modality,
    },
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("invalid configuration: {0}")]
    Config(String),
}
pub type Result<T, E = Error> = std::result::Result<T, E>;
