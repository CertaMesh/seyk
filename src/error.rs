use std::path::PathBuf;

/// Errors produced by the search engine and its components.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cache error: {0}")]
    Cache(#[from] rusqlite::Error),

    #[error("embedding backend error: {0}")]
    Embedding(String),

    #[error("model files not found: {0}")]
    ModelNotFound(String),

    #[error("{modality} resources are not supported by this embedder")]
    UnsupportedModality { modality: crate::Modality },

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("invalid configuration: {0}")]
    Config(String),

    #[error("invalid glob pattern: {0}")]
    Glob(#[from] globset::Error),
}

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
