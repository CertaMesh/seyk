use std::path::PathBuf;

/// Errors produced by the search engine and its components.
///
/// Non-exhaustive: new failure kinds can be added without breaking matches.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("STT model {reason}; run `seyk model download --stt`")]
    SttModel { reason: &'static str },
    #[error("speech-to-text failed: {reason}")]
    SpeechToText { reason: &'static str },
    #[error("audio extraction interval exceeds the 30-second request limit")]
    AudioIntervalTooLong,

    #[error("resource changed while reading: {resource}")]
    ChangedWhileReading { resource: crate::ResourceId },
    #[error("unsupported audio for {resource}: {reason}")]
    UnsupportedAudio {
        resource: crate::ResourceId,
        reason: &'static str,
    },
    #[error("PDF rendering failed for {resource}: {reason}")]
    PdfRender {
        resource: crate::ResourceId,
        reason: &'static str,
    },
    #[error(
        "readable text for scanned pages needs OCR, which is not available yet ({resource}, page {page})"
    )]
    PdfOcrUnavailable {
        resource: crate::ResourceId,
        page: u32,
    },
    #[error("PDF parse failed for {resource}: {reason}")]
    PdfParse {
        resource: crate::ResourceId,
        reason: &'static str,
    },
    #[error("PDF parser panicked for {resource}")]
    ParserPanicked { resource: crate::ResourceId },
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "cache changes committed, but privacy cleanup is pending; rerun the command or sync once other readers close (using the same cache path); cache clear is a fallback"
    )]
    CacheCleanupPending,

    #[error("cache error: {0}")]
    Cache(#[from] rusqlite::Error),

    #[error("embedding backend error: {0}")]
    Embedding(String),

    #[error("model files not found: {0}")]
    ModelNotFound(String),

    #[error("{modality} resources are not supported by this embedder")]
    UnsupportedModality { modality: crate::Modality },

    #[error("task {task:?} is not supported for {modality}")]
    UnsupportedTask {
        task: seyk_embed::EmbeddingTask,
        modality: crate::Modality,
    },

    #[error("resource changed since search: {resource} (expected {expected}, got {actual})")]
    StaleResource {
        resource: crate::ResourceId,
        expected: String,
        actual: String,
    },

    #[error("extraction is unsupported for {modality} resource {resource} at {locator:?}{note}", note = if *modality == crate::Modality::Audio { "; configure an audio extractor (feature `audio-stt`)" } else { "" })]
    UnsupportedExtraction {
        resource: crate::ResourceId,
        modality: crate::Modality,
        locator: crate::ResourceLocator,
    },

    #[error("invalid locator for {resource}: {reason}")]
    InvalidLocator {
        resource: crate::ResourceId,
        locator: crate::ResourceLocator,
        reason: &'static str,
    },

    #[error("resource {resource} is not valid UTF-8: {source}")]
    InvalidText {
        resource: crate::ResourceId,
        #[source]
        source: std::str::Utf8Error,
    },

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

// Preserve the established search error variants for downstream matches.
impl From<seyk_embed::Error> for Error {
    fn from(error: seyk_embed::Error) -> Self {
        match error {
            seyk_embed::Error::Io { path, source } => Self::Io { path, source },
            seyk_embed::Error::Embedding(message) => Self::Embedding(message),
            seyk_embed::Error::ModelNotFound(message) => Self::ModelNotFound(message),
            seyk_embed::Error::UnsupportedModality { modality } => {
                Self::UnsupportedModality { modality }
            }
            seyk_embed::Error::DimensionMismatch { expected, actual } => {
                Self::DimensionMismatch { expected, actual }
            }
            seyk_embed::Error::Config(message) => Self::Config(message),
            seyk_embed::Error::UnsupportedTask { task, modality } => {
                Self::UnsupportedTask { task, modality }
            }
            other => Self::Embedding(other.to_string()),
        }
    }
}
