//! Search, caching and verified evidence, independent of any concrete embedding backend.
//!
//! With `pdf`, applications installing a `log` logger MUST filter `pdf_extract`
//! to Off. With `pdf-render`, also filter `hayro_interpret`, `hayro_syntax` and
//! `hayro_font` to Off: upstream diagnostics can contain document bytes. This crate installs
//! neither a logger nor a panic hook. PDF panics become errors via catch_unwind,
//! but the application's panic hook still runs first; use a payload-free hook.
//! Unwinding does not contain aborts, OOM, hangs or excessive parser allocations.

pub mod cache;
pub mod discovery;
pub mod embedder {
    pub use seyk_embed::*;
}
mod embedding;
mod engine;
mod error;
pub mod extractor;
#[cfg(feature = "pdf")]
pub mod pdf;
#[cfg(feature = "pdf")]
pub use pdf::PdfTextExtractor;
pub mod index;
pub mod lexical;
pub mod provider;
mod resource;
pub mod segment;
#[cfg(feature = "watch")]
pub mod watch;

pub use cache::{
    CacheStats, EmbeddingCache, MemoryCache, ResourceWrite, SegmentWrite, SqliteCache,
};
pub use discovery::DiscoveryOptions;
pub use embedder::{Embedder, ModelIdentity, ResourceInput};
pub use embedding::{Dimension, Embedding};
pub use engine::{
    IndexReport, IndexRequest, MatchSignals, PageDiagnostic, ProgressEvent, ProgressFn,
    SearchEngine, SearchEngineBuilder, SearchHit, SearchMode, SearchRequest, SearchResults,
    SearchStats, SkippedResource, SyncStats, expand_tilde, make_preview,
};
pub use error::{Error, Result};
pub use resource::{ResourceId, ResourceLocator, ResourceMetadata};
pub use seyk_embed::{EmbeddingRequest, EmbeddingTask, Modality};

pub use provider::{
    FsProvider, ProvidedResource, ResourceProvider, ResourceScope, ResourceVersion,
};

pub use extractor::{ExtractedContent, Extractor, TextExtractor, extract_resource};

#[cfg(feature = "audio")]
pub mod audio;
