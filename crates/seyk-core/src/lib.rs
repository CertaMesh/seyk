//! Search, caching and verified evidence, independent of any concrete embedding backend.
//!
//! With `pdf`, applications installing a `log` logger MUST filter `pdf_extract`
//! to Off. With `pdf-render`, also filter `hayro_interpret`, `hayro_syntax` and
//! `hayro_font` to Off: upstream diagnostics can contain document bytes. This crate installs
//! no logger. PDF panics become errors via catch_unwind. A permanent dispatcher is
//! installed lazily once; RAII guards suppress and record parser panics only on
//! the parsing thread, delegating other threads to the captured previous hook.
//! Later external hooks replace it and may print; Seyk never removes or overwrites them.
//! First installation is deferred during unwinding, so that parse may also print.
//! With `audio`, Symphonia probe, construction, packet reads and decode panics
//! likewise become payload-free `UnsupportedAudio` errors; applications must use
//! a payload-free panic hook for audio. ALAC frame allocations are checked before construction.
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
#[cfg(unix)]
pub mod local_content;
mod progress;
pub mod provider;
mod resource;
pub use progress::{IndexProgress, KindProgress};
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
    IndexReport, IndexRequest, MatchReason, MatchSignals, PageDiagnostic, ProgressEvent,
    ProgressFn, SearchEngine, SearchEngineBuilder, SearchHit, SearchMode, SearchRequest,
    SearchResults, SearchStats, SkipCode, SkippedCount, SkippedResource, SyncStats, expand_tilde,
    make_preview,
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

mod cache_guards;

#[cfg(all(feature = "test-support", debug_assertions))]
pub use engine::LeaseEvent;
