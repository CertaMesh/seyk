//! Search, caching and verified evidence, independent of any concrete embedding backend.

pub mod cache;
pub mod discovery;
pub mod embedder {
    pub use seyk_embed::*;
}
mod embedding;
mod engine;
mod error;
pub mod extractor;
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
    IndexReport, IndexRequest, MatchSignals, ProgressEvent, ProgressFn, SearchEngine,
    SearchEngineBuilder, SearchHit, SearchMode, SearchRequest, SearchResults, SearchStats,
    SkippedResource, SyncStats, expand_tilde, make_preview,
};
pub use error::{Error, Result};
pub use resource::{ResourceId, ResourceLocator, ResourceMetadata};
pub use seyk_embed::{EmbeddingRequest, EmbeddingTask, Modality};

pub use provider::{
    FsProvider, ProvidedResource, ResourceProvider, ResourceScope, ResourceVersion,
};

pub use extractor::{ExtractedContent, Extractor, TextExtractor, extract_resource};
