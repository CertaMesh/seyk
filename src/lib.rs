//! Local-first semantic search over files.
//!
//! The crate combines resource discovery, segmentation, an embedding backend
//! ([`Embedder`]), an embedding cache and similarity search into a
//! [`SearchEngine`]. EmbeddingGemma 2 (through ONNX Runtime) is the default
//! backend, behind the `onnx` feature.
//!
//! ```no_run
//! # #[cfg(feature = "onnx")]
//! # fn main() -> seyk::Result<()> {
//! use seyk::{SearchEngine, SqliteCache};
//! use seyk::embedder::gemma::{GemmaConfig, GemmaEmbedder};
//!
//! let embedder = GemmaEmbedder::load(GemmaConfig::default())?;
//! let engine = SearchEngine::builder()
//!     .embedder(embedder)
//!     .cache(SqliteCache::open_default()?)
//!     .build()?;
//!
//! let results = engine
//!     .search("where did I discuss inherited behaviour?")
//!     .paths(["~/notes"])
//!     .limit(10)
//!     .execute()?;
//!
//! for hit in results.hits {
//!     println!("{:.3} {} {:?}", hit.score, hit.resource.path().display(), hit.locator);
//! }
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "onnx"))]
//! # fn main() {}
//! ```

pub mod cache;
pub mod discovery;
pub mod embedder;
mod embedding;
mod engine;
mod error;
pub mod index;
pub mod lexical;
mod resource;
pub mod segment;
#[cfg(feature = "watch")]
pub mod watch;

pub use cache::{CacheStats, EmbeddingCache, MemoryCache, SqliteCache};
pub use discovery::DiscoveryOptions;
pub use embedder::{Embedder, ModelIdentity, ResourceInput};
pub use embedding::{Dimension, Embedding};
pub use engine::{
    IndexReport, IndexRequest, MatchSignals, ProgressEvent, ProgressFn, SearchEngine,
    SearchEngineBuilder, SearchHit, SearchMode, SearchRequest, SearchResults, SearchStats,
    SkippedResource, SyncStats, expand_tilde, make_preview,
};
pub use error::{Error, Result};
pub use resource::{Modality, ResourceId, ResourceLocator, ResourceMetadata};
