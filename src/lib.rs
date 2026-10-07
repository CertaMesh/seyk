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
//!     println!("{:.3} {} {:?}", hit.score, hit.resource, hit.locator);
//! }
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "onnx"))]
//! # fn main() {}
//! ```

pub use seyk_core::*;
pub use seyk_embed::{
    EmbeddingRequest, EmbeddingTask, VideoFrame, cosine_similarity, dot_product, normalize,
};
pub mod embedder {
    pub use seyk_embed::*;
    #[cfg(feature = "onnx")]
    pub mod gemma {
        pub use crate::model_download::download;
        pub use seyk_embeddinggemma::*;
    }
}
#[cfg(feature = "onnx")]
mod model_download;
