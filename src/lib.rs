//! Local-first hybrid retrieval over files.
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
        #[cfg(feature = "audio")]
        pub use crate::model_download::{
            AudioAsset, audio_assets, audio_directory, cleanup_audio_stages, download_audio,
            verify_audio,
        };
        pub use seyk_embeddinggemma::*;
    }
}
#[cfg(feature = "onnx")]
mod model_download;

/// Speech backend and explicit provisioning. Runtime is offline.
#[cfg(feature = "audio-stt")]
pub mod stt {
    pub use crate::stt_download::download;
    pub use seyk_stt::*;
}
#[cfg(feature = "audio-stt")]
mod stt_download;
