use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The kind of content a resource holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
            Modality::Video => "video",
        })
    }
}

/// Identifies a searchable resource. Today every resource is a local file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResourceId(PathBuf);

impl ResourceId {
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self(path.into())
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

/// Where inside a resource a hit was found. Deliberately modality-independent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResourceLocator {
    /// A byte range of a text resource. Lines are 1-based and inclusive.
    Text {
        byte_start: u64,
        byte_end: u64,
        line_start: Option<u64>,
        line_end: Option<u64>,
    },
    /// A page of a paged document (reserved for PDF and similar).
    Page { page: u32 },
    /// A time window of an audio or video resource.
    Time { start_ms: u64, end_ms: u64 },
    /// The resource as a whole, for example a single image.
    WholeResource,
}

/// Facts about a resource at the time it was embedded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceMetadata {
    pub modality: Modality,
    pub size_bytes: u64,
    /// Modification time in nanoseconds since the Unix epoch, when available.
    pub modified_ns: Option<i64>,
    /// BLAKE3 hash of the full resource content, hex encoded.
    pub content_hash: String,
}
