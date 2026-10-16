//! Offline Photo Anchor 4-bit codec for already normalized photo embeddings.
//!
//! Tables: Pastel-Cloud OÜ, CC BY 4.0; see the bundled attribution.
//! This codec does not change Seyk's default f32 storage or search behavior.
use crate::Embedding;

mod sha256;
#[cfg(test)]
mod tests;

/// Exact upstream revision of all bundled Photo Anchor assets.
pub const REVISION: &str = "be9e066984345c92424c86bfcce3b17f3e3e8484";
/// Attribution must accompany packages and binaries that bundle these tables.
pub const ATTRIBUTION: &str = include_str!("../../assets/photo-anchor/NOTICE.md");
const MANIFEST: &[u8] = include_bytes!("../../assets/photo-anchor/manifest.json");
const MANIFEST_SHA: &str = "4dbde655fdd44a1feb4b010685fe065ffbe9c1f996331b4f4a40c9a5b2580c1e";

/// Invalid inputs or corrupted/unrecognized calibration assets fail closed.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("Photo Anchor tables are not bundled for dimension {0}")]
    UnsupportedDimension(usize),
    #[error("Photo Anchor asset integrity failure: {0}")]
    AssetIntegrity(&'static str),
    #[error("expected {expected} values/bytes, got {actual}")]
    Length { expected: usize, actual: usize },
    #[error("Photo Anchor requires a finite unit embedding")]
    NonUnitEmbedding,
}

/// Dimension-specific photo mean and 4-bit calibration, loaded without I/O.
pub struct PhotoAnchorQ4 {
    anchor: Vec<f32>,
    offset: Vec<f32>,
    scale: Vec<f32>,
}

impl PhotoAnchorQ4 {
    /// Load and validate the hash-pinned 256- or 768-dimensional tables.
    pub fn new(dim: usize) -> Result<Self, Error> {
        let (anchor, table, anchor_sha, table_sha) = match dim {
            256 => (
                include_bytes!("../../assets/photo-anchor/photo_anchor_256.f32.bin").as_slice(),
                include_bytes!("../../assets/photo-anchor/quant4_256.f32.bin").as_slice(),
                "4e30ed052129ee745bb810fc1673e360857557e4a97daffdb2734e2b8dc5dc28",
                "fe029c4cf510f6a628a24bcdc3db097ce8c387e948080d5065ef81ff904a2353",
            ),
            768 => (
                include_bytes!("../../assets/photo-anchor/photo_anchor_768.f32.bin").as_slice(),
                include_bytes!("../../assets/photo-anchor/quant4_768.f32.bin").as_slice(),
                "a3978221be716a22f0d738efdc93df0de90fb8c490339011422ab15c2e8a3bb6",
                "5088d57dbb8ae53da9f45f94ce20aa9177160fcbc499d41a8ecb755270ba5560",
            ),
            _ => return Err(Error::UnsupportedDimension(dim)),
        };
        Self::load(dim, MANIFEST, anchor, table, anchor_sha, table_sha)
    }

    fn load(
        dim: usize,
        manifest: &[u8],
        anchor: &[u8],
        table: &[u8],
        anchor_sha: &str,
        table_sha: &str,
    ) -> Result<Self, Error> {
        // Exact manifest identity pins the model revision, methods and bit order,
        // without a permissive JSON parser or a runtime download fallback.
        check_hash(manifest, MANIFEST_SHA, "manifest")?;
        let manifest =
            std::str::from_utf8(manifest).map_err(|_| Error::AssetIntegrity("manifest"))?;
        for expected in [
            "google/embeddinggemma-2",
            "914f7f89142e33e77833254d9c9b90c3cef7303b",
            anchor_sha,
            table_sha,
        ] {
            if !manifest.contains(expected) {
                return Err(Error::AssetIntegrity("manifest metadata"));
            }
        }
        check_hash(anchor, anchor_sha, "anchor")?;
        check_hash(table, table_sha, "calibration")?;
        check_len(anchor.len(), dim * 4)?;
        check_len(table.len(), dim * 8)?;
        let floats = |bytes: &[u8]| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>()
        };
        let anchor = floats(anchor);
        let offset = floats(&table[..dim * 4]);
        let scale = floats(&table[dim * 4..]);
        if anchor
            .iter()
            .chain(&offset)
            .chain(&scale)
            .any(|x| !x.is_finite())
            || scale.iter().any(|&x| x <= 0.0)
        {
            return Err(Error::AssetIntegrity(
                "non-finite values or nonpositive scale",
            ));
        }
        Ok(Self {
            anchor,
            offset,
            scale,
        })
    }

    pub fn dim(&self) -> usize {
        self.anchor.len()
    }

    /// Subtract the anchor, clip, round ties to even, and pack high nibble first.
    /// Rejects wrong dimensions and zero/non-unit deserialized embeddings.
    pub fn encode(&self, embedding: &Embedding) -> Result<Vec<u8>, Error> {
        check_embedding(embedding, self.dim())?;
        Ok(embedding
            .as_slice()
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .map(|(j, pair)| {
                let code = |i: usize, x: f32| {
                    let centered = (x - self.anchor[i])
                        .clamp(self.offset[i], self.offset[i] + 15.0 * self.scale[i]);
                    ((centered - self.offset[i]) / self.scale[i])
                        .round_ties_even()
                        .clamp(0.0, 15.0) as u8
                };
                (code(2 * j, pair[0]) << 4) | code(2 * j + 1, pair[1])
            })
            .collect())
    }
}

/// Asymmetric query scoring: the query remains f32 and is never centered.
/// Reconstruction is not renormalized, so scores are approximate dot products.
pub struct QueryLut {
    bias: f32,
    pairs: Vec<[f32; 256]>,
}

impl QueryLut {
    /// Validate the query and build one 256-entry table for each packed byte.
    /// Build once per query; reuse across all photo codes of the same dimension.
    pub fn new(query: &Embedding) -> Result<Self, Error> {
        let codec = PhotoAnchorQ4::new(query.dim())?;
        check_embedding(query, codec.dim())?;
        let q = query.as_slice();
        let bias = (0..codec.dim())
            .map(|i| (codec.anchor[i] + codec.offset[i]) * q[i])
            .sum();
        let pairs = (0..codec.dim())
            .step_by(2)
            .map(|i| {
                let (hi, lo) = (codec.scale[i] * q[i], codec.scale[i + 1] * q[i + 1]);
                std::array::from_fn(|byte| (byte >> 4) as f32 * hi + (byte & 15) as f32 * lo)
            })
            .collect();
        Ok(Self { bias, pairs })
    }

    /// Score exactly one packed vector, rejecting truncated or oversized codes.
    pub fn score(&self, codes: &[u8]) -> Result<f32, Error> {
        check_len(codes.len(), self.pairs.len())?;
        Ok(self.bias
            + self
                .pairs
                .iter()
                .zip(codes)
                .map(|(table, &code)| table[code as usize])
                .sum::<f32>())
    }
}

fn check_len(actual: usize, expected: usize) -> Result<(), Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Length { expected, actual })
    }
}

fn check_embedding(embedding: &Embedding, dim: usize) -> Result<(), Error> {
    check_len(embedding.dim(), dim)?;
    if embedding.normalized() {
        Ok(())
    } else {
        Err(Error::NonUnitEmbedding)
    }
}

fn check_hash(bytes: &[u8], expected: &str, name: &'static str) -> Result<(), Error> {
    if sha256::hex(bytes) == expected {
        Ok(())
    } else {
        Err(Error::AssetIntegrity(name))
    }
}
