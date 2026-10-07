use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Output dimensions supported by Matryoshka-trained models such as EmbeddingGemma 2.
///
/// Smaller dimensions keep the leading components of the full vector and are
/// re-normalised afterwards. Queries and resources must always share a dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(into = "usize", try_from = "usize")]
pub enum Dimension {
    D128,
    D256,
    D512,
    D768,
}

impl Dimension {
    pub const ALL: [Dimension; 4] = [Self::D128, Self::D256, Self::D512, Self::D768];

    pub fn get(self) -> usize {
        match self {
            Self::D128 => 128,
            Self::D256 => 256,
            Self::D512 => 512,
            Self::D768 => 768,
        }
    }
}

/// 256 is close to lossless on the official MTEB/MIEB numbers (60.41 vs 61.36
/// multilingual MTEB at 768) while storing a third of the bytes.
impl Default for Dimension {
    fn default() -> Self {
        Self::D256
    }
}

impl From<Dimension> for usize {
    fn from(d: Dimension) -> usize {
        d.get()
    }
}

impl TryFrom<usize> for Dimension {
    type Error = Error;

    fn try_from(value: usize) -> Result<Self> {
        Dimension::ALL
            .into_iter()
            .find(|d| d.get() == value)
            .ok_or_else(|| {
                Error::Config(format!(
                    "unsupported dimension {value}; use 128, 256, 512 or 768"
                ))
            })
    }
}

impl std::fmt::Display for Dimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get())
    }
}

impl std::str::FromStr for Dimension {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let n: usize = s
            .parse()
            .map_err(|_| Error::Config(format!("invalid dimension `{s}`")))?;
        Dimension::try_from(n)
    }
}

/// A unit-length embedding vector.
///
/// Constructed only through [`Embedding::from_raw`], which truncates and
/// L2-normalises, so cosine similarity reduces to a dot product.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding(Vec<f32>);

impl Embedding {
    /// Truncate a raw model output to `dim` leading components, then L2-normalise.
    ///
    /// Normalising after truncation matters: a slice of a unit vector is not
    /// unit length, and skipping this silently degrades ranking.
    pub fn from_raw(mut raw: Vec<f32>, dim: Dimension) -> Result<Self> {
        let d = dim.get();
        if raw.len() < d {
            return Err(Error::DimensionMismatch {
                expected: d,
                actual: raw.len(),
            });
        }
        raw.truncate(d);
        l2_normalize(&mut raw);
        if raw.iter().any(|x| !x.is_finite()) {
            return Err(Error::Embedding(
                "model produced a non-finite embedding".into(),
            ));
        }
        Ok(Self(raw))
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.0
    }

    /// Cosine similarity with another normalised embedding.
    pub fn cosine(&self, other: &Embedding) -> f32 {
        dot(&self.0, &other.0)
    }
}

pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_renormalizes() {
        let raw: Vec<f32> = (0..768).map(|i| (i as f32).sin()).collect();
        let e = Embedding::from_raw(raw, Dimension::D128).unwrap();
        assert_eq!(e.dim(), 128);
        let norm: f32 = e.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rejects_short_vectors() {
        assert!(Embedding::from_raw(vec![1.0; 100], Dimension::D128).is_err());
    }

    #[test]
    fn byte_roundtrip() {
        let v = vec![0.5, -1.25, 3.0];
        assert_eq!(from_bytes(&to_bytes(&v)), v);
    }

    #[test]
    fn dimension_parse() {
        assert_eq!("512".parse::<Dimension>().unwrap(), Dimension::D512);
        assert!("300".parse::<Dimension>().is_err());
    }
}
