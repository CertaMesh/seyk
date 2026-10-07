use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// A nonzero embedding width, independent of any model's supported dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(into = "usize", try_from = "usize")]
pub struct Dimension(usize);

impl Dimension {
    pub const D128: Self = Self(128);
    pub const D256: Self = Self(256);
    pub const D512: Self = Self(512);
    pub const D768: Self = Self(768);
    /// Common Matryoshka widths. Individual backends validate their own support.
    pub const ALL: [Self; 4] = [Self::D128, Self::D256, Self::D512, Self::D768];
    pub fn get(self) -> usize {
        self.0
    }
}

/// The historical default width; backends may choose a different default.
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
        if value == 0 {
            Err(Error::Config("embedding dimension must be nonzero".into()))
        } else {
            Ok(Self(value))
        }
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

/// A finite vector, truncated and L2-normalized on construction.
/// A zero vector remains zero and reports `normalized() == false`.
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

    /// Width derived from the vector, never duplicated mutable metadata.
    pub fn dimension(&self) -> usize {
        self.dim()
    }

    /// Whether this vector has unit L2 norm (within 1e-5).
    pub fn normalized(&self) -> bool {
        let norm = self.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        norm.is_finite() && (norm - 1.0).abs() < 1e-5
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

/// Dot product with eight independent accumulators.
///
/// A plain `iter().sum()` is a strict sequential float reduction that LLVM may
/// not reorder, so it does not vectorise. Separate lanes let it use SIMD; this
/// was 5-6x faster on 256- and 768-d vectors (benches/similarity.rs, dot_kernel).
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    let mut acc = [0.0f32; 8];
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// Dot product. Unequal dimensions are errors, including in release builds.
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(Error::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let value = dot(a, b);
    if !value.is_finite() {
        return Err(Error::Embedding("non-finite dot product".into()));
    }
    Ok(value)
}

/// Cosine similarity, including for vectors deserialized without normalization.
/// Zero vectors have no cosine and return an error.
pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> Result<f32> {
    let product = dot_product(a.as_slice(), b.as_slice())?;
    let norm_a = a.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b = b.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt();
    let denominator = norm_a * norm_b;
    if denominator == 0.0 || !denominator.is_finite() {
        return Err(Error::Embedding(
            "cosine requires finite nonzero norms".into(),
        ));
    }
    Ok(product / denominator)
}

/// L2-normalize in place. A zero vector remains zero; nonfinite values fail.
pub fn normalize(values: &mut [f32]) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Embedding(
            "cannot normalize non-finite values".into(),
        ));
    }
    l2_normalize(values);
    Ok(())
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
    fn dot_matches_naive_sum() {
        for n in [1, 7, 8, 9, 128, 131, 768] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos()).collect();
            let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot(&a, &b) - naive).abs() < 1e-4, "n={n}");
        }
    }

    #[test]
    fn dimension_parse() {
        assert_eq!("512".parse::<Dimension>().unwrap(), Dimension::D512);
        assert_eq!("300".parse::<Dimension>().unwrap().get(), 300);
        assert!("0".parse::<Dimension>().is_err());
    }
}
