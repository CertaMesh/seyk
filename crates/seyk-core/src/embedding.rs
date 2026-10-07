pub use seyk_embed::{Dimension, Embedding};
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

pub(crate) fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(crate) fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}
