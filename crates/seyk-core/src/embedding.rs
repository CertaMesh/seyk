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

pub(crate) fn from_bytes(b: &[u8]) -> std::io::Result<Vec<f32>> {
    let (chunks, remainder) = b.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "vector byte length must be a multiple of 4",
        ));
    }
    Ok(chunks.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_bytes_preserve_float_bits() {
        let bits = [
            0,
            0x8000_0000,
            0x3f80_0000,
            0xc020_0000,
            0x7f80_0000,
            0x7fc0_1234,
        ];
        let values: Vec<_> = bits.iter().map(|&bits| f32::from_bits(bits)).collect();
        let decoded = from_bytes(&to_bytes(&values)).unwrap();
        assert_eq!(
            decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            bits
        );
        assert!(from_bytes(&[]).unwrap().is_empty());
    }

    #[test]
    fn rejects_partial_floats_including_after_a_full_vector() {
        for width in [0, 1, 128] {
            for remainder in 1..=3 {
                let bytes = vec![0; width * 4 + remainder];
                assert_eq!(
                    from_bytes(&bytes).unwrap_err().kind(),
                    std::io::ErrorKind::InvalidData,
                );
            }
        }
    }
}
