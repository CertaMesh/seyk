//! Validate allocation-driving ALAC metadata before constructing Symphonia.

/// Four times Apple's usual 4096-sample frame; at eight channels this bounds
/// each i32 output/tail buffer to 512 KiB before any decoder allocation.
const MAX_FRAME_LENGTH: u32 = 16_384;

pub(super) fn validate_cookie(cookie: &[u8]) -> Result<(), &'static str> {
    // Symphonia 0.5.5 strips the MP4 atom header and passes only 24/48 bytes.
    if !matches!(cookie.len(), 24 | 48) {
        return Err("invalid ALAC magic cookie size");
    }
    let frame_length = u32::from_be_bytes(cookie[..4].try_into().unwrap());
    if !(1..=MAX_FRAME_LENGTH).contains(&frame_length) {
        return Err("ALAC frame length exceeds bounded decoder budget");
    }
    if cookie[4] != 0 || !matches!(cookie[5], 16 | 20 | 24 | 32) || !(1..=8).contains(&cookie[9]) {
        return Err("unsupported ALAC version, bit depth or channel count");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cookie(frames: u32) -> [u8; 24] {
        let mut bytes = [0; 24];
        bytes[..4].copy_from_slice(&frames.to_be_bytes());
        bytes[5] = 16;
        bytes[9] = 2;
        bytes
    }

    #[test]
    fn validates_allocation_bounds_without_decoding_media() {
        for frames in [1, 4096, MAX_FRAME_LENGTH] {
            assert!(validate_cookie(&cookie(frames)).is_ok());
        }
        for frames in [0, MAX_FRAME_LENGTH + 1, u32::MAX] {
            assert!(validate_cookie(&cookie(frames)).is_err());
        }
        let mut extended = [0; 48];
        extended[..24].copy_from_slice(&cookie(4096));
        assert!(validate_cookie(&extended).is_ok());
        for size in [0, 23, 25, 36, 47, 49] {
            assert!(validate_cookie(&vec![0; size]).is_err());
        }
        for (offset, values) in [(4, vec![1]), (5, vec![0, 8, 33]), (9, vec![0, 9])] {
            for value in values {
                let mut bytes = cookie(4096);
                bytes[offset] = value;
                assert!(validate_cookie(&bytes).is_err());
            }
        }
    }
}
