//! Reject SBR/PS before Symphonia can decode only its AAC-LC core.
//! Accept the bounded AAC-LC AudioSpecificConfig subset used by this phase.
pub(super) fn lc_config(bytes: &[u8]) -> bool {
    if bytes.len() < 2 || bytes.len() > 64 {
        return false;
    }
    let mut bits = Bits { bytes, at: 0 };
    if bits.take(5) != Some(2) {
        return false;
    }
    let rate = bits.take(4);
    if rate == Some(15) {
        if !matches!(bits.take(24), Some(8000..=192000)) {
            return false;
        }
    } else if !matches!(rate, Some(0..=12)) {
        return false;
    }
    if !matches!(bits.take(4), Some(1 | 2)) {
        return false;
    }
    // 1024 samples, no dependency on a core coder, no GA extension.
    if bits.take(3) != Some(0) {
        return false;
    }
    if bits.zero_tail() {
        return true;
    }
    // AAC-LC exports may explicitly signal that SBR is absent.
    if bits.take(11) != Some(0x2b7) || bits.take(5) != Some(5) || bits.take(1) != Some(0) {
        return false;
    }
    if bits.zero_tail() {
        return true;
    }
    bits.take(11) == Some(0x548) && bits.take(1) == Some(0) && bits.zero_tail()
}
struct Bits<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl Bits<'_> {
    fn take(&mut self, n: usize) -> Option<u32> {
        if self.at + n > self.bytes.len() * 8 {
            return None;
        }
        let mut value = 0;
        for i in self.at..self.at + n {
            value = (value << 1) | u32::from((self.bytes[i / 8] >> (7 - i % 8)) & 1);
        }
        self.at += n;
        Some(value)
    }
    fn zero_tail(&self) -> bool {
        (self.at..self.bytes.len() * 8).all(|i| (self.bytes[i / 8] >> (7 - i % 8)) & 1 == 0)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_and_backward_compatible_sbr_are_rejected() {
        assert!(lc_config(&[0x11, 0x90]));
        assert!(lc_config(&[0x11, 0x90, 0x56, 0xe5, 0x00]));
        assert!(!lc_config(&[0x11, 0x90, 0x56, 0xe5, 0x80]));
        assert!(!lc_config(&[0x2b, 0x92, 0x08]));
        assert!(!lc_config(&[0x11]));
    }
}
