//! Minimal bounded MP4 timing reader. No payloads escape as diagnostics.
type Result<T> = std::result::Result<T, &'static str>;
#[derive(Default)]
pub(super) struct Timing {
    edit: Option<(u64, u64, u32, u32)>,
    smpb: Option<(u64, u64)>,
    media_end: Option<(u64, u32)>,
}
impl Timing {
    pub fn samples(&self, rate: u32) -> Result<(u64, Option<u64>)> {
        if let Some((start, duration, media_scale, movie_scale)) = self.edit {
            let convert = |n: u64, scale: u32| -> Result<u64> {
                if scale == 0 {
                    return Err("zero MP4 timescale");
                }
                let n = (n as u128 * rate as u128 + scale as u128 / 2) / scale as u128;
                u64::try_from(n).map_err(|_| "MP4 timing overflow")
            };
            let start = convert(start, media_scale)?;
            let mut length = convert(duration, movie_scale)?;
            if let Some((end, scale)) = self.media_end {
                let remaining = convert(end, scale)?
                    .checked_sub(start)
                    .ok_or("MP4 edit starts beyond media duration")?;
                // Movie timescales often round up to milliseconds. mdhd retains
                // the final media-sample extent, so codec block padding cannot
                // become evidence merely because elst rounded the end upward.
                length = length.min(remaining);
            }
            return Ok((start, Some(length)));
        }
        Ok(self
            .smpb
            .map_or((0, None), |(delay, length)| (delay, Some(length))))
    }
}
#[derive(Clone, Copy)]
struct Atom<'a> {
    kind: &'a [u8],
    data: &'a [u8],
}
fn boxes(mut bytes: &[u8]) -> Result<Vec<Atom<'_>>> {
    let mut out = Vec::new();
    while !bytes.is_empty() {
        if out.len() >= 4096 || bytes.len() < 8 {
            return Err("invalid MP4 box table");
        }
        let size = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as u64;
        let (size, header) = if size == 1 {
            if bytes.len() < 16 {
                return Err("truncated MP4 box");
            }
            (u64::from_be_bytes(bytes[8..16].try_into().unwrap()), 16)
        } else if size == 0 {
            (bytes.len() as u64, 8)
        } else {
            (size, 8)
        };
        if size < header || size > bytes.len() as u64 {
            return Err("invalid MP4 box length");
        }
        let size = size as usize;
        out.push(Atom {
            kind: &bytes[4..8],
            data: &bytes[header as usize..size],
        });
        bytes = &bytes[size..];
    }
    Ok(out)
}
fn unique<'a>(atoms: &[Atom<'a>], kind: &[u8]) -> Result<Option<Atom<'a>>> {
    let mut found = atoms.iter().filter(|a| a.kind == kind);
    let first = found.next().copied();
    if found.next().is_some() {
        return Err("duplicate MP4 timing box");
    }
    Ok(first)
}
fn u32_at(bytes: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_be_bytes(
        bytes
            .get(at..at + 4)
            .ok_or("truncated MP4 timing")?
            .try_into()
            .unwrap(),
    ))
}
fn u64_at(bytes: &[u8], at: usize) -> Result<u64> {
    Ok(u64::from_be_bytes(
        bytes
            .get(at..at + 8)
            .ok_or("truncated MP4 timing")?
            .try_into()
            .unwrap(),
    ))
}
fn scale(atom: Atom<'_>) -> Result<u32> {
    let at = match atom.data.first() {
        Some(0) => 12,
        Some(1) => 20,
        _ => return Err("unsupported MP4 timing version"),
    };
    let scale = u32_at(atom.data, at)?;
    if scale == 0 {
        return Err("zero MP4 timescale");
    }
    Ok(scale)
}
pub(super) fn timing(bytes: &[u8]) -> Result<Timing> {
    if bytes.get(4..8) != Some(b"ftyp") {
        return Ok(Timing::default());
    }
    let root = boxes(bytes)?;
    let moov = unique(&root, b"moov")?.ok_or("missing MP4 movie header")?;
    let children = boxes(moov.data)?;
    let movie_scale = scale(unique(&children, b"mvhd")?.ok_or("missing MP4 movie timescale")?)?;
    let mut result = Timing::default();
    let mut audio_tracks = 0;
    for trak in children.iter().filter(|a| a.kind == b"trak") {
        let track = boxes(trak.data)?;
        let mdia = unique(&track, b"mdia")?.ok_or("missing MP4 media header")?;
        let media = boxes(mdia.data)?;
        let handler = unique(&media, b"hdlr")?.ok_or("missing MP4 handler")?;
        if handler.data.get(8..12) != Some(b"soun") {
            return Err("non-audio MP4 tracks are unsupported");
        }
        audio_tracks += 1;
        if audio_tracks > 1 {
            return Err("multiple audio tracks are unsupported");
        }
        let mdhd = unique(&media, b"mdhd")?.ok_or("missing MP4 media timescale")?;
        let media_scale = scale(mdhd)?;
        let duration = if mdhd.data[0] == 0 {
            u32_at(mdhd.data, 16)? as u64
        } else {
            u64_at(mdhd.data, 24)?
        };
        let unknown = if mdhd.data[0] == 0 {
            u32::MAX as u64
        } else {
            u64::MAX
        };
        if duration != unknown {
            result.media_end = Some((duration, media_scale));
        }
        if let Some(edts) = unique(&track, b"edts")? {
            let edits = boxes(edts.data)?;
            let elst = unique(&edits, b"elst")?.ok_or("missing MP4 edit list")?;
            if u32_at(elst.data, 4)? != 1 {
                return Err("multi-entry or empty MP4 edit lists are unsupported");
            }
            let (duration, start, rate_at) = match elst.data.first() {
                Some(0) => (
                    u32_at(elst.data, 8)? as u64,
                    u32_at(elst.data, 12)? as i32 as i64,
                    16,
                ),
                Some(1) => (u64_at(elst.data, 8)?, u64_at(elst.data, 16)? as i64, 24),
                _ => return Err("unsupported MP4 edit version"),
            };
            if start < 0 || duration == 0 {
                return Err("empty MP4 edits are unsupported");
            }
            if u32_at(elst.data, rate_at)? != 0x00010000 {
                return Err("MP4 edit playback rate is unsupported");
            }
            result.edit = Some((start as u64, duration, media_scale, movie_scale));
        }
    }
    if audio_tracks != 1 {
        return Err("no MP4 audio track");
    }
    result.smpb = smpb(moov.data, 0)?;
    Ok(result)
}
fn smpb(bytes: &[u8], depth: usize) -> Result<Option<(u64, u64)>> {
    if depth > 6 {
        return Err("MP4 metadata nesting exceeds budget");
    }
    for atom in boxes(bytes)? {
        if atom.kind == b"----" {
            let parts = boxes(atom.data)?;
            if let Some(name) = unique(&parts, b"name")?
                && name.data.get(4..) == Some(b"iTunSMPB")
            {
                let data = unique(&parts, b"data")?.ok_or("missing iTunSMPB data")?;
                let text = std::str::from_utf8(data.data.get(8..).ok_or("invalid iTunSMPB")?)
                    .map_err(|_| "invalid iTunSMPB")?;
                let fields: Vec<_> = text.split_ascii_whitespace().take(4).collect();
                if fields.len() < 4 {
                    return Err("invalid iTunSMPB fields");
                }
                let delay =
                    u64::from_str_radix(fields[1], 16).map_err(|_| "invalid iTunSMPB delay")?;
                let length =
                    u64::from_str_radix(fields[3], 16).map_err(|_| "invalid iTunSMPB length")?;
                return Ok(Some((delay, length)));
            }
        } else if [b"udta".as_slice(), b"meta", b"ilst"].contains(&atom.kind) {
            let data = if atom.kind == b"meta" {
                atom.data.get(4..).ok_or("invalid MP4 metadata")?
            } else {
                atom.data
            };
            if let Some(timing) = smpb(data, depth + 1)? {
                return Ok(Some(timing));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn atom(name: &[u8; 4], data: &[u8]) -> Vec<u8> {
        [(data.len() as u32 + 8).to_be_bytes().as_slice(), name, data].concat()
    }
    fn movie(edits: &[u8], version: u8) -> Vec<u8> {
        let mut header = vec![0; 32];
        header[0] = version;
        let off = if version == 0 { 12 } else { 20 };
        header[off..off + 4].copy_from_slice(&48000u32.to_be_bytes());
        if version == 0 {
            header[16..20].copy_from_slice(&50112u32.to_be_bytes());
        } else {
            header[24..32].copy_from_slice(&50112u64.to_be_bytes());
        }
        let mut handler = vec![0; 12];
        handler[8..12].copy_from_slice(b"soun");
        let mdia = atom(
            b"mdia",
            &[atom(b"mdhd", &header), atom(b"hdlr", &handler)].concat(),
        );
        let track = atom(
            b"trak",
            &[mdia, atom(b"edts", &atom(b"elst", edits))].concat(),
        );
        [
            atom(b"ftyp", b"M4A \0\0\0\0"),
            atom(b"moov", &[atom(b"mvhd", &header), track].concat()),
        ]
        .concat()
    }
    #[test]
    fn single_edit_v0_v1_and_rejections() {
        for version in [0, 1] {
            let mut edit = vec![version, 0, 0, 0];
            edit.extend(1u32.to_be_bytes());
            if version == 0 {
                edit.extend(48000u32.to_be_bytes());
                edit.extend(2112u32.to_be_bytes());
            } else {
                edit.extend(48000u64.to_be_bytes());
                edit.extend(2112u64.to_be_bytes());
            }
            edit.extend(0x10000u32.to_be_bytes());
            assert_eq!(
                timing(&movie(&edit, version))
                    .unwrap()
                    .samples(16000)
                    .unwrap(),
                (704, Some(16000))
            );
            edit[4..8].copy_from_slice(&2u32.to_be_bytes());
            assert!(timing(&movie(&edit, version)).is_err());
            edit[4..8].copy_from_slice(&0u32.to_be_bytes());
            assert!(timing(&movie(&edit, version)).is_err());
            edit[4..8].copy_from_slice(&1u32.to_be_bytes());
            let start = if version == 0 { 12..16 } else { 16..24 };
            edit[start].fill(255);
            assert!(timing(&movie(&edit, version)).is_err());
        }
        assert!(boxes(&[0; 7]).is_err());
        assert!(boxes(&[0xff; 16]).is_err());
    }
    #[test]
    fn metadata_fallback_and_no_metadata_policy() {
        let freeform = atom(
            b"----",
            &[
                atom(b"name", &[vec![0; 4], b"iTunSMPB".to_vec()].concat()),
                atom(
                    b"data",
                    &[
                        vec![0; 8],
                        b" 00000000 00000840 00000100 000000000000BB80".to_vec(),
                    ]
                    .concat(),
                ),
            ]
            .concat(),
        );
        let parsed = smpb(
            &atom(
                b"udta",
                &atom(b"meta", &[vec![0; 4], atom(b"ilst", &freeform)].concat()),
            ),
            0,
        )
        .unwrap();
        assert_eq!(parsed, Some((2112, 48000)));
        assert_eq!(
            Timing {
                edit: None,
                smpb: parsed,
                media_end: None,
            }
            .samples(48000)
            .unwrap(),
            (2112, Some(48000))
        );
        assert_eq!(Timing::default().samples(16000).unwrap(), (0, None));
    }
    #[test]
    fn rounded_edit_end_cannot_extend_the_media() {
        let timing = Timing {
            edit: Some((1024, 3000, 48000, 1000)),
            smpb: None,
            media_end: Some((145011, 48000)),
        };
        assert_eq!(timing.samples(48000).unwrap(), (1024, Some(143987)));
    }
}
