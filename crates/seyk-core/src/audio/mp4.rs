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
use std::io::{Read, Seek, SeekFrom};
/// Aggregate live allocation budget for Symphonia's parsed MP4 tables.
const TABLE_BUDGET: u64 = 64 * 1024 * 1024;
const LEAF_BUDGET: usize = 64 * 1024;
const PACKET_BUDGET: u32 = 1024 * 1024;
#[derive(Clone, Copy)]
struct Atom {
    kind: [u8; 4],
    start: u64,
    end: u64,
}
fn read_at<R: Read + Seek>(r: &mut R, start: u64, n: usize) -> Result<Vec<u8>> {
    r.seek(SeekFrom::Start(start))
        .map_err(|_| "MP4 seek failed")?;
    let mut b = vec![0; n];
    r.read_exact(&mut b).map_err(|_| "truncated MP4 box")?;
    Ok(b)
}
fn data<R: Read + Seek>(r: &mut R, a: Atom, max: usize) -> Result<Vec<u8>> {
    if a.end - a.start > max as u64 {
        return Err("MP4 metadata exceeds budget");
    }
    read_at(r, a.start, (a.end - a.start) as usize)
}
fn boxes<R: Read + Seek>(r: &mut R, mut at: u64, end: u64) -> Result<Vec<Atom>> {
    let mut out = Vec::new();
    while at < end {
        if out.len() >= 4096 || end - at < 8 {
            return Err("invalid MP4 box table");
        }
        let h = read_at(r, at, 8)?;
        let size = u32_at(&h, 0)? as u64;
        let (size, header) = if size == 1 {
            if end - at < 16 {
                return Err("truncated MP4 box");
            }
            (u64_at(&read_at(r, at + 8, 8)?, 0)?, 16)
        } else if size == 0 {
            // Symphonia represents zero-sized atom payloads as zero, rather
            // than resolving them to the parent's end. Only opaque padding or
            // media may use this form; parsed metadata could underflow upstream.
            if ![b"mdat".as_slice(), b"free", b"skip", b"wide"].contains(&&h[4..8]) {
                return Err("size-zero MP4 metadata is unsupported");
            }
            (end - at, 8)
        } else {
            (size, 8)
        };
        if size < header || size > end - at {
            return Err("invalid MP4 box length");
        }
        out.push(Atom {
            kind: h[4..8].try_into().unwrap(),
            start: at + header,
            end: at + size,
        });
        at += size;
    }
    Ok(out)
}
fn unique(atoms: &[Atom], kind: &[u8; 4]) -> Result<Option<Atom>> {
    let mut found = atoms.iter().filter(|a| &a.kind == kind);
    let first = found.next().copied();
    if found.next().is_some() {
        return Err("duplicate MP4 timing box");
    }
    Ok(first)
}
fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_be_bytes(
        b.get(at..at + 4)
            .ok_or("truncated MP4 timing")?
            .try_into()
            .unwrap(),
    ))
}
fn u64_at(b: &[u8], at: usize) -> Result<u64> {
    Ok(u64::from_be_bytes(
        b.get(at..at + 8)
            .ok_or("truncated MP4 timing")?
            .try_into()
            .unwrap(),
    ))
}
fn scale(b: &[u8]) -> Result<u32> {
    let at = match b.first() {
        Some(0) => 12,
        Some(1) => 20,
        _ => return Err("unsupported MP4 timing version"),
    };
    let n = u32_at(b, at)?;
    if n == 0 {
        return Err("zero MP4 timescale");
    }
    Ok(n)
}
// Walk every container parsed by the pinned isomp4 reader. Unknown leaves
// are capped too; unsupported codec/fragment structures are rejected outright.
fn table_budget<R: Read + Seek>(
    r: &mut R,
    parent: Atom,
    depth: usize,
    budget: &mut u64,
) -> Result<()> {
    if depth > 12 {
        return Err("MP4 metadata nesting exceeds budget");
    }
    for a in boxes(r, parent.start, parent.end)? {
        if parent.kind == *b"root" && [*b"mdat", *b"free", *b"skip", *b"wide"].contains(&a.kind) {
            continue;
        }
        if [*b"moof", *b"mvex", *b"sidx", *b"mfra"].contains(&a.kind) {
            return Err("fragmented MP4 is unsupported by bounded audio preparation");
        }
        // Every ilst item, including unknown tags and cover art, consists of
        // child atoms. Validate even items ignored by our timing reader.
        if parent.kind == *b"ilst" {
            if a.end - a.start > LEAF_BUDGET as u64 {
                return Err("MP4 metadata exceeds per-atom budget");
            }
            table_budget(
                r,
                Atom {
                    kind: *b"item",
                    ..a
                },
                depth + 1,
                budget,
            )?;
            continue;
        }
        if [
            *b"moov", *b"trak", *b"mdia", *b"minf", *b"stbl", *b"edts", *b"udta", *b"ilst",
            *b"wave",
        ]
        .contains(&a.kind)
        {
            table_budget(r, a, depth + 1, budget)?;
            continue;
        }
        if a.kind == *b"meta" {
            if a.end - a.start < 4 {
                return Err("truncated MP4 meta header");
            }
            table_budget(
                r,
                Atom {
                    start: a.start + 4,
                    ..a
                },
                depth + 1,
                budget,
            )?;
            continue;
        }
        if a.kind == *b"stsd" {
            sample_description(r, a, depth, budget)?;
            continue;
        }
        let (header, disk, live) = match &a.kind {
            b"stsz" => (12, 4, 4),
            b"stts" => (8, 8, 8),
            b"stsc" => (8, 12, 16),
            b"stco" => (8, 4, 4),
            b"co64" => (8, 8, 8),
            _ => {
                let bytes = data(r, a, LEAF_BUDGET)?;
                match &a.kind {
                    b"hdlr" if bytes.len() < 24 => return Err("truncated MP4 handler"),
                    b"data" if bytes.len() < 8 => return Err("truncated MP4 metadata data"),
                    b"name" | b"mean" if bytes.len() < 4 => {
                        return Err("truncated MP4 metadata namespace");
                    }
                    b"esds" => {
                        let body = bytes.get(4..).ok_or("truncated MP4 esds")?;
                        descriptors(body, 0)?;
                    }
                    b"elst" => {
                        // Only one edit is supported; bound even misplaced or
                        // duplicate edit lists before upstream can visit them.
                        if u32_at(&bytes, 4)? != 1 {
                            return Err("unsupported MP4 edit count");
                        }
                        let size = match bytes.first() {
                            Some(0) => 20,
                            Some(1) => 28,
                            _ => return Err("unsupported MP4 edit version"),
                        };
                        if bytes.len() < size {
                            return Err("truncated MP4 edit");
                        }
                    }
                    // These contain additional codec-specific lengths not in
                    // the AAC/ALAC contract. Never hand them to isomp4.
                    b"dfLa" | b"dOps" => return Err("unsupported MP4 codec configuration"),
                    _ => (),
                }
                continue;
            }
        };
        if a.end - a.start < header {
            return Err("truncated MP4 sample table");
        }
        let b = read_at(r, a.start, header as usize)?;
        let count = u32_at(&b, (header - 4) as usize)? as u64;
        let constant = if a.kind == *b"stsz" {
            u32_at(&b, 4)?
        } else {
            0
        };
        if constant > PACKET_BUDGET {
            return Err("MP4 sample exceeds packet budget");
        }
        let count = if constant != 0 { 0 } else { count };
        if count > (a.end - a.start - header) / disk {
            return Err("invalid MP4 sample table count");
        }
        *budget = budget
            .checked_add(count * live)
            .ok_or("MP4 sample table budget overflow")?;
        if *budget > TABLE_BUDGET {
            return Err("MP4 parsed tables exceed 64 MiB live allocation budget");
        }
        if a.kind == *b"stsz" && constant == 0 {
            // Fixed scratch buffer: large legitimate tables never become PCM-
            // sized allocations, and every packet size is checked before demux.
            r.seek(SeekFrom::Start(a.start + header))
                .map_err(|_| "MP4 seek failed")?;
            let mut buf = [0; 4096];
            let mut left = count;
            while left != 0 {
                let n = left.min((buf.len() / 4) as u64) as usize;
                r.read_exact(&mut buf[..n * 4])
                    .map_err(|_| "truncated MP4 sample sizes")?;
                if buf[..n * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|v| u32::from_be_bytes(*v) > PACKET_BUDGET)
                {
                    return Err("MP4 sample exceeds packet budget");
                }
                left -= n as u64;
            }
        }
    }
    Ok(())
}

fn sample_description<R: Read + Seek>(
    r: &mut R,
    a: Atom,
    depth: usize,
    budget: &mut u64,
) -> Result<()> {
    if a.end - a.start < 8 {
        return Err("truncated MP4 sample description");
    }
    let header = read_at(r, a.start, 8)?;
    if u32_at(&header, 4)? != 1 {
        return Err("unsupported MP4 sample entry count");
    }
    let entries = boxes(r, a.start + 8, a.end)?;
    if entries.len() != 1 {
        return Err("invalid MP4 sample entry count");
    }
    let entry = entries[0];
    if ![*b"mp4a", *b"alac"].contains(&entry.kind) {
        return Err("unsupported MP4 sample entry");
    }
    let bytes = data(r, entry, LEAF_BUDGET)?;
    let version = bytes.get(8..10).ok_or("truncated MP4 sample entry")?;
    let prefix = match version {
        [0, 0] => 28,
        [0, 1] => 44,
        _ => return Err("unsupported MP4 sample entry version"),
    };
    if bytes.len() < prefix {
        return Err("truncated MP4 sample entry");
    }
    table_budget(
        r,
        Atom {
            start: entry.start + prefix as u64,
            ..entry
        },
        depth + 1,
        budget,
    )
}

// ESDS has its own variable-length nested descriptor headers. Bounding the
// enclosing atom alone does not bound DecoderSpecificInfo's upstream allocation.
fn descriptors(mut bytes: &[u8], depth: usize) -> Result<()> {
    if depth > 3 {
        return Err("MP4 descriptor nesting exceeds budget");
    }
    while !bytes.is_empty() {
        let tag = bytes[0];
        bytes = &bytes[1..];
        let mut len = 0usize;
        let mut terminated = false;
        for _ in 0..4 {
            let (&b, rest) = bytes
                .split_first()
                .ok_or("truncated MP4 descriptor length")?;
            bytes = rest;
            len = (len << 7) | (b & 0x7f) as usize;
            if b & 0x80 == 0 {
                terminated = true;
                break;
            }
        }
        if !terminated || len > bytes.len() {
            return Err("invalid MP4 descriptor length");
        }
        let (body, rest) = bytes.split_at(len);
        match tag {
            3 => {
                // Symphonia's ES flags change cursor position without adjusting
                // its scope. Reject optional fields to keep our bounds identical.
                if body.len() < 3 || body[2] & 0xe0 != 0 {
                    return Err("unsupported MP4 ES descriptor flags");
                }
                descriptors(&body[3..], depth + 1)?;
            }
            4 => descriptors(
                body.get(13..).ok_or("truncated MP4 decoder descriptor")?,
                depth + 1,
            )?,
            // Upstream consumes exactly one byte regardless of declared len.
            // Requiring that extent keeps subsequent descriptor origins aligned.
            6 if body != [2] => return Err("unsupported MP4 SL descriptor"),
            _ => (), // Allocation length (including tag 5) is now within this bounded leaf.
        }
        bytes = rest;
    }
    Ok(())
}
pub(super) fn timing<R: Read + Seek>(r: &mut R) -> Result<Timing> {
    let len = r
        .seek(SeekFrom::End(0))
        .map_err(|_| "MP4 length unavailable")?;
    if len < 8 || read_at(r, 4, 4)? != b"ftyp" {
        return Err("MP4 requires ftyp at canonical origin");
    }
    let root = boxes(r, 0, len)?;
    // Fragment sample-run tables are a different allocation contract; this
    // phase supports the bounded, non-fragmented sample-table path only.
    if root.iter().any(|a| a.kind == *b"moof") {
        return Err("fragmented MP4 is unsupported by bounded audio preparation");
    }
    let metadata_bytes = root
        .iter()
        .filter(|a| ![*b"mdat", *b"free", *b"skip", *b"wide"].contains(&a.kind))
        .try_fold(0u64, |sum, a| {
            sum.checked_add(a.end - a.start)
                .ok_or("MP4 metadata size overflow")
        })?;
    if metadata_bytes > TABLE_BUDGET {
        return Err("MP4 metadata exceeds 64 MiB budget");
    }
    let moov = unique(&root, b"moov")?.ok_or("missing MP4 movie header")?;
    // Bounds all non-table metadata too; table allocation is checked separately
    // (stsc has larger in-memory entries than its on-disk representation).
    if moov.end - moov.start > TABLE_BUDGET {
        return Err("MP4 metadata exceeds 64 MiB budget");
    }
    table_budget(
        r,
        Atom {
            kind: *b"root",
            start: 0,
            end: len,
        },
        0,
        &mut 0,
    )?;
    let children = boxes(r, moov.start, moov.end)?;
    if children.iter().any(|a| a.kind == *b"mvex") {
        return Err("fragmented MP4 is unsupported by bounded audio preparation");
    }
    let movie = unique(&children, b"mvhd")?.ok_or("missing MP4 movie timescale")?;
    let movie_scale = scale(&data(r, movie, 256)?)?;
    let mut result = Timing::default();
    let mut audio_tracks = 0;
    for trak in children.iter().filter(|a| a.kind == *b"trak") {
        let track = boxes(r, trak.start, trak.end)?;
        let mdia = unique(&track, b"mdia")?.ok_or("missing MP4 media header")?;
        let media = boxes(r, mdia.start, mdia.end)?;
        let handler = unique(&media, b"hdlr")?.ok_or("missing MP4 handler")?;
        if data(r, handler, 65536)?.get(8..12) != Some(b"soun") {
            return Err("non-audio MP4 tracks are unsupported");
        }
        audio_tracks += 1;
        if audio_tracks > 1 {
            return Err("multiple audio tracks are unsupported");
        }
        let mdhd = data(
            r,
            unique(&media, b"mdhd")?.ok_or("missing MP4 media timescale")?,
            256,
        )?;
        let media_scale = scale(&mdhd)?;
        let duration = if mdhd[0] == 0 {
            u32_at(&mdhd, 16)? as u64
        } else {
            u64_at(&mdhd, 24)?
        };
        let unknown = if mdhd[0] == 0 {
            u32::MAX as u64
        } else {
            u64::MAX
        };
        if duration != unknown {
            result.media_end = Some((duration, media_scale));
        }
        if let Some(edts) = unique(&track, b"edts")? {
            let edits = boxes(r, edts.start, edts.end)?;
            let elst = data(
                r,
                unique(&edits, b"elst")?.ok_or("missing MP4 edit list")?,
                64,
            )?;
            if u32_at(&elst, 4)? != 1 {
                return Err("multi-entry or empty MP4 edit lists are unsupported");
            }
            let (duration, start, rate_at) = match elst.first() {
                Some(0) => (
                    u32_at(&elst, 8)? as u64,
                    u32_at(&elst, 12)? as i32 as i64,
                    16,
                ),
                Some(1) => (u64_at(&elst, 8)?, u64_at(&elst, 16)? as i64, 24),
                _ => return Err("unsupported MP4 edit version"),
            };
            if start < 0 || duration == 0 {
                return Err("empty MP4 edits are unsupported");
            }
            if u32_at(&elst, rate_at)? != 0x00010000 {
                return Err("MP4 edit playback rate is unsupported");
            }
            result.edit = Some((start as u64, duration, media_scale, movie_scale));
        }
    }
    if audio_tracks != 1 {
        return Err("no MP4 audio track");
    }
    result.smpb = smpb(r, moov.start, moov.end, 0)?;
    Ok(result)
}
fn smpb<R: Read + Seek>(
    r: &mut R,
    start: u64,
    end: u64,
    depth: usize,
) -> Result<Option<(u64, u64)>> {
    if depth > 6 {
        return Err("MP4 metadata nesting exceeds budget");
    }
    for a in boxes(r, start, end)? {
        if a.kind == *b"----" {
            let parts = boxes(r, a.start, a.end)?;
            if let Some(name) = unique(&parts, b"name")?
                && data(r, name, 65536)?.get(4..) == Some(b"iTunSMPB")
            {
                let b = data(
                    r,
                    unique(&parts, b"data")?.ok_or("missing iTunSMPB data")?,
                    65536,
                )?;
                let text = std::str::from_utf8(b.get(8..).ok_or("invalid iTunSMPB")?)
                    .map_err(|_| "invalid iTunSMPB")?;
                let fields: Vec<_> = text.split_ascii_whitespace().take(4).collect();
                if fields.len() < 4 {
                    return Err("invalid iTunSMPB fields");
                }
                return Ok(Some((
                    u64::from_str_radix(fields[1], 16).map_err(|_| "invalid iTunSMPB delay")?,
                    u64::from_str_radix(fields[3], 16).map_err(|_| "invalid iTunSMPB length")?,
                )));
            }
        } else if [*b"udta", *b"meta", *b"ilst"].contains(&a.kind) {
            let start = if a.kind == *b"meta" {
                a.start
                    .checked_add(4)
                    .filter(|n| *n <= a.end)
                    .ok_or("invalid MP4 metadata")?
            } else {
                a.start
            };
            if let Some(t) = smpb(r, start, a.end, depth + 1)? {
                return Ok(Some(t));
            }
        }
    }
    Ok(None)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    fn timing(b: &[u8]) -> Result<Timing> {
        super::timing(&mut Cursor::new(b))
    }
    fn boxes(b: &[u8]) -> Result<Vec<Atom>> {
        super::boxes(&mut Cursor::new(b), 0, b.len() as u64)
    }
    fn smpb(b: &[u8], depth: usize) -> Result<Option<(u64, u64)>> {
        super::smpb(&mut Cursor::new(b), 0, b.len() as u64, depth)
    }
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
        let mut handler = vec![0; 24];
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
    // Defensive parser tests: all adversarial containers below are built as
    // small byte vectors. Never materialize their attacker-declared sizes.
    fn reject_before_upstream(bytes: Vec<u8>, reason: &'static str) {
        let id = "mem://defensive-parser/input.m4a".parse().unwrap();
        let result = crate::audio::windows(bytes, &id, |_, _| panic!("must not emit"));
        assert!(
            matches!(result, Err(crate::Error::UnsupportedAudio { reason: actual, .. }) if actual == reason)
        );
    }
    fn with_table(table: Vec<u8>) -> Vec<u8> {
        [
            atom(b"ftyp", b"M4A \0\0\0\0"),
            atom(
                b"moov",
                &atom(
                    b"trak",
                    &atom(b"mdia", &atom(b"minf", &atom(b"stbl", &table))),
                ),
            ),
        ]
        .concat()
    }
    #[test]
    fn prefixed_ftyp_cannot_reach_upstream_mp4() {
        let table = atom(
            b"stsz",
            &[vec![0; 8], u32::MAX.to_be_bytes().to_vec()].concat(),
        );
        let movie = with_table(table);
        // ID3 selects the MP3-only fallback; its scanner must never reach isomp4.
        reject_before_upstream(
            [vec![b'I', b'D', b'3', 4, 0, 0, 0, 0, 0, 0], movie.clone()].concat(),
            "unrecognized or corrupt audio container",
        );
        for prefix in [atom(b"free", &[]), b"prefix".to_vec()] {
            reject_before_upstream(
                [prefix, movie.clone()].concat(),
                "unrecognized audio header at canonical origin",
            );
        }
    }
    #[test]
    fn constant_and_variable_sample_sizes_are_bounded_before_demux() {
        for constant in [true, false] {
            let mut body = vec![0; 4];
            body.extend(if constant { u32::MAX } else { 0 }.to_be_bytes());
            body.extend(1u32.to_be_bytes());
            if !constant {
                body.extend(u32::MAX.to_be_bytes());
            }
            reject_before_upstream(
                with_table(atom(b"stsz", &body)),
                "MP4 sample exceeds packet budget",
            );
        }
    }
    #[test]
    fn nested_extended_metadata_cannot_escape_its_parent() {
        let malicious = [
            1u32.to_be_bytes().to_vec(),
            b"data".to_vec(),
            (1u64 << 34).to_be_bytes().to_vec(),
            vec![0; 8],
        ]
        .concat();
        for item in [*b"covr", *b"----", *b"zzzz"] {
            let meta = atom(
                b"meta",
                &[vec![0; 4], atom(b"ilst", &atom(&item, &malicious))].concat(),
            );
            for nested in [false, true] {
                let metadata = if nested {
                    atom(b"moov", &atom(b"udta", &meta))
                } else {
                    meta.clone()
                };
                let movie = if nested { vec![] } else { atom(b"moov", &[]) };
                reject_before_upstream(
                    [atom(b"ftyp", b"M4A \0\0\0\0"), metadata, movie].concat(),
                    "invalid MP4 box length",
                );
            }
        }
    }
    #[test]
    fn descriptor_sizes_and_metadata_minimums_are_checked() {
        assert_eq!(
            descriptors(&[5, 0xff, 0xff, 0xff, 0x7f], 0),
            Err("invalid MP4 descriptor length")
        );
        for (kind, payload, error) in [
            (*b"hdlr", vec![0; 12], "truncated MP4 handler"),
            (*b"data", vec![0; 4], "truncated MP4 metadata data"),
            (*b"name", vec![], "truncated MP4 metadata namespace"),
            (*b"mean", vec![], "truncated MP4 metadata namespace"),
        ] {
            let bytes = atom(&kind, &payload);
            let parent = Atom {
                kind: *b"item",
                start: 0,
                end: bytes.len() as u64,
            };
            assert_eq!(
                table_budget(&mut Cursor::new(bytes), parent, 0, &mut 0),
                Err(error)
            );
        }
        // A zero-sized parsed metadata atom is unsafe even when its outer
        // extent fits: upstream assigns it data_len=0 and subtracts a prefix.
        assert!(boxes(&[0, 0, 0, 0, b'd', b'a', b't', b'a', 0, 0, 0, 0, 0, 0, 0, 0]).is_err());
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
    #[test]
    fn parent_bounds_extended_and_zero_sized_boxes() {
        let mut bytes = Vec::new();
        bytes.extend(1u32.to_be_bytes());
        bytes.extend(b"mdat");
        bytes.extend(24u64.to_be_bytes());
        bytes.extend([0; 8]);
        bytes.extend(0u32.to_be_bytes());
        bytes.extend(b"free");
        bytes.extend([0; 3]);
        let atoms = boxes(&bytes).unwrap();
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms[0].start, 16);
        assert_eq!(atoms[0].end, 24);
        assert_eq!(atoms[1].end, bytes.len() as u64);
        bytes[8..16].copy_from_slice(&u64::MAX.to_be_bytes());
        assert!(boxes(&bytes).is_err());
        assert!(super::boxes(&mut Cursor::new([0; 16]), 0, 7).is_err());
    }
    #[test]
    fn table_counts_are_checked_before_demuxer_allocations() {
        for kind in [*b"stsz", *b"stts", *b"stsc", *b"stco", *b"co64"] {
            let mut body = vec![0; if kind == *b"stsz" { 12 } else { 8 }];
            let at = body.len() - 4;
            body[at..].copy_from_slice(&u32::MAX.to_be_bytes());
            let bytes = atom(&kind, &body);
            let parent = Atom {
                kind: *b"stbl",
                start: 0,
                end: bytes.len() as u64,
            };
            assert!(table_budget(&mut Cursor::new(bytes), parent, 0, &mut 0).is_err());
        }
        let bytes = atom(
            b"stsc",
            &[vec![0; 4], 1u32.to_be_bytes().to_vec(), vec![0; 12]].concat(),
        );
        let parent = Atom {
            kind: *b"stbl",
            start: 0,
            end: bytes.len() as u64,
        };
        // stsc is 12 bytes on disk but 16 bytes in the parsed live table.
        assert!(
            table_budget(&mut Cursor::new(bytes), parent, 0, &mut (TABLE_BUDGET - 12)).is_err()
        );
    }
    #[test]
    fn fragment_tables_cannot_bypass_the_sample_table_budget() {
        let bytes = [atom(b"ftyp", b"M4A "), atom(b"moof", &[])].concat();
        assert!(matches!(
            timing(&bytes),
            Err("fragmented MP4 is unsupported by bounded audio preparation")
        ));
    }
}
