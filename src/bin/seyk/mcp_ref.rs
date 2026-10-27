//! Stateless selection codec. Decoding does not authorize or read a resource.
use anyhow::{Result, anyhow, bail, ensure};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::{Deserialize, Serialize};
use seyk::{Modality, ResourceId, ResourceLocator, ResourceMetadata};

pub const PREFIX: &str = "seyk-ref-v1:";
pub const MAX_BYTES: usize = 65_536;
pub const MAX_REF: usize = PREFIX.len() + (4 * MAX_BYTES).div_ceil(3);
pub const MAX_PATH: usize = 16_384;

#[derive(Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Selection {
    pub resource: ResourceId,
    pub locator: ResourceLocator,
    pub metadata: ResourceMetadata,
}

fn var(out: &mut Vec<u8>, mut n: u64) {
    while n >= 128 {
        out.push((n as u8 & 127) | 128);
        n >>= 7;
    }
    out.push(n as u8);
}

fn string(out: &mut Vec<u8>, s: &str, limit: usize) -> Result<()> {
    ensure!(s.len() <= limit, "reference string exceeds its bound");
    var(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

fn optional(out: &mut Vec<u8>, n: Option<u64>) {
    out.push(u8::from(n.is_some()));
    if let Some(n) = n {
        var(out, n);
    }
}

fn hash(out: &mut Vec<u8>, s: &str) -> Result<()> {
    ensure!(
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid reference hash"
    );
    for pair in s.as_bytes().as_chunks::<2>().0 {
        let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
        out.push(digit(pair[0]) * 16 + digit(pair[1]));
    }
    Ok(())
}

fn path_resource(path: &str) -> Result<ResourceId> {
    ensure!(
        path.len() <= MAX_PATH && std::path::Path::new(path).is_absolute(),
        "reference requires a bounded absolute path"
    );
    Ok(ResourceId::from_path(path)?)
}

pub fn encode(s: &Selection, path: Option<&str>) -> Result<String> {
    let omitted = path.is_some_and(|p| path_resource(p).is_ok_and(|r| r == s.resource));
    let mut out = vec![0x53, 0x52, 1, u8::from(omitted)];
    if !omitted {
        string(&mut out, s.resource.as_str(), MAX_PATH)?;
    }
    use ResourceLocator::*;
    match &s.locator {
        Text {
            byte_start,
            byte_end,
            line_start,
            line_end,
        } => {
            out.push(0);
            var(&mut out, *byte_start);
            var(&mut out, *byte_end);
            optional(&mut out, *line_start);
            optional(&mut out, *line_end);
        }
        PageText {
            page,
            byte_start,
            byte_end,
            text_version,
            page_text_hash,
        } => {
            out.push(1);
            var(&mut out, u64::from(*page));
            var(&mut out, *byte_start);
            var(&mut out, *byte_end);
            string(&mut out, text_version, 4096)?;
            hash(&mut out, page_text_hash)?;
        }
        Page { page } => {
            out.push(2);
            var(&mut out, u64::from(*page));
        }
        Time { start_ms, end_ms } => {
            out.push(3);
            var(&mut out, *start_ms);
            var(&mut out, *end_ms);
        }
        ImageRegion {
            x,
            y,
            width,
            height,
        } => {
            out.push(4);
            for n in [x, y, width, height] {
                var(&mut out, u64::from(*n));
            }
        }
        WholeResource => out.push(5),
        _ => bail!("unsupported reference locator"),
    }
    out.push(match s.metadata.modality {
        Modality::Document => 0,
        Modality::Text => 1,
        Modality::Image => 2,
        Modality::Audio => 3,
        Modality::Video => 4,
        _ => bail!("unsupported reference modality"),
    });
    var(&mut out, s.metadata.size_bytes);
    optional(
        &mut out,
        s.metadata
            .modified_ns
            .map(|n| ((n as u64) << 1) ^ ((n >> 63) as u64)),
    );
    match s.metadata.media_type.as_deref() {
        None => out.push(0),
        Some("application/pdf") => out.push(1),
        Some(s) => {
            out.push(2);
            string(&mut out, s, 1024)?;
        }
    }
    hash(&mut out, &s.metadata.content_hash)?;
    ensure!(out.len() <= MAX_BYTES, "reference exceeds decoded bound");
    Ok(format!("{PREFIX}{}", URL_SAFE_NO_PAD.encode(out)))
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(n <= self.0.len(), "truncated reference");
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn var(&mut self) -> Result<u64> {
        let mut n = 0;
        for i in 0..10 {
            let b = self.byte()?;
            ensure!(i != 9 || b <= 1, "reference integer overflow");
            n |= u64::from(b & 127) << (i * 7);
            if b & 128 == 0 {
                ensure!(i == 0 || b != 0, "noncanonical reference integer");
                return Ok(n);
            }
        }
        bail!("reference integer overflow")
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(self.var()?.try_into()?)
    }
    fn optional(&mut self) -> Result<Option<u64>> {
        match self.byte()? {
            0 => Ok(None),
            1 => Ok(Some(self.var()?)),
            _ => bail!("invalid reference option"),
        }
    }
    fn string(&mut self, limit: usize) -> Result<String> {
        let n = usize::try_from(self.var()?)?;
        ensure!(n <= limit, "reference string exceeds its bound");
        Ok(std::str::from_utf8(self.take(n)?)?.into())
    }
    fn hash(&mut self) -> Result<String> {
        use std::fmt::Write;
        let mut hex = String::with_capacity(64);
        for b in self.take(32)? {
            write!(&mut hex, "{b:02x}")?;
        }
        Ok(hex)
    }
}

pub fn decode(value: &str, path: Option<&str>) -> Result<Selection> {
    ensure!(value.len() <= MAX_REF, "reference exceeds encoded bound");
    let encoded = value
        .strip_prefix(PREFIX)
        .ok_or_else(|| anyhow!("unsupported reference version"))?;
    let bytes = URL_SAFE_NO_PAD.decode(encoded)?;
    ensure!(
        bytes.len() <= MAX_BYTES && URL_SAFE_NO_PAD.encode(&bytes) == encoded,
        "noncanonical reference"
    );
    let mut r = Reader(&bytes);
    ensure!(
        r.take(3)? == [0x53, 0x52, 1],
        "unsupported reference domain/version"
    );
    let resource = match r.byte()? {
        0 => {
            let uri = r.string(MAX_PATH)?;
            let resource: ResourceId = uri.parse()?;
            ensure!(resource.as_str() == uri, "noncanonical resource URI");
            resource
        }
        1 => path_resource(path.ok_or_else(|| anyhow!("reference omits URI; path is required"))?)?,
        _ => bail!("unsupported reference flags"),
    };
    use ResourceLocator::*;
    let locator = match r.byte()? {
        0 => Text {
            byte_start: r.var()?,
            byte_end: r.var()?,
            line_start: r.optional()?,
            line_end: r.optional()?,
        },
        1 => PageText {
            page: r.u32()?,
            byte_start: r.var()?,
            byte_end: r.var()?,
            text_version: r.string(4096)?,
            page_text_hash: r.hash()?,
        },
        2 => Page { page: r.u32()? },
        3 => Time {
            start_ms: r.var()?,
            end_ms: r.var()?,
        },
        4 => ImageRegion {
            x: r.u32()?,
            y: r.u32()?,
            width: r.u32()?,
            height: r.u32()?,
        },
        5 => WholeResource,
        _ => bail!("unsupported reference locator"),
    };
    let modality = match r.byte()? {
        0 => Modality::Document,
        1 => Modality::Text,
        2 => Modality::Image,
        3 => Modality::Audio,
        4 => Modality::Video,
        _ => bail!("unsupported reference modality"),
    };
    let size_bytes = r.var()?;
    let modified_ns = r.optional()?.map(|n| ((n >> 1) as i64) ^ -((n & 1) as i64));
    let media_type = match r.byte()? {
        0 => None,
        1 => Some("application/pdf".into()),
        2 => {
            let s = r.string(1024)?;
            ensure!(s != "application/pdf", "noncanonical media type");
            Some(s)
        }
        _ => bail!("unsupported reference media type"),
    };
    let content_hash = r.hash()?;
    ensure!(r.0.is_empty(), "trailing reference bytes");
    Ok(Selection {
        resource,
        locator,
        metadata: ResourceMetadata {
            media_type,
            modality,
            size_bytes,
            modified_ns,
            content_hash,
        },
    })
}

/// Semantic selection checks are deliberately separate from lossless decoding.
pub fn validate(s: &Selection) -> Result<()> {
    use ResourceLocator::*;
    match &s.locator {
        Text {
            byte_start,
            byte_end,
            line_start,
            line_end,
        } => {
            ensure!(byte_start <= byte_end, "reversed text range");
            ensure!(
                *line_start != Some(0) && *line_end != Some(0),
                "lines are 1-based"
            );
            if let (Some(a), Some(b)) = (line_start, line_end) {
                ensure!(a <= b, "reversed line range");
            }
        }
        PageText {
            page,
            byte_start,
            byte_end,
            ..
        } => {
            ensure!(
                *page > 0 && byte_start <= byte_end,
                "invalid page-text range"
            );
        }
        Page { page } => ensure!(*page > 0, "pages are 1-based"),
        Time { start_ms, end_ms } => ensure!(start_ms <= end_ms, "reversed time range"),
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
#[path = "mcp_ref_tests.rs"]
mod tests;
