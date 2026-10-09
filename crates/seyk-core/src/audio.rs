//! Bounded audio decoding, presentation timing and deterministic 16 kHz windows.
use crate::{Error, ResourceId, ResourceLocator, Result};
use std::{
    collections::VecDeque,
    io::{Cursor, Read, Seek, SeekFrom},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CodecParameters, Decoder, DecoderOptions},
    formats::{FormatOptions, FormatReader},
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};
mod aac;
mod alac;
mod mp4;
mod verified;
pub use verified::{VerificationStats, VerifiedReader};

pub const SAMPLE_RATE: usize = 16_000;
pub const WINDOW: usize = 10 * SAMPLE_RATE;
pub const STRIDE: usize = 8 * SAMPLE_RATE;
/// Default maximum canonical 16 kHz presentation samples (two hours).
pub const MAX_SAMPLES: u64 = 7200 * SAMPLE_RATE as u64;
/// PREPARATION's historical cap suffix is a frozen compatibility token, not
/// the effective acceptance limit. Limits must never change input hashes.
pub const PREPARATION: &str = "audio-prep-v1;symphonia=0.5.5;gapless=true;elst-v1-single-edit-mdhd-end-itunsmpb-fallback-else-decoded-timeline;mean-channels-f64;sinc64-blackman-guard0.94-v1;zero-edges;16k-f32;window160000-stride128000;tail-pad320-stop-at-first-short-v2;alac-frame-cap16384;cap16777216";

fn unsupported(id: &ResourceId, reason: &'static str) -> Error {
    Error::UnsupportedAudio {
        resource: id.clone(),
        reason,
    }
}

/// Contain upstream unwinding panics without exposing their payloads. As with
/// PDF parsing, applications must install a payload-free panic hook.
fn guarded<T>(id: &ResourceId, f: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err(unsupported(id, "audio decoder or parser panicked")))
}

pub fn media_type(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(b"RIFF") {
        "audio/wav"
    } else if bytes.starts_with(b"fLaC") {
        "audio/flac"
    } else if bytes.get(4..8) == Some(b"ftyp") {
        "audio/mp4"
    } else {
        "audio/mpeg"
    }
}

/// Calls `emit` with one owned window at a time. No decoded file is retained.
/// Endpoints describe the post-edit presentation timeline, with outward rounding.
pub fn windows(
    bytes: Vec<u8>,
    id: &ResourceId,
    emit: impl FnMut(ResourceLocator, Vec<f32>) -> Result<()>,
) -> Result<()> {
    let source = VerifiedReader::new(
        Box::new(Cursor::new(bytes)),
        id,
        crate::discovery::audio_byte_limit(7200)?,
    )?;
    windows_verified(source, id, 7200, emit)
}

/// Decode verified encoded bytes with bounded PCM storage and explicit duration.
pub fn windows_verified(
    source: VerifiedReader,
    id: &ResourceId,
    seconds: u64,
    mut emit: impl FnMut(ResourceLocator, Vec<f32>) -> Result<()>,
) -> Result<()> {
    let changed = source.changed_flag();
    let result = windows_inner(source, id, seconds, &mut emit);
    if changed.load(Ordering::Relaxed) {
        return Err(Error::ChangedWhileReading {
            resource: id.clone(),
        });
    }
    result
}
fn windows_inner(
    source: VerifiedReader,
    id: &ResourceId,
    seconds: u64,
    emit: &mut impl FnMut(ResourceLocator, Vec<f32>) -> Result<()>,
) -> Result<()> {
    let mut reader = Reader::new(source, id, seconds)?;
    let mut resampler = Resampler::new(reader.rate);
    let mut buf = VecDeque::with_capacity(WINDOW);
    let mut start = 0u64;
    loop {
        while buf.len() < WINDOW {
            match resampler.next(&mut reader)? {
                Some(sample) => buf.push_back(sample),
                None => break,
            }
        }
        if buf.is_empty() {
            break;
        }
        let real_len = buf.len();
        let end = start + real_len as u64;
        let locator = ResourceLocator::Time {
            start_ms: start * 1000 / SAMPLE_RATE as u64,
            end_ms: (end * 1000).div_ceil(SAMPLE_RATE as u64),
        };
        if let ResourceLocator::Time { start_ms, end_ms } = locator
            && start_ms >= end_ms
        {
            return Err(Error::InvalidLocator {
                resource: id.clone(),
                locator,
                reason: "empty audio interval",
            });
        }
        let mut samples: Vec<f32> = buf.iter().copied().collect();
        // The feature extractor needs at least one frame. Padding is model input
        // only: a short tail's locator still ends at its real final sample.
        samples.resize(real_len.max(320), 0.0);
        emit(locator, samples)?;
        if real_len < WINDOW {
            break;
        }
        buf.drain(..STRIDE);
        start += STRIDE as u64;
    }
    if start == 0 && resampler.output == 0 {
        return Err(unsupported(id, "empty audio"));
    }
    Ok(())
}

struct Reader {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn Decoder>,
    id: ResourceId,
    track: u32,
    rate: u32,
    channels: usize,
    packet: VecDeque<f32>,
    decoded: u64,
    emitted: u64,
    skip: u64,
    length: Option<u64>,
    max_samples: u64,
    changed: Arc<AtomicBool>,
}
impl Reader {
    fn new(mut source: VerifiedReader, id: &ResourceId, seconds: u64) -> Result<Self> {
        crate::discovery::audio_byte_limit(seconds)?;
        let max_samples = seconds
            .checked_mul(SAMPLE_RATE as u64)
            .ok_or_else(|| unsupported(id, "audio duration overflow"))?;
        let changed = source.changed_flag();
        let mut header = [0; 12];
        let n = source
            .read(&mut header)
            .map_err(|_| unsupported(id, "audio source read failed"))?;
        let reliable_length = matches!(media_type(&header[..n]), "audio/wav" | "audio/flac");
        let timing = mp4::timing(&mut source).map_err(|reason| unsupported(id, reason))?;
        source
            .seek(SeekFrom::Start(0))
            .map_err(|_| unsupported(id, "audio source seek failed"))?;
        let stream = MediaSourceStream::new(Box::new(source), Default::default());
        let mut format = guarded(id, || {
            symphonia::default::get_probe()
                .format(
                    &Hint::new(),
                    stream,
                    &FormatOptions {
                        enable_gapless: true,
                        ..Default::default()
                    },
                    &MetadataOptions::default(),
                )
                .map_err(|_| unsupported(id, "unrecognized or corrupt audio container"))
        })?
        .format;
        if format.tracks().len() != 1 {
            return Err(unsupported(id, "multiple audio tracks are unsupported"));
        }
        let track = format
            .default_track()
            .ok_or_else(|| unsupported(id, "no audio track"))?;
        let params: &CodecParameters = &track.codec_params;
        if params.codec == symphonia::core::codecs::CODEC_TYPE_AAC
            && !params.extra_data.as_deref().is_some_and(aac::lc_config)
        {
            return Err(unsupported(
                id,
                "only AAC-LC is supported; HE-AAC/SBR and unknown AAC profiles are unsupported",
            ));
        }
        let rate = params
            .sample_rate
            .ok_or_else(|| unsupported(id, "missing sample rate"))?;
        let channels = params.channels.map_or(0, |c| c.count());
        if !(8_000..=192_000).contains(&rate) || channels > 8 {
            return Err(unsupported(id, "unsupported sample rate or channel count"));
        }
        let (skip, length) = timing
            .samples(rate)
            .map_err(|reason| unsupported(id, reason))?;
        let length = length.or(if reliable_length {
            params.n_frames
        } else {
            None
        });
        if skip > rate as u64 * 30 {
            return Err(unsupported(
                id,
                "audio priming/edit skip exceeds 30 second budget",
            ));
        }
        if length.is_some_and(|len| {
            len as u128 * SAMPLE_RATE as u128 / rate as u128 > max_samples as u128
        }) {
            return Err(unsupported(
                id,
                "decoded audio exceeds configured duration limit",
            ));
        }
        if params.codec == symphonia::core::codecs::CODEC_TYPE_ALAC {
            alac::validate_cookie(params.extra_data.as_deref().unwrap_or_default())
                .map_err(|reason| unsupported(id, reason))?;
        }
        let decoder = guarded(id, || {
            symphonia::default::get_codecs()
                .make(params, &DecoderOptions::default())
                .map_err(|_| {
                    unsupported(id, "unsupported audio codec (HE-AAC/SBR is not supported)")
                })
        })?;
        let track = track.id;
        if length.is_none() {
            // Unknown-length media must be rejected before the first embedding.
            // Packet durations use this track's time base, including gapless trim.
            let time_base = params
                .time_base
                .ok_or_else(|| unsupported(id, "missing packet time base"))?;
            if time_base.numer == 0 || time_base.denom == 0 {
                return Err(unsupported(id, "invalid packet time base"));
            }
            let mut ticks = 0u64;
            loop {
                let packet = match guarded(id, || Ok(format.next_packet()))? {
                    Ok(p) => p,
                    Err(symphonia::core::errors::Error::IoError(e))
                        if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                    {
                        break;
                    }
                    Err(_) => {
                        return Err(unsupported(id, "corrupt audio packet during duration scan"));
                    }
                };
                if packet.track_id() != track || packet.data.len() > 1024 * 1024 {
                    return Err(unsupported(id, "invalid audio duration scan packet"));
                }
                ticks = ticks
                    .checked_add(packet.dur)
                    .ok_or_else(|| unsupported(id, "audio duration overflow"))?;
                let samples = ticks as u128 * time_base.numer as u128 * SAMPLE_RATE as u128
                    / time_base.denom as u128;
                // Account for manual MP4 presentation skip, never more than 30s.
                let skip_samples = skip as u128 * SAMPLE_RATE as u128 / rate as u128;
                if samples.saturating_sub(skip_samples) > max_samples as u128 {
                    return Err(unsupported(
                        id,
                        "decoded audio exceeds configured duration limit",
                    ));
                }
            }
            let mut stream = format.into_inner();
            stream
                .seek(SeekFrom::Start(0))
                .map_err(|_| unsupported(id, "audio rewind failed"))?;
            format = guarded(id, || {
                symphonia::default::get_probe()
                    .format(
                        &Hint::new(),
                        stream,
                        &FormatOptions {
                            enable_gapless: true,
                            ..Default::default()
                        },
                        &MetadataOptions::default(),
                    )
                    .map(|p| p.format)
                    .map_err(|_| unsupported(id, "audio reprobe failed"))
            })?;
        }
        Ok(Self {
            format,
            decoder,
            id: id.clone(),
            track,
            rate,
            channels,
            packet: VecDeque::new(),
            decoded: 0,
            emitted: 0,
            skip,
            length,
            max_samples,
            changed,
        })
    }
    fn next(&mut self) -> Result<Option<f32>> {
        loop {
            if self.changed.load(Ordering::Relaxed) {
                return Err(Error::ChangedWhileReading {
                    resource: self.id.clone(),
                });
            }
            if self.length.is_some_and(|len| self.emitted >= len) {
                return Ok(None);
            }
            if let Some(sample) = self.packet.pop_front() {
                self.decoded += 1;
                if self.decoded <= self.skip {
                    continue;
                }
                self.emitted += 1;
                if self.emitted as u128 * SAMPLE_RATE as u128 / self.rate as u128
                    > self.max_samples as u128
                {
                    return Err(unsupported(
                        &self.id,
                        "decoded audio exceeds configured duration limit",
                    ));
                }
                return Ok(Some(sample));
            }
            let packet = match guarded(&self.id, || Ok(self.format.next_packet()))? {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    if self.length.is_some_and(|len| self.emitted + 1 < len) {
                        return Err(unsupported(&self.id, "truncated audio presentation"));
                    }
                    return Ok(None);
                }
                Err(_) => return Err(unsupported(&self.id, "corrupt audio packet")),
            };
            if packet.track_id() != self.track {
                return Err(unsupported(&self.id, "unexpected audio track"));
            }
            // Reject pathological packets before the decoder allocates its output.
            if packet.data.len() > 1024 * 1024 || packet.dur > self.rate as u64 * 2 {
                return Err(unsupported(
                    &self.id,
                    "audio packet exceeds bounded decoder budget",
                ));
            }
            let decoded = guarded(&self.id, || {
                self.decoder.decode(&packet).map_err(|_| {
                    unsupported(
                        &self.id,
                        "unsupported or corrupt audio frame (HE-AAC/SBR is not supported)",
                    )
                })
            })?;
            let channels = decoded.spec().channels.count();
            if self.channels == 0 {
                self.channels = channels;
            }
            if !(1..=8).contains(&channels)
                || decoded.spec().rate != self.rate
                || channels != self.channels
            {
                return Err(unsupported(
                    &self.id,
                    "audio format changed during decode (HE-AAC/SBR is not supported)",
                ));
            }
            if decoded.capacity() > 384_000 {
                return Err(unsupported(&self.id, "decoded packet exceeds budget"));
            }
            let mut samples = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
            samples.copy_interleaved_ref(decoded);
            for frame in samples.samples().chunks_exact(self.channels) {
                let mean =
                    (frame.iter().map(|x| *x as f64).sum::<f64>() / self.channels as f64) as f32;
                if !mean.is_finite() {
                    return Err(unsupported(&self.id, "non-finite audio samples"));
                }
                self.packet.push_back(mean);
            }
        }
    }
}

struct Resampler {
    rate: u32,
    samples: VecDeque<f32>,
    base: u64,
    read: u64,
    output: u64,
    eof: bool,
}
impl Resampler {
    fn new(rate: u32) -> Self {
        Self {
            rate,
            samples: VecDeque::new(),
            base: 0,
            read: 0,
            output: 0,
            eof: false,
        }
    }
    fn next(&mut self, reader: &mut Reader) -> Result<Option<f32>> {
        if self.rate == SAMPLE_RATE as u32 {
            let sample = reader.next()?;
            self.output += u64::from(sample.is_some());
            return Ok(sample);
        }
        let pos = self.output as f64 * self.rate as f64 / SAMPLE_RATE as f64;
        let center = pos.floor() as i64;
        while !self.eof && self.read <= (center + 32) as u64 {
            match reader.next()? {
                Some(sample) => {
                    self.samples.push_back(sample);
                    self.read += 1;
                }
                None => self.eof = true,
            }
        }
        if self.eof && self.output >= self.read * SAMPLE_RATE as u64 / self.rate as u64 {
            return Ok(None);
        }
        let cutoff = (SAMPLE_RATE as f64 / self.rate as f64).min(1.0) * 0.94;
        let mut value = 0.0;
        let mut norm = 0.0;
        for k in center - 31..=center + 32 {
            let x = k as f64 - pos;
            let t = x / 32.0;
            if t.abs() > 1.0 {
                continue;
            }
            let a = std::f64::consts::PI * x * cutoff;
            let sinc = if a.abs() < 1e-12 { 1.0 } else { a.sin() / a };
            let win = 0.42
                + 0.5 * (std::f64::consts::PI * t).cos()
                + 0.08 * (2.0 * std::f64::consts::PI * t).cos();
            let w = cutoff * sinc * win;
            norm += w;
            if k >= self.base as i64 && (k as u64) < self.read {
                value += self.samples[(k as u64 - self.base) as usize] as f64 * w;
            }
        }
        self.output += 1;
        let retain = (center - 31).max(0) as u64;
        while self.base < retain {
            self.samples.pop_front();
            self.base += 1;
        }
        Ok(Some((value / norm) as f32))
    }
}

/// Hash-verify a source and return only its chosen canonical interval. No STT.
pub fn extract_samples(
    provider: &dyn crate::ResourceProvider,
    id: &ResourceId,
    locator: &ResourceLocator,
    metadata: &crate::ResourceMetadata,
    seconds: u64,
) -> Result<Vec<f32>> {
    if metadata.modality != crate::Modality::Audio
        || !matches!(locator, ResourceLocator::Time { .. })
    {
        return Err(Error::UnsupportedExtraction {
            resource: id.clone(),
            modality: metadata.modality,
            locator: locator.clone(),
        });
    }
    let source = VerifiedReader::new(
        provider.open(id)?,
        id,
        crate::discovery::audio_byte_limit(seconds)?,
    )?;
    if source.content_hash() != metadata.content_hash {
        return Err(Error::StaleResource {
            resource: id.clone(),
            expected: metadata.content_hash.clone(),
            actual: source.content_hash().into(),
        });
    }
    slice_verified(source, id, locator, seconds)
}

/// Sequential slicing preserves the retrieval resampler's absolute phase.
pub fn slice_verified(
    source: VerifiedReader,
    id: &ResourceId,
    locator: &ResourceLocator,
    seconds: u64,
) -> Result<Vec<f32>> {
    let invalid = |reason| Error::InvalidLocator {
        resource: id.clone(),
        locator: locator.clone(),
        reason,
    };
    let ResourceLocator::Time { start_ms, end_ms } = *locator else {
        return Err(invalid("expected audio time interval"));
    };
    if start_ms >= end_ms || end_ms - start_ms > 30_000 {
        return Err(invalid(
            "audio slice must be nonempty and at most 30 seconds",
        ));
    }
    let start = start_ms
        .checked_mul(16)
        .ok_or_else(|| invalid("audio time overflow"))?;
    let end = end_ms
        .checked_mul(16)
        .ok_or_else(|| invalid("audio time overflow"))?;
    let changed = source.changed_flag();
    let result = (|| {
        let mut reader = Reader::new(source, id, seconds)?;
        let known = reader
            .length
            .map(|n| (n as u128 * SAMPLE_RATE as u128 / reader.rate as u128) as u64);
        let mut resampler = Resampler::new(reader.rate);
        let mut samples = Vec::new();
        while let Some(sample) = resampler.next(&mut reader)? {
            let at = resampler.output - 1;
            if at >= start && at < end {
                samples.push(sample);
            }
            // Unknown lengths are fully validated before returning the slice.
            if known.is_some() && resampler.output >= end {
                break;
            }
        }
        let len = known.unwrap_or(resampler.output);
        if start >= len || end_ms > len.div_ceil(16) {
            return Err(invalid("audio interval exceeds presentation duration"));
        }
        Ok(samples)
    })();
    if changed.load(Ordering::Relaxed) {
        return Err(Error::ChangedWhileReading {
            resource: id.clone(),
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dependency_panic_becomes_payload_free_audio_error() {
        let id: ResourceId = "mem://audio/panic".parse().unwrap();
        let result = guarded::<()>(&id, || panic!("private decoder payload"));
        assert!(
            matches!(result, Err(Error::UnsupportedAudio { resource, reason })
            if resource == id && reason == "audio decoder or parser panicked")
        );
        assert_eq!(guarded(&id, || Ok(42)).unwrap(), 42);
    }
}
