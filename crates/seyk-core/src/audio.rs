//! Bounded audio decoding, presentation timing and deterministic 16 kHz windows.
use crate::{Error, ResourceId, ResourceLocator, Result};
use std::{collections::VecDeque, io::Cursor};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CodecParameters, Decoder, DecoderOptions},
    formats::{FormatOptions, FormatReader},
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};
mod aac;
mod mp4;

pub const SAMPLE_RATE: usize = 16_000;
pub const WINDOW: usize = 10 * SAMPLE_RATE;
pub const STRIDE: usize = 8 * SAMPLE_RATE;
/// The existing 32 MiB encoded limit, expressed as mono 16-bit PCM at 16 kHz.
/// Also caps decoding work for highly compressed or malicious inputs: 1048.576 s.
pub const MAX_SAMPLES: u64 = 32 * 1024 * 1024 / 2;
pub const PREPARATION: &str = "audio-prep-v1;symphonia=0.5.5;gapless=true;elst-v1-single-edit-mdhd-end-itunsmpb-fallback-else-decoded-timeline;mean-channels-f64;sinc64-blackman-guard0.94-v1;zero-edges;16k-f32;window160000-stride128000;tail-pad320;cap16777216";

fn unsupported(id: &ResourceId, reason: &'static str) -> Error {
    Error::UnsupportedAudio {
        resource: id.clone(),
        reason,
    }
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
    mut emit: impl FnMut(ResourceLocator, Vec<f32>) -> Result<()>,
) -> Result<()> {
    let mut reader = Reader::new(bytes, id)?;
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
        if real_len <= STRIDE {
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
}
impl Reader {
    fn new(bytes: Vec<u8>, id: &ResourceId) -> Result<Self> {
        if bytes.len() > 32 * 1024 * 1024 {
            return Err(unsupported(id, "audio exceeds 32 MiB input limit"));
        }
        let reliable_length = matches!(media_type(&bytes), "audio/wav" | "audio/flac");
        let timing = mp4::timing(&bytes).map_err(|reason| unsupported(id, reason))?;
        let stream = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
        let format = symphonia::default::get_probe()
            .format(
                &Hint::new(),
                stream,
                &FormatOptions {
                    enable_gapless: true,
                    ..Default::default()
                },
                &MetadataOptions::default(),
            )
            .map_err(|_| unsupported(id, "unrecognized or corrupt audio container"))?
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
        if length.is_some_and(|len| {
            (len as u128 + skip as u128) * SAMPLE_RATE as u128 > MAX_SAMPLES as u128 * rate as u128
        }) {
            return Err(unsupported(
                id,
                "decoded audio exceeds 1048.576 second duration cap",
            ));
        }
        let decoder = symphonia::default::get_codecs()
            .make(params, &DecoderOptions::default())
            .map_err(|_| {
                unsupported(id, "unsupported audio codec (HE-AAC/SBR is not supported)")
            })?;
        let track = track.id;
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
        })
    }
    fn next(&mut self) -> Result<Option<f32>> {
        loop {
            if self.length.is_some_and(|len| self.emitted >= len) {
                return Ok(None);
            }
            if let Some(sample) = self.packet.pop_front() {
                self.decoded += 1;
                if self.decoded * SAMPLE_RATE as u64 > MAX_SAMPLES * self.rate as u64 {
                    return Err(unsupported(
                        &self.id,
                        "decoded audio exceeds 1048.576 second duration cap",
                    ));
                }
                if self.decoded <= self.skip {
                    continue;
                }
                self.emitted += 1;
                return Ok(Some(sample));
            }
            let packet = match self.format.next_packet() {
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
            let decoded = self.decoder.decode(&packet).map_err(|_| {
                unsupported(
                    &self.id,
                    "unsupported or corrupt audio frame (HE-AAC/SBR is not supported)",
                )
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
