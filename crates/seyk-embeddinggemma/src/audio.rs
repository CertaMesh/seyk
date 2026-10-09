//! Gemma 4 audio preprocessing and the optional pinned q8 encoder.
use super::*;
use rustfft::{FftPlanner, num_complex::Complex};

/// Version of the reference feature and token policy.
pub const FEATURE_POLICY: &str = "gemma4-audio-v1;16k;magnitude;hann-periodic320;hop160;fft512;mel128-htk;ln+0.001;pad-left160;frame321;audio-begin-end;no-prefix";
/// Only these q8 files are provisioned for audio (independent of text precision).
pub const FILES: [&str; 2] = [
    "onnx/audio_encoder_quantized.onnx",
    "onnx/audio_encoder_quantized.onnx_data",
];

pub(super) struct AudioEncoder {
    pub identity: String,
    dir: PathBuf,
    available: bool,
    session: Mutex<Option<Session>>,
}
impl AudioEncoder {
    pub fn new(dir: &Path) -> Self {
        use std::io::Read;
        let mut hash = blake3::Hasher::new();
        let mut available = true;
        hash.update(FEATURE_POLICY.as_bytes());
        for name in FILES {
            hash.update(name.as_bytes());
            if let Ok(mut file) = std::fs::File::open(dir.join(name)) {
                let mut buf = [0; 65536];
                loop {
                    match file.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            hash.update(&buf[..n]);
                        }
                        Err(_) => {
                            available = false;
                            hash.update(b"unreadable");
                            break;
                        }
                    }
                }
            } else {
                available = false;
                hash.update(b"missing");
            }
        }
        Self {
            identity: hash.finalize().to_hex().to_string(),
            dir: dir.into(),
            available,
            session: Mutex::new(None),
        }
    }
}

pub struct Features {
    pub values: Vec<f32>,
    pub mask: Vec<bool>,
}

/// Cached Gemma4AudioFeatureExtractor: 128 HTK mel filters, magnitude (not power),
/// periodic Hann, semicausal zero padding, 321-sample analysis window.
pub fn preprocess(samples: &[f32]) -> Result<Features> {
    if samples.len() < 161 || samples.len() > 160000 || samples.iter().any(|x| !x.is_finite()) {
        return Err(Error::Config(
            "audio requires 161..160000 finite mono samples".into(),
        ));
    }
    let padded = samples.len().div_ceil(128) * 128;
    let frames = (padded + 160).saturating_sub(321) / 160 + 1;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(512);
    let window: Vec<f32> = (0..320)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 320.0).cos()) as f32)
        .collect();
    let max_mel = 2595.0_f64 * (1.0_f64 + 8000.0 / 700.0).log10();
    let freqs: Vec<f64> = (0..130)
        .map(|i| 700.0 * (10.0_f64.powf(max_mel * i as f64 / 129.0 / 2595.0) - 1.0))
        .collect();
    let mut filters = vec![0.0_f64; 257 * 128];
    for k in 0..257 {
        for m in 0..128 {
            let hz = k as f64 * 8000.0 / 256.0;
            filters[k * 128 + m] = ((hz - freqs[m]) / (freqs[m + 1] - freqs[m]))
                .min((freqs[m + 2] - hz) / (freqs[m + 2] - freqs[m + 1]))
                .max(0.0);
        }
    }
    let mut values = Vec::with_capacity(frames * 128);
    let mut mask = Vec::with_capacity(frames);
    for frame in 0..frames {
        let valid = frame * 160 + 160 < samples.len();
        mask.push(valid);
        let mut buf = vec![Complex::new(0.0, 0.0); 512];
        for i in 0..320 {
            let pos = (frame * 160 + i) as isize - 160;
            if pos >= 0 && (pos as usize) < samples.len() {
                buf[i].re = (samples[pos as usize] * window[i]) as f64;
            }
        }
        fft.process(&mut buf);
        let mag: Vec<f64> = buf[..257].iter().map(|x| x.norm()).collect();
        for m in 0..128 {
            let mel: f64 = (0..257).map(|k| mag[k] * filters[k * 128 + m]).sum();
            values.push(if valid {
                (mel + 0.001).ln() as f32
            } else {
                0.0
            });
        }
    }
    Ok(Features { values, mask })
}

impl EmbeddingGemma2 {
    pub fn embed_audio(&self, samples: &[f32], sample_rate: u32) -> Result<Embedding> {
        if sample_rate != 16000 {
            return Err(Error::Config("audio requires 16000 Hz".into()));
        }
        let pre = preprocess(samples)?;
        self.embed_audio_features(pre)
    }
    fn embed_audio_features(&self, pre: Features) -> Result<Embedding> {
        let audio = self.audio.as_ref().ok_or(Error::UnsupportedModality {
            modality: Modality::Audio,
        })?;
        let mut mask = pre.mask.clone();
        for _ in 0..2 {
            mask = mask.iter().step_by(2).copied().collect();
        }
        let tokens = mask.iter().filter(|x| **x).count();
        let features = {
            let mut state = audio.session.lock().map_err(ort_err)?;
            if state.is_none() {
                if !audio.available {
                    return Err(Error::ModelNotFound("audio encoder is missing; run seyk model download --audio, then reload the model".into()));
                }
                // Lazy loading must use exactly the weights identified at model
                // construction, even when a local directory changes meanwhile.
                if AudioEncoder::new(&audio.dir).identity != audio.identity {
                    return Err(Error::Embedding(
                        "audio weights changed; reload the model before indexing".into(),
                    ));
                }
                for name in FILES {
                    if !audio.dir.join(name).is_file() {
                        return Err(Error::ModelNotFound(
                            "audio encoder is missing; run seyk model download --audio".into(),
                        ));
                    }
                }
                let dir = onnx_files_without_symlinks(
                    &audio.dir,
                    FILES.iter().map(|s| s.to_string()).collect(),
                    &audio.identity,
                )?;
                *state = Some(
                    session(
                        &dir.join("audio_encoder_quantized.onnx"),
                        self.config.threads,
                    )
                    .map_err(|_| Error::Embedding("cannot load audio encoder".into()))?,
                );
            }
            let s = state.as_mut().expect("initialized");
            let out = s.run(ort::inputs![
                "input_features" => Tensor::from_array(([1usize,pre.mask.len(),128],pre.values)).map_err(ort_err)?,
                "input_features_mask" => Tensor::from_array(([1usize,pre.mask.len()],pre.mask)).map_err(ort_err)?,
            ]).map_err(ort_err)?;
            let (shape, data) = out["audio_features"]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            if shape.as_ref() != [tokens as i64, 512] {
                return Err(Error::Embedding(format!(
                    "audio shape mismatch {shape:?} vs {tokens}"
                )));
            }
            data.to_vec()
        };
        let ids = self
            .token_ids(&format!("<|audio>{}<audio|>", "<|audio|>".repeat(tokens)))?
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        if ids.iter().filter(|x| **x == 258881).count() != tokens
            || !ids.contains(&256000)
            || !ids.contains(&258883)
        {
            return Err(Error::Embedding(
                "audio token layout truncated or invalid".into(),
            ));
        }
        Ok(self
            .run_text(&[&ids], None, Some((features, tokens)))?
            .remove(0))
    }
}
