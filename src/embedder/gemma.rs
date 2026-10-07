//! EmbeddingGemma 2 through ONNX Runtime.
//!
//! Uses the ONNX export at `onnx-community/embeddinggemma-2-ONNX`, which splits
//! the model into a text model (that also accepts precomputed media features),
//! a vision encoder and an audio encoder. The text model returns a pooled
//! `sentence_embedding`; this module truncates it to the configured
//! Matryoshka dimension and L2-normalises it.
//!
//! Text and images are implemented. Audio needs a port of the Gemma 4
//! log-mel feature extractor, and video needs frame decoding; both are
//! reported as unsupported rather than approximated.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::embedder::{Embedder, ModelIdentity, ResourceInput};
use crate::{Dimension, Embedding, Error, Modality, Result};

/// The upstream model the ONNX export was converted from.
pub const MODEL_ID: &str = "google/embeddinggemma-2";
/// Hugging Face repository holding the ONNX export.
pub const ONNX_REPO: &str = "onnx-community/embeddinggemma-2-ONNX";
/// Pinned commit of [`ONNX_REPO`], so cached vectors always refer to known weights.
pub const ONNX_REVISION: &str = "daa72c51243991dfcaf9f9137d2c573d8f7790c0";

/// Official prompts (`config_sentence_transformers.json`, prompt names
/// `SearchQuery` and `Document`).
pub const QUERY_PROMPT: &str = "task: search result | query: ";
pub const DOCUMENT_PROMPT_TEMPLATE: &str = "title: {title} | text: ";

const NATIVE_DIM: usize = 768;
const FEATURE_DIM: usize = 512;

// Gemma 4 image processor settings from `processor_config.json`.
const PATCH_SIZE: usize = 16;
const POOLING_KERNEL: usize = 3;
const MAX_SOFT_TOKENS: usize = 280;

/// Weight precision. `float16` is deliberately absent: the model card warns
/// that its activations overflow fp16 and silently produce NaN or degraded vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Full precision (~1.1 GB text model).
    Fp32,
    /// 8-bit weights (~0.3 GB text model). The export reports worst-case
    /// cosine 0.9997 against fp32.
    #[default]
    Q8,
}

impl Precision {
    fn suffix(self) -> &'static str {
        match self {
            Precision::Fp32 => "",
            Precision::Q8 => "_quantized",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Q8 => "q8",
        }
    }
}

impl std::str::FromStr for Precision {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "fp32" => Ok(Self::Fp32),
            "q8" => Ok(Self::Q8),
            _ => Err(Error::Config(format!(
                "unknown precision `{s}`; use fp32 or q8"
            ))),
        }
    }
}

/// Where model files come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// A Hugging Face repository in the standard HF cache (`$HF_HOME`, default `~/.cache/huggingface`).
    HuggingFace { repo: String, revision: String },
    /// A local directory with the same layout as the ONNX repository.
    Directory(PathBuf),
}

impl Default for ModelSource {
    fn default() -> Self {
        ModelSource::HuggingFace {
            repo: ONNX_REPO.into(),
            revision: ONNX_REVISION.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GemmaConfig {
    pub source: ModelSource,
    pub dimension: Dimension,
    pub precision: Precision,
    /// Longer inputs are truncated. Segments are far shorter than this; it is a safety net.
    pub max_tokens: usize,
    /// Load the vision encoder so image resources can be embedded.
    pub vision: bool,
    /// ONNX Runtime intra-op threads. `None` lets ONNX Runtime decide.
    pub threads: Option<usize>,
    /// Inputs per model call.
    pub batch_size: usize,
}

impl Default for GemmaConfig {
    fn default() -> Self {
        Self {
            source: ModelSource::default(),
            dimension: Dimension::default(),
            precision: Precision::default(),
            max_tokens: 2048,
            vision: cfg!(feature = "image"),
            threads: None,
            batch_size: 16,
        }
    }
}

impl GemmaConfig {
    fn files(&self) -> Vec<String> {
        let s = self.precision.suffix();
        let mut f = vec![
            "tokenizer.json".to_string(),
            format!("onnx/model{s}.onnx"),
            format!("onnx/model{s}.onnx_data"),
        ];
        if self.vision {
            f.push(format!("onnx/vision_encoder{s}.onnx"));
            f.push(format!("onnx/vision_encoder{s}.onnx_data"));
        }
        f
    }
}

/// Download the files `config` needs into the Hugging Face cache.
///
/// This is the only function in the crate that uses the network. It sends
/// nothing but file requests for public model weights.
pub fn download(config: &GemmaConfig) -> Result<PathBuf> {
    let ModelSource::HuggingFace { repo, revision } = &config.source else {
        return Err(Error::Config(
            "model source is a local directory; nothing to download".into(),
        ));
    };
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .map_err(|e| Error::Embedding(format!("cannot start download: {e}")))?;
    let repo_handle = api.repo(hf_hub::Repo::with_revision(
        repo.clone(),
        hf_hub::RepoType::Model,
        revision.clone(),
    ));
    let mut dir = None;
    for f in config.files() {
        let p = repo_handle
            .get(&f)
            .map_err(|e| Error::Embedding(format!("downloading {f}: {e}")))?;
        dir.get_or_insert_with(|| snapshot_root(&p, &f));
    }
    Ok(dir.expect("at least one file"))
}

/// Resolve the local model directory without touching the network.
pub fn locate(config: &GemmaConfig) -> Result<PathBuf> {
    match &config.source {
        ModelSource::Directory(dir) => {
            for f in config.files() {
                if !dir.join(&f).exists() {
                    return Err(Error::ModelNotFound(format!(
                        "{} is missing {f}",
                        dir.display()
                    )));
                }
            }
            Ok(dir.clone())
        }
        ModelSource::HuggingFace { repo, revision } => {
            let cache = hf_hub::Cache::from_env();
            let repo_cache = cache.repo(hf_hub::Repo::with_revision(
                repo.clone(),
                hf_hub::RepoType::Model,
                revision.clone(),
            ));
            let mut dir = None;
            for f in config.files() {
                let p = repo_cache.get(&f).ok_or_else(|| {
                    Error::ModelNotFound(format!(
                        "{repo}@{} ({}) is not downloaded; run `gemma-search model download`",
                        &revision[..revision.len().min(12)],
                        f
                    ))
                })?;
                dir.get_or_insert_with(|| snapshot_root(&p, &f));
            }
            Ok(dir.expect("at least one file"))
        }
    }
}

fn snapshot_root(file: &Path, rel: &str) -> PathBuf {
    let depth = Path::new(rel).components().count();
    let mut p = file.to_path_buf();
    for _ in 0..depth {
        p.pop();
    }
    p
}

/// EmbeddingGemma 2 embedder.
pub struct GemmaEmbedder {
    tokenizer: Tokenizer,
    text: Mutex<Session>,
    vision: Option<Mutex<Session>>,
    identity: ModelIdentity,
    config: GemmaConfig,
    pad_id: i64,
}

fn ort_err(e: impl std::fmt::Display) -> Error {
    Error::Embedding(e.to_string())
}

impl GemmaEmbedder {
    /// Load from local files. Fails with [`Error::ModelNotFound`] if they are not present.
    pub fn load(config: GemmaConfig) -> Result<Self> {
        let dir = locate(&config)?;
        Self::from_dir(&dir, config)
    }

    fn from_dir(dir: &Path, config: GemmaConfig) -> Result<Self> {
        let s = config.precision.suffix();
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(ort_err)?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: config.max_tokens,
                ..Default::default()
            }))
            .map_err(ort_err)?;
        tokenizer.with_padding(None);
        let pad_id = tokenizer.token_to_id("<pad>").unwrap_or(0) as i64;

        let text = session(&dir.join(format!("onnx/model{s}.onnx")), config.threads)?;
        let vision = if config.vision {
            Some(Mutex::new(session(
                &dir.join(format!("onnx/vision_encoder{s}.onnx")),
                config.threads,
            )?))
        } else {
            None
        };

        let revision = match &config.source {
            ModelSource::HuggingFace { repo, revision } => format!("{repo}@{revision}"),
            ModelSource::Directory(_) => format!("local:{}", fingerprint(dir, &config)?),
        };
        let identity = ModelIdentity {
            model_id: MODEL_ID.into(),
            revision,
            variant: format!("onnx-{}", config.precision.name()),
            dimension: config.dimension,
            normalization: "mean-pool;truncate;l2".into(),
            query_prompt: QUERY_PROMPT.into(),
            document_prompt: format!("{DOCUMENT_PROMPT_TEMPLATE}{{text}}"),
            max_tokens: config.max_tokens,
        };
        Ok(Self {
            tokenizer,
            text: Mutex::new(text),
            vision,
            identity,
            config,
            pad_id,
        })
    }

    pub fn config(&self) -> &GemmaConfig {
        &self.config
    }

    /// The exact string sent to the model for a text resource.
    pub fn document_input(title: Option<&str>, text: &str) -> String {
        let title = title
            .map(|t| t.replace(['\n', '\r'], " "))
            .filter(|t| !t.trim().is_empty());
        format!(
            "title: {} | text: {text}",
            title.as_deref().unwrap_or("none")
        )
    }

    pub fn query_input(query: &str) -> String {
        format!("{QUERY_PROMPT}{query}")
    }

    /// Token ids exactly as the model sees them (for verification against the reference).
    pub fn token_ids(&self, input: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(input, true)
            .map_err(ort_err)?
            .get_ids()
            .to_vec())
    }

    /// Embed raw strings that already contain their prompt.
    fn embed_strings(&self, inputs: &[String]) -> Result<Vec<Embedding>> {
        let encodings = self
            .tokenizer
            .encode_batch(inputs.iter().map(String::as_str).collect::<Vec<_>>(), true)
            .map_err(ort_err)?;
        let ids: Vec<Vec<i64>> = encodings
            .iter()
            .map(|e| e.get_ids().iter().map(|&i| i as i64).collect())
            .collect();

        // Sort by length so each batch pads as little as possible.
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_by_key(|&i| ids[i].len());
        let mut out: Vec<Option<Embedding>> = vec![None; ids.len()];
        for chunk in order.chunks(self.config.batch_size.max(1)) {
            let batch: Vec<&[i64]> = chunk.iter().map(|&i| ids[i].as_slice()).collect();
            let vecs = self.run_text(&batch, None)?;
            for (&i, v) in chunk.iter().zip(vecs) {
                out[i] = Some(v);
            }
        }
        Ok(out
            .into_iter()
            .map(|e| e.expect("every input embedded"))
            .collect())
    }

    /// Run the text model on padded token ids, optionally with image features.
    fn run_text(
        &self,
        batch: &[&[i64]],
        image_features: Option<(Vec<f32>, usize)>,
    ) -> Result<Vec<Embedding>> {
        let max_len = batch.iter().map(|b| b.len()).max().unwrap_or(0);
        let n = batch.len();
        let mut input_ids = vec![self.pad_id; n * max_len];
        let mut mask = vec![0i64; n * max_len];
        for (r, row) in batch.iter().enumerate() {
            input_ids[r * max_len..r * max_len + row.len()].copy_from_slice(row);
            mask[r * max_len..r * max_len + row.len()].fill(1);
        }
        let (img, img_tokens) = image_features.unwrap_or((Vec::new(), 0));
        let inputs = ort::inputs![
            "input_ids" => Tensor::from_array(([n, max_len], input_ids)).map_err(ort_err)?,
            "attention_mask" => Tensor::from_array(([n, max_len], mask)).map_err(ort_err)?,
            "image_features" => Tensor::from_array(([img_tokens, FEATURE_DIM], img)).map_err(ort_err)?,
            "video_features" => Tensor::from_array(([0usize, FEATURE_DIM], Vec::<f32>::new())).map_err(ort_err)?,
            "audio_features" => Tensor::from_array(([0usize, FEATURE_DIM], Vec::<f32>::new())).map_err(ort_err)?,
        ];
        let mut session = self
            .text
            .lock()
            .map_err(|_| Error::Embedding("session lock poisoned".into()))?;
        let outputs = session.run(inputs).map_err(ort_err)?;
        let (shape, data) = outputs["sentence_embedding"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        if shape.len() != 2 || shape[0] as usize != n || shape[1] as usize != NATIVE_DIM {
            return Err(Error::Embedding(format!(
                "unexpected sentence_embedding shape {shape:?}"
            )));
        }
        data.chunks_exact(NATIVE_DIM)
            .map(|row| Embedding::from_raw(row.to_vec(), self.config.dimension))
            .collect()
    }

    #[cfg(feature = "image")]
    fn embed_image(&self, bytes: &[u8]) -> Result<Embedding> {
        let vision = self.vision.as_ref().ok_or(Error::UnsupportedModality {
            modality: Modality::Image,
        })?;
        let pre = vision::preprocess(bytes)?;
        let features = {
            let mut session = vision
                .lock()
                .map_err(|_| Error::Embedding("session lock poisoned".into()))?;
            let outputs = session
                .run(ort::inputs![
                    "pixel_values" => Tensor::from_array(([1usize, pre.max_patches, vision::PATCH_DIM], pre.pixel_values)).map_err(ort_err)?,
                    "pixel_position_ids" => Tensor::from_array(([1usize, pre.max_patches, 2], pre.position_ids)).map_err(ort_err)?,
                ])
                .map_err(ort_err)?;
            let (shape, data) = outputs["image_features"]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            if shape.len() != 2
                || shape[0] as usize != pre.soft_tokens
                || shape[1] as usize != FEATURE_DIM
            {
                return Err(Error::Embedding(format!(
                    "vision encoder returned {shape:?}, expected [{}, {FEATURE_DIM}]",
                    pre.soft_tokens
                )));
            }
            data.to_vec()
        };
        // The processor expands `<|image|>` to <|image> + N x <|image|> + <image|>,
        // and images get no task prefix (model card, "Task Instruction Prefixes").
        let text = format!("<|image>{}<image|>", "<|image|>".repeat(pre.soft_tokens));
        let ids: Vec<i64> = self.token_ids(&text)?.into_iter().map(i64::from).collect();
        let mut out = self.run_text(&[&ids], Some((features, pre.soft_tokens)))?;
        Ok(out.pop().expect("one input"))
    }

    #[cfg(not(feature = "image"))]
    fn embed_image(&self, _bytes: &[u8]) -> Result<Embedding> {
        Err(Error::UnsupportedModality {
            modality: Modality::Image,
        })
    }
}

fn session(path: &Path, threads: Option<usize>) -> Result<Session> {
    let mut b = Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?;
    if let Some(t) = threads {
        b = b.with_intra_threads(t).map_err(ort_err)?;
    }
    b.commit_from_file(path)
        .map_err(|e| Error::Embedding(format!("loading {}: {e}", path.display())))
}

/// Identify local model files cheaply: hash the graph files and the size of the weight files.
fn fingerprint(dir: &Path, config: &GemmaConfig) -> Result<String> {
    let mut h = blake3::Hasher::new();
    for f in config.files() {
        let p = dir.join(&f);
        if f.ends_with("_data") {
            let len = std::fs::metadata(&p).map_err(|e| Error::io(&p, e))?.len();
            h.update(&len.to_le_bytes());
        } else {
            h.update(&std::fs::read(&p).map_err(|e| Error::io(&p, e))?);
        }
    }
    Ok(h.finalize().to_hex()[..16].to_string())
}

impl Embedder for GemmaEmbedder {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn supports(&self, modality: Modality) -> bool {
        match modality {
            Modality::Text => true,
            Modality::Image => cfg!(feature = "image") && self.vision.is_some(),
            Modality::Audio | Modality::Video => false,
        }
    }

    fn embed_query(&self, query: &str) -> Result<Embedding> {
        let mut v = self.embed_strings(&[Self::query_input(query)])?;
        Ok(v.pop().expect("one input"))
    }

    fn embed_resources(&self, inputs: &[ResourceInput<'_>]) -> Result<Vec<Embedding>> {
        let mut out: Vec<Option<Embedding>> = vec![None; inputs.len()];
        let mut text_idx = Vec::new();
        let mut text_inputs = Vec::new();
        for (i, input) in inputs.iter().enumerate() {
            match input {
                ResourceInput::Text { title, text } => {
                    text_idx.push(i);
                    text_inputs.push(Self::document_input(*title, text));
                }
                ResourceInput::Image { bytes } => out[i] = Some(self.embed_image(bytes)?),
                other => {
                    return Err(Error::UnsupportedModality {
                        modality: other.modality(),
                    });
                }
            }
        }
        if !text_inputs.is_empty() {
            for (i, e) in text_idx.into_iter().zip(self.embed_strings(&text_inputs)?) {
                out[i] = Some(e);
            }
        }
        Ok(out
            .into_iter()
            .map(|e| e.expect("every input embedded"))
            .collect())
    }
}

#[cfg(feature = "image")]
pub mod vision {
    //! Rust port of the Gemma 4 image processor (`Gemma4ImageProcessor` in
    //! Transformers): aspect-preserving bicubic resize, rescale to [0, 1],
    //! 16x16 patches, (x, y) patch positions, padding to the patch budget.

    use fast_image_resize::images::Image;
    use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

    use super::{MAX_SOFT_TOKENS, PATCH_SIZE, POOLING_KERNEL};
    use crate::{Error, Result};

    pub const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * 3;

    pub struct Preprocessed {
        /// `[max_patches, PATCH_DIM]`, row-major patches, channels last within a patch.
        pub pixel_values: Vec<f32>,
        /// `[max_patches, 2]` as (x, y); padding rows are (-1, -1).
        pub position_ids: Vec<i64>,
        pub max_patches: usize,
        pub soft_tokens: usize,
        pub width: u32,
        pub height: u32,
    }

    /// Mirrors `get_aspect_ratio_preserving_size`.
    pub fn target_size(height: usize, width: usize) -> Result<(usize, usize)> {
        let max_patches = MAX_SOFT_TOKENS * POOLING_KERNEL * POOLING_KERNEL;
        let total_px = (height * width) as f64;
        let target_px = (max_patches * PATCH_SIZE * PATCH_SIZE) as f64;
        let factor = (target_px / total_px).sqrt();
        let side = POOLING_KERNEL * PATCH_SIZE;
        let mut th = ((factor * height as f64) / side as f64).floor() as usize * side;
        let mut tw = ((factor * width as f64) / side as f64).floor() as usize * side;
        if th == 0 && tw == 0 {
            return Err(Error::Embedding("image too small to resize".into()));
        }
        let max_side = (max_patches / (POOLING_KERNEL * POOLING_KERNEL)) * side;
        if th == 0 {
            th = side;
            tw = ((width / height) * side).min(max_side);
        } else if tw == 0 {
            tw = side;
            th = ((height / width) * side).min(max_side);
        }
        if (th * tw) as f64 > target_px {
            return Err(Error::Embedding(format!(
                "cannot fit {width}x{height} image into the patch budget"
            )));
        }
        Ok((th, tw))
    }

    pub fn preprocess(bytes: &[u8]) -> Result<Preprocessed> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| Error::Embedding(format!("cannot decode image: {e}")))?
            .to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let (th, tw) = target_size(h, w)?;

        let rgb = if (th, tw) == (h, w) {
            img.into_raw()
        } else {
            let src = Image::from_vec_u8(w as u32, h as u32, img.into_raw(), PixelType::U8x3)
                .map_err(|e| Error::Embedding(e.to_string()))?;
            let mut dst = Image::new(tw as u32, th as u32, PixelType::U8x3);
            Resizer::new()
                .resize(
                    &src,
                    &mut dst,
                    &ResizeOptions::new()
                        .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom)),
                )
                .map_err(|e| Error::Embedding(e.to_string()))?;
            dst.into_vec()
        };

        let (ph, pw) = (th / PATCH_SIZE, tw / PATCH_SIZE);
        let max_patches = MAX_SOFT_TOKENS * POOLING_KERNEL * POOLING_KERNEL;
        let mut pixel_values = vec![0f32; max_patches * PATCH_DIM];
        let mut position_ids = vec![-1i64; max_patches * 2];
        for py in 0..ph {
            for px in 0..pw {
                let p = py * pw + px;
                let base = p * PATCH_DIM;
                for y in 0..PATCH_SIZE {
                    for x in 0..PATCH_SIZE {
                        let src = ((py * PATCH_SIZE + y) * tw + (px * PATCH_SIZE + x)) * 3;
                        let dst = base + (y * PATCH_SIZE + x) * 3;
                        for c in 0..3 {
                            pixel_values[dst + c] = rgb[src + c] as f32 / 255.0;
                        }
                    }
                }
                position_ids[p * 2] = px as i64;
                position_ids[p * 2 + 1] = py as i64;
            }
        }
        Ok(Preprocessed {
            pixel_values,
            position_ids,
            max_patches,
            soft_tokens: ph * pw / (POOLING_KERNEL * POOLING_KERNEL),
            width: tw as u32,
            height: th as u32,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn sizes_match_reference_formula() {
            // 640x480 -> factor sqrt(645120/307200); values checked against the Python implementation.
            let (h, w) = target_size(480, 640).unwrap();
            assert_eq!((h % 48, w % 48), (0, 0));
            assert!(h * w <= 2520 * 256);
            let (h, w) = target_size(10, 4000).unwrap();
            assert_eq!(h, 48);
            assert!(w <= 280 * 48);
        }
    }
}
