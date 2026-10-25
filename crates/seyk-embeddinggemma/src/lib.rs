//! EmbeddingGemma 2 through ONNX Runtime.
//!
//! Uses the ONNX export at `onnx-community/embeddinggemma-2-ONNX`, which splits
//! the model into a text model (that also accepts precomputed media features),
//! a vision encoder and an audio encoder. The text model returns a pooled
//! `sentence_embedding`; this module truncates it to the configured
//! Matryoshka dimension and L2-normalises it.
//!
//! Text, images and opt-in audio are implemented. Video remains unsupported.

//! # Standalone text embedding
//!
//! ```no_run
//! use seyk_embed::{Embedder, EmbeddingRequest, EmbeddingTask, ResourceInput};
//! use seyk_embeddinggemma::EmbeddingGemma2;
//! # fn main() -> seyk_embed::Result<()> {
//! let model = EmbeddingGemma2::builder().dimension(256).vision(false).build()?;
//! let query = model.embed_text("account access", EmbeddingTask::SearchQuery)?;
//! let batch = model.embed_batch(&[EmbeddingRequest {
//!     input: ResourceInput::Text { title: Some("Support"), text: "Reset your password." },
//!     task: EmbeddingTask::SearchDocument,
//! }])?;
//! assert_eq!(query.dim(), batch[0].dim());
//! # Ok(())
//! # }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

mod materialize;
#[cfg(test)]
mod session_tests;

use rayon::prelude::*;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;

mod text_tokenization;

pub use seyk_embed::{
    Dimension, Embedder, Embedding, EmbeddingRequest, EmbeddingTask, Error, Modality,
    ModelIdentity, ResourceInput, Result, cosine_similarity, dot_product, normalize,
};

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
#[cfg(feature = "image")]
const PATCH_SIZE: usize = 16;
#[cfg(feature = "image")]
const POOLING_KERNEL: usize = 3;
#[cfg(feature = "image")]
const MAX_SOFT_TOKENS: usize = 280;

/// Weight precision. `float16` is deliberately absent: the model card warns
/// that its activations overflow fp16 and silently produce NaN or degraded vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
pub struct GemmaConfig {
    pub source: ModelSource,
    pub dimension: Dimension,
    pub precision: Precision,
    /// Longer inputs are truncated. Segments are far shorter than this; it is a safety net.
    pub max_tokens: usize,
    /// Load the vision encoder so image resources can be embedded.
    pub vision: bool,
    /// Enable audio inputs; missing weights fail individual audio requests.
    pub audio: bool,
    /// ONNX Runtime intra-op threads (1..=i32::MAX). `None` lets ONNX Runtime decide.
    pub threads: Option<usize>,
    /// Inputs per model call.
    pub batch_size: usize,
    /// Independent CPU text sessions. Each owns an allocator arena and thread team.
    /// Defaults to one; choose explicit intra-op threads when using several.
    /// Bounds concurrent inference, not queued memory across callers. Callers must
    /// limit their own concurrency and input window sizes.
    pub sessions: usize,
}

impl Default for GemmaConfig {
    fn default() -> Self {
        Self {
            source: ModelSource::default(),
            dimension: Dimension::default(),
            precision: Precision::default(),
            max_tokens: 2048,
            vision: cfg!(feature = "image"),
            audio: cfg!(feature = "audio"),
            threads: None,
            batch_size: 16,
            sessions: 1,
        }
    }
}

impl GemmaConfig {
    pub fn files(&self) -> Vec<String> {
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

/// Resolve the local model directory without touching the network.
pub fn locate(config: &GemmaConfig) -> Result<PathBuf> {
    Ok(resolve(config)?.dir)
}

/// Derive cache identity without loading weights, downloading files or creating a cache.
/// HF sources read local refs when present; local directories use the same file
/// fingerprint as model loading (graph/config bytes and external weight sizes).
pub fn model_identity(config: &GemmaConfig) -> Result<ModelIdentity> {
    let resolved = match &config.source {
        ModelSource::Directory(_) => resolve(config)?,
        ModelSource::HuggingFace { repo, revision } => {
            hf_snapshot(&hf_cache_path()?, repo, revision)
        }
    };
    resolved.identity(config)
}

/// Required model files that are absent, checked using filesystem metadata only.
/// Does not load weights, materialize files, download or create a cache.
pub fn missing_model_files(config: &GemmaConfig) -> Result<Vec<String>> {
    let dir = match &config.source {
        ModelSource::Directory(dir) => dir.clone(),
        ModelSource::HuggingFace { repo, revision } => {
            hf_snapshot(&hf_cache_path()?, repo, revision).dir
        }
    };
    let missing: Vec<_> = config
        .files()
        .into_iter()
        .filter(|file| !dir.join(file).exists())
        .collect();
    Ok(missing)
}

/// A snapshot and the commit selected in the same cache lookup. Local directories
/// have no HF revision; their identity is still derived from their files.
struct ResolvedModel {
    dir: PathBuf,
    revision: Option<String>,
}

fn resolve(config: &GemmaConfig) -> Result<ResolvedModel> {
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
            Ok(ResolvedModel {
                dir: dir.clone(),
                revision: None,
            })
        }
        ModelSource::HuggingFace { repo, revision } => {
            resolve_hf(config, &hf_cache_path()?, repo, revision)
        }
    }
}

fn resolve_hf(
    config: &GemmaConfig,
    cache: &Path,
    repo: &str,
    revision: &str,
) -> Result<ResolvedModel> {
    let resolved = hf_snapshot(cache, repo, revision);
    let dir = &resolved.dir;
    for f in config.files() {
        if !dir.join(&f).exists() {
            return Err(Error::ModelNotFound(format!(
                "{repo}@{} ({f}) is not present; provision the model files or select a local model directory",
                &revision[..revision.len().min(12)],
            )));
        }
    }
    Ok(resolved)
}

fn hf_snapshot(cache: &Path, repo: &str, revision: &str) -> ResolvedModel {
    let repo_dir = cache.join(format!("models--{}", repo.replace('/', "--")));
    // Branch names resolve through refs/. A pinned commit often has no
    // refs/ entry (the Python client never writes one), only snapshots/<sha>,
    // so `hf_hub::CacheRepo::get` cannot be used here.
    let commit = std::fs::read_to_string(repo_dir.join("refs").join(revision))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| revision.to_string());
    let dir = repo_dir.join("snapshots").join(&commit);
    ResolvedModel {
        dir,
        revision: Some(commit),
    }
}

impl ResolvedModel {
    fn identity(&self, config: &GemmaConfig) -> Result<ModelIdentity> {
        let revision = match &config.source {
            ModelSource::HuggingFace { repo, .. } => {
                let commit = self.revision.as_deref().ok_or_else(|| {
                    Error::Config("Hugging Face model has no resolved commit".into())
                })?;
                format!("{repo}@{commit}")
            }
            ModelSource::Directory(_) => format!("local:{}", fingerprint(&self.dir, config)?),
        };
        Ok(ModelIdentity {
            model_id: MODEL_ID.into(),
            revision,
            // A literal control token could previously embed successfully. Those
            // cached vectors must not be reused under the new text policy.
            variant: format!("onnx-{};text-literals-v1", config.precision.name()),
            dimension: config.dimension,
            normalization: "mean-pool;truncate;l2".into(),
            query_prompt: QUERY_PROMPT.into(),
            document_prompt: format!("{DOCUMENT_PROMPT_TEMPLATE}{{text}}"),
            max_tokens: config.max_tokens,
        })
    }
}

fn hf_cache_path() -> Result<PathBuf> {
    let home = match std::env::var("HF_HOME") {
        Ok(home) => PathBuf::from(home),
        Err(_) => dirs::home_dir()
            .ok_or_else(|| Error::Config("no home directory".into()))?
            .join(".cache")
            .join("huggingface"),
    };
    Ok(home.join("hub"))
}

fn io_error(path: impl Into<PathBuf>, source: std::io::Error) -> Error {
    Error::Io {
        path: path.into(),
        source,
    }
}

/// Aggregate timings for a diagnostic text batch. Worker times can overlap.
#[derive(Debug, Default, Clone)]
pub struct TextBatchMetrics {
    pub prompt: Duration,
    pub tokenize: Duration,
    pub padding: Duration,
    pub inference: Duration,
    pub normalize: Duration,
    pub tokens: usize,
    pub padded_tokens: usize,
}

fn elapsed(start: Option<Instant>) -> Duration {
    start.map_or(Duration::ZERO, |t| t.elapsed())
}

/// EmbeddingGemma 2 embedder.
pub struct EmbeddingGemma2 {
    tokenizer: Tokenizer,
    text_tokenizer: Tokenizer,
    text: Vec<Mutex<Session>>,
    text_workers: Option<rayon::ThreadPool>,
    vision: Option<Mutex<Session>>,
    #[cfg(feature = "audio")]
    audio: Option<audio::AudioEncoder>,
    identity: ModelIdentity,
    config: GemmaConfig,
    pad_id: i64,
}

fn ort_err(e: impl std::fmt::Display) -> Error {
    Error::Embedding(e.to_string())
}

/// Compatibility name for existing callers.
pub type GemmaEmbedder = EmbeddingGemma2;

/// Configures a local, synchronous model. Building never downloads weights.
#[derive(Debug, Default)]
pub struct EmbeddingGemma2Builder {
    config: GemmaConfig,
    dimension: Option<usize>,
}
impl EmbeddingGemma2Builder {
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.source = ModelSource::Directory(path.into());
        self
    }
    pub fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }
    pub fn precision(mut self, precision: Precision) -> Self {
        self.config.precision = precision;
        self
    }
    pub fn vision(mut self, enabled: bool) -> Self {
        self.config.vision = enabled;
        self
    }
    pub fn audio(mut self, enabled: bool) -> Self {
        self.config.audio = enabled;
        self
    }
    pub fn max_tokens(mut self, count: usize) -> Self {
        self.config.max_tokens = count;
        self
    }
    pub fn batch_size(mut self, count: usize) -> Self {
        self.config.batch_size = count;
        self
    }
    pub fn sessions(mut self, count: usize) -> Self {
        self.config.sessions = count;
        self
    }
    pub fn threads(mut self, count: usize) -> Self {
        self.config.threads = Some(count);
        self
    }
    pub fn build(mut self) -> Result<EmbeddingGemma2> {
        if let Some(dimension) = self.dimension {
            self.config.dimension = Dimension::try_from(dimension)?;
        }
        EmbeddingGemma2::load(self.config)
    }
}

impl EmbeddingGemma2 {
    /// Load the pinned model from the local Hugging Face cache. Never downloads.
    pub fn load_default() -> Result<Self> {
        Self::load(GemmaConfig::default())
    }
    pub fn builder() -> EmbeddingGemma2Builder {
        EmbeddingGemma2Builder::default()
    }

    /// Load from local files. Fails with [`Error::ModelNotFound`] if they are not present.
    pub fn load(config: GemmaConfig) -> Result<Self> {
        if !Dimension::ALL.contains(&config.dimension) {
            return Err(Error::Config(format!(
                "unsupported dimension {}; use 128, 256, 512 or 768",
                config.dimension
            )));
        }
        if config.max_tokens == 0
            || config.batch_size == 0
            || config.sessions == 0
            || config.threads == Some(0)
        {
            return Err(Error::Config(
                "max_tokens, batch_size, sessions and threads must be nonzero".into(),
            ));
        }
        if config.threads.is_some_and(|n| n > i32::MAX as usize) {
            return Err(Error::Config("threads must not exceed i32::MAX".into()));
        }
        if config.audio && !cfg!(feature = "audio") {
            return Err(Error::UnsupportedModality {
                modality: Modality::Audio,
            });
        }
        if config.vision && !cfg!(feature = "image") {
            return Err(Error::UnsupportedModality {
                modality: Modality::Image,
            });
        }
        let resolved = resolve(&config)?;
        Self::from_resolved(resolved, config)
    }

    fn from_resolved(resolved: ResolvedModel, config: GemmaConfig) -> Result<Self> {
        let dir = &resolved.dir;
        let s = config.precision.suffix();
        let onnx_dir = onnx_dir_without_symlinks(dir, &config)?;
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(ort_err)?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: config.max_tokens,
                ..Default::default()
            }))
            .map_err(ort_err)?;
        tokenizer.with_padding(None);
        let text_tokenizer = text_tokenization::plain_text_tokenizer(&tokenizer);
        let pad_id = tokenizer.token_to_id("<pad>").unwrap_or(0) as i64;

        let text = (0..config.sessions)
            .map(|_| {
                session(&onnx_dir.join(format!("model{s}.onnx")), config.threads).map(Mutex::new)
            })
            .collect::<Result<Vec<_>>>()?;
        let text_workers = if config.sessions > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(config.sessions)
                    .thread_name(|i| format!("seyk-text-{i}"))
                    .build()
                    .map_err(|e| Error::Config(format!("building CPU text pool: {e}")))?,
            )
        } else {
            None
        };
        let vision = if config.vision {
            Some(Mutex::new(session(
                &onnx_dir.join(format!("vision_encoder{s}.onnx")),
                config.threads,
            )?))
        } else {
            None
        };

        let identity = resolved.identity(&config)?;
        Ok(Self {
            #[cfg(feature = "audio")]
            audio: config.audio.then(|| audio::AudioEncoder::new(dir)),
            tokenizer,
            text_tokenizer,
            text,
            text_workers,
            vision,
            identity,
            config,
            pad_id,
        })
    }

    /// Build information for the loaded ONNX Runtime, for local benchmark provenance.
    pub fn runtime_info(&self) -> &'static str {
        ort::info()
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

    /// Exact prompted text passed to tokenization.
    ///
    /// Source: [pinned official ONNX prompt config](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/blob/daa72c51243991dfcaf9f9137d2c573d8f7790c0/config_sentence_transformers.json).
    /// SearchQuery (Retrieval-query/query/Retrieval/Reranking/BitextMining): `task: search result | query: `.
    /// SearchDocument (Document/document/Retrieval-document): `title: none | text: `, or the supplied title.
    /// Classification (MultilabelClassification): `task: classification | query: `.
    /// Clustering: `task: clustering | query: `.
    /// CodeRetrieval (InstructionRetrieval): `task: code retrieval | query: `.
    /// FactChecking: `task: fact checking | query: `.
    /// QuestionAnswering: `task: question answering | query: `.
    /// SentenceSimilarity (STS/PairClassification/Summarization): `task: sentence similarity | query: `.
    /// Aliases describe equivalent embedding prompts, not text generation capabilities.
    pub fn text_input(title: Option<&str>, text: &str, task: EmbeddingTask) -> Result<String> {
        let prefix = match task {
            EmbeddingTask::SearchDocument => return Ok(Self::document_input(title, text)),
            EmbeddingTask::SearchQuery => QUERY_PROMPT,
            EmbeddingTask::Classification => "task: classification | query: ",
            EmbeddingTask::Clustering => "task: clustering | query: ",
            EmbeddingTask::CodeRetrieval => "task: code retrieval | query: ",
            EmbeddingTask::FactChecking => "task: fact checking | query: ",
            EmbeddingTask::QuestionAnswering => "task: question answering | query: ",
            EmbeddingTask::SentenceSimilarity => "task: sentence similarity | query: ",
            _ => {
                return Err(Error::UnsupportedTask {
                    task,
                    modality: Modality::Text,
                });
            }
        };
        Ok(format!("{prefix}{text}"))
    }

    /// Raw processor token ids, including literal special-token recognition.
    ///
    /// For verifying trusted image/audio templates against the reference only.
    /// Text embedding uses a separate plain-text tokenizer, so user-controlled
    /// text and titles cannot insert special tokens. Ordinary text ids agree.
    pub fn template_token_ids(&self, input: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(input, true)
            .map_err(ort_err)?
            .get_ids()
            .to_vec())
    }

    /// Token ids for an already-formatted text prompt, using the same tokenizer
    /// and truncation as text embedding. Literal special-token spellings remain
    /// ordinary text; the postprocessor still adds the template's BOS/EOS.
    /// Use `document_input`, `query_input`, or `text_input` to format the prompt.
    pub fn text_token_ids(&self, input: &str) -> Result<Vec<u32>> {
        Ok(self
            .text_tokenizer
            .encode(input, true)
            .map_err(ort_err)?
            .get_ids()
            .to_vec())
    }

    /// Measure text stages without changing the embedding profile. Durations for
    /// worker stages are summed CPU-side elapsed times, not end-to-end wall time.
    /// This diagnostic entry point rejects media and never records source text.
    pub fn embed_text_batch_profiled(
        &self,
        requests: &[EmbeddingRequest<'_>],
    ) -> Result<(Vec<Embedding>, TextBatchMetrics)> {
        let started = Instant::now();
        let inputs = requests
            .iter()
            .map(|r| match r.input {
                ResourceInput::Text { title, text } => Self::text_input(title, text, r.task),
                _ => Err(Error::Config(
                    "text profiling accepts only text requests".into(),
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let prompt = started.elapsed();
        let (vectors, mut metrics) = self.embed_strings_measured(&inputs, true)?;
        metrics.prompt = prompt;
        Ok((vectors, metrics))
    }

    /// Embed raw strings that already contain their prompt.
    fn embed_strings(&self, inputs: &[String]) -> Result<Vec<Embedding>> {
        self.embed_strings_measured(inputs, false).map(|(v, _)| v)
    }

    fn embed_strings_measured(
        &self,
        inputs: &[String],
        measure: bool,
    ) -> Result<(Vec<Embedding>, TextBatchMetrics)> {
        let start = measure.then(Instant::now);
        // Tokenize before entering our inference pool: tokenizers may use its own
        // Rayon work, which must never recursively schedule while a session is locked.
        let encodings = self
            .text_tokenizer
            .encode_batch(inputs.iter().map(String::as_str).collect::<Vec<_>>(), true)
            .map_err(ort_err)?;
        let ids: Vec<Vec<i64>> = encodings
            .iter()
            .map(|e| e.get_ids().iter().map(|&i| i as i64).collect())
            .collect();
        let tokenize = elapsed(start);
        let run = |batch: &[&[i64]]| {
            let index = self
                .text_workers
                .as_ref()
                .and_then(rayon::ThreadPool::current_thread_index)
                .unwrap_or(0);
            self.run_text_on(index, batch, None, None, measure)
        };
        let (vectors, mut metrics) = embed_token_ids(
            &ids,
            self.config.batch_size,
            self.text_workers.as_ref(),
            run,
        )?;
        metrics.tokenize = tokenize;
        Ok((vectors, metrics))
    }

    /// Run the text model on padded token ids, optionally with media features.
    #[cfg(any(feature = "image", feature = "audio"))]
    fn run_text(
        &self,
        batch: &[&[i64]],
        image_features: Option<(Vec<f32>, usize)>,
        audio_features: Option<(Vec<f32>, usize)>,
    ) -> Result<Vec<Embedding>> {
        // Keep media's text stage serialized on session zero, as before. Only
        // plain-text batches are distributed across the additional sessions.
        self.run_text_on(0, batch, image_features, audio_features, false)
            .map(|(v, _)| v)
    }

    fn run_text_on(
        &self,
        index: usize,
        batch: &[&[i64]],
        image_features: Option<(Vec<f32>, usize)>,
        audio_features: Option<(Vec<f32>, usize)>,
        measure: bool,
    ) -> Result<(Vec<Embedding>, TextBatchMetrics)> {
        let start = measure.then(Instant::now);
        let max_len = batch.iter().map(|b| b.len()).max().unwrap_or(0);
        let n = batch.len();
        let mut input_ids = vec![self.pad_id; n * max_len];
        let mut mask = vec![0i64; n * max_len];
        for (r, row) in batch.iter().enumerate() {
            input_ids[r * max_len..r * max_len + row.len()].copy_from_slice(row);
            mask[r * max_len..r * max_len + row.len()].fill(1);
        }
        let (img, img_tokens) = image_features.unwrap_or((Vec::new(), 0));
        let (aud, aud_tokens) = audio_features.unwrap_or((Vec::new(), 0));
        let inputs = ort::inputs![
            "input_ids" => Tensor::from_array(([n, max_len], input_ids)).map_err(ort_err)?,
            "attention_mask" => Tensor::from_array(([n, max_len], mask)).map_err(ort_err)?,
            "image_features" => Tensor::from_array(([img_tokens, FEATURE_DIM], img)).map_err(ort_err)?,
            "video_features" => Tensor::from_array(([0usize, FEATURE_DIM], Vec::<f32>::new())).map_err(ort_err)?,
            "audio_features" => Tensor::from_array(([aud_tokens, FEATURE_DIM], aud)).map_err(ort_err)?,
        ];
        let mut metrics = TextBatchMetrics {
            padding: elapsed(start),
            padded_tokens: n * max_len,
            ..Default::default()
        };
        let mut session = self.text[index]
            .lock()
            .map_err(|_| Error::Embedding("session lock poisoned".into()))?;
        let start = measure.then(Instant::now);
        let outputs = session.run(inputs).map_err(ort_err)?;
        metrics.inference = elapsed(start);
        let start = measure.then(Instant::now);
        let (shape, data) = outputs["sentence_embedding"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        if shape.len() != 2 || shape[0] as usize != n || shape[1] as usize != NATIVE_DIM {
            return Err(Error::Embedding(format!(
                "unexpected sentence_embedding shape {shape:?}"
            )));
        }
        let vectors = data
            .as_chunks::<NATIVE_DIM>()
            .0
            .iter()
            .map(|row| Embedding::from_raw(row.to_vec(), self.config.dimension))
            .collect::<Result<Vec<_>>>()?;
        metrics.normalize = elapsed(start);
        Ok((vectors, metrics))
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
        let ids: Vec<i64> = self
            .template_token_ids(&text)?
            .into_iter()
            .map(i64::from)
            .collect();
        let mut out = self.run_text(&[&ids], Some((features, pre.soft_tokens)), None)?;
        Ok(out.pop().expect("one input"))
    }

    #[cfg(not(feature = "image"))]
    fn embed_image(&self, _bytes: &[u8]) -> Result<Embedding> {
        Err(Error::UnsupportedModality {
            modality: Modality::Image,
        })
    }
}

/// ONNX Runtime refuses external weight files that resolve outside the
/// directory of the `.onnx` file. The Hugging Face cache stores every file as a
/// symlink into a shared blob store, which trips that check. In that case,
/// materialize the resolved files side by side in a cache directory. Links are
/// created without clobbering; replacement copies are verified and published
/// atomically, so competing callers never write through a source hard link.
/// Plain directories are used as they are.
fn onnx_dir_without_symlinks(dir: &Path, config: &GemmaConfig) -> Result<PathBuf> {
    onnx_files_without_symlinks(dir, config.files(), "")
}

fn onnx_files_without_symlinks(
    dir: &Path,
    files: Vec<String>,
    discriminator: &str,
) -> Result<PathBuf> {
    let onnx: Vec<String> = files
        .into_iter()
        .filter_map(|f| f.strip_prefix("onnx/").map(str::to_string))
        .collect();
    let src = dir.join("onnx");
    let any_link = onnx
        .iter()
        .any(|f| std::fs::symlink_metadata(src.join(f)).is_ok_and(|m| m.file_type().is_symlink()));
    if !any_link {
        return Ok(src);
    }
    let base = dirs::cache_dir()
        .ok_or_else(|| Error::Config("no cache directory on this platform".into()))?;
    let mut key = blake3::Hasher::new();
    key.update(src.to_string_lossy().as_bytes());
    key.update(discriminator.as_bytes());
    let key = key.finalize().to_hex();
    let cache = base.join("seyk");
    materialize::create_private_dir(&cache)?;
    let models = cache.join("models");
    materialize::create_private_dir(&models)?;
    let dst = models.join(&key[..16]);
    materialize::create_private_dir(&dst)?;
    for f in &onnx {
        let real = src
            .join(f)
            .canonicalize()
            .map_err(|e| io_error(src.join(f), e))?;
        let target = dst.join(f);
        materialize::publish(&real, &target)?;
    }
    Ok(dst)
}

// Private scheduling seam: tests inject model-free inference, while production
// uses the same ordering, short-circuit collection and output restoration.
fn embed_token_ids(
    ids: &[Vec<i64>],
    batch_size: usize,
    workers: Option<&rayon::ThreadPool>,
    run: impl Fn(&[&[i64]]) -> Result<(Vec<Embedding>, TextBatchMetrics)> + Sync,
) -> Result<(Vec<Embedding>, TextBatchMetrics)> {
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&i| ids[i].len());
    let run_chunk = |chunk: &[usize]| {
        let batch: Vec<&[i64]> = chunk.iter().map(|&i| ids[i].as_slice()).collect();
        run(&batch)
    };
    // Result collection discards successes on error and short-circuits admission.
    // Rayon joins already-running work before returning; simultaneous failures
    // may return either error. Successful results retain indexed chunk order.
    let results = if let Some(workers) = workers {
        workers.install(|| {
            order
                .par_chunks(batch_size)
                .map(run_chunk)
                .collect::<Result<Vec<_>>>()
        })
    } else {
        order
            .chunks(batch_size)
            .map(run_chunk)
            .collect::<Result<Vec<_>>>()
    }?;
    let mut metrics = TextBatchMetrics {
        tokens: ids.iter().map(Vec::len).sum(),
        ..Default::default()
    };
    let mut out: Vec<Option<Embedding>> = vec![None; ids.len()];
    for (chunk, (vectors, timing)) in order.chunks(batch_size).zip(results) {
        metrics.padding += timing.padding;
        metrics.inference += timing.inference;
        metrics.normalize += timing.normalize;
        metrics.padded_tokens += timing.padded_tokens;
        for (&i, v) in chunk.iter().zip(vectors) {
            out[i] = Some(v);
        }
    }
    Ok((
        out.into_iter()
            .map(|v| v.expect("every input embedded"))
            .collect(),
        metrics,
    ))
}

fn session(path: &Path, threads: Option<usize>) -> Result<Session> {
    let mut b = Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?;
    if let Some(t) = threads {
        b = b.with_intra_threads(t).map_err(ort_err)?;
    }
    // ORT 1.28's GetDirNameFromFilePath uses dirname(), whose result is shared
    // static storage on Apple platforms. Concurrent loads from different model
    // directories can tear external-weight paths. Keep file-backed initialization
    // serialized across text/vision/audio sessions until upstream fixes that use.
    // The guard ends before returning the session; inference remains concurrent.
    #[cfg(target_vendor = "apple")]
    let _init = {
        static SESSION_INIT: Mutex<()> = Mutex::new(());
        // The lock guards no data, so a panic in another load must not poison it.
        SESSION_INIT.lock().unwrap_or_else(|e| e.into_inner())
    };
    b.commit_from_file(path)
        .map_err(|e| Error::Embedding(format!("loading {}: {e}", path.display())))
}

/// Identify local model files cheaply: hash the graph files and the size of the weight files.
fn fingerprint(dir: &Path, config: &GemmaConfig) -> Result<String> {
    let mut h = blake3::Hasher::new();
    for f in config.files() {
        let p = dir.join(&f);
        if f.ends_with("_data") {
            let len = std::fs::metadata(&p).map_err(|e| io_error(&p, e))?.len();
            h.update(&len.to_le_bytes());
        } else {
            h.update(&std::fs::read(&p).map_err(|e| io_error(&p, e))?);
        }
    }
    Ok(h.finalize().to_hex()[..16].to_string())
}

impl Embedder for EmbeddingGemma2 {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn input_identity(&self, modality: Modality) -> Result<&str> {
        #[cfg(feature = "audio")]
        if modality == Modality::Audio {
            return self
                .audio
                .as_ref()
                .map_or(Ok("audio-disabled"), |a| a.identity());
        }
        let _ = modality;
        Ok("")
    }

    fn supports(&self, modality: Modality) -> bool {
        match modality {
            Modality::Text => true,
            #[cfg(feature = "audio")]
            Modality::Audio => self.config.audio,
            Modality::Image => cfg!(feature = "image") && self.vision.is_some(),
            _ => false,
        }
    }

    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> Result<Vec<Embedding>> {
        // Validate the entire batch before doing inference.
        for request in requests {
            let modality = request.input.modality();
            if !self.supports(modality) {
                return Err(Error::UnsupportedModality { modality });
            }
            if modality == Modality::Image && request.task != EmbeddingTask::SearchDocument {
                return Err(Error::UnsupportedTask {
                    task: request.task,
                    modality,
                });
            }
        }
        let mut out: Vec<Option<Embedding>> = vec![None; requests.len()];
        let mut text_idx = Vec::new();
        let mut text_inputs = Vec::new();
        for (i, request) in requests.iter().enumerate() {
            match &request.input {
                ResourceInput::Text { title, text } => {
                    text_idx.push(i);
                    text_inputs.push(Self::text_input(*title, text, request.task)?);
                }
                ResourceInput::Image { bytes } => out[i] = Some(self.embed_image(bytes)?),
                #[cfg(feature = "audio")]
                ResourceInput::Audio {
                    samples,
                    sample_rate,
                } => out[i] = Some(self.embed_audio(samples, *sample_rate)?),
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

#[cfg(test)]
mod scheduler_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Barrier, mpsc};

    fn success(batch: &[&[i64]]) -> Result<(Vec<Embedding>, TextBatchMetrics)> {
        let vectors = batch
            .iter()
            .map(|ids| {
                let mut vector = vec![0.0; 128];
                vector[ids[0] as usize] = 1.0;
                Embedding::from_raw(vector, Dimension::D128)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((vectors, TextBatchMetrics::default()))
    }

    #[test]
    fn serial_and_pool_stop_after_first_failure() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let ids = vec![vec![0]; 32];
        for workers in [None, Some(&pool)] {
            let calls = AtomicUsize::new(0);
            let result = embed_token_ids(&ids, 1, workers, |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(Error::Embedding("injected failure".into()))
            });
            assert!(
                matches!(result, Err(Error::Embedding(message)) if message == "injected failure")
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn reversed_completion_restores_order_and_discards_successes_on_failure() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        // Stable length sort produces chunks [1, 2] then [0, 3].
        let ids = vec![vec![0; 3], vec![1], vec![2; 2], vec![3; 4]];
        for fail_earlier in [false, true] {
            let (send, recv) = mpsc::channel();
            let recv = Mutex::new(recv);
            let completed = Mutex::new(Vec::new());
            let result = embed_token_ids(&ids, 2, Some(&pool), |batch| {
                let first = batch[0][0];
                if first == 1 {
                    recv.lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(10))
                        .unwrap();
                    completed.lock().unwrap().push(first);
                    if fail_earlier {
                        return Err(Error::Embedding("earlier chunk failed".into()));
                    }
                    success(batch)
                } else {
                    let vectors = success(batch)?;
                    completed.lock().unwrap().push(first);
                    send.send(()).unwrap();
                    Ok(vectors)
                }
            });
            assert_eq!(*completed.lock().unwrap(), vec![0, 1]);
            if fail_earlier {
                // The sole public outcome is Err: no successful partial vector list.
                assert!(
                    matches!(result, Err(Error::Embedding(message)) if message == "earlier chunk failed")
                );
            } else {
                let expected = success(&ids.iter().map(Vec::as_slice).collect::<Vec<_>>())
                    .unwrap()
                    .0;
                assert_eq!(result.unwrap().0, expected);
            }
        }
        // The same scheduler remains usable after a failed window, including retries.
        for id in &ids {
            let result =
                embed_token_ids(std::slice::from_ref(id), 1, Some(&pool), success).unwrap();
            assert_eq!(result.0, success(&[id]).unwrap().0);
        }
    }

    #[test]
    fn failure_joins_already_running_chunks() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let entered = Barrier::new(2);
        let (send, recv) = mpsc::channel();
        let recv = Mutex::new(recv);
        let finished = AtomicBool::new(false);
        let result = embed_token_ids(&[vec![0], vec![1]], 1, Some(&pool), |batch| {
            entered.wait();
            if batch[0][0] == 0 {
                send.send(()).unwrap();
                Err(Error::Embedding("injected failure".into()))
            } else {
                recv.lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
                let vectors = success(batch)?;
                finished.store(true, Ordering::SeqCst);
                Ok(vectors)
            }
        });
        assert!(result.is_err());
        assert!(finished.load(Ordering::SeqCst));
    }
}

#[cfg(test)]
mod resolution_tests {
    use super::*;

    const A: &str = ONNX_REVISION;
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn config(revision: &str) -> GemmaConfig {
        GemmaConfig {
            source: ModelSource::HuggingFace {
                repo: ONNX_REPO.into(),
                revision: revision.into(),
            },
            vision: false,
            ..GemmaConfig::default()
        }
    }

    fn write_files(dir: &Path, config: &GemmaConfig) {
        for file in config.files() {
            let path = dir.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"synthetic model file").unwrap();
        }
    }

    fn repo_dir(cache: &Path) -> PathBuf {
        cache.join(format!("models--{}", ONNX_REPO.replace('/', "--")))
    }

    #[test]
    fn cpu_tuning_keeps_identity_and_rejects_zero_sessions_before_model_loading() {
        let resolved = ResolvedModel {
            dir: PathBuf::new(),
            revision: Some(ONNX_REVISION.into()),
        };
        let mut config = GemmaConfig::default();
        let original = resolved.identity(&config).unwrap();
        config.sessions = 3;
        config.threads = Some(2);
        config.batch_size = 8;
        assert_eq!(original, resolved.identity(&config).unwrap());
        config.sessions = 0;
        assert!(matches!(
            EmbeddingGemma2::load(config),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn thread_range_is_checked_before_model_loading() {
        let missing = tempfile::tempdir().unwrap();
        for threads in [0, i32::MAX as usize + 1, usize::MAX] {
            let config = GemmaConfig {
                source: ModelSource::Directory(missing.path().into()),
                threads: Some(threads),
                ..GemmaConfig::default()
            };
            assert!(matches!(
                EmbeddingGemma2::load(config),
                Err(Error::Config(_))
            ));
        }
        for threads in [1, i32::MAX as usize] {
            let config = GemmaConfig {
                source: ModelSource::Directory(missing.path().into()),
                threads: Some(threads),
                ..GemmaConfig::default()
            };
            assert!(matches!(
                EmbeddingGemma2::load(config),
                Err(Error::ModelNotFound(_))
            ));
        }
    }

    #[test]
    fn moving_ref_changes_identity_but_not_an_already_resolved_model() {
        let cache = tempfile::tempdir().unwrap();
        let config = config("main");
        let repo = repo_dir(cache.path());
        for commit in [A, B] {
            write_files(&repo.join("snapshots").join(commit), &config);
        }
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs/main"), format!("{A}\n")).unwrap();
        let first = resolve_hf(&config, cache.path(), ONNX_REPO, "main").unwrap();
        std::fs::write(repo.join("refs/main"), B).unwrap();
        // Identity is built after the alias moves: it must still describe A.
        let first_identity = first.identity(&config).unwrap();
        assert_eq!(first.dir, repo.join("snapshots").join(A));
        assert_eq!(first_identity.revision, format!("{ONNX_REPO}@{A}"));
        let second = resolve_hf(&config, cache.path(), ONNX_REPO, "main").unwrap();
        assert_eq!(second.dir, repo.join("snapshots").join(B));
        assert_eq!(
            second.identity(&config).unwrap().revision,
            format!("{ONNX_REPO}@{B}")
        );
        assert_ne!(first_identity, second.identity(&config).unwrap());
    }

    #[test]
    fn missing_resolved_commit_returns_an_error() {
        let resolved = ResolvedModel {
            dir: PathBuf::new(),
            revision: None,
        };
        assert!(matches!(
            resolved.identity(&config("main")),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn pinned_sha_without_ref_keeps_exact_identity() {
        let cache = tempfile::tempdir().unwrap();
        let config = config(A);
        let dir = repo_dir(cache.path()).join("snapshots").join(A);
        write_files(&dir, &config);
        let resolved = resolve_hf(&config, cache.path(), ONNX_REPO, A).unwrap();
        assert_eq!(resolved.dir, dir);
        assert_eq!(
            resolved.identity(&config).unwrap(),
            ModelIdentity {
                model_id: "google/embeddinggemma-2".into(),
                revision:
                    "onnx-community/embeddinggemma-2-ONNX@daa72c51243991dfcaf9f9137d2c573d8f7790c0"
                        .into(),
                variant: "onnx-q8;text-literals-v1".into(),
                dimension: config.dimension,
                normalization: "mean-pool;truncate;l2".into(),
                query_prompt: "task: search result | query: ".into(),
                document_prompt: "title: {title} | text: {text}".into(),
                max_tokens: 2048,
            }
        );
    }

    #[test]
    fn snapshot_must_contain_every_required_file() {
        let cache = tempfile::tempdir().unwrap();
        let config = config(A);
        let dir = repo_dir(cache.path()).join("snapshots").join(A);
        write_files(&dir, &config);
        let missing = config.files().pop().unwrap();
        std::fs::remove_file(dir.join(&missing)).unwrap();
        let Err(Error::ModelNotFound(message)) = resolve_hf(&config, cache.path(), ONNX_REPO, A)
        else {
            panic!("an incomplete snapshot must fail");
        };
        assert!(message.contains(&missing));
    }

    #[test]
    fn local_directory_keeps_fingerprint_identity_and_path_only_locate() {
        let dir = tempfile::tempdir().unwrap();
        let config = GemmaConfig {
            source: ModelSource::Directory(dir.path().to_path_buf()),
            vision: false,
            ..GemmaConfig::default()
        };
        write_files(dir.path(), &config);
        let expected = format!("local:{}", fingerprint(dir.path(), &config).unwrap());
        let resolved = resolve(&config).unwrap();
        assert!(resolved.revision.is_none());
        assert_eq!(locate(&config).unwrap(), dir.path());
        assert_eq!(resolved.identity(&config).unwrap().revision, expected);
        std::fs::write(dir.path().join("tokenizer.json"), b"changed graph").unwrap();
        assert_ne!(
            resolve(&config)
                .unwrap()
                .identity(&config)
                .unwrap()
                .revision,
            expected
        );
        std::fs::remove_file(dir.path().join("tokenizer.json")).unwrap();
        assert!(matches!(locate(&config), Err(Error::ModelNotFound(_))));
    }
}

#[cfg(feature = "audio")]
pub mod audio;
