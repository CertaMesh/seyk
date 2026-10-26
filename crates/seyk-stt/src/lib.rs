//! Pinned, offline Whisper tiny transcription. Construct lazily; only explicit
//! extraction loads sessions. Content and token IDs never enter diagnostics.
//! Applications must install a payload-free panic hook (catch_unwind runs it).
pub mod manifest;
mod mel;
use manifest::{manifest, model_error, read_asset};
use ort::{
    logging::LogLevel,
    session::{
        OutputSelector, RunOptions, Session, SessionInputValue, builder::GraphOptimizationLevel,
    },
    value::{DynValue, Tensor, TensorElementType, ValueType},
};
use seyk_core::{
    Error, Result,
    audio::{SpeechToText, Transcript},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Mutex,
};
use tokenizers::Tokenizer;

pub const IMPLEMENTATION: &str = "seyk-whisper-onnx-v1";
const VOCAB: usize = 51865;
const EOT: usize = 50257;
const SOT: usize = 50258;
const TRANSCRIBE: usize = 50359;
const NO_TIMESTAMPS: usize = 50363;
const NO_SPEECH: usize = 50362;
fn failure(reason: &'static str) -> Error {
    Error::SpeechToText { reason }
}
fn ort_error<T>(_: T) -> Error {
    failure("ONNX Runtime operation failed")
}

/// One lazy, serialized CPU session set. No network, disk transcript cache,
/// profiling, beam search or previous-interval conditioning.
pub struct Whisper {
    directory: Option<PathBuf>,
    threads: usize,
    state: Mutex<Option<State>>,
}
impl Whisper {
    /// A local directory is subject to the same manifest as an installed model.
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: Some(directory.into()),
            threads: 1,
            state: Mutex::new(None),
        }
    }
    pub fn installed() -> Self {
        Self {
            directory: None,
            threads: 1,
            state: Mutex::new(None),
        }
    }
    /// Explicit eager load for applications that want to prepare extraction sessions.
    pub fn load(directory: impl Into<PathBuf>) -> Result<Self> {
        let value = Self::new(directory);
        *value
            .state
            .lock()
            .map_err(|_| failure("session lock poisoned"))? = Some(State::load(
            value.directory.as_ref().expect("explicit directory"),
            value.threads,
        )?);
        Ok(value)
    }
    fn run(&self, samples: &[f32], language: Option<&str>) -> Result<Transcript> {
        if samples.len() > 480_000 {
            return Err(Error::AudioIntervalTooLong);
        }
        if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
            return Err(failure("invalid canonical PCM"));
        }
        let language_token = language
            .map(|code| {
                manifest()
                    .policy
                    .lang_to_id
                    .get(&format!("<|{code}|>"))
                    .copied()
                    .ok_or_else(|| failure("unsupported language code"))
            })
            .transpose()?;
        let mut guard = self
            .state
            .lock()
            .map_err(|_| failure("session lock poisoned"))?;
        if guard.is_none() {
            *guard = Some(State::load(
                &self
                    .directory
                    .clone()
                    .map(Ok)
                    .unwrap_or_else(manifest::default_directory)?,
                self.threads,
            )?);
        }
        // A valid installation is still required; no inference or guessed language
        // for exact zero PCM. The manifest identifies this policy as well.
        if samples.iter().all(|x| *x == 0.0) {
            return Ok(no_speech());
        }
        guard
            .as_mut()
            .expect("loaded state")
            .decode(samples, language_token, false)
            .map(|x| x.0)
    }
}
impl SpeechToText for Whisper {
    fn implementation(&self) -> &'static str {
        IMPLEMENTATION
    }
    fn manifest_sha256(&self) -> String {
        manifest::manifest_sha256()
    }
    fn transcribe(&self, samples: &[f32], language: Option<&str>) -> Result<Transcript> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.run(samples, language)))
            .unwrap_or_else(|_| Err(failure("backend panicked")))
    }
}
fn no_speech() -> Transcript {
    Transcript {
        text: String::new(),
        language: None,
        no_speech: true,
    }
}

struct State {
    encoder: Session,
    decoder: Session,
    past: Session,
    tokenizer: Tokenizer,
}
struct Step {
    logits: Vec<f32>,
    cache: BTreeMap<String, DynValue>,
}
impl State {
    fn load(dir: &Path, threads: usize) -> Result<Self> {
        manifest::verify_layout(dir)?;
        let mut tokenizer = None;
        // Validate all small runtime assets, even those not used by this policy.
        for asset in manifest().files.iter().filter(|asset| {
            !manifest()
                .graphs
                .iter()
                .any(|graph| graph.file == asset.path)
        }) {
            let bytes = read_asset(dir, asset)?;
            if asset.path == "tokenizer.json" {
                tokenizer = Some(
                    Tokenizer::from_bytes(&bytes).map_err(|_| model_error("tokenizer invalid"))?,
                );
            }
        }
        let load_graph = |file| -> Result<Session> {
            let (asset, graph) = manifest().graph(file);
            let bytes = read_asset(dir, asset)?;
            let session = Session::builder()
                .map_err(ort_error)?
                .with_execution_providers([ort::ep::CPU::default().build()])
                .map_err(ort_error)?
                .with_intra_threads(threads)
                .map_err(ort_error)?
                .with_inter_threads(1)
                .map_err(ort_error)?
                .with_log_level(LogLevel::Fatal)
                .map_err(ort_error)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(ort_error)?
                .commit_from_memory(&bytes)
                .map_err(|_| model_error("graph load failed"))?;
            for (actual, expected) in [
                (session.inputs(), &graph.inputs),
                (session.outputs(), &graph.outputs),
            ] {
                if actual.len() != expected.len() {
                    return Err(model_error("graph contract mismatch"));
                }
                for (actual, expected) in actual.iter().zip(expected) {
                    let ValueType::Tensor { ty, shape, .. } = actual.dtype() else {
                        return Err(model_error("graph contract mismatch"));
                    };
                    let dtype = if expected.dtype == "FLOAT" {
                        TensorElementType::Float32
                    } else {
                        TensorElementType::Int64
                    };
                    if actual.name() != expected.name
                        || *ty != dtype
                        || shape.len() != expected.shape.len()
                        || shape
                            .iter()
                            .zip(&expected.shape)
                            .any(|(n, e)| e.as_i64().is_some_and(|v| *n != v))
                    {
                        return Err(model_error("graph contract mismatch"));
                    }
                }
            }
            Ok(session)
        };
        let encoder = load_graph("onnx/encoder_model.onnx")?;
        let decoder = load_graph("onnx/decoder_model.onnx")?;
        let past = load_graph("onnx/decoder_with_past_model.onnx")?;
        Ok(Self {
            encoder,
            decoder,
            past,
            tokenizer: tokenizer.expect("manifest tokenizer"),
        })
    }
    fn encode(&mut self, samples: &[f32]) -> Result<DynValue> {
        let tensor =
            Tensor::from_array(([1usize, 80, 3000], mel::features(samples)?)).map_err(ort_error)?;
        let options = RunOptions::new()
            .map_err(ort_error)?
            .with_outputs(OutputSelector::no_default().with("last_hidden_state"));
        let mut output = self
            .encoder
            .run_with_options(ort::inputs!["input_features" => tensor], &options)
            .map_err(ort_error)?;
        let hidden = output
            .remove("last_hidden_state")
            .ok_or_else(|| failure("encoder output missing"))?;
        check_tensor(&hidden, &[1, 1500, 384])?;
        Ok(hidden)
    }
    fn sot_logits(&mut self, hidden: &DynValue) -> Result<Vec<f32>> {
        let tokens = Tensor::from_array(([1usize, 1], vec![SOT as i64])).map_err(ort_error)?;
        let options = RunOptions::new()
            .map_err(ort_error)?
            .with_outputs(OutputSelector::no_default().with("logits"));
        let outputs = self
            .decoder
            .run_with_options(
                ort::inputs!["input_ids" => tokens, "encoder_hidden_states" => hidden],
                &options,
            )
            .map_err(ort_error)?;
        let logits = outputs
            .get("logits")
            .ok_or_else(|| failure("SOT logits missing"))?;
        check_tensor(logits, &[1, 1, VOCAB as i64])?;
        Ok(logits
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?
            .1
            .to_vec())
    }
    fn initial(&mut self, ids: &[usize], hidden: &DynValue) -> Result<Step> {
        let tokens = Tensor::from_array((
            [1usize, ids.len()],
            ids.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        ))
        .map_err(ort_error)?;
        forward(
            &mut self.decoder,
            vec![
                ("input_ids".into(), tokens.into()),
                ("encoder_hidden_states".into(), hidden.into()),
            ],
            ids.len(),
            ids.len(),
            true,
        )
    }
    fn cached(&mut self, token: usize, previous: Step, length: usize) -> Result<Step> {
        let tokens = Tensor::from_array(([1usize, 1], vec![token as i64])).map_err(ort_error)?;
        let mut inputs = vec![("input_ids".into(), tokens.into())];
        for (name, value) in &previous.cache {
            inputs.push((name.replace("present.", "past_key_values."), value.into()));
        }
        let mut next = forward(&mut self.past, inputs, 1, length, false)?;
        for (name, value) in previous.cache {
            if name.contains(".encoder.") {
                next.cache.insert(name, value);
            }
        }
        Ok(next)
    }
    /// `parity` is used only by the ignored model gate; no token data is logged.
    fn decode(
        &mut self,
        samples: &[f32],
        forced_language: Option<usize>,
        parity: bool,
    ) -> Result<(Transcript, f32, Vec<usize>)> {
        let hidden = self.encode(samples)?;
        let sot = self.sot_logits(&hidden)?;
        let p_no_speech = log_probability(&sot, NO_SPEECH)?.exp();
        let language_token = forced_language.unwrap_or_else(|| {
            manifest()
                .policy
                .lang_to_id
                .values()
                .copied()
                .max_by(|a, b| sot[*a].total_cmp(&sot[*b]).then_with(|| b.cmp(a)))
                .expect("languages")
        });
        drop(sot);
        let language = manifest()
            .policy
            .lang_to_id
            .iter()
            .find(|(_, id)| **id == language_token)
            .map(|(name, _)| {
                name.trim_start_matches("<|")
                    .trim_end_matches("|>")
                    .to_string()
            })
            .ok_or_else(|| failure("language selection failed"))?;
        let prefix = [SOT, language_token, TRANSCRIBE, NO_TIMESTAMPS];
        let mut step = self.initial(&prefix, &hidden)?;
        let mut generated = Vec::new();
        let mut sum = 0.0;
        let mut max_error = 0.0f32;
        loop {
            let mut logits = step.logits.clone();
            suppress(&mut logits, generated.is_empty());
            let token = argmax(&logits)?;
            sum += log_probability(&logits, token)?;
            generated.push(token);
            if token == EOT {
                break;
            }
            if prefix.len() + generated.len() >= 448 {
                return Err(failure("token context exhausted"));
            }
            step = self.cached(token, step, prefix.len() + generated.len())?;
            if parity {
                let ids: Vec<usize> = prefix.iter().chain(&generated).copied().collect();
                let full = self.initial(&ids, &hidden)?;
                for (a, b) in step.logits.iter().zip(&full.logits) {
                    max_error = max_error.max((a - b).abs());
                }
                let mut a = step.logits.clone();
                let mut b = full.logits;
                suppress(&mut a, false);
                suppress(&mut b, false);
                if argmax(&a)? != argmax(&b)? {
                    return Err(failure("cached decoder parity mismatch"));
                }
            }
        }
        // Binding policy: includes EOT, denominator generated count + 1.
        if classify_no_speech(p_no_speech, sum, generated.len()) {
            return Ok((no_speech(), max_error, generated));
        }
        let ids: Vec<u32> = generated.iter().map(|x| *x as u32).collect();
        let text = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|_| failure("detokenization failed"))?;
        Ok((
            Transcript {
                text,
                language: Some(language),
                no_speech: false,
            },
            max_error,
            generated,
        ))
    }
}
fn check_tensor(value: &DynValue, expected: &[i64]) -> Result<()> {
    let (shape, values) = value.try_extract_tensor::<f32>().map_err(ort_error)?;
    if shape.as_ref() != expected || values.iter().any(|x| !x.is_finite()) {
        return Err(failure("invalid tensor output"));
    }
    Ok(())
}
fn forward(
    session: &mut Session,
    inputs: Vec<(String, SessionInputValue<'_>)>,
    token_count: usize,
    length: usize,
    initial: bool,
) -> Result<Step> {
    let mut names = Vec::new();
    for layer in 0..4 {
        for side in ["decoder", "encoder"] {
            if !initial && side == "encoder" {
                continue;
            }
            for kv in ["key", "value"] {
                names.push(format!("present.{layer}.{side}.{kv}"));
            }
        }
    }
    let mut selector = OutputSelector::no_default().with("logits");
    for name in &names {
        selector = selector.with(name.clone());
    }
    let options = RunOptions::new().map_err(ort_error)?.with_outputs(selector);
    let mut outputs = session
        .run_with_options(inputs, &options)
        .map_err(ort_error)?;
    let output = outputs
        .get("logits")
        .ok_or_else(|| failure("logits missing"))?;
    check_tensor(output, &[1, token_count as i64, VOCAB as i64])?;
    let (_, values) = output.try_extract_tensor::<f32>().map_err(ort_error)?;
    let logits = values[values.len() - VOCAB..].to_vec();
    let mut cache = BTreeMap::new();
    for name in names {
        let value = outputs
            .remove(&name)
            .ok_or_else(|| failure("KV output missing"))?;
        check_tensor(
            &value,
            &[
                1,
                6,
                if name.contains(".encoder.") {
                    1500
                } else {
                    length as i64
                },
                64,
            ],
        )?;
        cache.insert(name, value);
    }
    Ok(Step { logits, cache })
}
fn classify_no_speech(
    p_no_speech: f64,
    generated_logprob_sum: f64,
    generated_count: usize,
) -> bool {
    p_no_speech > 0.6 && generated_logprob_sum / (generated_count + 1) as f64 <= -1.0
}
fn suppress(logits: &mut [f32], first: bool) {
    logits[EOT + 1..].fill(f32::NEG_INFINITY);
    for &id in &manifest().policy.suppress_tokens {
        logits[id] = f32::NEG_INFINITY;
    }
    if first {
        logits[220] = f32::NEG_INFINITY;
        logits[EOT] = f32::NEG_INFINITY;
    }
}
fn argmax(logits: &[f32]) -> Result<usize> {
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    if !logits[best].is_finite() {
        return Err(failure("all logits suppressed or nonfinite"));
    }
    Ok(best)
}
fn log_probability(logits: &[f32], token: usize) -> Result<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    if !max.is_finite() || !logits[token].is_finite() {
        return Err(failure("invalid probability"));
    }
    let denom: f64 = logits.iter().map(|x| (*x as f64 - max).exp()).sum();
    Ok(logits[token] as f64 - max - denom.ln())
}

#[cfg(test)]
mod tests;
