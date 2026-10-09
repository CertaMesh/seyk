//! Paired CPU throughput measurements; all model files must already be local.
//! cargo run --release --example bench_cpu_pool -- --baseline q8:1:8:16:32
//!     --candidate q8:2:4:16:32 --pairs 5 --segments 64
//! Configurations are precision:sessions:intra_threads:microbatch:engine_window.
//! Use `auto` for intra_threads. Every A/B or B/A pair uses fresh child processes.
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use seyk::embedder::gemma::{GemmaConfig, GemmaEmbedder, ModelSource, Precision};
use seyk::{Embedder, EmbeddingRequest, EmbeddingTask, MemoryCache, ResourceInput, SearchEngine};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug)]
struct Config {
    precision: Precision,
    sessions: usize,
    threads: Option<usize>,
    batch: usize,
    window: usize,
}

fn positive(s: &str) -> Result<usize> {
    let n = s.parse::<usize>()?;
    if n == 0 {
        return Err("counts must be positive".into());
    }
    Ok(n)
}

fn config(s: &str) -> Result<Config> {
    let parts: Vec<_> = s.split(':').collect();
    if parts.len() != 5 {
        return Err("expected precision:sessions:threads|auto:microbatch:window".into());
    }
    Ok(Config {
        precision: parts[0].parse()?,
        sessions: positive(parts[1])?,
        threads: if parts[2] == "auto" {
            None
        } else {
            let threads = positive(parts[2])?;
            if threads > i32::MAX as usize {
                return Err("threads must not exceed i32::MAX".into());
            }
            Some(threads)
        },
        batch: positive(parts[3])?,
        window: positive(parts[4])?,
    })
}

#[derive(Clone)]
struct Args {
    baseline: String,
    candidate: String,
    cell: Option<String>,
    pairs: usize,
    segments: usize,
    mixed: bool,
    model_dir: Option<PathBuf>,
    corpus: usize,
    queries: usize,
}

fn args() -> Result<Args> {
    let mut a = Args {
        baseline: "q8:1:8:16:32".into(),
        candidate: "q8:2:4:16:32".into(),
        cell: None,
        pairs: 5,
        segments: 64,
        mixed: false,
        model_dir: None,
        corpus: 0,
        queries: 5,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--mixed" {
            a.mixed = true;
            continue;
        }
        let value = it.next().ok_or("option requires a value")?;
        match flag.as_str() {
            "--baseline" => a.baseline = value,
            "--candidate" => a.candidate = value,
            "--cell" => a.cell = Some(value),
            "--pairs" => a.pairs = positive(&value)?,
            "--segments" => a.segments = positive(&value)?,
            "--queries" => a.queries = positive(&value)?,
            "--corpus" => a.corpus = value.parse()?,
            "--model-dir" => a.model_dir = Some(value.into()),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    config(&a.baseline)?;
    config(&a.candidate)?;
    Ok(a)
}

#[derive(Serialize, Deserialize)]
struct Measurement {
    seconds: f64,
    load_before: Option<[f64; 3]>,
    load_after: Option<[f64; 3]>,
    peak_rss_mib: Option<f64>,
}

impl Measurement {
    fn run<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, Self)> {
        let load_before = load();
        let start = Instant::now();
        let result = f()?;
        Ok((
            result,
            Self {
                seconds: start.elapsed().as_secs_f64(),
                load_before,
                load_after: load(),
                peak_rss_mib: rss(),
            },
        ))
    }
}

#[cfg(unix)]
fn load() -> Option<[f64; 3]> {
    let mut values = [0.0; 3];
    // SAFETY: getloadavg writes at most the three allocated f64 values.
    (unsafe { libc::getloadavg(values.as_mut_ptr(), 3) } == 3).then_some(values)
}
#[cfg(not(unix))]
fn load() -> Option<[f64; 3]> {
    None
}

#[cfg(unix)]
fn rss() -> Option<f64> {
    let mut u = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: valid writable rusage; only read after successful initialization.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, u.as_mut_ptr()) } != 0 {
        return None;
    }
    let value = unsafe { u.assume_init() }.ru_maxrss as f64;
    let bytes = if cfg!(target_os = "macos") {
        value
    } else {
        value * 1024.0
    };
    Some(bytes / 1048576.0)
}
#[cfg(not(unix))]
fn rss() -> Option<f64> {
    None
}

#[derive(Serialize, Deserialize)]
struct Cell {
    configuration: String,
    revision: String,
    runtime: String,
    dimension: usize,
    segments: usize,
    mixed: bool,
    tokens: usize,
    padded_tokens: usize,
    // prompt, tokenization, padding/tensor construction, ORT, normalization.
    // Summed worker wall times, which overlap when sessions > 1.
    stage_seconds: [f64; 5],
    model_load: Measurement,
    first_inference: Measurement,
    warmed: Measurement,
    queries: Vec<Measurement>,
    concurrent_queries: Vec<Measurement>,
    concurrent_documents: Option<Measurement>,
    cold_index: Option<Measurement>,
    cached_search: Option<Measurement>,
}

fn texts(count: usize, mixed: bool) -> Vec<String> {
    // Fixed public synthetic prose. Variation avoids repeated-input cache effects.
    (0..count).map(|i| {
        let repetitions = if mixed { [1, 4, 12, 36][i % 4] } else { 12 };
        format!("Entry {i}. {}", "The winter garden holds an old letter. Her brother remembers the harbour and a promise made at the kitchen table. ".repeat(repetitions))
    }).collect()
}

fn requests(texts: &[String]) -> Vec<EmbeddingRequest<'_>> {
    texts
        .iter()
        .map(|text| EmbeddingRequest {
            input: ResourceInput::Text {
                title: Some("Scene"),
                text,
            },
            task: EmbeddingTask::SearchDocument,
        })
        .collect()
}

fn cell(a: &Args, label: &str) -> Result<Cell> {
    let c = config(label)?;
    let mut conf = GemmaConfig::default();
    conf.precision = c.precision;
    conf.sessions = c.sessions;
    conf.threads = c.threads;
    conf.batch_size = c.batch;
    conf.vision = false;
    conf.audio = false;
    if let Some(dir) = &a.model_dir {
        conf.source = ModelSource::Directory(dir.clone());
    }
    let (model, model_load) = Measurement::run(|| Ok(Arc::new(GemmaEmbedder::load(conf)?)))?;
    let text = texts(a.segments, a.mixed);
    let input = requests(&text);
    let (_, first_inference) = Measurement::run(|| {
        for chunk in input.chunks(c.window) {
            model.embed_batch(chunk)?;
        }
        Ok(())
    })?;
    let mut stages = [0.0; 5];
    let mut tokens = 0;
    let mut padded_tokens = 0;
    let (_, warmed) = Measurement::run(|| {
        for chunk in input.chunks(c.window) {
            let (vectors, timing) = model.embed_text_batch_profiled(chunk)?;
            std::hint::black_box(vectors);
            for (total, elapsed) in stages.iter_mut().zip([
                timing.prompt,
                timing.tokenize,
                timing.padding,
                timing.inference,
                timing.normalize,
            ]) {
                *total += elapsed.as_secs_f64();
            }
            tokens += timing.tokens;
            padded_tokens += timing.padded_tokens;
        }
        Ok(())
    })?;
    let mut queries = Vec::new();
    let query = "Where did I describe a winter garden?";
    // Warm the batch-1 query shape separately.
    model.embed_query(query)?;
    for _ in 0..a.queries {
        queries.push(Measurement::run(|| Ok(model.embed_query(query)?))?.1);
    }
    let mut concurrent_queries = Vec::new();
    let mut concurrent_documents = None;
    let mut cold_index = None;
    let mut cached_search = None;
    if a.corpus > 0 {
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| -> Result<()> {
            let worker = scope.spawn(|| -> seyk::Result<()> {
                barrier.wait();
                for chunk in input.chunks(c.window) {
                    model.embed_batch(chunk)?;
                }
                Ok(())
            });
            barrier.wait();
            let load_before = load();
            let start = Instant::now();
            for _ in 0..a.queries {
                concurrent_queries.push(Measurement::run(|| Ok(model.embed_query(query)?))?.1);
            }
            worker.join().map_err(|_| "document worker panicked")??;
            concurrent_documents = Some(Measurement {
                seconds: start.elapsed().as_secs_f64(),
                load_before,
                load_after: load(),
                peak_rss_mib: rss(),
            });
            Ok(())
        })?;
        let dir = tempfile::tempdir()?;
        for (i, text) in texts(a.corpus, a.mixed).iter().enumerate() {
            std::fs::write(dir.path().join(format!("scene-{i}.txt")), text)?;
        }
        let engine = SearchEngine::builder()
            .shared_embedder(model.clone())
            .batch_size(c.window)
            .cache(MemoryCache::new())
            .build()?;
        cold_index =
            Some(Measurement::run(|| Ok(engine.search(query).paths([dir.path()]).execute()?))?.1);
        cached_search =
            Some(Measurement::run(|| Ok(engine.search(query).paths([dir.path()]).execute()?))?.1);
    }
    Ok(Cell {
        configuration: label.into(),
        revision: model.identity().revision.clone(),
        runtime: model.runtime_info().into(),
        dimension: model.identity().dimension.get(),
        segments: a.segments,
        mixed: a.mixed,
        tokens,
        padded_tokens,
        stage_seconds: stages,
        model_load,
        first_inference,
        warmed,
        queries,
        concurrent_queries,
        concurrent_documents,
        cold_index,
        cached_search,
    })
}

fn child(a: &Args, label: &str) -> Result<Cell> {
    let mut cmd = Command::new(std::env::current_exe()?);
    cmd.args([
        "--cell",
        label,
        "--segments",
        &a.segments.to_string(),
        "--queries",
        &a.queries.to_string(),
        "--corpus",
        &a.corpus.to_string(),
    ]);
    if a.mixed {
        cmd.arg("--mixed");
    }
    if let Some(dir) = &a.model_dir {
        cmd.arg("--model-dir").arg(dir);
    }
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(format!(
            "cell {label} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn main() -> Result<()> {
    let a = args()?;
    if let Some(label) = &a.cell {
        println!("{}", serde_json::to_string(&cell(&a, label)?)?);
        return Ok(());
    }
    eprintln!(
        "INDICATIVE unless run in a coordinated quiet window; ratios only compare this run. N/A means unavailable, not zero. Fresh process per cell; no model downloads."
    );
    println!(
        "{}",
        serde_json::json!({"kind":"environment", "os":std::env::consts::OS, "arch":std::env::consts::ARCH, "available_parallelism":std::thread::available_parallelism().ok().map(usize::from), "baseline":a.baseline, "candidate":a.candidate, "pairs":a.pairs})
    );
    let mut ratios = Vec::new();
    for pair in 0..a.pairs {
        let labels = if pair % 2 == 0 {
            [&a.baseline, &a.candidate]
        } else {
            [&a.candidate, &a.baseline]
        };
        let first = child(&a, labels[0])?;
        println!(
            "{}",
            serde_json::json!({"kind":"cell", "pair":pair, "data":first})
        );
        let second = child(&a, labels[1])?;
        println!(
            "{}",
            serde_json::json!({"kind":"cell", "pair":pair, "data":second})
        );
        let (base, candidate) = if pair % 2 == 0 {
            (&first, &second)
        } else {
            (&second, &first)
        };
        let ratio = base.warmed.seconds / candidate.warmed.seconds;
        ratios.push(ratio);
        eprintln!(
            "pair {pair}: candidate/baseline {ratio:.3}x; baseline load {:?}→{:?}; candidate load {:?}→{:?}",
            base.warmed.load_before,
            base.warmed.load_after,
            candidate.warmed.load_before,
            candidate.warmed.load_after
        );
    }
    let median = median(&mut ratios);
    println!(
        "{}",
        serde_json::json!({"kind":"summary", "candidate_over_baseline_median":median, "min":ratios[0], "max":ratios[ratios.len()-1], "ratios":ratios})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn medians_handle_odd_even_and_single_pairs() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&mut [4.0, 1.0]), 2.5);
        assert_eq!(median(&mut [2.0]), 2.0);
    }

    #[test]
    fn thread_count_respects_ort_range() {
        for threads in [1, i32::MAX as usize] {
            assert!(config(&format!("q8:2:{threads}:16:32")).is_ok());
        }
        for threads in [0, i32::MAX as usize + 1, usize::MAX] {
            assert!(config(&format!("q8:2:{threads}:16:32")).is_err());
        }
    }

    #[test]
    fn configurations_validate_every_count() {
        assert!(config("q8:2:4:16:32").is_ok());
        assert!(config("fp32:1:auto:16:16").is_ok());
        for input in [
            "q8:0:4:16:32",
            "q8:2:0:16:32",
            "q8:2:4:0:32",
            "q8:2:4:16:0",
            "q8:2:4:16",
            "fp16:1:4:16:16",
        ] {
            assert!(config(input).is_err());
        }
    }
}
