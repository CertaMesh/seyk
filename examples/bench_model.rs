//! End-to-end measurements with the real EmbeddingGemma 2 model.
//!
//! ```text
//! cargo run --release --example bench_model -- [--precision q8|fp32] [--model-dir DIR] [--threads N]
//!     [--segments 64] [--batch-sizes 1,8,16,32] [--corpus 200,1000]
//! ```
//!
//! Peak RSS is a process-wide high-water mark, so run once per precision
//! to get a separate memory figure for each. Model files must already be in
//! the Hugging Face cache (`seyk model download`) or `--model-dir`;
//! nothing is fetched here.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use seyk::embedder::gemma::{GemmaConfig, GemmaEmbedder, ModelSource, Precision};
use seyk::{Embedder, ResourceInput, SearchEngine, SqliteCache};

struct Args {
    precision: Precision,
    model_dir: Option<PathBuf>,
    threads: Option<usize>,
    segments: usize,
    batch_sizes: Vec<usize>,
    corpus: Vec<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        precision: Precision::Q8,
        model_dir: None,
        threads: None,
        segments: 64,
        batch_sizes: vec![1, 8, 16, 32],
        corpus: vec![200, 1000],
    };
    let list = |v: String| -> Result<Vec<usize>, String> {
        v.split(',')
            .map(|n| n.trim().parse().map_err(|_| format!("bad number `{n}`")))
            .collect()
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--precision" => args.precision = value()?.parse().map_err(|e| format!("{e}"))?,
            "--model-dir" => args.model_dir = Some(PathBuf::from(value()?)),
            "--threads" => args.threads = Some(value()?.parse().map_err(|_| "bad --threads")?),
            "--segments" => args.segments = value()?.parse().map_err(|_| "bad --segments")?,
            "--batch-sizes" => args.batch_sizes = list(value()?)?,
            "--corpus" => args.corpus = list(value()?)?,
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(args)
}

/// Peak resident set size of this process so far, in MiB.
#[cfg(unix)]
fn peak_rss_mib() -> Option<f64> {
    let mut u = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: a writable rusage for getrusage to fill.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, u.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: getrusage succeeded, and the struct was zeroed beforehand.
    let maxrss = unsafe { u.assume_init() }.ru_maxrss as f64;
    // macOS reports bytes, Linux kilobytes.
    let bytes = if cfg!(target_os = "macos") {
        maxrss
    } else {
        maxrss * 1024.0
    };
    Some(bytes / (1024.0 * 1024.0))
}

#[cfg(not(unix))]
fn peak_rss_mib() -> Option<f64> {
    None
}

fn rss() -> String {
    peak_rss_mib().map_or("n/a".into(), |m| format!("{m:.0} MiB"))
}

/// 1, 5 and 15 minute load averages. Timings on a busy machine are not
/// comparable, so every row records how busy it was.
#[cfg(unix)]
fn load_average() -> String {
    let mut l = [0.0f64; 3];
    // SAFETY: the buffer holds the three values requested.
    if unsafe { libc::getloadavg(l.as_mut_ptr(), 3) } != 3 {
        return "n/a".into();
    }
    format!("{:.1} {:.1} {:.1}", l[0], l[1], l[2])
}

#[cfg(not(unix))]
fn load_average() -> String {
    "n/a".into()
}

/// Deterministic generator of plausible but meaningless notebook prose.
struct Prose(u64);

const SUBJECTS: &[&str] = &[
    "The old house",
    "My grandmother",
    "The narrator",
    "Her brother",
    "The harbour town",
    "The winter",
    "The letter",
    "His father",
    "The garden",
    "Nobody in the village",
    "The second chapter",
    "The train station",
];
const VERBS: &[&str] = &[
    "remembers",
    "avoids",
    "keeps returning to",
    "quietly rewrites",
    "cannot forgive",
    "measures everything against",
    "slowly turns into",
    "hides",
    "argues about",
    "waits for",
    "forgets",
    "listens to",
];
const OBJECTS: &[&str] = &[
    "the smell of wet stone",
    "a promise made at the kitchen table",
    "the sound of the ferry in fog",
    "an unopened box of photographs",
    "the year the mill closed",
    "a song nobody else knows",
    "the map with the wrong coastline",
    "every unanswered phone call",
    "the long walk back from the hospital",
    "a borrowed coat that never fit",
    "the silence after dinner",
    "the names carved into the bench",
];
const TAILS: &[&str] = &[
    "and nobody mentions it again",
    "while the light changes outside",
    "as if it happened to someone else",
    "before the rain starts",
    "even though the story says otherwise",
    "until the end of the summer",
    "without explaining why",
    "and the reader should feel the weight of it",
];

impl Prose {
    fn pick<'a>(&mut self, words: &[&'a str]) -> &'a str {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        words[(self.0 >> 33) as usize % words.len()]
    }

    fn paragraph(&mut self, chars: usize) -> String {
        let mut p = String::new();
        while p.len() < chars {
            let (s, v, o, t) = (
                self.pick(SUBJECTS),
                self.pick(VERBS),
                self.pick(OBJECTS),
                self.pick(TAILS),
            );
            let _ = write!(p, "{s} {v} {o} {t}. ");
        }
        p.trim_end().to_string()
    }
}

/// Paragraph length that tokenises to roughly 250 tokens with the real tokenizer.
const SEGMENT_CHARS: usize = 1200;

/// Write a Markdown corpus with about `segments` sections of one paragraph each.
fn write_corpus(dir: &Path, segments: usize, prose: &mut Prose) -> std::io::Result<()> {
    const SECTIONS_PER_FILE: usize = 10;
    for f in 0..segments.div_ceil(SECTIONS_PER_FILE) {
        let mut doc = format!("# Notes {f}\n\n");
        for s in 0..SECTIONS_PER_FILE.min(segments - f * SECTIONS_PER_FILE) {
            let _ = write!(
                doc,
                "## Scene {s}\n\n{}\n\n",
                prose.paragraph(SEGMENT_CHARS)
            );
        }
        std::fs::write(dir.join(format!("notes-{f:04}.md")), doc)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let mut prose = Prose(42);
    let mut rows: Vec<[String; 4]> = Vec::new();
    let mut row = |what: String, value: String| {
        let r = [what, value, rss(), load_average()];
        eprintln!("{} | {} | {} | {}", r[0], r[1], r[2], r[3]);
        rows.push(r);
    };

    let max_batch = args.batch_sizes.iter().copied().max().unwrap_or(1);
    let config = {
        let mut config = GemmaConfig::default();
        config.precision = args.precision;
        config.source = args
            .model_dir
            .clone()
            .map_or_else(ModelSource::default, ModelSource::Directory);
        config.threads = args.threads;
        config.vision = false;
        config.batch_size = max_batch;
        config
    };
    let started = Instant::now();
    let embedder = Arc::new(GemmaEmbedder::load(config)?);
    row(
        "model load".into(),
        format!("{:.2} s", started.elapsed().as_secs_f64()),
    );

    let texts: Vec<String> = (0..args.segments)
        .map(|_| prose.paragraph(SEGMENT_CHARS))
        .collect();
    let mut tokens = 0;
    for t in &texts {
        tokens += embedder
            .text_token_ids(&GemmaEmbedder::document_input(Some("Scene"), t))?
            .len();
    }
    let mean_tokens = tokens as f64 / texts.len().max(1) as f64;
    let inputs: Vec<ResourceInput> = texts
        .iter()
        .map(|t| ResourceInput::Text {
            title: Some("Scene"),
            text: t,
        })
        .collect();

    // Warm up ONNX Runtime (first run allocates and specialises kernels).
    embedder.embed_resources(&inputs[..inputs.len().min(max_batch)])?;
    for &b in &args.batch_sizes {
        let started = Instant::now();
        // The embedder runs at most `max_batch` inputs per model call, so
        // chunking by `b` here gives exactly `b` per call.
        for chunk in inputs.chunks(b) {
            embedder.embed_resources(chunk)?;
        }
        let secs = started.elapsed().as_secs_f64();
        row(
            format!(
                "embed {} segments (~{mean_tokens:.0} tokens), batch {b}",
                inputs.len()
            ),
            format!("{:.1} segments/s", inputs.len() as f64 / secs),
        );
    }

    let started = Instant::now();
    let runs = 20;
    for _ in 0..runs {
        embedder.embed_query("How do I prepare tomato soup?")?;
    }
    row(
        "embed query".into(),
        format!(
            "{:.1} ms",
            started.elapsed().as_secs_f64() * 1000.0 / runs as f64
        ),
    );

    for &n in &args.corpus {
        let dir = tempfile::tempdir()?;
        let corpus = dir.path().join("corpus");
        std::fs::create_dir(&corpus)?;
        write_corpus(&corpus, n, &mut prose)?;
        let engine = SearchEngine::builder()
            .shared_embedder(embedder.clone())
            .cache(SqliteCache::open(dir.path().join("cache.sqlite3"))?)
            .build()?;
        let query = "Which notes deal with somebody repeating their parents' behaviour?";

        let first = engine.search(query).paths([&corpus]).execute()?;
        let s = &first.stats;
        row(
            format!(
                "search, uncached, {} segments in {} files",
                s.segments_searched, s.sync.resources_discovered
            ),
            format!(
                "{:.2} s total ({:.2} s embedding, {:.1} segments/s)",
                s.total_ms as f64 / 1000.0,
                s.sync.embed_ms as f64 / 1000.0,
                s.sync.segments_embedded as f64 / (s.sync.embed_ms.max(1) as f64 / 1000.0)
            ),
        );

        let started = Instant::now();
        let second = engine.search(query).paths([&corpus]).execute()?;
        assert_eq!(
            second.stats.sync.segments_embedded, 0,
            "second run hit the model"
        );
        row(
            format!(
                "search, cached, {} segments",
                second.stats.segments_searched
            ),
            format!("{:.1} ms", started.elapsed().as_secs_f64() * 1000.0),
        );
    }

    println!(
        "\nEmbeddingGemma 2, {} precision, dimension {}, threads {}\n",
        args.precision.name(),
        embedder.dimension(),
        args.threads.map_or("auto".into(), |t| t.to_string())
    );
    println!("| Measurement | Result | Peak RSS so far | Load avg 1/5/15 min |");
    println!("|---|---|---|---|");
    for [a, b, c, d] in rows {
        println!("| {a} | {b} | {c} | {d} |");
    }
    Ok(())
}
