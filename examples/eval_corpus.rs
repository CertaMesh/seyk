//! Compare semantic search with a naive keyword baseline on a real corpus.
//!
//! ```text
//! cargo run --release --example eval_corpus -- --queries queries.txt [--limit 5]
//!     [--precision q8|fp32] [--model-dir DIR] [--dimension 256] [--no-cache | --cache FILE] PATHS...
//! ```
//!
//! For every query (one per line, `#` starts a comment) it prints the top
//! semantic hits and the top keyword hits. Semantic hits that share no content
//! word with the query are marked: those are the passages a keyword search
//! would not have surfaced. Uses the persistent cache unless `--no-cache`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use gemma_search::discovery::{ResourceKind, discover};
use gemma_search::embedder::gemma::{GemmaConfig, GemmaEmbedder, ModelSource, Precision};
use gemma_search::segment::TextSegmenter;
use gemma_search::{
    Dimension, DiscoveryOptions, MemoryCache, ResourceLocator, SearchEngine, SqliteCache,
    expand_tilde, make_preview,
};

const PREVIEW_CHARS: usize = 160;

/// Words that carry no topic: function words plus the phrasing people use
/// when asking about their own notes ("where did I write about ...").
const STOPWORDS: &[&str] = &[
    "a",
    "about",
    "after",
    "again",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "more",
    "most",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "other",
    "our",
    "out",
    "she",
    "should",
    "so",
    "some",
    "something",
    "somebody",
    "someone",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    // Meta phrasing about the notes themselves.
    "write",
    "wrote",
    "written",
    "describe",
    "described",
    "discuss",
    "discussed",
    "mention",
    "mentioned",
    "note",
    "notes",
    "idea",
    "ideas",
    "deal",
    "deals",
    "talk",
    "talked",
    "find",
];

struct Args {
    queries: PathBuf,
    limit: usize,
    precision: Precision,
    model_dir: Option<PathBuf>,
    dimension: Dimension,
    no_cache: bool,
    cache: Option<PathBuf>,
    paths: Vec<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut queries = None;
    let mut args = Args {
        queries: PathBuf::new(),
        limit: 5,
        precision: Precision::Q8,
        model_dir: None,
        dimension: Dimension::default(),
        no_cache: false,
        cache: None,
        paths: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--queries" => queries = Some(PathBuf::from(value()?)),
            "--limit" => args.limit = value()?.parse().map_err(|_| "bad --limit")?,
            "--precision" => args.precision = value()?.parse().map_err(|e| format!("{e}"))?,
            "--model-dir" => args.model_dir = Some(PathBuf::from(value()?)),
            "--dimension" => args.dimension = value()?.parse().map_err(|e| format!("{e}"))?,
            "--no-cache" => args.no_cache = true,
            "--cache" => args.cache = Some(PathBuf::from(value()?)),
            flag if flag.starts_with("--") => return Err(format!("unknown flag `{flag}`")),
            path => args.paths.push(expand_tilde(Path::new(path))),
        }
    }
    args.queries = queries.ok_or("--queries FILE is required")?;
    if args.paths.is_empty() {
        return Err("give at least one path to search".into());
    }
    Ok(args)
}

/// Lowercase alphanumeric words.
fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
}

fn content_words(query: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    words(query)
        .filter(|w| w.len() > 1 && !STOPWORDS.contains(&w.as_str()))
        .filter(|w| seen.insert(w.clone()))
        .collect()
}

/// Crude stemming: compare the first five characters, so "fathers" matches
/// "father" and "memories" matches "memory". Generous on purpose, so that
/// "no keyword overlap" is a conservative label.
fn stem(word: &str) -> &str {
    match word.char_indices().nth(5) {
        Some((i, _)) => &word[..i],
        None => word,
    }
}

fn stems_of(text: &str) -> HashSet<String> {
    words(text).map(|w| stem(&w).to_string()).collect()
}

/// How many distinct query content words occur among `present` stems.
fn keyword_score(stems: &[&str], present: &HashSet<String>) -> usize {
    stems.iter().filter(|s| present.contains(**s)).count()
}

/// One text segment as the engine sees it, for the keyword baseline.
struct Segment {
    path: PathBuf,
    lines: (u64, u64),
    text: String,
    /// Computed once, since every query scores every segment.
    stems: HashSet<String>,
}

fn segment_corpus(paths: &[PathBuf], segmenter: &TextSegmenter) -> Vec<Segment> {
    let resources = match discover(paths, &DiscoveryOptions::default()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("keyword baseline: discovery failed: {e}");
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    for r in resources {
        let ResourceKind::Text(format) = r.kind else {
            continue;
        };
        let Ok(source) = std::fs::read_to_string(&r.path) else {
            continue;
        };
        out.extend(
            segmenter
                .segment(&source, format)
                .into_iter()
                .map(|s| Segment {
                    path: r.path.clone(),
                    lines: (s.line_start, s.line_end),
                    stems: stems_of(&s.text),
                    text: s.text,
                }),
        );
    }
    out
}

/// The text of a hit, read back from the file (the cache stores no content).
fn hit_text(
    path: &Path,
    locator: &ResourceLocator,
    files: &mut HashMap<PathBuf, Vec<u8>>,
) -> String {
    let ResourceLocator::Text {
        byte_start,
        byte_end,
        ..
    } = locator
    else {
        return String::new();
    };
    let bytes = files
        .entry(path.to_path_buf())
        .or_insert_with(|| std::fs::read(path).unwrap_or_default());
    let end = (*byte_end as usize).min(bytes.len());
    let start = (*byte_start as usize).min(end);
    String::from_utf8_lossy(&bytes[start..end]).into_owned()
}

fn display_path(path: &Path, cwd: &Path) -> String {
    path.strip_prefix(cwd).unwrap_or(path).display().to_string()
}

fn lines_of(locator: &ResourceLocator) -> String {
    match locator {
        ResourceLocator::Text {
            line_start: Some(a),
            line_end: Some(b),
            ..
        } => format!(":{a}-{b}"),
        _ => String::new(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| {
        format!("{e}\nusage: eval_corpus --queries FILE [--limit N] [--precision q8|fp32] [--model-dir DIR] [--dimension 256] [--no-cache | --cache FILE] PATHS...")
    })?;
    let queries: Vec<String> = std::fs::read_to_string(&args.queries)?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect();
    let cwd = std::env::current_dir()?;

    let embedder = GemmaEmbedder::load(GemmaConfig {
        precision: args.precision,
        source: args
            .model_dir
            .clone()
            .map_or_else(ModelSource::default, ModelSource::Directory),
        dimension: args.dimension,
        vision: false,
        ..Default::default()
    })?;
    let builder = SearchEngine::builder().embedder(embedder);
    let engine = if args.no_cache {
        builder.cache(MemoryCache::new()).build()?
    } else {
        let cache = match &args.cache {
            Some(path) => SqliteCache::open(path)?,
            None => SqliteCache::open_default()?,
        };
        builder.cache(cache).build()?
    };

    // Embed once up front; each search then only re-checks file metadata.
    let report = engine.index().paths(&args.paths).execute()?;
    eprintln!(
        "indexed {} files ({} embedded segments, {} unchanged files) in {:.1} s",
        report.sync.resources_discovered,
        report.sync.segments_embedded,
        report.sync.resources_unchanged,
        report.total_ms as f64 / 1000.0
    );
    for s in &report.sync.skipped {
        eprintln!("skipped {}: {}", s.path.display(), s.reason);
    }

    let segments = segment_corpus(&args.paths, &TextSegmenter::default());
    let mut files = HashMap::new();
    let mut total_no_overlap = 0;
    let mut total_hits = 0;

    for (qi, query) in queries.iter().enumerate() {
        let content = content_words(query);
        let stems: Vec<&str> = content.iter().map(|w| stem(w)).collect();
        println!("## Q{}: {query}", qi + 1);
        println!("content words: {}\n", content.join(", "));

        let results = engine
            .search(query)
            .paths(&args.paths)
            .limit(args.limit)
            .previews(Some(PREVIEW_CHARS))
            .execute()?;
        println!("semantic (top {}):", args.limit);
        let mut semantic_keys = HashSet::new();
        let mut no_overlap = 0;
        for (i, hit) in results.hits.iter().enumerate() {
            let path = hit.resource.path();
            let text = hit_text(path, &hit.locator, &mut files);
            let overlap = keyword_score(&stems, &stems_of(&text));
            let mark = if overlap == 0 {
                no_overlap += 1;
                "  [no keyword overlap]".to_string()
            } else {
                format!("  [{overlap}/{} words]", stems.len())
            };
            semantic_keys.insert((path.to_path_buf(), lines_of(&hit.locator)));
            println!(
                "  {}. {:.3}  {}{}{mark}",
                i + 1,
                hit.score,
                display_path(path, &cwd),
                lines_of(&hit.locator)
            );
            if let Some(p) = &hit.preview {
                println!("       {p}");
            }
        }
        total_no_overlap += no_overlap;
        total_hits += results.hits.len();

        let mut ranked: Vec<(usize, &Segment)> = segments
            .iter()
            .map(|s| (keyword_score(&stems, &s.stems), s))
            .filter(|(score, _)| *score > 0)
            .collect();
        // Stable sort keeps file order among ties, so output is reproducible.
        ranked.sort_by_key(|r| std::cmp::Reverse(r.0));
        println!("keyword baseline (top {}):", args.limit);
        if ranked.is_empty() {
            println!("  (no segment contains any content word)");
        }
        let mut shared = 0;
        for (i, (score, s)) in ranked.iter().take(args.limit).enumerate() {
            let lines = format!(":{}-{}", s.lines.0, s.lines.1);
            if semantic_keys.contains(&(s.path.clone(), lines.clone())) {
                shared += 1;
            }
            println!(
                "  {}. {score}/{}  {}{lines}",
                i + 1,
                stems.len(),
                display_path(&s.path, &cwd)
            );
            println!("       {}", make_preview(&s.text, PREVIEW_CHARS));
        }
        println!(
            "\n{no_overlap} of {} semantic hits share no content word with the query; {shared} also in the keyword top {}.\n",
            results.hits.len(),
            args.limit
        );
    }

    println!(
        "summary: {} queries, {total_no_overlap} of {total_hits} semantic hits had no keyword overlap.",
        queries.len()
    );
    Ok(())
}
