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
//! would not have surfaced under this overlap rule. Unavailable extractions are
//! reported separately and excluded from the overlap denominator. Uses the
//! persistent cache unless `--no-cache`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use seyk::discovery::{ResourceKind, discover};
use seyk::embedder::gemma::{GemmaConfig, GemmaEmbedder, ModelSource, Precision};
use seyk::segment::TextSegmenter;
use seyk::{
    Dimension, DiscoveryOptions, MemoryCache, ResourceLocator, SearchEngine, SearchHit, SearchMode,
    SqliteCache, expand_tilde, make_preview,
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

/// Crude stemming: compare the first five characters, so "gardens" matches
/// "garden" and "memories" matches "memory". Generous on purpose, so that
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

/// Only verified extraction is an observation; even empty text can be observed.
fn hit_text(engine: &SearchEngine, hit: &SearchHit) -> seyk::Result<String> {
    engine.extract(hit).map(|content| content.text)
}

#[derive(Debug, Default, PartialEq, Eq)]
struct OverlapCounts {
    evaluable: usize,
    no_overlap: usize,
    unavailable: usize,
}

impl OverlapCounts {
    fn observe(&mut self, text: seyk::Result<String>, stems: &[&str]) -> String {
        match text {
            Ok(text) => {
                self.evaluable += 1;
                let overlap = keyword_score(stems, &stems_of(&text));
                if overlap == 0 {
                    self.no_overlap += 1;
                    "  [no keyword overlap]".into()
                } else {
                    format!("  [{overlap}/{} words]", stems.len())
                }
            }
            Err(reason) => {
                self.unavailable += 1;
                format!("  [keyword overlap unavailable: {reason}]")
            }
        }
    }

    fn add(&mut self, other: &Self) {
        self.evaluable += other.evaluable;
        self.no_overlap += other.no_overlap;
        self.unavailable += other.unavailable;
    }

    fn summary(&self) -> String {
        let overlap = if self.evaluable == 0 {
            "no evaluable semantic hits (keyword overlap not measured)".into()
        } else {
            format!(
                "{} of {} evaluable semantic hits had no keyword overlap",
                self.no_overlap, self.evaluable
            )
        };
        format!("{overlap}; {} unavailable", self.unavailable)
    }
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

    let embedder = GemmaEmbedder::load({
        let mut config = GemmaConfig::default();
        config.precision = args.precision;
        config.source = args
            .model_dir
            .clone()
            .map_or_else(ModelSource::default, ModelSource::Directory);
        config.dimension = args.dimension;
        config.vision = false;
        config
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
        eprintln!("skipped {}: {}", s.resource, s.reason);
    }

    let segments = segment_corpus(&args.paths, &TextSegmenter::default());
    let mut totals = OverlapCounts::default();

    for (qi, query) in queries.iter().enumerate() {
        let content = content_words(query);
        let stems: Vec<&str> = content.iter().map(|w| stem(w)).collect();
        println!("## Q{}: {query}", qi + 1);
        println!("content words: {}\n", content.join(", "));

        let results = engine
            .search(query)
            .mode(SearchMode::Semantic)
            .paths(&args.paths)
            .limit(args.limit)
            .previews(Some(PREVIEW_CHARS))
            .execute()?;
        println!("semantic (top {}):", args.limit);
        let mut semantic_keys = HashSet::new();
        let mut counts = OverlapCounts::default();
        for (i, hit) in results.hits.iter().enumerate() {
            let path = hit.resource.to_file_path().expect("file resource");
            let mark = counts.observe(hit_text(&engine, hit), &stems);
            semantic_keys.insert((path.to_path_buf(), lines_of(&hit.locator)));
            println!(
                "  {}. {:.3}  {}{}{mark}",
                i + 1,
                hit.score,
                display_path(&path, &cwd),
                lines_of(&hit.locator)
            );
            if let Some(p) = &hit.preview {
                println!("       {p}");
            }
        }
        totals.add(&counts);

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
            "\n{}; {shared} also in the keyword top {}.\n",
            counts.summary(),
            args.limit
        );
    }

    println!("summary: {} queries, {}.", queries.len(), totals.summary());
    Ok(())
}

#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod common;

#[cfg(test)]
mod tests {
    use super::*;
    use seyk::{Error, Modality, ResourceId, ResourceMetadata};

    fn engine() -> SearchEngine {
        SearchEngine::builder()
            .embedder(common::FakeEmbedder::new(Dimension::D128, "eval-overlap"))
            .build()
            .unwrap()
    }

    fn locator(start: u64, end: u64) -> ResourceLocator {
        ResourceLocator::Text {
            byte_start: start,
            byte_end: end,
            line_start: None,
            line_end: None,
        }
    }

    fn hit(path: &Path, text: &str) -> SearchHit {
        SearchHit {
            resource: ResourceId::from_path(path).unwrap(),
            locator: locator(0, text.len() as u64),
            score: 1.0,
            signals: Default::default(),
            r#match: Default::default(),
            metadata: ResourceMetadata {
                media_type: Some("text/plain".into()),
                modality: Modality::Text,
                size_bytes: text.len() as u64,
                modified_ns: None,
                content_hash: blake3::hash(text.as_bytes()).to_hex().to_string(),
            },
            heading: None,
            preview: None,
            query_index: 0,
        }
    }

    #[test]
    fn empty_text_is_observed_but_unsupported_locator_is_unavailable() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("empty.txt");
        std::fs::write(&path, "").unwrap();
        let engine = engine();
        let mut hit = hit(&path, "");
        let mut counts = OverlapCounts::default();
        assert_eq!(hit_text(&engine, &hit).unwrap(), "");
        assert_eq!(
            counts.observe(hit_text(&engine, &hit), &["alpha"]),
            "  [no keyword overlap]"
        );
        hit.locator = ResourceLocator::WholeResource;
        let unavailable = hit_text(&engine, &hit);
        assert!(matches!(
            unavailable,
            Err(Error::UnsupportedExtraction { .. })
        ));
        assert!(
            counts
                .observe(unavailable, &["alpha"])
                .contains("unavailable:")
        );
        assert_eq!(
            counts,
            OverlapCounts {
                evaluable: 1,
                no_overlap: 1,
                unavailable: 1
            }
        );
    }

    #[test]
    fn read_failure_and_stale_content_are_unavailable() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("note.txt");
        let engine = engine();
        let hit = hit(&path, "alpha");
        let mut counts = OverlapCounts::default();
        let missing = hit_text(&engine, &hit);
        assert!(matches!(missing, Err(Error::Io { .. })));
        counts.observe(missing, &["alpha"]);
        std::fs::write(&path, "bravo").unwrap();
        let stale = hit_text(&engine, &hit);
        assert!(matches!(stale, Err(Error::StaleResource { .. })));
        counts.observe(stale, &["alpha"]);
        assert_eq!(
            counts,
            OverlapCounts {
                evaluable: 0,
                no_overlap: 0,
                unavailable: 2
            }
        );
        assert_eq!(
            counts.summary(),
            "no evaluable semantic hits (keyword overlap not measured); 2 unavailable"
        );
    }

    #[test]
    fn invalid_ranges_are_unavailable_instead_of_truncated() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("note.txt");
        let text = "a🙂b";
        std::fs::write(&path, text).unwrap();
        let engine = engine();
        let mut hit = hit(&path, text);
        let mut counts = OverlapCounts::default();
        for range in [locator(2, 1), locator(0, 99), locator(2, 5)] {
            hit.locator = range;
            let invalid = hit_text(&engine, &hit);
            assert!(matches!(invalid, Err(Error::InvalidLocator { .. })));
            counts.observe(invalid, &["alpha"]);
        }
        assert_eq!(
            counts,
            OverlapCounts {
                evaluable: 0,
                no_overlap: 0,
                unavailable: 3
            }
        );
    }

    #[test]
    fn aggregate_counts_exclude_unavailable_observations() {
        let mut first = OverlapCounts::default();
        assert_eq!(
            first.observe(Ok("alpha".into()), &["alpha"]),
            "  [1/1 words]"
        );
        first.observe(Ok("bravo".into()), &["alpha"]);
        let mut second = OverlapCounts::default();
        second.observe(Ok(String::new()), &["alpha"]);
        let mark = second.observe(Err(Error::Config("extraction failed".into())), &["alpha"]);
        assert_eq!(
            mark,
            "  [keyword overlap unavailable: invalid configuration: extraction failed]"
        );
        let mut totals = OverlapCounts::default();
        totals.add(&first);
        totals.add(&second);
        assert_eq!(
            totals,
            OverlapCounts {
                evaluable: 3,
                no_overlap: 2,
                unavailable: 1
            }
        );
        assert_eq!(
            totals.summary(),
            "2 of 3 evaluable semantic hits had no keyword overlap; 1 unavailable"
        );
    }

    #[test]
    fn no_hits_does_not_claim_an_overlap_measurement() {
        assert_eq!(
            OverlapCounts::default().summary(),
            "no evaluable semantic hits (keyword overlap not measured); 0 unavailable"
        );
    }
}
