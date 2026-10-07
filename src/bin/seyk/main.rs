//! `seyk`: a thin command-line front end for the `seyk` crate.

#[cfg(feature = "mcp")]
mod mcp;

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use seyk::embedder::gemma::{self, GemmaConfig, GemmaEmbedder, ModelSource, Precision};
use seyk::{
    CacheStats, Dimension, DiscoveryOptions, EmbeddingCache, IndexReport, MemoryCache,
    ProgressEvent, ResourceLocator, SearchEngine, SearchHit, SearchResults, SqliteCache, SyncStats,
    expand_tilde,
};

#[derive(Parser)]
#[command(
    name = "seyk",
    version,
    about = "Local semantic search over your files with EmbeddingGemma 2"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Clone)]
struct ModelArgs {
    /// Model directory, or Hugging Face repo as REPO[@REVISION]. Defaults to the pinned ONNX export.
    #[arg(long, env = "SEYK_MODEL", global = true)]
    model: Option<String>,

    /// Embedding dimension: 768, 512, 256 or 128. Queries and cache entries always share it.
    #[arg(long, default_value = "256", global = true)]
    dimension: Dimension,

    /// Weight precision: q8 (smaller, faster) or fp32.
    #[arg(long, default_value = "q8", global = true)]
    precision: Precision,

    /// Do not load the vision encoder; image files are skipped.
    #[arg(long, global = true)]
    text_only: bool,

    /// Cache file. Defaults to the platform cache directory.
    #[arg(long, env = "SEYK_CACHE", global = true)]
    cache_path: Option<PathBuf>,
}

#[derive(Args, Clone)]
struct FilterArgs {
    /// Only search files matching this glob (repeatable). Without `/`, matches the file name.
    #[arg(long, value_name = "GLOB")]
    include: Vec<String>,

    /// Skip files or directories matching this glob (repeatable).
    #[arg(long, value_name = "GLOB")]
    exclude: Vec<String>,

    /// Also visit hidden files and directories.
    #[arg(long)]
    hidden: bool,
}

impl FilterArgs {
    fn discovery(&self) -> DiscoveryOptions {
        DiscoveryOptions {
            include: self.include.clone(),
            exclude: self.exclude.clone(),
            include_hidden: self.hidden,
            ..Default::default()
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Search files and directories with a natural-language query.
    Search {
        /// What you are looking for, in your own words.
        query: String,
        /// Another phrasing of the same need (repeatable). Each result keeps its best score.
        #[arg(long = "also", short = 'q', value_name = "QUERY")]
        also: Vec<String>,
        /// Files or directories to search. Defaults to the current directory.
        paths: Vec<PathBuf>,
        /// Maximum number of results.
        #[arg(long, short = 'n', default_value_t = 10)]
        limit: usize,
        /// Hide results scoring below this cosine similarity.
        #[arg(long)]
        min_score: Option<f32>,
        /// Embed everything in memory and keep nothing on disk.
        #[arg(long)]
        no_cache: bool,
        /// Print results as JSON.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        filter: FilterArgs,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Embed files ahead of time so later searches are fast. Optional.
    Index {
        paths: Vec<PathBuf>,
        #[arg(long)]
        json: bool,
        /// Keep running and re-index whenever files change.
        #[cfg(feature = "watch")]
        #[arg(long)]
        watch: bool,
        #[command(flatten)]
        filter: FilterArgs,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Inspect or clear the embedding cache.
    Cache {
        #[command(subcommand)]
        action: CacheAction,
        #[arg(long, env = "SEYK_CACHE", global = true)]
        cache_path: Option<PathBuf>,
    },
    /// Serve semantic search to agents over MCP (stdio). The model stays loaded.
    #[cfg(feature = "mcp")]
    Mcp {
        /// Folders the server may search. Defaults to the current directory.
        roots: Vec<PathBuf>,
        /// Do not index in the background; each search embeds what it needs first.
        #[arg(long)]
        no_watch: bool,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Download or inspect the model files.
    Model {
        #[command(subcommand)]
        action: ModelAction,
        #[command(flatten)]
        model: ModelArgs,
    },
}

#[derive(Subcommand)]
enum CacheAction {
    /// Show what the cache holds.
    Stats {
        #[arg(long)]
        json: bool,
    },
    /// Delete every cached embedding.
    Clear,
}

#[derive(Subcommand)]
enum ModelAction {
    /// Download the model weights from Hugging Face into the local HF cache.
    Download,
    /// Show where the model files are and whether they are present.
    Info,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Command::Search {
            query,
            also,
            paths,
            limit,
            min_score,
            no_cache,
            json,
            filter,
            model,
        } => {
            let engine = engine(&model, no_cache)?;
            let paths = default_paths(paths)?;
            let progress = Progress::new(!json);
            let report = |e: ProgressEvent| progress.event(e);
            let mut req = also
                .into_iter()
                .fold(engine.search(&query), |r, q| r.query(q))
                .paths(&paths)
                .limit(limit)
                .discovery(filter.discovery())
                .progress(&report);
            if let Some(s) = min_score {
                req = req.min_score(s);
            }
            let results = req.execute()?;
            progress.finish();
            let mut out = std::io::stdout().lock();
            if json {
                serde_json::to_writer_pretty(&mut out, &results)?;
                writeln!(out)?;
            } else {
                write!(
                    out,
                    "{}",
                    format_results(&results, &cwd(), color(&std::io::stdout()))
                )?;
                eprint!("{}", format_summary(&results));
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Index {
            paths,
            json,
            #[cfg(feature = "watch")]
            watch,
            filter,
            model,
        } => {
            let engine = engine(&model, false)?;
            let paths = default_paths(paths)?;
            #[cfg(feature = "watch")]
            if watch {
                return watch_and_index(&engine, &paths, &filter.discovery(), json);
            }
            let progress = Progress::new(true);
            let report = |e: ProgressEvent| progress.event(e);
            let result = engine
                .index()
                .paths(&paths)
                .discovery(filter.discovery())
                .progress(&report)
                .execute()?;
            progress.finish();
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print!("{}", format_index_report(&result));
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Cache { action, cache_path } => {
            let mut cache = open_cache(cache_path.as_deref())?;
            match action {
                CacheAction::Stats { json } => {
                    let stats = cache.stats()?;
                    if json {
                        println!("{}", serde_json::to_string_pretty(&stats)?);
                    } else {
                        print!("{}", format_cache_stats(&stats));
                    }
                }
                CacheAction::Clear => {
                    cache.clear()?;
                    println!("Cleared {}", cache.path().display());
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        #[cfg(feature = "mcp")]
        Command::Mcp {
            roots,
            no_watch,
            model,
        } => {
            let roots = default_paths(roots)?;
            mcp::serve(roots, Box::new(move || engine(&model, false)), !no_watch)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Model { action, model } => {
            let config = gemma_config(&model)?;
            match action {
                ModelAction::Download => {
                    let dir = gemma::download(&config)?;
                    println!("Model files are in {}", dir.display());
                }
                ModelAction::Info => match gemma::locate(&config) {
                    Ok(dir) => println!(
                        "present: {}\nprecision: {}",
                        dir.display(),
                        config.precision.name()
                    ),
                    Err(e) => {
                        println!("{e}");
                        return Ok(ExitCode::FAILURE);
                    }
                },
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Runs until interrupted. Each pass prints one line (or one JSON object).
#[cfg(feature = "watch")]
fn watch_and_index(
    engine: &SearchEngine,
    paths: &[PathBuf],
    discovery: &DiscoveryOptions,
    json: bool,
) -> Result<ExitCode> {
    use seyk::watch::{WatchOptions, watch};
    let progress = Progress::new(!json);
    let report = |e: ProgressEvent| progress.event(e);
    if !json {
        eprintln!(
            "Watching {} for changes. Ctrl-C to stop.",
            paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let mut first = true;
    watch(
        engine,
        paths,
        discovery,
        &WatchOptions::default(),
        Some(&report),
        |pass| {
            progress.finish();
            let quiet_ok = !std::mem::take(&mut first);
            match pass {
                Ok(r) if json => println!("{}", serde_json::to_string(&r).unwrap_or_default()),
                // Quiet when a burst of events changed nothing we index.
                Ok(r)
                    if quiet_ok
                        && r.sync.resources_updated == 0
                        && r.resources_removed == 0
                        && r.sync.skipped.is_empty() => {}
                Ok(r) => print!("{}", format_index_report(&r)),
                Err(e) => eprintln!("error: {e}"),
            }
        },
    )?;
    Ok(ExitCode::SUCCESS)
}

fn gemma_config(args: &ModelArgs) -> Result<GemmaConfig> {
    let source = match &args.model {
        None => ModelSource::default(),
        Some(m) => {
            let p = expand_tilde(Path::new(m));
            if p.is_dir() {
                ModelSource::Directory(p)
            } else if m.contains('/') && !m.starts_with(['.', '/', '~']) {
                let (repo, revision) = m.split_once('@').unwrap_or((m.as_str(), "main"));
                ModelSource::HuggingFace {
                    repo: repo.into(),
                    revision: revision.into(),
                }
            } else {
                bail!("--model {m}: not a directory or a Hugging Face repo id");
            }
        }
    };
    Ok(GemmaConfig {
        source,
        dimension: args.dimension,
        precision: args.precision,
        vision: !args.text_only,
        ..Default::default()
    })
}

fn engine(args: &ModelArgs, ephemeral: bool) -> Result<SearchEngine> {
    let embedder = GemmaEmbedder::load(gemma_config(args)?)?;
    let builder = SearchEngine::builder().embedder(embedder);
    let builder = if ephemeral {
        builder.cache(MemoryCache::new())
    } else {
        builder.cache(open_cache(args.cache_path.as_deref())?)
    };
    Ok(builder.build()?)
}

fn open_cache(path: Option<&Path>) -> Result<SqliteCache> {
    let path = match path {
        Some(p) => expand_tilde(p),
        None => SqliteCache::default_path()?,
    };
    SqliteCache::open(&path).with_context(|| format!("opening cache {}", path.display()))
}

fn default_paths(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    Ok(if paths.is_empty() { vec![cwd()] } else { paths })
}

fn cwd() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn color(stream: &impl IsTerminal) -> bool {
    stream.is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

/// Progress on stderr, only when it is a terminal.
struct Progress {
    enabled: bool,
}

impl Progress {
    fn new(wanted: bool) -> Self {
        Self {
            enabled: wanted && std::io::stderr().is_terminal(),
        }
    }

    fn event(&self, e: ProgressEvent) {
        if !self.enabled {
            return;
        }
        if let ProgressEvent::Embedding { done, total } = e
            && total > 0
        {
            eprint!("\r\x1b[2Kembedding {done}/{total} segments");
        }
    }

    fn finish(&self) {
        if self.enabled {
            eprint!("\r\x1b[2K");
        }
    }
}

/// Show a path relative to `base` when it is inside it, else with `~` for the home directory.
fn display_path(path: &Path, base: &Path) -> String {
    if let Ok(rel) = path.strip_prefix(base) {
        return rel.display().to_string();
    }
    if let Some(home) = dirs::home_dir()
        && let Ok(rel) = path.strip_prefix(&home)
    {
        return format!("~/{}", rel.display());
    }
    path.display().to_string()
}

fn format_locator(locator: &ResourceLocator) -> String {
    match locator {
        ResourceLocator::Text {
            line_start: Some(a),
            line_end: Some(b),
            ..
        } if a == b => format!(":{a}"),
        ResourceLocator::Text {
            line_start: Some(a),
            line_end: Some(b),
            ..
        } => format!(":{a}-{b}"),
        ResourceLocator::Text {
            byte_start,
            byte_end,
            ..
        } => format!(" bytes {byte_start}-{byte_end}"),
        ResourceLocator::Page { page } => format!(" page {page}"),
        ResourceLocator::Time { start_ms, end_ms } => {
            format!(" @ {}–{}", clock(*start_ms), clock(*end_ms))
        }
        ResourceLocator::WholeResource => String::new(),
    }
}

fn clock(ms: u64) -> String {
    let s = ms / 1000;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

/// `via` names the query that matched, shown only when several were searched.
fn format_hit(hit: &SearchHit, via: Option<&str>, base: &Path, color: bool) -> String {
    let (bold, dim, reset) = if color {
        ("\x1b[1m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    let mut s = format!(
        "{:.3}  {bold}{}{}{reset}",
        hit.score,
        display_path(hit.resource.path(), base),
        format_locator(&hit.locator)
    );
    if let Some(h) = &hit.heading {
        s.push_str(&format!("  {dim}§ {h}{reset}"));
    }
    if let Some(q) = via {
        s.push_str(&format!("  {dim}← \"{q}\"{reset}"));
    }
    s.push('\n');
    if let Some(p) = &hit.preview {
        s.push_str(&format!("       {dim}{p}{reset}\n"));
    }
    s
}

fn format_results(results: &SearchResults, base: &Path, color: bool) -> String {
    if results.hits.is_empty() {
        return "No results.\n".to_string();
    }
    results
        .hits
        .iter()
        .map(|h| {
            let via = (results.queries.len() > 1)
                .then(|| results.queries.get(h.query_index))
                .flatten()
                .map(String::as_str);
            format_hit(h, via, base, color)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_sync(s: &SyncStats) -> String {
    let mut out = format!(
        "{} files ({} unchanged, {} new or changed), {} segments embedded",
        s.resources_discovered, s.resources_unchanged, s.resources_updated, s.segments_embedded
    );
    if !s.skipped.is_empty() {
        out.push_str(&format!(", {} skipped", s.skipped.len()));
    }
    out
}

fn format_summary(r: &SearchResults) -> String {
    let mut out = format!(
        "\nSearched {} segments: {} in {:.1}s\n",
        r.stats.segments_searched,
        format_sync(&r.stats.sync),
        r.stats.total_ms as f64 / 1000.0
    );
    for s in r.stats.sync.skipped.iter().take(5) {
        out.push_str(&format!("  skipped {}: {}\n", s.path.display(), s.reason));
    }
    out
}

fn format_index_report(r: &IndexReport) -> String {
    let mut out = format!(
        "Indexed {} in {:.1}s",
        format_sync(&r.sync),
        r.total_ms as f64 / 1000.0
    );
    if r.resources_removed > 0 {
        out.push_str(&format!(
            "; removed {} deleted files from the cache",
            r.resources_removed
        ));
    }
    out.push('\n');
    for s in &r.sync.skipped {
        out.push_str(&format!("  skipped {}: {}\n", s.path.display(), s.reason));
    }
    out
}

fn format_cache_stats(stats: &CacheStats) -> String {
    let mut out = String::new();
    if let Some(p) = &stats.location {
        out.push_str(&format!("Cache: {}\n", p.display()));
    }
    if let Some(b) = stats.size_bytes {
        out.push_str(&format!("Size:  {:.1} MB\n", b as f64 / 1_048_576.0));
    }
    if stats.profiles.is_empty() {
        out.push_str("Empty.\n");
    }
    for p in &stats.profiles {
        out.push_str(&format!("\nProfile {}\n", p.profile_id));
        if let Some(prof) = &p.profile {
            out.push_str(&format!(
                "  model:     {} ({}, {})\n  dimension: {}\n  segmenter: {}\n",
                prof.model.model_id,
                prof.model.revision,
                prof.model.variant,
                prof.model.dimension,
                prof.segmenter_version
            ));
        }
        out.push_str(&format!(
            "  {} files, {} segments, {} vectors\n",
            p.resources, p.segments, p.embeddings
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use seyk::{Modality, ResourceId, ResourceMetadata, SearchStats};

    fn hit(path: &str, score: f32, locator: ResourceLocator, preview: Option<&str>) -> SearchHit {
        SearchHit {
            resource: ResourceId::from_path(path),
            locator,
            score,
            metadata: ResourceMetadata {
                modality: Modality::Text,
                size_bytes: 10,
                modified_ns: None,
                content_hash: "h".into(),
            },
            heading: None,
            preview: preview.map(str::to_string),
            query_index: 0,
        }
    }

    fn results() -> SearchResults {
        SearchResults {
            queries: vec!["father".into()],
            hits: vec![
                hit(
                    "/book/characters/father.md",
                    0.9137,
                    ResourceLocator::Text {
                        byte_start: 0,
                        byte_end: 9,
                        line_start: Some(81),
                        line_end: Some(96),
                    },
                    Some("He notices himself repeating the same gestures..."),
                ),
                hit(
                    "/book/photos/kitchen.jpg",
                    0.5,
                    ResourceLocator::WholeResource,
                    None,
                ),
            ],
            stats: SearchStats::default(),
        }
    }

    #[test]
    fn human_output() {
        let out = format_results(&results(), Path::new("/book"), false);
        assert_eq!(
            out,
            "0.914  characters/father.md:81-96\n       He notices himself repeating the same gestures...\n\n0.500  photos/kitchen.jpg\n"
        );
    }

    #[test]
    fn names_matching_query_when_several() {
        let mut r = results();
        r.queries.push("dad".into());
        r.hits[1].query_index = 1;
        let out = format_results(&r, Path::new("/book"), false);
        assert!(out.contains("father.md:81-96  ← \"father\"\n"), "{out}");
        assert!(out.contains("kitchen.jpg  ← \"dad\"\n"), "{out}");
    }

    #[test]
    fn json_output_mirrors_library_types() {
        let v: serde_json::Value = serde_json::to_value(results()).unwrap();
        let first = &v["hits"][0];
        assert_eq!(first["resource"], "/book/characters/father.md");
        assert_eq!(first["locator"]["kind"], "text");
        assert_eq!(first["locator"]["line_start"], 81);
        assert_eq!(v["hits"][1]["locator"]["kind"], "whole_resource");
        let back: SearchResults = serde_json::from_value(v).unwrap();
        assert_eq!(back, results());
    }

    #[test]
    fn time_locator() {
        assert_eq!(
            format_locator(&ResourceLocator::Time {
                start_ms: 2_530_000,
                end_ms: 2_585_000
            }),
            " @ 00:42:10–00:43:05"
        );
    }
}
