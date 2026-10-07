//! `seyk`: a thin command-line front end for the `seyk` crate.

#[cfg(all(feature = "mcp", unix))]
mod daemon;
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
    ProgressEvent, ResourceLocator, SearchEngine, SearchHit, SearchMode, SearchResults,
    SqliteCache, SyncStats, expand_tilde,
};

/// Machine-output protocol; legacy output had no version field.
const JSON_SCHEMA_VERSION: u32 = 2;

#[derive(serde::Serialize)]
struct JsonOutput<'a, T: serde::Serialize> {
    schema_version: u32,
    #[serde(flatten)]
    payload: &'a T,
}

fn json_output<T: serde::Serialize>(payload: &T) -> JsonOutput<'_, T> {
    JsonOutput {
        schema_version: JSON_SCHEMA_VERSION,
        payload,
    }
}

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
        /// Hide semantic matches below this cosine similarity. Keyword and file
        /// name matches are kept.
        #[arg(long)]
        min_score: Option<f32>,
        /// hybrid (meaning, keywords and file names), semantic or lexical.
        #[arg(long, default_value = "hybrid")]
        mode: SearchMode,
        /// Keyword weight against semantic (1.0) in hybrid mode. For tuning.
        #[arg(long, hide = true)]
        lexical_weight: Option<f32>,
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
    ///
    /// Sessions share one background daemon, which loads the model once and
    /// exits after a quiet period. It starts on first use.
    Mcp {
        /// Folders the server may search. Defaults to the current directory.
        roots: Vec<PathBuf>,
        /// Run the server in this process instead of the shared daemon.
        #[arg(long)]
        standalone: bool,
        /// Do not index in the background; each search embeds what it needs
        /// first. Implies --standalone.
        #[arg(long)]
        no_watch: bool,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// The shared background process behind `seyk mcp`.
    #[cfg(all(feature = "mcp", unix))]
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
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

#[cfg(all(feature = "mcp", unix))]
#[derive(Subcommand)]
enum DaemonAction {
    /// Run the daemon in the foreground. `seyk mcp` starts one automatically.
    Run {
        /// Exit after this many minutes without sessions.
        #[arg(long, default_value_t = 15)]
        idle_minutes: u64,
    },
    /// Show the running daemon, its sessions and folders.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Stop the running daemon. Connected sessions lose their search tool.
    Stop,
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
            mode,
            lexical_weight,
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
                .mode(mode)
                .paths(&paths)
                .limit(limit)
                .discovery(filter.discovery())
                .progress(&report);
            if let Some(s) = min_score {
                req = req.min_score(s);
            }
            if let Some(w) = lexical_weight {
                req = req.lexical_weight(w);
            }
            let results = req.execute()?;
            progress.finish();
            let mut out = std::io::stdout().lock();
            if json {
                serde_json::to_writer_pretty(&mut out, &json_output(&results))?;
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
                println!("{}", serde_json::to_string_pretty(&json_output(&result))?);
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
                        println!("{}", serde_json::to_string_pretty(&json_output(&stats))?);
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
            standalone,
            no_watch,
            model,
        } => {
            let roots = default_paths(roots)?;
            #[cfg(unix)]
            if !standalone && !no_watch {
                daemon::relay(&daemon::Endpoint::new(model.to_args()?)?, roots)?;
                return Ok(ExitCode::SUCCESS);
            }
            let _ = standalone;
            mcp::serve_standalone(roots, Box::new(move || engine(&model, false)), !no_watch)?;
            Ok(ExitCode::SUCCESS)
        }
        #[cfg(all(feature = "mcp", unix))]
        Command::Daemon { action, model } => {
            let endpoint = daemon::Endpoint::new(model.to_args()?)?;
            match action {
                DaemonAction::Run { idle_minutes } => daemon::run(
                    &endpoint,
                    Box::new(move || engine(&model, false)),
                    std::time::Duration::from_secs(idle_minutes * 60),
                )?,
                DaemonAction::Status { json } => match daemon::status(&endpoint)? {
                    None if json => println!("null"),
                    None => println!("No daemon running for this configuration."),
                    Some(s) if json => {
                        println!("{}", serde_json::to_string_pretty(&json_output(&s))?)
                    }
                    Some(s) => print!("{}", format_daemon_status(&s)),
                },
                DaemonAction::Stop => {
                    if daemon::stop(&endpoint)? {
                        println!("Stopped.");
                    } else {
                        println!("No daemon running for this configuration.");
                    }
                }
            }
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
                Ok(r) if json => println!(
                    "{}",
                    serde_json::to_string(&json_output(&r)).unwrap_or_default()
                ),
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

impl ModelArgs {
    /// These options as flags, with paths made absolute, so a daemon started
    /// from another directory resolves them the same way.
    #[cfg(all(feature = "mcp", unix))]
    fn to_args(&self) -> Result<Vec<String>> {
        let mut a = vec![
            "--dimension".to_string(),
            self.dimension.get().to_string(),
            "--precision".to_string(),
            self.precision.name().to_string(),
        ];
        if let Some(m) = &self.model {
            let p = expand_tilde(Path::new(m));
            let m = if p.is_dir() {
                p.canonicalize()?.display().to_string()
            } else {
                m.clone()
            };
            a.extend(["--model".to_string(), m]);
        }
        if self.text_only {
            a.push("--text-only".to_string());
        }
        if let Some(c) = &self.cache_path {
            let c = std::path::absolute(expand_tilde(c))?;
            a.extend(["--cache-path".to_string(), c.display().to_string()]);
        }
        Ok(a)
    }
}

#[cfg(all(feature = "mcp", unix))]
fn format_daemon_status(s: &daemon::DaemonStatus) -> String {
    let mut out = format!(
        "seyk daemon {} (pid {}), {} session(s)\n",
        s.version, s.pid, s.sessions
    );
    for (folder, state) in &s.folders {
        let state = serde_json::to_value(state)
            .ok()
            .and_then(|v| v["state"].as_str().map(str::to_string))
            .unwrap_or_default();
        out.push_str(&format!("  {state:<14} {}\n", folder.display()));
    }
    out
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
        ResourceLocator::ImageRegion {
            x,
            y,
            width,
            height,
        } => format!(" region {x},{y} {width}x{height}"),
        ResourceLocator::WholeResource => String::new(),
    }
}

fn clock(ms: u64) -> String {
    let s = ms / 1000;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

/// `via` names the query that matched, shown only when several were searched.
/// What found a hybrid hit, e.g. `meaning 0.69+words` or `name`. The hybrid
/// score only orders hits, so the cosine shows how strong the meaning match is.
fn signal_tag(hit: &SearchHit) -> Option<String> {
    let s = &hit.signals;
    let mut tags: Vec<String> = Vec::new();
    if let Some(c) = s.semantic {
        tags.push(format!("meaning {c:.2}"));
    }
    if s.lexical.is_some() {
        tags.push("words".into());
    }
    if s.filename {
        tags.push("name".into());
    }
    (!tags.is_empty()).then(|| tags.join("+"))
}

fn format_hit(
    hit: &SearchHit,
    tag: Option<&str>,
    via: Option<&str>,
    base: &Path,
    color: bool,
) -> String {
    let (bold, dim, reset) = if color {
        ("\x1b[1m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    let mut s = format!(
        "{:.3}  {bold}{}{}{reset}",
        hit.score,
        hit.resource
            .to_file_path()
            .map(|p| display_path(&p, base))
            .unwrap_or_else(|| hit.resource.to_string()),
        format_locator(&hit.locator)
    );
    if let Some(h) = &hit.heading {
        s.push_str(&format!("  {dim}§ {h}{reset}"));
    }
    if let Some(t) = tag {
        s.push_str(&format!("  {dim}[{t}]{reset}"));
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
            // Only a semantic match has one best phrasing.
            let via = (results.queries.len() > 1 && h.signals.semantic.is_some())
                .then(|| results.queries.get(h.query_index))
                .flatten()
                .map(String::as_str);
            let tag = (results.mode == SearchMode::Hybrid)
                .then(|| signal_tag(h))
                .flatten();
            format_hit(h, tag.as_deref(), via, base, color)
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
        out.push_str(&format!(
            "  skipped {}: {}\n",
            s.path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| s.resource.to_string()),
            s.reason
        ));
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
        out.push_str(&format!(
            "  skipped {}: {}\n",
            s.path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| s.resource.to_string()),
            s.reason
        ));
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
    use seyk::{MatchSignals, Modality, ResourceId, ResourceMetadata, SearchStats};

    fn hit(path: &str, score: f32, locator: ResourceLocator, preview: Option<&str>) -> SearchHit {
        SearchHit {
            resource: ResourceId::from_path(path).unwrap(),
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
            signals: MatchSignals {
                semantic: Some(score),
                ..Default::default()
            },
        }
    }

    fn results() -> SearchResults {
        SearchResults {
            queries: vec!["father".into()],
            mode: SearchMode::Semantic,
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
    fn hybrid_hits_say_what_found_them() {
        let mut r = results();
        r.mode = SearchMode::Hybrid;
        r.hits[0].signals.lexical = Some(4.2);
        r.hits[1].signals = MatchSignals {
            filename: true,
            ..Default::default()
        };
        let out = format_results(&r, Path::new("/book"), false);
        assert!(
            out.contains("father.md:81-96  [meaning 0.91+words]\n"),
            "{out}"
        );
        assert!(out.contains("kitchen.jpg  [name]\n"), "{out}");
    }

    #[test]
    fn json_output_mirrors_library_types() {
        let v: serde_json::Value = serde_json::to_value(json_output(&results())).unwrap();
        assert_eq!(v["schema_version"], 2);
        let first = &v["hits"][0];
        assert_eq!(first["resource"], "file:///book/characters/father.md");
        assert_eq!(first["locator"]["kind"], "text");
        assert_eq!(first["locator"]["line_start"], 81);
        assert_eq!(v["hits"][1]["locator"]["kind"], "whole_resource");
        let back: SearchResults = serde_json::from_value(v).unwrap();
        assert_eq!(back, results());
    }

    #[test]
    fn generic_hit_json_has_uri_and_human_output_has_a_fallback() {
        let mut r = results();
        r.hits[0].resource = "mem://book/chapter".parse().unwrap();
        let value = serde_json::to_value(json_output(&r)).unwrap();
        assert_eq!(value["schema_version"], 2);
        assert_eq!(value["hits"][0]["resource"], "mem://book/chapter");
        assert!(format_results(&r, Path::new("/"), false).contains("mem://book/chapter"));
        assert_eq!(
            serde_json::to_value(json_output(&IndexReport::default())).unwrap()["schema_version"],
            2
        );
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
