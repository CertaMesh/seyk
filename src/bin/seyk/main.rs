//! `seyk`: a thin command-line front end for the `seyk` crate.

#[cfg(all(feature = "mcp", unix))]
mod daemon;
#[cfg(feature = "mcp")]
mod mcp;

use std::io::{IsTerminal, Read, Write};
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
/// Any field addition, removal or rename after release requires a version bump.
const JSON_SCHEMA_VERSION: u32 = 8;

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
    /// Local pinned Whisper directory; defaults to the verified STT installation.
    #[cfg(feature = "audio-stt")]
    #[arg(long, env = "SEYK_STT_MODEL_DIR", global = true)]
    stt_model_dir: Option<PathBuf>,
    /// Transcription language code; omitted detects language from the chosen hit.
    #[cfg(feature = "audio-stt")]
    #[arg(long, global = true)]
    stt_language: Option<String>,

    /// Maximum audio presentation duration in seconds (inclusive).
    #[arg(long, default_value_t = 7200, value_parser = audio_duration, global = true)]
    max_audio_duration_secs: u64,

    /// Model directory, or Hugging Face repo as REPO[@REVISION]. Defaults to the pinned ONNX export.
    #[arg(long, env = "SEYK_MODEL", global = true)]
    model: Option<String>,

    /// Embedding dimension: 768, 512, 256 or 128. Queries and cache entries always share it.
    #[arg(long, default_value = "256", global = true)]
    dimension: Dimension,

    /// Weight precision: q8 (smaller) or fp32; benchmark speed on your machine.
    #[arg(long, default_value = "q8", global = true)]
    precision: Precision,

    /// Independent CPU text sessions (several GiB each; measure memory use).
    #[arg(long, default_value_t = 1, value_parser = positive_count, global = true)]
    sessions: usize,

    /// Intra-op threads per session; omitted lets ONNX Runtime decide.
    #[arg(long, value_parser = thread_count, global = true)]
    threads: Option<usize>,

    /// Text inputs per model call, independently of the engine window.
    #[arg(long, default_value_t = 16, value_parser = positive_count, global = true)]
    batch_size: usize,

    /// Inputs per engine dispatch. Defaults to sessions × batch-size.
    #[arg(long, value_parser = positive_count, global = true)]
    embedding_window: Option<usize>,

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
        /// Do not read previews. JSON hits retain preview: null.
        #[arg(long)]
        no_preview: bool,
        /// Include returned-hit counts by locator kind in JSON.
        #[arg(long)]
        kind_counts: bool,
        /// Print results as JSON.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        filter: FilterArgs,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Extract readable evidence from one selected hit (audio uses optional local STT).
    Extract {
        /// Local pinned Whisper directory; defaults to the verified STT installation.
        #[cfg(feature = "audio-stt")]
        #[arg(long, env = "SEYK_STT_MODEL_DIR")]
        stt_model_dir: Option<PathBuf>,
        /// Transcription language code; omitted detects language from the chosen hit.
        #[cfg(feature = "audio-stt")]
        #[arg(long)]
        stt_language: Option<String>,
        /// Search JSON file, or - to read standard input.
        #[arg(long, value_name = "FILE|-")]
        from_json: PathBuf,
        /// Zero-based hit index in the search results.
        #[arg(long, default_value_t = 0)]
        hit: usize,
        /// Emit a versioned extraction JSON document instead of raw text.
        #[arg(long)]
        json: bool,
        /// Maximum audio presentation duration in seconds (inclusive).
        #[arg(long, default_value_t = 7200, value_parser = audio_duration)]
        max_audio_duration_secs: u64,
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
        /// Fixed local-content policy v1 with confined no-follow reads. Requires
        /// explicit real directory roots; implies --standalone (Unix only).
        #[arg(long)]
        local_content: bool,
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
    /// Erase deleted keyword postings and old WAL frames without clearing live records.
    Scrub,
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
    Download {
        /// Provision only the pinned q8 audio encoder (requires feature `audio`).
        #[arg(long)]
        audio: bool,
        /// Provision only the pinned Whisper STT manifest (feature `audio-stt`).
        #[arg(long, conflicts_with = "audio")]
        stt: bool,
    },
    /// Show where the model files are and whether they are present.
    Info,
}

fn main() -> ExitCode {
    // Never include a panic payload: dependency panics can contain source bytes.
    // Core catches PDF unwinds; this hook also runs before catch_unwind returns.
    std::panic::set_hook(Box::new(|info| {
        if let Some(location) = info.location() {
            eprintln!(
                "seyk: internal failure at {}:{}",
                location.file(),
                location.line()
            );
        } else {
            eprintln!("seyk: internal failure");
        }
    }));
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    let model = match &cli.command {
        Command::Search { model, .. }
        | Command::Index { model, .. }
        | Command::Model { model, .. } => Some(model),
        #[cfg(feature = "mcp")]
        Command::Mcp { model, .. } => Some(model),
        #[cfg(all(feature = "mcp", unix))]
        Command::Daemon { model, .. } => Some(model),
        _ => None,
    };
    if model.is_some_and(|m| m.sessions > 1 && m.threads.is_none()) {
        eprintln!(
            "warning: multiple CPU sessions with automatic threads may oversubscribe the machine; set --threads per session"
        );
    }
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
            no_preview,
            kind_counts,
            json,
            filter,
            model,
        } => {
            let paths = default_paths(paths)?;
            let engine = search_engine(&model, no_cache)?;
            let progress = Progress::new(!json);
            let report = |e: ProgressEvent| progress.event(e);
            let mut req = also
                .into_iter()
                .fold(engine.search(&query), |r, q| r.query(q))
                .mode(mode)
                .kind_counts(kind_counts)
                .paths(&paths)
                .limit(limit)
                .discovery(filter.discovery())
                .progress(&report);
            if no_preview {
                req = req.previews(None);
            }
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
            out.flush()?;
            drop(out);
            report_scrub(engine.scrub_cache());
            Ok(ExitCode::SUCCESS)
        }
        Command::Extract {
            #[cfg(feature = "audio-stt")]
            stt_model_dir,
            #[cfg(feature = "audio-stt")]
            stt_language,
            from_json,
            hit,
            json,
            max_audio_duration_secs,
        } => {
            let input: Box<dyn Read> = if from_json == Path::new("-") {
                Box::new(std::io::stdin().lock())
            } else {
                Box::new(
                    std::fs::File::open(&from_json)
                        .with_context(|| format!("reading {}", from_json.display()))?,
                )
            };
            let selected = read_search_hit(input, hit)?;
            let extractor: &dyn seyk::Extractor = &seyk::TextExtractor;
            #[cfg(feature = "pdf")]
            let extractor: &dyn seyk::Extractor =
                if matches!(selected.locator, ResourceLocator::PageText { .. }) {
                    &seyk::PdfTextExtractor
                } else {
                    extractor
                };
            let extracted = if selected.metadata.modality == seyk::Modality::Audio {
                #[cfg(feature = "audio-stt")]
                {
                    audio_extractor(stt_model_dir.as_deref(), stt_language.as_deref()).extract(
                        &seyk::FsProvider,
                        &selected.resource,
                        &selected.locator,
                        &selected.metadata,
                        max_audio_duration_secs,
                    )?
                }
                #[cfg(not(feature = "audio-stt"))]
                {
                    #[cfg(feature = "audio")]
                    seyk::audio::extract_samples(
                        &seyk::FsProvider,
                        &selected.resource,
                        &selected.locator,
                        &selected.metadata,
                        max_audio_duration_secs,
                    )?;
                    #[cfg(not(feature = "audio"))]
                    let _ = max_audio_duration_secs;
                    return Err(seyk::Error::UnsupportedExtraction {
                        resource: selected.resource,
                        modality: seyk::Modality::Audio,
                        locator: selected.locator,
                    }
                    .into());
                }
            } else {
                seyk::extract_resource(
                    &seyk::FsProvider,
                    extractor,
                    &selected.resource,
                    &selected.locator,
                    &selected.metadata,
                )?
            };
            let mut out = std::io::stdout().lock();
            if json {
                serde_json::to_writer_pretty(&mut out, &json_output(&extracted))?;
                writeln!(out)?;
            } else {
                if extracted
                    .provenance
                    .as_ref()
                    .is_some_and(|p| p.speech.is_some())
                {
                    writeln!(out, "Machine transcript (may contain errors)")?;
                }
                out.write_all(extracted.text.as_bytes())?;
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
            let paths = default_paths(paths)?;
            let engine = engine(&model, false)?;
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
            std::io::stdout().flush()?;
            report_scrub(engine.scrub_cache());
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
                    match cache.clear() {
                        Ok(()) => {}
                        Err(seyk::Error::CacheCleanupPending) => scrub_warning(),
                        Err(error) => return Err(error.into()),
                    }
                    println!("Cleared {}", cache.path().display());
                }
                CacheAction::Scrub => {
                    let report = cache.scrub()?;
                    if report.pending {
                        scrub_warning();
                    } else {
                        println!("Scrubbed {}", cache.path().display());
                    }
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        #[cfg(feature = "mcp")]
        Command::Mcp {
            roots,
            standalone,
            no_watch,
            local_content,
            model,
        } => {
            if local_content {
                #[cfg(not(unix))]
                bail!("--local-content is only supported on Unix");
                #[cfg(unix)]
                {
                    if roots.is_empty() {
                        bail!("--local-content requires explicit directory roots");
                    }
                    let roots = roots
                        .into_iter()
                        .map(|p| {
                            let p = seyk::expand_tilde(&p);
                            if p.is_absolute() { p } else { cwd().join(p) }
                        })
                        .collect::<Vec<_>>();
                    let provider = std::sync::Arc::new(
                        seyk::local_content::LocalContentProvider::new(&roots)?,
                    );
                    let engine_provider = provider.clone();
                    mcp::serve_local_content(
                        provider,
                        Box::new(move || {
                            search_engine_with_provider(
                                &model,
                                false,
                                Some(engine_provider.clone()),
                            )
                        }),
                        !no_watch,
                    )?;
                    return Ok(ExitCode::SUCCESS);
                }
            }
            let roots = default_paths(roots)?;
            #[cfg(unix)]
            if !standalone && !no_watch {
                daemon::relay(&daemon::Endpoint::new(model.to_args()?)?, roots)?;
                return Ok(ExitCode::SUCCESS);
            }
            let _ = standalone;
            mcp::serve_standalone(
                roots,
                Box::new(move || search_engine(&model, false)),
                !no_watch,
            )?;
            Ok(ExitCode::SUCCESS)
        }
        #[cfg(all(feature = "mcp", unix))]
        Command::Daemon { action, model } => {
            let endpoint = daemon::Endpoint::new(model.to_args()?)?;
            match action {
                DaemonAction::Run { idle_minutes } => daemon::run(
                    &endpoint,
                    Box::new(move || search_engine(&model, false)),
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
                ModelAction::Download { audio, stt } => {
                    #[cfg(feature = "audio-stt")]
                    if model.stt_model_dir.is_some() || model.stt_language.is_some() {
                        anyhow::bail!(
                            "--stt-model-dir and --stt-language configure extraction, not model download"
                        );
                    }
                    let dir = if stt {
                        #[cfg(feature = "audio-stt")]
                        {
                            seyk::stt::download()?
                        }
                        #[cfg(not(feature = "audio-stt"))]
                        {
                            anyhow::bail!("STT support is not compiled in (feature `audio-stt`)")
                        }
                    } else if audio {
                        #[cfg(feature = "audio")]
                        {
                            gemma::download_audio()?
                        }
                        #[cfg(not(feature = "audio"))]
                        {
                            anyhow::bail!("audio support is not compiled in (feature `audio`)")
                        }
                    } else {
                        gemma::download(&config)?
                    };
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

fn audio_duration(value: &str) -> std::result::Result<u64, String> {
    let seconds = value
        .parse::<u64>()
        .map_err(|_| "must be a positive integer".to_string())?;
    seyk::discovery::audio_byte_limit(seconds).map_err(|e| e.to_string())?;
    Ok(seconds)
}

fn positive_count(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| "must be a positive integer".to_string())
}

fn thread_count(value: &str) -> std::result::Result<usize, String> {
    positive_count(value).and_then(|n| {
        (n <= i32::MAX as usize)
            .then_some(n)
            .ok_or_else(|| "threads must not exceed i32::MAX".into())
    })
}

impl ModelArgs {
    fn embedding_window(&self) -> Result<usize> {
        self.embedding_window.map(Ok).unwrap_or_else(|| {
            self.sessions
                .checked_mul(self.batch_size)
                .context("sessions × batch-size exceeds usize")
        })
    }

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
        // Omit defaults to retain the existing default daemon endpoint identity.
        for (flag, value) in [
            ("--sessions", (self.sessions != 1).then_some(self.sessions)),
            ("--threads", self.threads),
            (
                "--batch-size",
                (self.batch_size != 16).then_some(self.batch_size),
            ),
            ("--embedding-window", self.embedding_window),
        ] {
            if let Some(value) = value {
                a.extend([flag.to_string(), value.to_string()]);
            }
        }
        // Include the default too: do not attach a new 2 h client to an older
        // daemon whose implementation still has the fixed 5a duration limit.
        a.extend([
            "--max-audio-duration-secs".to_string(),
            self.max_audio_duration_secs.to_string(),
        ]);
        if let Some(m) = &self.model {
            let p = expand_tilde(Path::new(m));
            let m = if p.is_dir() {
                p.canonicalize()?.display().to_string()
            } else {
                m.clone()
            };
            a.extend(["--model".to_string(), m]);
        }
        #[cfg(feature = "audio-stt")]
        {
            if let Some(dir) = &self.stt_model_dir {
                a.extend([
                    "--stt-model-dir".to_string(),
                    std::path::absolute(expand_tilde(dir))?
                        .display()
                        .to_string(),
                ]);
            }
            if let Some(language) = &self.stt_language {
                a.extend(["--stt-language".to_string(), language.clone()]);
            }
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
        if let Some(progress) = s
            .index_progress
            .iter()
            .find(|p| p.root.to_file_path().as_ref() == Some(folder))
            .and_then(|p| p.progress.as_ref())
        {
            for (kind, counts) in &progress.by_kind {
                out.push_str(&format!(
                    "    {kind}: {} indexed, {} waiting, {} skipped\n",
                    counts.indexed, counts.waiting, counts.skipped
                ));
            }
            if let Some(seconds) = progress.eta_seconds {
                out.push_str(&format!("    estimated remaining: {seconds}s\n"));
            }
        }
    }
    out
}

/// A search envelope is required so incompatible versions cannot be mistaken
/// for usable byte locators. Parsing and selection happen before any source read.
fn read_search_hit(input: impl Read, index: usize) -> Result<SearchHit> {
    let value: serde_json::Value = serde_json::from_reader(input).context("invalid search JSON")?;
    if !matches!(
        value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64),
        Some(2..=8)
    ) {
        bail!("expected search JSON schema_version 2, 3, 4, 5, 6, 7 or 8");
    }
    let results: SearchResults = serde_json::from_value(value).context("invalid search results")?;
    results
        .hits
        .into_iter()
        .nth(index)
        .with_context(|| format!("no hit at index {index}"))
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
    Ok({
        let mut config = GemmaConfig::default();
        config.source = source;
        config.dimension = args.dimension;
        config.precision = args.precision;
        config.sessions = args.sessions;
        config.threads = args.threads;
        config.batch_size = args.batch_size;
        config.vision = cfg!(feature = "image") && !args.text_only;
        config
    })
}

fn search_engine(args: &ModelArgs, ephemeral: bool) -> Result<SearchEngine> {
    search_engine_with_provider(args, ephemeral, None)
}

fn search_engine_with_provider(
    args: &ModelArgs,
    ephemeral: bool,
    provider: Option<std::sync::Arc<dyn seyk::ResourceProvider>>,
) -> Result<SearchEngine> {
    match engine_with_provider(args, ephemeral, provider.clone()) {
        Ok(engine) => Ok(engine),
        Err(error)
            if matches!(
                error.downcast_ref::<seyk_embed::Error>(),
                Some(seyk_embed::Error::ModelNotFound(_))
            ) =>
        {
            Ok(match provider {
                Some(provider) => SearchEngine::lexical_with_provider(provider)?,
                None => SearchEngine::lexical()?,
            })
        }
        Err(error) => Err(error),
    }
}

fn engine(args: &ModelArgs, ephemeral: bool) -> Result<SearchEngine> {
    engine_with_provider(args, ephemeral, None)
}

fn engine_with_provider(
    args: &ModelArgs,
    ephemeral: bool,
    provider: Option<std::sync::Arc<dyn seyk::ResourceProvider>>,
) -> Result<SearchEngine> {
    let window = args.embedding_window()?;
    let embedder = GemmaEmbedder::load(gemma_config(args)?).map_err(|error| {
        if matches!(error, seyk_embed::Error::ModelNotFound(_)) {
            anyhow::Error::new(error).context("model is missing; run `seyk model download`")
        } else {
            anyhow::Error::new(error)
        }
    })?;
    let builder = SearchEngine::builder()
        .embedder(embedder)
        .batch_size(window)
        .max_audio_duration_secs(args.max_audio_duration_secs);
    let builder = match provider {
        Some(provider) => builder.shared_provider(provider),
        None => builder,
    };
    #[cfg(feature = "audio-stt")]
    let builder = builder.audio_extractor(audio_extractor(
        args.stt_model_dir.as_deref(),
        args.stt_language.as_deref(),
    ));
    let builder = if ephemeral {
        builder.cache(MemoryCache::new())
    } else {
        builder.cache(open_cache(args.cache_path.as_deref())?)
    };
    Ok(builder.build()?)
}

#[cfg(feature = "audio-stt")]
fn audio_extractor(
    directory: Option<&Path>,
    language: Option<&str>,
) -> seyk::audio::AudioExtractor {
    let speech = directory
        .map(|p| seyk::stt::Whisper::new(expand_tilde(p)))
        .unwrap_or_else(seyk::stt::Whisper::installed);
    let extractor = seyk::audio::AudioExtractor::new(speech);
    match language {
        Some(code) => extractor.language(code),
        None => extractor,
    }
}

fn open_cache(path: Option<&Path>) -> Result<SqliteCache> {
    let path = match path {
        Some(p) => expand_tilde(p),
        None => SqliteCache::default_path()?,
    };
    SqliteCache::open(&path).with_context(|| format!("opening cache {}", path.display()))
}

fn default_paths(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    let paths = if paths.is_empty() { vec![cwd()] } else { paths };
    for path in &paths {
        seyk::discovery::ensure_allowed_path(&seyk::expand_tilde(path))?;
    }
    Ok(paths)
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
        ResourceLocator::PageText {
            page,
            byte_start,
            byte_end,
            ..
        } => format!(" page {page} bytes {byte_start}-{byte_end}"),
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
        _ => String::new(),
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
        let mut counts = std::collections::BTreeMap::new();
        for count in &s.skipped_counts {
            *counts.entry(count.code).or_insert(0usize) += count.count;
        }
        for (code, count) in counts.iter().take(5) {
            let code = match code {
                seyk::SkipCode::PdfParserPanicked => "pdf_parser_panicked",
                seyk::SkipCode::PdfParseFailed => "pdf_parse_failed",
                seyk::SkipCode::Other => "other",
            };
            out.push_str(&format!("; {count}: {code}"));
        }
        if counts.len() > 5 {
            out.push_str("; further codes omitted");
        }
    }
    if !s.page_diagnostics.is_empty() {
        out.push_str(&format!(", {} pages not indexed", s.page_diagnostics.len()));
    }
    out
}

fn format_page_diagnostics(s: &SyncStats) -> String {
    let mut counts = std::collections::BTreeMap::new();
    for d in &s.page_diagnostics {
        *counts.entry((&d.resource, &d.reason)).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .map(|((resource, reason), count)| {
            let label = resource
                .to_file_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| resource.to_string());
            format!(
                "  {label}: {count} page{} not indexed: {reason}\n",
                if count == 1 { "" } else { "s" }
            )
        })
        .collect()
}

fn format_summary(r: &SearchResults) -> String {
    let mut out = format!(
        "\nSearched {} segments: {} in {:.1}s\n",
        r.stats.segments_searched,
        format_sync(&r.stats.sync),
        r.stats.total_ms as f64 / 1000.0
    );
    if let Some(degraded) = r.degraded {
        let reason = match degraded.reason {
            seyk::DegradedReason::ModelLoading => "model loading",
            seyk::DegradedReason::NotYetEmbedded => "files awaiting embeddings",
            seyk::DegradedReason::NoModel => "model unavailable",
        };
        out.push_str(&format!("  Keyword and filename matches only ({reason})\n"));
    }
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
    out.push_str(&format_page_diagnostics(&r.stats.sync));
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
    out.push_str(&format_page_diagnostics(&r.sync));
    out
}

fn scrub_warning() {
    eprintln!(
        "warning: cache scrub pending; run seyk cache scrub once other readers close (using the same cache path), or rerun the one-shot command"
    );
}

// Cleanup must never turn a successful one-shot sync or search into a failure.
fn report_scrub(result: seyk::Result<seyk::cache::CacheCleanup>) {
    match result {
        Ok(report) if report.pending => scrub_warning(),
        Err(error) => {
            eprintln!("warning: cache scrub pending: {error}; run seyk cache scrub to retry")
        }
        _ => {}
    }
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
    if stats.scrub_pending {
        out.push_str(
            "  warning: cache scrub pending; run seyk cache scrub once other readers close\n",
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use seyk::{MatchSignals, Modality, ResourceId, ResourceMetadata, SearchStats};

    fn test_path(path: &str) -> PathBuf {
        std::env::temp_dir().join(path.trim_start_matches('/'))
    }

    fn hit(path: &str, score: f32, locator: ResourceLocator, preview: Option<&str>) -> SearchHit {
        SearchHit {
            resource: ResourceId::from_path(test_path(path)).unwrap(),
            locator,
            score,
            metadata: ResourceMetadata {
                media_type: None,
                modality: Modality::Text,
                size_bytes: 10,
                modified_ns: None,
                content_hash: "h".into(),
            },
            heading: None,
            preview: preview.map(str::to_string),
            query_index: 0,
            r#match: seyk::MatchReason {
                semantic_rank: Some(1),
                ..Default::default()
            },
            signals: MatchSignals {
                semantic: Some(score),
                ..Default::default()
            },
        }
    }

    fn results() -> SearchResults {
        SearchResults {
            degraded: None,
            kind_counts: None,
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
    fn threads_reject_out_of_range_counts() {
        for value in ["1".to_string(), i32::MAX.to_string()] {
            assert!(Cli::try_parse_from(["seyk", "search", "test", "--threads", &value]).is_ok());
        }
        for value in [
            "0".to_string(),
            (i32::MAX as usize + 1).to_string(),
            usize::MAX.to_string(),
        ] {
            assert!(Cli::try_parse_from(["seyk", "search", "test", "--threads", &value]).is_err());
        }
    }

    #[test]
    fn cpu_controls_keep_defaults_and_forward_to_daemon() {
        let Cli {
            command: Command::Search {
                model: defaults, ..
            },
        } = Cli::try_parse_from(["seyk", "search", "test"]).unwrap()
        else {
            panic!()
        };
        assert_eq!(
            (
                defaults.sessions,
                defaults.threads,
                defaults.batch_size,
                defaults.embedding_window().unwrap()
            ),
            (1, None, 16, 16)
        );
        let Cli {
            command: Command::Search { model, .. },
        } = Cli::try_parse_from([
            "seyk",
            "search",
            "test",
            "--sessions",
            "2",
            "--threads",
            "4",
            "--batch-size",
            "8",
            "--embedding-window",
            "24",
        ])
        .unwrap()
        else {
            panic!()
        };
        let config = gemma_config(&model).unwrap();
        assert_eq!(
            (
                config.sessions,
                config.threads,
                config.batch_size,
                model.embedding_window().unwrap()
            ),
            (2, Some(4), 8, 24)
        );
        #[cfg(all(feature = "mcp", unix))]
        {
            let args = model.to_args().unwrap();
            for flag in [
                "--sessions",
                "--threads",
                "--batch-size",
                "--embedding-window",
            ] {
                assert!(args.iter().any(|a| a == flag));
                assert!(!defaults.to_args().unwrap().iter().any(|a| a == flag));
            }
        }
        for flag in [
            "--sessions",
            "--threads",
            "--batch-size",
            "--embedding-window",
        ] {
            assert!(Cli::try_parse_from(["seyk", "search", "test", flag, "0"]).is_err());
        }
    }

    #[test]
    fn human_output() {
        let out = format_results(&results(), &test_path("/book"), false);
        assert_eq!(
            out.replace('\\', "/"),
            "0.914  characters/father.md:81-96\n       He notices himself repeating the same gestures...\n\n0.500  photos/kitchen.jpg\n"
        );
    }

    #[test]
    fn names_matching_query_when_several() {
        let mut r = results();
        r.queries.push("dad".into());
        r.hits[1].query_index = 1;
        let out = format_results(&r, &test_path("/book"), false);
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
        let out = format_results(&r, &test_path("/book"), false);
        assert!(
            out.contains("father.md:81-96  [meaning 0.91+words]\n"),
            "{out}"
        );
        assert!(out.contains("kitchen.jpg  [name]\n"), "{out}");
    }

    #[cfg(unix)]
    #[test]
    fn missing_model_fallback_preserves_local_content_provider() {
        let temp = tempfile::Builder::new()
            .prefix("lexical-policy-")
            .tempdir()
            .unwrap();
        let root = temp.path().canonicalize().unwrap();
        std::fs::write(root.join("safe.txt"), "quartz lighthouse").unwrap();
        std::fs::create_dir(root.join("Library")).unwrap();
        std::fs::write(root.join("Library/hidden.txt"), "quartz excluded").unwrap();
        let missing = root.join("missing-model");
        std::fs::create_dir(&missing).unwrap();
        let cli = Cli::try_parse_from([
            "seyk",
            "search",
            "quartz",
            "--model",
            missing.to_str().unwrap(),
        ])
        .unwrap();
        let Command::Search { model, .. } = cli.command else {
            unreachable!()
        };
        let provider = std::sync::Arc::new(
            seyk::local_content::LocalContentProvider::new(std::slice::from_ref(&root)).unwrap(),
        );
        let engine = search_engine_with_provider(&model, true, Some(provider)).unwrap();
        let result = engine.search("quartz").paths([&root]).execute().unwrap();
        assert_eq!(
            result.degraded.unwrap().reason,
            seyk::DegradedReason::NoModel
        );
        assert_eq!(result.hits.len(), 1);
        assert_eq!(
            result.hits[0].resource.to_file_path().unwrap(),
            root.join("safe.txt")
        );
    }

    #[test]
    fn json_output_mirrors_library_types() {
        let v: serde_json::Value = serde_json::to_value(json_output(&results())).unwrap();
        assert_eq!(v["schema_version"], 8);
        let first = &v["hits"][0];
        assert_eq!(
            first["resource"],
            ResourceId::from_path(test_path("/book/characters/father.md"))
                .unwrap()
                .as_str()
        );
        assert_eq!(first["match"]["semantic_rank"], 1);
        assert_eq!(first["match"]["matched_terms"], serde_json::json!([]));
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
        assert_eq!(value["schema_version"], 8);
        assert_eq!(value["hits"][0]["resource"], "mem://book/chapter");
        assert!(format_results(&r, &test_path("/"), false).contains("mem://book/chapter"));
        assert_eq!(
            serde_json::to_value(json_output(&IndexReport::default())).unwrap()["schema_version"],
            8
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
    #[test]
    fn no_preview_parses_and_keeps_v2_null_shape() {
        let cli =
            Cli::try_parse_from(["seyk", "search", "alpha", "--no-preview", "--json"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Search {
                no_preview: true,
                json: true,
                ..
            }
        ));
        let mut results = results();
        for hit in &mut results.hits {
            hit.preview = None;
        }
        let value = serde_json::to_value(json_output(&results)).unwrap();
        assert_eq!(value["schema_version"], 8);
        assert!(
            value["hits"]
                .as_array()
                .unwrap()
                .iter()
                .all(|hit| hit.get("preview").is_some_and(serde_json::Value::is_null))
        );
        let input = serde_json::to_vec(&value).unwrap();
        assert_eq!(
            read_search_hit(input.as_slice(), 1).unwrap(),
            results.hits[1]
        );
        let mut malformed = value;
        malformed["hits"][0]["locator"] = serde_json::json!({"kind": "text"});
        assert!(read_search_hit(malformed.to_string().as_bytes(), 0).is_err());
    }
    #[test]
    fn summary_counts_every_skipped_reason_even_when_details_are_capped() {
        let mut r = results();
        r.stats.sync.skipped_counts = vec![seyk::SkippedCount {
            code: seyk::SkipCode::PdfParserPanicked,
            reason: "PDF parser panicked".into(),
            count: 7,
        }];
        r.stats.sync.skipped = (0..7)
            .map(|i| seyk::SkippedResource {
                code: seyk::SkipCode::PdfParserPanicked,
                resource: format!("mem://pdf/{i}").parse().unwrap(),
                path: None,
                reason: "PDF parser panicked".into(),
            })
            .collect();
        let summary = format_summary(&r);
        assert!(summary.contains("7 skipped; 7: pdf_parser_panicked"));
        assert_eq!(summary.matches("  skipped ").count(), 5);
    }

    #[test]
    fn sync_summary_groups_codes_without_inlining_resource_reasons() {
        let stats = SyncStats {
            skipped: (0..100)
                .map(|i| seyk::SkippedResource {
                    code: seyk::SkipCode::Other,
                    resource: format!("mem://{i}").parse().unwrap(),
                    path: None,
                    reason: format!("private long path {i}"),
                })
                .collect(),
            skipped_counts: (0..100)
                .map(|i| seyk::SkippedCount {
                    code: seyk::SkipCode::Other,
                    count: 1,
                    reason: format!("private long path {i}"),
                })
                .collect(),
            ..Default::default()
        };
        let summary = format_sync(&stats);
        assert!(summary.contains("100 skipped; 100: other"));
        assert!(!summary.contains("private"));
        assert!(summary.len() < 150);
    }

    #[test]
    fn pdf_page_diagnostics_are_not_skipped_resources() {
        let resource: ResourceId = "mem://mixed".parse().unwrap();
        let report = IndexReport {
            sync: SyncStats {
                resources_discovered: 1,
                resources_updated: 1,
                page_diagnostics: vec![seyk::PageDiagnostic {
                    code: seyk::SkipCode::Other,
                    resource,
                    page: 2,
                    reason: "no usable embedded text".into(),
                }],
                ..Default::default()
            },
            ..Default::default()
        };
        let output = format_index_report(&report);
        assert!(!output.contains("skipped"));
        assert!(output.contains("mem://mixed: 1 page not indexed: no usable embedded text"));
        assert_eq!(
            format_locator(&ResourceLocator::PageText {
                page: 3,
                byte_start: 0,
                byte_end: 120,
                text_version: "test".into(),
                page_text_hash: "test".into()
            }),
            " page 3 bytes 0-120"
        );
    }
    #[test]
    fn audio_duration_configuration_and_daemon_round_trip() {
        let cli = Cli::try_parse_from(["seyk", "search", "q", "--max-audio-duration-secs", "123"])
            .unwrap();
        let Command::Search { model, .. } = cli.command else {
            panic!()
        };
        assert_eq!(model.max_audio_duration_secs, 123);
        #[cfg(all(feature = "mcp", unix))]
        {
            let args = model.to_args().unwrap();
            let at = args
                .iter()
                .position(|a| a == "--max-audio-duration-secs")
                .unwrap();
            assert_eq!(args[at + 1], "123");
            let mut daemon = vec!["seyk".to_string(), "daemon".into(), "run".into()];
            daemon.extend(args);
            assert!(Cli::try_parse_from(daemon).is_ok());
        }
        for bad in ["0", "-1", "18446744073709551615"] {
            assert!(
                Cli::try_parse_from(["seyk", "search", "q", "--max-audio-duration-secs", bad])
                    .is_err()
            );
        }
    }
}
