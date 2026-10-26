//! The MCP search tool: one [`Hub`] holding the model and the folder watchers,
//! and one [`SeykServer`] per agent session.
//!
//! The server only finds passages. The calling agent reads them and writes any
//! answer, so no generation happens here.
//!
//! A hub indexes each session's folders in the background and re-indexes on
//! change. Searches use keyword fallback while the model or indexed resources
//! are not ready; they report both the fallback reason and indexing progress.
//! `seyk mcp` normally runs the hub inside the shared daemon (see `daemon.rs`), so every session shares
//! one model; `--standalone` runs it in-process.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, anyhow};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use seyk::discovery::{PolicyRoot, RootRevalidationError};
use seyk::watch::{WatchOptions, watch};
use seyk::{
    DiscoveryOptions, ProgressEvent, ResourceId, ResourceLocator, SearchEngine, SearchResults,
};

const MAX_QUERIES: usize = 8;
const MAX_LIMIT: usize = 50;

const INSTRUCTIONS: &str = "Local search over the user's files, by meaning (EmbeddingGemma 2), \
keywords and file names at once. Describe what the passage would say rather than only asking \
the question, or name a file or identifier directly (\"pricing.md\", \"INC-18372\"). Pass 2-4 \
different phrasings in one call. Each hit says what matched: \"meaning\", \"words\" or \"name\". \
Hits are pointers: read the file at the given lines before answering, and cite path:lines. \
The server indexes its folders in the background. A `degraded` marker means keywords and \
file names only: semantic retrieval is unavailable while the model loads, files await \
embedding, or weights are absent.";

pub type EngineFactory = Box<dyn Fn() -> Result<SearchEngine> + Send + Sync>;

/// What every session shares: the model, loaded once, and one watcher per
/// folder. Sessions whose folders are already watched reuse that watcher.
pub struct Hub {
    engine: Mutex<Option<Arc<SearchEngine>>>,
    ready_engine: OnceLock<Arc<SearchEngine>>,
    factory: EngineFactory,
    /// `false` turns background indexing off; searches then embed inline.
    background: bool,
    watched: Mutex<Vec<(PathBuf, Arc<Mutex<IndexStatus>>)>>,
    diagnostics: Mutex<std::collections::HashMap<PathBuf, WatcherDiagnostics>>,
    progress: Mutex<std::collections::BTreeMap<PathBuf, MeasuredProgress>>,
}

/// A watcher's progress and the canonical root its pass walked. Exposure is
/// tied to that identity, never to whatever the logical root names later.
#[derive(Clone)]
struct MeasuredProgress {
    identity: PathBuf,
    progress: seyk::IndexProgress,
}

/// The resolved root a watcher pass walked: `walked` arrives with Walked and
/// becomes the pass's `measured` identity at Discovered.
#[derive(Default)]
struct PassIdentity {
    walked: Option<PathBuf>,
    measured: Option<PathBuf>,
}

/// One agent session: the folders it may search, and the watchers covering them.
#[derive(Clone)]
pub struct SeykServer {
    #[cfg(unix)]
    local_content: Option<Arc<seyk::local_content::LocalContentProvider>>,
    // Pin authorization even if a logical root symlink is later retargeted.
    roots: Arc<Vec<PolicyRoot>>,
    logical_paths: Arc<[PathBuf]>,
    hub: Arc<Hub>,
    /// Status of each watcher covering `roots`; empty when background indexing is off.
    watchers: Arc<Vec<Arc<Mutex<IndexStatus>>>>,
    tool_router: ToolRouter<Self>,
}

/// Background indexing, as agents see it. Every transition replaces it whole,
/// so a stale phase cannot outlive the event that ended it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum IndexStatus {
    LoadingModel,
    Indexing {
        segments_done: usize,
        segments_total: usize,
    },
    Ready,

    /// The last pass failed; the watcher still runs and retries on the next change.
    PassFailed {
        error: String,
    },
    /// This watcher no longer keeps its root fresh; overlapping watchers may still do so.
    Stopped {
        error: String,
    },
}

const MAX_PAGE_DIAGNOSTICS: usize = 128;

/// Diagnostics from the last completed watcher pass, separate from its state.
#[derive(Debug, Clone, Default, Serialize)]
struct IndexDiagnostics {
    skipped: Vec<seyk::SkippedCount>,
    page_diagnostics: Vec<seyk::PageDiagnostic>,
    /// Total before the page-detail cap; counts are never truncated.
    pages_total: usize,
}

#[derive(Default)]
struct WatcherDiagnostics {
    skipped: Vec<seyk::SkippedResource>,
    page_counts: std::collections::HashMap<ResourceId, usize>,
    pages: Vec<seyk::PageDiagnostic>,
}

impl WatcherDiagnostics {
    fn from_sync(sync: &seyk::SyncStats) -> Self {
        let mut page_counts = std::collections::HashMap::new();
        for page in &sync.page_diagnostics {
            *page_counts.entry(page.resource.clone()).or_insert(0usize) += 1;
        }
        Self {
            skipped: sync.skipped.clone(),
            page_counts,
            pages: sync
                .page_diagnostics
                .iter()
                .take(MAX_PAGE_DIAGNOSTICS)
                .cloned()
                .collect(),
        }
    }

    fn confined(&self, roots: &[PathBuf]) -> IndexDiagnostics {
        let allowed = |resource: &ResourceId| {
            resource.to_file_path().is_some_and(|path| {
                roots.iter().any(|root| {
                    let real = root.canonicalize().unwrap_or_else(|_| root.clone());
                    // File URIs use ordinary drive paths; Windows canonicalize returns
                    // verbatim paths. Round-trip the root before comparing components.
                    let real = ResourceId::from_path(&real)
                        .ok()
                        .and_then(|id| id.to_file_path());
                    real.and_then(|real| path.strip_prefix(real).ok())
                        .is_some_and(|relative| {
                            seyk::discovery::ensure_allowed_path(&root.join(relative)).is_ok()
                        })
                })
            })
        };
        let mut counts = std::collections::BTreeMap::new();
        for entry in self.skipped.iter().filter(|entry| allowed(&entry.resource)) {
            *counts
                .entry((entry.code, entry.reason.clone()))
                .or_insert(0usize) += 1;
        }
        IndexDiagnostics {
            skipped: counts
                .into_iter()
                .map(|((code, reason), count)| seyk::SkippedCount {
                    code,
                    reason,
                    count,
                })
                .collect(),
            page_diagnostics: self
                .pages
                .iter()
                .filter(|page| allowed(&page.resource))
                .cloned()
                .collect(),
            pages_total: self
                .page_counts
                .iter()
                .filter(|(resource, _)| allowed(resource))
                .map(|(_, count)| count)
                .sum(),
        }
    }
}

fn combine_diagnostics(records: impl IntoIterator<Item = IndexDiagnostics>) -> IndexDiagnostics {
    let mut result = IndexDiagnostics::default();
    let mut counts = std::collections::BTreeMap::new();
    for record in records {
        result.pages_total += record.pages_total;
        for count in record.skipped {
            *counts.entry((count.code, count.reason)).or_insert(0usize) += count.count;
        }
        let remaining = MAX_PAGE_DIAGNOSTICS.saturating_sub(result.page_diagnostics.len());
        result
            .page_diagnostics
            .extend(record.page_diagnostics.into_iter().take(remaining));
    }
    result.skipped = counts
        .into_iter()
        .map(|((code, reason), count)| seyk::SkippedCount {
            code,
            reason,
            count,
        })
        .collect();
    result
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    /// Include counts of returned hits by locator kind (after the limit).
    #[serde(default)]
    pub kind_counts: bool,
    /// One to eight phrasings of what you are looking for. Describe the passage
    /// ("the protagonist's name and role"), not only the question.
    pub queries: Vec<String>,
    /// Files or folders to search, absolute or relative to the first server root.
    /// Must lie inside the server's roots. Defaults to all roots.
    #[serde(default)]
    pub paths: Vec<String>,
    /// Maximum number of hits (default 10, at most 50).
    pub limit: Option<usize>,
    /// Drop hits whose cosine similarity is below this value.
    pub min_score: Option<f32>,
    /// Only search files matching these globs, e.g. "*.md" or "src/**/*.rs".
    #[serde(default)]
    pub include: Vec<String>,
    /// Skip files or folders matching these globs.
    #[serde(default)]
    pub exclude: Vec<String>,
    /// "hybrid" (default: meaning, keywords and file names), "semantic" or
    /// "lexical" (keywords and file names only).
    pub mode: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ExtractParams {
    /// The complete selected hit object returned by search, including locator and metadata.
    pub hit: serde_json::Value,
}
#[derive(Deserialize)]
struct SelectedHit {
    resource: ResourceId,
    locator: ResourceLocator,
    metadata: seyk::ResourceMetadata,
}

#[derive(Debug, Serialize)]
struct Hit {
    resource: ResourceId,
    locator: ResourceLocator,
    metadata: seyk::ResourceMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    /// "start-end" line range, when the hit is a text segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    lines: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    page: Option<u32>,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    heading: Option<String>,
    /// What found it: "meaning", "words" (keywords) and/or "name" (file name).
    matched: Vec<&'static str>,
    r#match: seyk::MatchReason,
    /// The phrasing that matched best by meaning, when meaning matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    matched_query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    preview: Option<String>,
}

#[derive(Debug, Serialize)]
struct Response {
    schema_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    degraded: Option<seyk::SearchDegradation>,
    hits: Vec<Hit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kind_counts: Option<std::collections::BTreeMap<String, usize>>,
    index_progress: Vec<RootProgress>,
    segments_searched: usize,
    files: usize,
    skipped: Vec<seyk::SkippedCount>,
    page_diagnostics: Vec<seyk::PageDiagnostic>,
    page_diagnostics_total: usize,
    index_diagnostics: IndexDiagnostics,
    segments_embedded: usize,
    took_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<IndexStatus>,
}

/// Per watched root. A session restricted to a child gets no parent inventory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootProgress {
    pub root: ResourceId,
    pub progress: Option<seyk::IndexProgress>,
}

impl Hub {
    /// `allowed` pairs a session's logical roots with their pinned canonical roots.
    pub fn root_progress(&self, allowed: Option<&[PolicyRoot]>) -> Vec<RootProgress> {
        let Ok(watched) = self.watched.lock() else {
            return Vec::new();
        };
        let Ok(progress) = self.progress.lock() else {
            return Vec::new();
        };
        let mut result = std::collections::BTreeMap::new();
        for (root, _) in watched.iter() {
            let snapshot = progress.get(root);
            let shown = match (allowed, snapshot) {
                (None, snapshot) => Some(snapshot.map(|s| s.progress.clone())),
                (Some(roots), Some(snapshot)) => roots
                    .iter()
                    .any(|session| snapshot_visible(session, root, &snapshot.identity))
                    .then(|| Some(snapshot.progress.clone())),
                (Some(roots), None) => roots
                    .iter()
                    .any(|r| progress_visible(r.uri_path(), root))
                    .then_some(None),
            };
            if let Some(shown) = shown {
                result.insert(root.clone(), shown);
            } else if let Some(allowed) = allowed {
                for child in allowed.iter().filter(|r| r.uri_path().starts_with(root)) {
                    result.entry(child.uri_path().to_path_buf()).or_insert(None);
                }
            }
        }
        result
            .into_iter()
            .filter_map(|(root, progress)| {
                Some(RootProgress {
                    root: ResourceId::from_path(root).ok()?,
                    progress,
                })
            })
            .collect()
    }
    pub fn new(factory: EngineFactory, background: bool) -> Arc<Self> {
        Arc::new(Self {
            engine: Mutex::new(None),
            ready_engine: OnceLock::new(),
            factory,
            background,
            watched: Mutex::new(Vec::new()),
            diagnostics: Mutex::new(std::collections::HashMap::new()),
            progress: Mutex::new(std::collections::BTreeMap::new()),
        })
    }

    #[cfg(unix)]
    pub fn cache_status(&self) -> Result<Option<seyk::CacheStats>> {
        self.engine
            .lock()
            .map_err(|_| anyhow!("engine lock poisoned"))?
            .as_ref()
            .map(|engine| engine.with_cache(|cache| cache.stats()))
            .transpose()
            .map_err(Into::into)
    }

    fn engine(&self) -> Result<Arc<SearchEngine>> {
        let mut slot = self
            .engine
            .lock()
            .map_err(|_| anyhow!("engine lock poisoned"))?;
        if let Some(e) = slot.as_ref() {
            return Ok(e.clone());
        }
        let e = Arc::new((self.factory)()?);
        e.start_cache_scrubber();
        *slot = Some(e.clone());
        let _ = self.ready_engine.set(e.clone());
        Ok(e)
    }

    /// Status cells for the watchers covering `roots`, starting a watcher for
    /// any root no existing one covers. Empty when background indexing is off.
    fn watch_roots(self: &Arc<Self>, roots: &[PathBuf]) -> Result<Vec<Arc<Mutex<IndexStatus>>>> {
        if !self.background {
            return Ok(Vec::new());
        }
        let mut watched = self
            .watched
            .lock()
            .map_err(|_| anyhow!("watcher list lock poisoned"))?;
        let mut cells = Vec::new();
        for root in roots {
            if let Some((_, cell)) = watched.iter().find(|(w, _)| watcher_covers(w, root)) {
                cells.push(cell.clone());
                continue;
            }
            let cell = Arc::new(Mutex::new(IndexStatus::LoadingModel));
            watched.push((root.clone(), cell.clone()));
            let (hub, root, c) = (self.clone(), root.clone(), cell.clone());
            std::thread::spawn(move || hub.background_index(&root, &c));
            cells.push(cell);
        }
        Ok(cells)
    }

    /// Inline sync owns the request only when every overlapping hub watcher
    /// has stopped. Check both directions: registering a child before its parent
    /// can leave two watchers, and a parent request includes its watched children.
    fn may_sync_inline(&self, paths: &[PathBuf]) -> Result<bool> {
        let watched = self
            .watched
            .lock()
            .map_err(|_| anyhow!("watcher list lock poisoned"))?;
        for (root, cell) in watched.iter() {
            if paths.iter().any(|path| paths_overlap(path, root)) {
                let status = cell
                    .lock()
                    .map_err(|_| anyhow!("index status lock poisoned"))?;
                if !matches!(*status, IndexStatus::Stopped { .. }) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Index `root`, then keep it fresh, reporting into `cell`. Runs on its own
    /// thread for the hub's lifetime.
    fn background_index(&self, root: &Path, cell: &Mutex<IndexStatus>) {
        self.background_index_with_options(root, cell, &WatchOptions::default());
    }

    fn background_index_with_options(
        &self,
        root: &Path,
        cell: &Mutex<IndexStatus>,
        options: &WatchOptions,
    ) {
        let set = |s: IndexStatus| {
            if let Ok(mut g) = cell.lock() {
                *g = s;
            }
        };
        let engine = match self.engine() {
            Ok(e) => e,
            Err(e) => {
                return set(IndexStatus::Stopped {
                    error: format!("{e:#}"),
                });
            }
        };
        let indexing = |done, total| {
            set(IndexStatus::Indexing {
                segments_done: done,
                segments_total: total,
            })
        };
        indexing(0, 0);
        let identity = Mutex::new(PassIdentity::default());
        let progress = |e: ProgressEvent| {
            self.record_progress(root, &identity, &e);
            if matches!(e, ProgressEvent::Discovered { .. }) {
                indexing(0, 0);
            }
            if let ProgressEvent::Embedding { done, total } = e {
                indexing(done, total);
            }
        };
        let result = watch(
            &engine,
            &[root.to_path_buf()],
            &DiscoveryOptions::default(),
            options,
            Some(&progress),
            |pass| {
                set(match pass {
                    Ok(report) => {
                        if let Ok(mut diagnostics) = self.diagnostics.lock() {
                            diagnostics.insert(
                                root.to_path_buf(),
                                WatcherDiagnostics::from_sync(&report.sync),
                            );
                        }
                        IndexStatus::Ready
                    }
                    Err(e) => {
                        if let Ok(mut roots) = self.progress.lock() {
                            roots.remove(root);
                        }
                        IndexStatus::PassFailed {
                            error: e.to_string(),
                        }
                    }
                })
            },
        );
        if let Ok(mut roots) = self.progress.lock() {
            roots.remove(root);
        }
        set(IndexStatus::Stopped {
            error: match result {
                Ok(()) => "file watcher stopped".into(),
                Err(e) => e.to_string(),
            },
        });
    }

    /// Keep a pass's snapshot under the root its listing actually walked, as
    /// carried by Walked. Never re-resolve `root` here: the alias may name a
    /// different directory by now. Discovered starts the pass's counts; without
    /// a single carried root for that pass, nothing is recorded.
    fn record_progress(&self, root: &Path, identity: &Mutex<PassIdentity>, e: &ProgressEvent) {
        match e {
            ProgressEvent::Walked { roots } => {
                if let Ok(mut identity) = identity.lock() {
                    identity.walked = match roots.as_slice() {
                        [walked] => walked.to_file_path(),
                        _ => None,
                    };
                }
            }
            ProgressEvent::Discovered { .. } => {
                if let Ok(mut identity) = identity.lock() {
                    identity.measured = identity.walked.take();
                }
                if let Ok(mut roots) = self.progress.lock() {
                    roots.remove(root);
                }
            }
            _ => {}
        }
        if let ProgressEvent::IndexProgress { progress } = e
            && let Some(identity) = identity.lock().ok().and_then(|id| id.measured.clone())
            && let Ok(mut roots) = self.progress.lock()
        {
            roots.insert(
                root.to_path_buf(),
                MeasuredProgress {
                    identity,
                    progress: progress.clone(),
                },
            );
        }
    }

    /// Every watched folder and its status, for `seyk daemon status`.
    #[cfg(any(unix, test))]
    pub fn watched(&self) -> Vec<(PathBuf, IndexStatus)> {
        self.watched
            .lock()
            .map(|w| {
                w.iter()
                    .filter_map(|(p, c)| Some((p.clone(), c.lock().ok()?.clone())))
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn logical_root_paths(roots: &[PolicyRoot]) -> Vec<PathBuf> {
    roots
        .iter()
        .map(|root| root.uri_path().to_path_buf())
        .collect()
}

fn root_revalidation_into_anyhow(error: RootRevalidationError) -> anyhow::Error {
    match error {
        RootRevalidationError::Policy(error) => error.into(),
        RootRevalidationError::Identity(error) => error.into(),
    }
}

impl SeykServer {
    pub fn new(roots: Vec<PathBuf>, hub: Arc<Hub>) -> Result<Self> {
        let roots = roots
            .into_iter()
            .map(|r| {
                seyk::discovery::ensure_allowed_path(&r)?;
                r.canonicalize()
                    .with_context(|| format!("root {}", r.display()))?;
                PolicyRoot::new(&r).with_context(|| format!("root {}", r.display()))
            })
            .collect::<Result<Vec<_>>>()?;
        if roots.is_empty() {
            return Err(anyhow!("no folders to serve"));
        }
        let paths = logical_root_paths(&roots);
        let watchers = hub.watch_roots(&paths)?;
        Ok(Self {
            #[cfg(unix)]
            local_content: None,
            roots: Arc::new(roots),
            logical_paths: paths.into(),
            hub,
            watchers: Arc::new(watchers),
            tool_router: Self::tool_router(),
        })
    }

    fn logical_paths(&self) -> &[PathBuf] {
        &self.logical_paths
    }

    fn validate_roots(&self) -> Result<()> {
        for root in self.roots.iter() {
            // Preserve the errors exposed before revalidation owned both checks.
            let current = root.revalidate().map_err(root_revalidation_into_anyhow)?;
            if !current {
                return Err(anyhow!(
                    "server root target changed: {}",
                    root.uri_path().display()
                ));
            }
        }
        Ok(())
    }

    fn run_extract(&self, params: ExtractParams) -> Result<seyk::ExtractedContent> {
        self.validate_roots()?;
        let hit: SelectedHit = serde_json::from_value(params.hit).map_err(|_| {
            anyhow!("invalid selected hit: resource, locator and metadata are required")
        })?;
        let path = hit
            .resource
            .to_file_path()
            .ok_or_else(|| anyhow!("only local resources may be extracted"))?;
        // Authorization is independent of the caller's claimed content hash.
        let path = self.confine(&path)?;
        let hit = seyk::SearchHit {
            resource: ResourceId::from_path(path.canonicalize()?)?,
            locator: hit.locator,
            metadata: hit.metadata,
            score: 0.0,
            signals: Default::default(),
            r#match: Default::default(),
            heading: None,
            preview: None,
            query_index: 0,
        };
        Ok(self.hub.engine()?.extract(&hit)?)
    }

    /// The least settled state among this session's watchers, so an agent never
    /// sees `ready` while one of its folders is still indexing. `None` means
    /// background indexing is off, never that a lock failed.
    fn status(&self) -> Result<Option<IndexStatus>> {
        let mut states = Vec::with_capacity(self.watchers.len());
        for cell in self.watchers.iter() {
            let s = cell
                .lock()
                .map_err(|_| anyhow!("index status lock poisoned"))?;
            states.push(s.clone());
        }
        Ok(combine(states))
    }

    /// Callers validate roots first, so the pinned canonical roots are current.
    fn root_progress(&self) -> Vec<RootProgress> {
        self.hub.root_progress(Some(&self.roots))
    }

    fn index_status_json(&self) -> Result<serde_json::Value> {
        self.validate_roots()?;
        Ok(serde_json::json!({
            "schema_version": crate::JSON_SCHEMA_VERSION,
            "index": self.status()?,
            "index_progress": self.root_progress(),
            "model_diagnostics": self.hub.ready_engine.get().and_then(|engine| engine.embedder().unavailable_reason(seyk::Modality::Image)).into_iter().collect::<Vec<_>>(),
        }))
    }

    fn diagnostics(&self) -> Result<IndexDiagnostics> {
        self.validate_roots()?;
        let records = self
            .hub
            .diagnostics
            .lock()
            .map_err(|_| anyhow!("index diagnostics lock poisoned"))?;
        let mut matching: Vec<_> = records
            .iter()
            .filter(|(root, _)| {
                self.roots
                    .iter()
                    .any(|session| paths_overlap(session.uri_path(), root))
            })
            .collect();
        matching.sort_by_key(|(root, _)| *root);
        let roots = self.logical_paths();
        Ok(combine_diagnostics(
            matching
                .into_iter()
                .map(|(_, record)| record.confined(roots)),
        ))
    }

    fn run_search(&self, p: SearchParams) -> Result<Response> {
        self.validate_roots()?;
        let queries: Vec<String> = p
            .queries
            .into_iter()
            .map(|q| q.trim().to_string())
            .filter(|q| !q.is_empty())
            .collect();
        if queries.is_empty() {
            return Err(anyhow!("give at least one non-empty query"));
        }
        if queries.len() > MAX_QUERIES {
            return Err(anyhow!("at most {MAX_QUERIES} queries per call"));
        }
        let explicit_paths;
        let paths = if p.paths.is_empty() {
            self.logical_paths()
        } else {
            explicit_paths = p
                .paths
                .iter()
                .map(|s| self.confine(Path::new(s)))
                .collect::<Result<Vec<_>>>()?;
            &explicit_paths
        };

        let status = self.status()?;
        // Display status is session-wide; ownership depends on the requested
        // paths and every overlapping hub watcher, including other sessions.
        // PassFailed still owns its root because it retries on the next change.
        let inline = self.hub.may_sync_inline(paths)?;
        // Loading holds the model mutex. Never wait on it in a search: the
        // background worker will publish the semantic engine when it is ready.
        let available = if let Some(engine) = self.hub.ready_engine.get() {
            // Ordinary ready-state mutex contention must never degrade ranking.
            Some(Some(engine.clone()))
        } else {
            match self.hub.engine.try_lock() {
                Ok(slot) => Some(slot.clone()),
                Err(std::sync::TryLockError::WouldBlock) => None,
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(anyhow!("engine lock poisoned"));
                }
            }
        };
        let (engine, loading) = match available {
            Some(Some(engine)) => (engine, false),
            // No active indexer: preserve inline loading/retry and its errors.
            Some(None) if inline => (self.hub.engine()?, false),
            // The model mutex, not the folded watcher status, is authoritative:
            // a failed/stopped sibling may mask another root's LoadingModel.
            _ => {
                #[cfg(unix)]
                let lexical = match &self.local_content {
                    Some(provider) => SearchEngine::lexical_with_provider(provider.clone())?,
                    None => SearchEngine::lexical()?,
                };
                #[cfg(not(unix))]
                let lexical = SearchEngine::lexical()?;
                (Arc::new(lexical), true)
            }
        };
        let mut queries = queries.into_iter();
        let first = queries.next().expect("checked non-empty");
        let mut req = queries
            .fold(engine.search(first), |r, q| r.query(q))
            .paths(paths)
            .kind_counts(p.kind_counts)
            .limit(p.limit.unwrap_or(10).clamp(1, MAX_LIMIT))
            .sync(inline || loading)
            .lexical_fallback(true);
        if let Some(m) = p.mode.as_deref() {
            req = req.mode(m.parse()?);
        }
        if let Some(s) = p.min_score {
            req = req.min_score(s);
        }
        for g in p.include {
            req = req.include(g);
        }
        for g in p.exclude {
            req = req.exclude(g);
        }
        let mut resp = to_response(req.execute()?);
        if loading {
            resp.degraded = Some(seyk::SearchDegradation::lexical(
                seyk::DegradedReason::ModelLoading,
            ));
        }
        resp.index = status;
        resp.index_progress = self.root_progress();
        resp.index_diagnostics = self.diagnostics()?;
        Ok(resp)
    }

    fn confine(&self, path: &Path) -> Result<PathBuf> {
        #[cfg(unix)]
        if let Some(provider) = &self.local_content {
            let path = seyk::expand_tilde(path);
            let path = if path.is_absolute() {
                path
            } else {
                self.roots[0].uri_path().join(path)
            };
            provider.validate_path(&path)?;
            return Ok(path);
        }
        confine(path, &self.roots)
    }
}

/// Fold several watchers' states into one: stopped, then failed, then loading,
/// then indexing (progress summed), and `ready` only if all are ready.
fn combine(states: Vec<IndexStatus>) -> Option<IndexStatus> {
    use IndexStatus::*;
    let rank = |s: &IndexStatus| match s {
        Stopped { .. } => 0,
        PassFailed { .. } => 1,
        LoadingModel => 2,
        Indexing { .. } => 3,
        Ready => 4,
    };
    let worst = states.iter().min_by_key(|s| rank(s))?.clone();
    Some(match worst {
        Indexing { .. } => {
            let (done, total) = states.iter().fold((0, 0), |(d, t), s| match s {
                Indexing {
                    segments_done,
                    segments_total,
                } => (d + segments_done, t + segments_total),
                _ => (d, t),
            });
            Indexing {
                segments_done: done,
                segments_total: total,
            }
        }
        other => other,
    })
}

// A no-follow discovery walk covers ordinary descendants, but not a root
// reached through a directory symlink. That root needs its own logical walk.
fn watcher_covers(watched: &Path, root: &Path) -> bool {
    if watched == root {
        return true;
    }
    let Ok(relative) = root.strip_prefix(watched) else {
        return false;
    };
    match (watched.canonicalize(), root.canonicalize()) {
        (Ok(watched), Ok(root)) => watched.join(relative) == root,
        _ => false,
    }
}

// A populated snapshot is shown only while its watcher root still resolves to
// the identity it was measured under, and that identity is the session's
// pinned root joined with the same relative path (the watcher_covers rule).
// A root that is missing, dangling or retargeted hides its snapshot.
fn snapshot_visible(session: &PolicyRoot, watched: &Path, measured: &Path) -> bool {
    let Ok(relative) = watched.strip_prefix(session.uri_path()) else {
        return false;
    };
    seyk::discovery::ensure_allowed_path(watched).is_ok()
        && watched
            .canonicalize()
            .is_ok_and(|current| current == measured)
        && session.authorize_path().join(relative) == measured
}

// Whether a watcher with no snapshot may be listed (null progress, no
// inventory): `watched` must sit below `session` without crossing a directory
// symlink, and pass the deny policy. A tail that does not exist yet cannot
// redirect anywhere.
fn progress_visible(session: &Path, watched: &Path) -> bool {
    let Ok(relative) = watched.strip_prefix(session) else {
        return false;
    };
    if seyk::discovery::ensure_allowed_path(watched).is_err() {
        return false;
    }
    match (resolve_existing(session), resolve_existing(watched)) {
        (Some(session), Some(watched)) => session.join(relative) == watched,
        _ => false,
    }
}

// Canonicalize the longest existing ancestor and keep the missing tail as
// spelled. Anything present but unresolvable, such as a dangling link, fails.
fn resolve_existing(path: &Path) -> Option<PathBuf> {
    let mut missing = Vec::new();
    let mut current = path;
    loop {
        if let Ok(real) = current.canonicalize() {
            return Some(missing.iter().rev().fold(real, |p, name| p.join(name)));
        }
        if current.symlink_metadata().is_ok() {
            return None;
        }
        missing.push(current.file_name()?);
        current = current.parent()?;
    }
}

// Compare filesystem identities for watcher ownership without discarding the
// logical roots used by discovery and policy checks.
fn paths_overlap(a: &Path, b: &Path) -> bool {
    let a = a.canonicalize().unwrap_or_else(|_| a.to_path_buf());
    let b = b.canonicalize().unwrap_or_else(|_| b.to_path_buf());
    a.starts_with(&b) || b.starts_with(&a)
}

/// Resolve `path` against the first root and refuse anything outside the roots,
/// so an agent cannot read previews of arbitrary files through this server.
/// Return an absolute logical path without `..`, retaining policy-bearing aliases.
fn confine(path: &Path, roots: &[PolicyRoot]) -> Result<PathBuf> {
    let path = seyk::expand_tilde(path);
    let joined = if path.is_absolute() {
        path
    } else {
        roots[0].uri_path().join(path)
    };
    seyk::discovery::ensure_allowed_path(&joined)?;
    let real = joined
        .canonicalize()
        .with_context(|| format!("{} does not exist", joined.display()))?;
    let mut logical = None;
    for root in roots {
        let root = root
            .current()
            .with_context(|| format!("root {}", root.uri_path().display()))?;
        if let Some(candidate) = root.logical_child(&real) {
            // Check every applicable logical root, including for absolute cached IDs.
            seyk::discovery::ensure_allowed_path(&candidate)?;
            let supplied = if joined.starts_with(root.uri_path()) {
                joined.clone()
            } else if let Ok(relative) = joined.strip_prefix(root.authorize_path()) {
                root.uri_path().join(relative)
            } else {
                candidate
            };
            // Keep directory symlinks supplied below a broader server root too.
            let candidate = seyk::discovery::absolute_logical_path(&supplied)?;
            logical.get_or_insert(candidate);
        }
    }
    if let Some(logical) = logical {
        Ok(logical)
    } else {
        Err(anyhow!(
            "{} is outside the server roots ({})",
            real.display(),
            roots
                .iter()
                .map(|r| r.uri_path().display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

fn to_response(r: SearchResults) -> Response {
    let hits = r
        .hits
        .into_iter()
        .map(|h| Hit {
            page: match &h.locator {
                ResourceLocator::Page { page } | ResourceLocator::PageText { page, .. } => {
                    Some(*page)
                }
                _ => None,
            },
            lines: match h.locator {
                ResourceLocator::Text {
                    line_start: Some(a),
                    line_end: Some(b),
                    ..
                } => Some(if a == b {
                    a.to_string()
                } else {
                    format!("{a}-{b}")
                }),
                _ => None,
            },
            path: h
                .resource
                .to_file_path()
                .map(|p| p.to_string_lossy().into_owned()),
            resource: h.resource,
            locator: h.locator,
            metadata: h.metadata,
            score: (h.score * 1000.0).round() / 1000.0,
            heading: h.heading,
            r#match: h.r#match,
            matched: [
                (h.signals.semantic.is_some(), "meaning"),
                (h.signals.lexical.is_some(), "words"),
                (h.signals.filename, "name"),
            ]
            .into_iter()
            .filter_map(|(on, t)| on.then_some(t))
            .collect(),
            matched_query: h
                .signals
                .semantic
                .and_then(|_| r.queries.get(h.query_index).cloned()),
            preview: h.preview,
        })
        .collect();
    Response {
        schema_version: crate::JSON_SCHEMA_VERSION,
        degraded: r.degraded,
        hits,
        kind_counts: r.kind_counts,
        index_progress: Vec::new(),
        segments_searched: r.stats.segments_searched,
        files: r.stats.sync.resources_discovered,
        skipped: r.stats.sync.skipped_counts,
        page_diagnostics_total: r.stats.sync.page_diagnostics.len(),
        page_diagnostics: r
            .stats
            .sync
            .page_diagnostics
            .into_iter()
            .take(MAX_PAGE_DIAGNOSTICS)
            .collect(),
        index_diagnostics: IndexDiagnostics::default(),
        segments_embedded: r.stats.sync.segments_embedded,
        took_ms: r.stats.total_ms,
        index: None,
    }
}

#[tool_router]
impl SeykServer {
    #[tool(
        description = "Current indexing state, model availability diagnostics, and per-root resource counts with an observed-throughput ETA. Counts describe the current or last completed pass; null progress is unavailable."
    )]
    async fn index_status(&self) -> Result<CallToolResult, McpError> {
        Ok(match self.index_status_json() {
            Ok(status) => CallToolResult::success(vec![ContentBlock::json(status)?]),
            Err(e) => CallToolResult::error(vec![ContentBlock::text(format!("{e:#}"))]),
        })
    }

    #[tool(
        description = "Extract readable evidence from one selected search hit. Pass the entire hit object. Audio returns a machine transcript that may contain errors, with provenance, and is limited to 30 seconds. Never called by search."
    )]
    async fn extract(
        &self,
        Parameters(params): Parameters<ExtractParams>,
    ) -> Result<CallToolResult, McpError> {
        let server = self.clone();
        let outcome = tokio::task::spawn_blocking(move || server.run_extract(params))
            .await
            .map_err(|_| McpError::internal_error("extraction worker failed", None))?;
        Ok(match outcome {
            Ok(extracted) => {
                CallToolResult::success(vec![ContentBlock::json(crate::json_output(&extracted))?])
            }
            Err(error) => CallToolResult::error(vec![ContentBlock::text(error.to_string())]),
        })
    }

    #[tool(
        description = "Semantic search over local files (Markdown, text, code, images). Returns \
        ranked passages as path + line range + preview. Read the files at those lines before \
        answering. Pass several phrasings of the same need in `queries`."
    )]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let server = self.clone();
        let outcome = tokio::task::spawn_blocking(move || server.run_search(params))
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(match outcome {
            Ok(resp) => CallToolResult::success(vec![ContentBlock::json(resp)?]),
            // Tool errors go back to the agent as results it can read and correct.
            Err(e) => CallToolResult::error(vec![ContentBlock::text(format!("{e:#}"))]),
        })
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SeykServer {
    fn get_info(&self) -> ServerConfig {
        let name = "seyk".to_owned();
        #[cfg(unix)]
        let name = if self.local_content.is_some() {
            format!(
                "seyk-local-content-v{}",
                seyk::local_content::LOCAL_CONTENT_POLICY_VERSION
            )
        } else {
            name
        };
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(name, env!("SEYK_BUILD_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }
}

/// Serve one session over stdio with an in-process hub.
pub fn serve_standalone(
    roots: Vec<PathBuf>,
    factory: EngineFactory,
    background: bool,
) -> Result<()> {
    let server = SeykServer::new(roots, Hub::new(factory, background))?;
    serve(server)
}

/// Dedicated in-process mode; its provider and capabilities cannot leak into a
/// shared daemon session. Roots were validated without canonicalizing links.
#[cfg(unix)]
pub fn serve_local_content(
    provider: Arc<seyk::local_content::LocalContentProvider>,
    factory: EngineFactory,
    background: bool,
) -> Result<()> {
    serve(local_content_server(
        provider,
        Hub::new(factory, background),
    )?)
}

#[cfg(unix)]
fn local_content_server(
    provider: Arc<seyk::local_content::LocalContentProvider>,
    hub: Arc<Hub>,
) -> Result<SeykServer> {
    let roots = provider.roots();
    let policy_roots = roots
        .iter()
        .map(|root| PolicyRoot::new(root))
        .collect::<seyk::Result<Vec<_>>>()?;
    let paths = logical_root_paths(&policy_roots);
    let watchers = hub.watch_roots(&paths)?;
    Ok(SeykServer {
        local_content: Some(provider),
        roots: Arc::new(policy_roots),
        logical_paths: paths.into(),
        hub,
        watchers: Arc::new(watchers),
        tool_router: SeykServer::tool_router(),
    })
}

fn serve(server: SeykServer) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let service = server.serve(rmcp::transport::stdio()).await?;
        service.waiting().await?;
        Ok(())
    })
}

#[cfg(all(test, feature = "pdf"))]
#[path = "../../../crates/seyk-core/tests/support/pdf_fixture.rs"]
mod pdf_fixture;

#[cfg(test)]
mod tests {
    use super::*;
    use seyk::{
        Dimension, Embedder, Embedding, MemoryCache, Modality, ModelIdentity, ResourceInput,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// Bag-of-words hashing, enough for lifecycle tests without model weights.
    struct WordHash(ModelIdentity, Arc<AtomicUsize>);

    impl WordHash {
        fn vector(&self, text: &str) -> seyk_embed::Result<Embedding> {
            let mut v = vec![0.0; 128];
            for w in text.split_whitespace() {
                v[blake3::hash(w.to_lowercase().as_bytes()).as_bytes()[0] as usize % 128] += 1.0;
            }
            Embedding::from_raw(v, Dimension::D128)
        }
    }

    impl Embedder for WordHash {
        fn identity(&self) -> &ModelIdentity {
            &self.0
        }
        fn supports(&self, m: Modality) -> bool {
            m == Modality::Text
        }
        fn embed_query(&self, q: &str) -> seyk_embed::Result<Embedding> {
            self.vector(q)
        }
        fn embed_resources(
            &self,
            inputs: &[ResourceInput<'_>],
        ) -> seyk_embed::Result<Vec<Embedding>> {
            self.1.fetch_add(inputs.len(), Ordering::SeqCst);
            inputs
                .iter()
                .map(|i| match i {
                    ResourceInput::Text { text, .. } => self.vector(text),
                    _ => unreachable!(),
                })
                .collect()
        }

        fn embed_batch(
            &self,
            requests: &[seyk_embed::EmbeddingRequest<'_>],
        ) -> seyk_embed::Result<Vec<Embedding>> {
            if requests
                .iter()
                .all(|r| r.task == seyk_embed::EmbeddingTask::SearchDocument)
            {
                let inputs: Vec<_> = requests.iter().map(|r| r.input.clone()).collect();
                return self.embed_resources(&inputs);
            }
            requests
                .iter()
                .map(|r| match &r.input {
                    ResourceInput::Text { text, .. }
                        if r.task == seyk_embed::EmbeddingTask::SearchQuery =>
                    {
                        self.embed_query(text)
                    }
                    _ => self
                        .embed_resources(std::slice::from_ref(&r.input))
                        .map(|mut v| v.remove(0)),
                })
                .collect()
        }
    }

    fn fake_engine() -> Result<SearchEngine> {
        counted_engine(Arc::new(AtomicUsize::new(0)))
    }

    fn counted_engine(documents: Arc<AtomicUsize>) -> Result<SearchEngine> {
        engine_with_cache(documents, MemoryCache::new())
    }

    fn engine_with_cache(
        documents: Arc<AtomicUsize>,
        cache: impl seyk::EmbeddingCache + 'static,
    ) -> Result<SearchEngine> {
        engine_with_provider(documents, cache, None)
    }

    fn engine_with_provider(
        documents: Arc<AtomicUsize>,
        cache: impl seyk::EmbeddingCache + 'static,
        provider: Option<Arc<dyn seyk::ResourceProvider>>,
    ) -> Result<SearchEngine> {
        let builder = SearchEngine::builder()
            .embedder(WordHash(
                ModelIdentity {
                    model_id: "test/word-hash".into(),
                    revision: "1".into(),
                    variant: "test".into(),
                    dimension: Dimension::D128,
                    normalization: "l2".into(),
                    query_prompt: String::new(),
                    document_prompt: String::new(),
                    max_tokens: 512,
                },
                documents,
            ))
            .cache(cache);
        let builder = match provider {
            Some(provider) => builder.shared_provider(provider),
            None => builder,
        };
        Ok(builder.build()?)
    }

    #[cfg(unix)]
    #[test]
    fn local_content_root_validation_uses_stored_spelling() {
        let temp = tempfile::Builder::new()
            .prefix("local-content-case-")
            .tempdir()
            .unwrap();
        let stored = temp.path().canonicalize().unwrap().join("scope");
        std::fs::create_dir(&stored).unwrap();
        let supplied = stored.with_file_name("SCOPE");
        // Exercise the case-insensitive filesystem boundary when available.
        if supplied.canonicalize().ok().as_ref() != Some(&stored) {
            return;
        }
        let provider =
            Arc::new(seyk::local_content::LocalContentProvider::new(&[supplied]).unwrap());
        let server =
            local_content_server(provider, Hub::new(Box::new(fake_engine), false)).unwrap();
        server.validate_roots().unwrap();
        assert_eq!(
            server
                .roots
                .iter()
                .map(|r| r.authorize_path().to_path_buf())
                .collect::<Vec<_>>()
                .as_slice(),
            &[stored]
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn local_content_tools_refuse_original_link_paths_and_generated_aliases() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::Builder::new()
            .prefix("local-content-mcp-")
            .tempdir()
            .unwrap();
        let root = temp.path().canonicalize().unwrap();
        std::fs::write(root.join("local.rs"), "ordinary quartz notes").unwrap();
        let generated = root.join("Targets/gaze/Debug/Deps/out.rs");
        std::fs::create_dir_all(generated.parent().unwrap()).unwrap();
        std::fs::write(&generated, "generated quartz notes").unwrap();
        symlink(root.join("local.rs"), root.join("alias.rs")).unwrap();
        symlink(root.join("Targets"), root.join("linked-dir")).unwrap();
        symlink(&generated, root.join("generated-alias.rs")).unwrap();
        for rel in ["id_rsa.txt", "vault.opvault/notes.txt"] {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "synthetic secret quartz").unwrap();
        }
        assert!(
            seyk::local_content::LocalContentProvider::new(&[root.join("vault.opvault")]).is_err()
        );
        let provider = Arc::new(
            seyk::local_content::LocalContentProvider::new(std::slice::from_ref(&root)).unwrap(),
        );
        let engine_provider = provider.clone();
        let hub = Hub::new(
            Box::new(move || {
                engine_with_provider(
                    Arc::new(AtomicUsize::new(0)),
                    MemoryCache::new(),
                    Some(engine_provider.clone()),
                )
            }),
            false,
        );
        let server = local_content_server(provider, hub).unwrap();
        assert_eq!(server.get_info().server_info.name, "seyk-local-content-v1");
        assert_eq!(
            server.get_info().server_info.version,
            env!("SEYK_BUILD_VERSION")
        );
        let valid = search(&server, "quartz");
        assert_eq!(valid.schema_version, 10);
        assert_eq!(valid.hits.len(), 1);
        let hit = serde_json::to_value(&valid.hits[0]).unwrap();
        assert!(
            !server
                .extract(Parameters(ExtractParams { hit: hit.clone() }))
                .await
                .unwrap()
                .is_error
                .unwrap_or(false)
        );
        for rel in [
            "alias.rs",
            "linked-dir",
            "generated-alias.rs",
            "id_rsa.txt",
            "vault.opvault",
        ] {
            let params: SearchParams = serde_json::from_value(serde_json::json!({
                "queries": ["quartz"], "paths": [root.join(rel)]
            }))
            .unwrap();
            assert!(
                server
                    .search(Parameters(params))
                    .await
                    .unwrap()
                    .is_error
                    .unwrap_or(false),
                "search {rel}"
            );
            let mut alias_hit = hit.clone();
            alias_hit["resource"] =
                serde_json::to_value(ResourceId::from_path(root.join(rel)).unwrap()).unwrap();
            assert!(
                server
                    .extract(Parameters(ExtractParams { hit: alias_hit }))
                    .await
                    .unwrap()
                    .is_error
                    .unwrap_or(false),
                "extract {rel}"
            );
        }
        // Existing default mode still supports an explicitly requested local link.
        let ordinary =
            SeykServer::new(vec![root.clone()], Hub::new(Box::new(fake_engine), false)).unwrap();
        assert_eq!(
            ordinary.confine(&root.join("alias.rs")).unwrap(),
            root.join("alias.rs")
        );
    }

    #[cfg(feature = "image")]
    #[test]
    fn missing_vision_is_visible_in_status_without_no_model_degradation() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        std::fs::write(root.join("note.txt"), "quartz lighthouse").unwrap();
        std::fs::write(root.join("photo.png"), b"skipped before decoding").unwrap();
        let reason = "image embedding unavailable: missing onnx/vision_encoder_quantized.onnx, onnx/vision_encoder_quantized.onnx_data; run `seyk model download`; semantic text search remains available";
        let hub = Hub::new(
            Box::new(move || {
                Ok(SearchEngine::builder()
                    .embedder(
                        crate::common::FakeEmbedder::new(Dimension::D128, "missing-vision")
                            .with_missing_vision(Some(reason.into())),
                    )
                    .build()?)
            }),
            false,
        );
        let server = SeykServer::new(vec![root], hub).unwrap();
        // Status must not initialize or wait for the model.
        assert_eq!(
            server.index_status_json().unwrap()["model_diagnostics"],
            serde_json::json!([])
        );
        assert!(server.hub.ready_engine.get().is_none());
        let response = search(&server, "quartz");
        assert!(response.degraded.is_none());
        assert_eq!(response.hits.len(), 1);
        assert_eq!(
            server.index_status_json().unwrap()["model_diagnostics"],
            serde_json::json!([reason])
        );
        assert!(
            serde_json::to_string(&response)
                .unwrap()
                .contains("vision_encoder_quantized.onnx_data")
        );
    }

    fn wait_for(server: &SeykServer, done: impl Fn(&IndexStatus) -> bool) -> IndexStatus {
        let started = Instant::now();
        loop {
            let s = server.status().unwrap().unwrap();
            if done(&s) {
                return s;
            }
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "stuck at {s:?}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn search(server: &SeykServer, q: &str) -> Response {
        server
            .run_search(SearchParams {
                kind_counts: false,
                queries: vec![q.into()],
                paths: vec![],
                limit: None,
                min_score: None,
                include: vec![],
                exclude: vec![],
                mode: None,
            })
            .unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn mcp_status_reuses_guard_snapshot_after_directory_changes() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("dedicated");
        let cache = seyk::SqliteCache::open(directory.join("cache.sqlite3")).unwrap();
        let hub = Hub::new(Box::new(|| Err(anyhow!("already initialized"))), false);
        *hub.engine.lock().unwrap() = Some(Arc::new(
            engine_with_cache(Arc::new(AtomicUsize::new(0)), cache).unwrap(),
        ));
        let initial = hub.cache_status().unwrap().unwrap();
        assert!(!initial.in_synced_folder);
        std::fs::create_dir(directory.join(".stfolder")).unwrap();
        #[cfg(target_os = "macos")]
        {
            assert_eq!(initial.backup_excluded, Some(true));
            // Independent mutation: a fresh native read would now return false.
            assert!(
                std::process::Command::new("/usr/bin/xattr")
                    .args(["-d", "com.apple.metadata:com_apple_backup_excludeItem"])
                    .arg(&directory)
                    .status()
                    .unwrap()
                    .success()
            );
        }
        for _ in 0..4 {
            let status = hub.cache_status().unwrap().unwrap();
            assert_eq!(status.backup_excluded, initial.backup_excluded);
            assert_eq!(status.in_synced_folder, initial.in_synced_folder);
        }
    }

    #[test]
    fn inline_mcp_sync_leaves_scrub_pending_and_never_runs_it_in_the_request() {
        let temp = tempfile::tempdir().unwrap();
        let notes = temp.path().join("notes");
        std::fs::create_dir(&notes).unwrap();
        let file = notes.join("note.txt");
        std::fs::write(&file, "syntheticprivatequartzsentinel old passage").unwrap();
        let database = temp.path().join("cache.sqlite3");
        let factory_path = database.clone();
        let hub = Hub::new(
            Box::new(move || {
                engine_with_cache(
                    Arc::new(AtomicUsize::new(0)),
                    seyk::SqliteCache::open(&factory_path)?,
                )
            }),
            false,
        );
        let server = SeykServer::new(vec![notes], hub.clone()).unwrap();
        search(&server, "old passage");
        assert!(
            !hub.engine()
                .unwrap()
                .with_cache(|c| c.stats())
                .unwrap()
                .scrub_pending
        );
        std::fs::write(&file, "new ordinary synthetic passage with different bytes").unwrap();
        let response = search(&server, "ordinary passage");
        assert_eq!(response.segments_embedded, 1);
        assert!(
            hub.engine()
                .unwrap()
                .with_cache(|c| c.stats())
                .unwrap()
                .scrub_pending
        );
        let wal = std::fs::read(database.with_extension("sqlite3-wal")).unwrap();
        assert!(
            wal.windows("quartzsentinel".len())
                .any(|w| w == b"quartzsentinel")
        );
        assert!(!hub.engine().unwrap().scrub_cache().unwrap().pending);
    }

    #[test]
    fn response_keeps_openable_path_and_adds_versioned_uri() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("my note.md");
        std::fs::write(&file, "lighthouse keeper counts ships").unwrap();
        let server = SeykServer::new(
            vec![dir.path().into()],
            Hub::new(Box::new(fake_engine), false),
        )
        .unwrap();
        let response = serde_json::to_value(search(&server, "lighthouse")).unwrap();
        assert_eq!(response["schema_version"], 10);
        assert_eq!(response["hits"][0]["match"]["semantic_rank"], 1);
        assert_eq!(response["hits"][0]["match"]["keyword_rank"], 1);
        assert_eq!(
            response["hits"][0]["match"]["matched_terms"],
            serde_json::json!(["lighthouse"])
        );
        assert!(response.get("cache_status").is_none());
        let canonical = file.canonicalize().unwrap();
        assert_eq!(
            Path::new(response["hits"][0]["path"].as_str().unwrap())
                .canonicalize()
                .unwrap(),
            canonical
        );
        assert_eq!(
            response["hits"][0]["resource"],
            ResourceId::from_path(&canonical).unwrap().as_str()
        );
        assert!(response["hits"][0]["lines"].is_string());
        assert!(response["hits"][0]["preview"].is_string());
    }

    #[test]
    fn explicit_extraction_uses_selected_hash_and_session_roots() {
        let inside = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let file = inside.path().join("note.txt");
        std::fs::write(&file, "quartz pebble").unwrap();
        let server = SeykServer::new(
            vec![inside.path().into()],
            Hub::new(Box::new(fake_engine), false),
        )
        .unwrap();
        let response = serde_json::to_value(search(&server, "quartz")).unwrap();
        let hit = response["hits"][0].clone();
        let result = server
            .run_extract(ExtractParams { hit: hit.clone() })
            .unwrap();
        assert_eq!(result.text, "quartz pebble");
        std::fs::write(&file, "changed bytes").unwrap();
        assert!(
            server
                .run_extract(ExtractParams { hit: hit.clone() })
                .is_err()
        );
        let other = outside.path().join("other.txt");
        std::fs::write(&other, "quartz pebble").unwrap();
        let mut escaped = hit;
        escaped["resource"] = serde_json::to_value(ResourceId::from_path(&other).unwrap()).unwrap();
        assert!(server.run_extract(ExtractParams { hit: escaped }).is_err());
    }

    #[test]
    fn sessions_search_only_their_roots_even_with_a_shared_cache() {
        let outer = tempfile::tempdir().unwrap();
        let inner = tempfile::tempdir().unwrap();
        std::fs::write(outer.path().join("a.txt"), "quartz pebble").unwrap();
        std::fs::write(inner.path().join("b.txt"), "quartz pebble").unwrap();
        let hub = Hub::new(Box::new(fake_engine), false);
        let a = SeykServer::new(vec![outer.path().into()], hub.clone()).unwrap();
        let b = SeykServer::new(vec![inner.path().into()], hub).unwrap();
        assert_eq!(search(&a, "quartz").hits.len(), 1);
        assert_eq!(search(&b, "quartz").hits.len(), 1);
        let hits = search(&a, "quartz").hits;
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0]
                .resource
                .to_file_path()
                .unwrap()
                .canonicalize()
                .unwrap()
                .starts_with(outer.path().canonicalize().unwrap())
        );
    }

    #[cfg(unix)]
    #[test]
    fn docker_symlink_session_rejects_logical_and_resolved_secret_paths() {
        for chained in [false, true] {
            docker_symlink_session_rejects_logical_and_resolved_secret_paths_case(chained);
        }
    }

    #[cfg(unix)]
    fn docker_symlink_session_rejects_logical_and_resolved_secret_paths_case(chained: bool) {
        let temp = tempfile::tempdir().unwrap();
        let store = temp.path().join("config-store");
        std::fs::create_dir(&store).unwrap();
        let config = store.join("config.json");
        std::fs::write(
            &config,
            r#"{"auths":{"example.invalid":{"auth":"synthetic quartz"}}}"#,
        )
        .unwrap();
        std::fs::write(store.join("safe.txt"), "quartz pebble").unwrap();
        let docker = temp.path().join(".docker");
        std::os::unix::fs::symlink(&store, &docker).unwrap();
        let docker = if chained {
            let middle = temp.path().join("middle");
            let alias = temp.path().join("docker-alias");
            std::os::unix::fs::symlink(".docker", &middle).unwrap();
            std::os::unix::fs::symlink("middle", &alias).unwrap();
            alias
        } else {
            docker
        };
        let hub = Hub::new(Box::new(fake_engine), false);
        let old = hub
            .engine()
            .unwrap()
            .search("quartz")
            .paths([&config])
            .execute()
            .unwrap();
        let server = SeykServer::new(vec![docker.clone()], hub.clone()).unwrap();
        // A watcher owns this root, so this session must use cached-only search.
        hub.watched
            .lock()
            .unwrap()
            .push((docker.clone(), Arc::new(Mutex::new(IndexStatus::Ready))));
        for path in [
            PathBuf::from("config.json"),
            docker.join("config.json"),
            config.canonicalize().unwrap(),
        ] {
            assert!(confine(&path, &server.roots).is_err(), "{}", path.display());
        }
        // The permitted but unembedded sibling now triggers lexical fallback.
        // The logical Docker config must still never enter the candidate set.
        let response = search(&server, "quartz");
        assert_eq!(response.hits.len(), 1);
        assert_eq!(
            response.hits[0].resource.to_file_path().unwrap(),
            store.join("safe.txt").canonicalize().unwrap()
        );
        assert_eq!(
            response.degraded.unwrap().reason,
            seyk::DegradedReason::NotYetEmbedded
        );
        let old = serde_json::to_value(to_response(old)).unwrap();
        assert!(
            server
                .run_extract(ExtractParams {
                    hit: old["hits"][0].clone()
                })
                .is_err()
        );
        let safe = confine(&store.join("safe.txt"), &server.roots).unwrap();
        assert_eq!(safe, docker.join("safe.txt"));
        // Explicit directory requests must preserve the root context too.
        assert_eq!(confine(&store, &server.roots).unwrap(), docker);
        assert!(!watcher_covers(temp.path(), &docker));
        assert!(watcher_covers(temp.path(), &store));
        let broad = SeykServer::new(vec![temp.path().into()], hub).unwrap();
        assert_eq!(confine(&docker, &broad.roots).unwrap(), docker);
        let elsewhere = tempfile::tempdir().unwrap();
        std::fs::remove_file(&docker).unwrap();
        std::os::unix::fs::symlink(elsewhere.path(), &docker).unwrap();
        assert!(server.validate_roots().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn secret_chain_mcp_roots_requests_and_retargeting_are_refused() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let store = temp.path().join("store");
        std::fs::create_dir(&store).unwrap();
        let file = store.join("note.txt");
        std::fs::write(&file, "quartz pebble").unwrap();
        let alias = temp.path().join("alias-dir");
        symlink("store", &alias).unwrap();
        let hub = Hub::new(Box::new(fake_engine), false);
        let server = SeykServer::new(vec![alias.clone()], hub.clone()).unwrap();
        let old = serde_json::to_value(search(&server, "quartz")).unwrap();
        assert_eq!(old["hits"].as_array().unwrap().len(), 1);
        symlink("store", temp.path().join("vault.opvault")).unwrap();
        std::fs::remove_file(&alias).unwrap();
        symlink("vault.opvault", &alias).unwrap();
        // The final canonical directory is unchanged, but its new chain is secret.
        assert_eq!(alias.canonicalize().unwrap(), store.canonicalize().unwrap());
        assert!(server.validate_roots().is_err());
        assert!(
            server
                .run_extract(ExtractParams {
                    hit: old["hits"][0].clone()
                })
                .is_err()
        );
        assert!(SeykServer::new(vec![alias.clone()], hub.clone()).is_err());
        symlink("store/note.txt", temp.path().join("id_rsa")).unwrap();
        symlink("id_rsa", temp.path().join("alias.txt")).unwrap();
        let roots = vec![PolicyRoot::new(temp.path()).unwrap()];
        for path in [
            alias,
            temp.path().join("alias-dir/note.txt"),
            temp.path().join("alias.txt"),
        ] {
            assert!(confine(&path, &roots).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_cannot_expand_the_mcp_session_scope() {
        let inside = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("outside.txt");
        std::fs::write(&secret, "outside quartz").unwrap();
        let link = inside.path().join("link.txt");
        std::os::unix::fs::symlink(&secret, &link).unwrap();
        let roots = vec![PolicyRoot::new(&inside.path().canonicalize().unwrap()).unwrap()];
        assert!(confine(&link, &roots).is_err());
        let server = SeykServer::new(
            roots.iter().map(|r| r.uri_path().to_path_buf()).collect(),
            Hub::new(Box::new(fake_engine), false),
        )
        .unwrap();
        assert!(search(&server, "quartz").hits.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_filename_keeps_a_lossless_uri_and_readable_path() {
        use std::os::unix::ffi::OsStringExt;
        let dir = tempfile::tempdir().unwrap();
        let valid = dir.path().join("note.txt");
        std::fs::write(&valid, "quartz pebble").unwrap();
        let engine = fake_engine().unwrap();
        let mut results = engine.search("quartz").paths([&valid]).execute().unwrap();
        // Some Unix filesystems reject invalid UTF-8 filenames. Exercise the
        // output boundary without requiring such a file to exist on this host.
        let file = dir
            .path()
            .canonicalize()
            .unwrap()
            .join(std::ffi::OsString::from_vec(b"note-\xff.txt".to_vec()));
        results.hits[0].resource = ResourceId::from_path(&file).unwrap();
        let value = serde_json::to_value(to_response(results)).unwrap();
        let id: ResourceId = value["hits"][0]["resource"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(id.to_file_path().unwrap(), file);
        assert!(value["hits"][0]["path"].as_str().unwrap().contains('�'));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn background_index_reaches_ready_and_follows_changes() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("keeper.md"),
            "lighthouse keeper counts ships",
        )
        .unwrap();
        let hub = Hub::new(Box::new(fake_engine), true);
        let cell = Arc::new(Mutex::new(IndexStatus::LoadingModel));
        hub.watched
            .lock()
            .unwrap()
            .push((dir.path().into(), cell.clone()));
        let server = SeykServer::new(vec![dir.path().into()], hub.clone()).unwrap();
        let (events, changes, passes) = seyk::watch::TestEvents::channel();
        let options = WatchOptions {
            debounce: Duration::ZERO,
            test_events: Some(events),
            ..Default::default()
        };
        let root = dir.path().to_path_buf();
        let worker = std::thread::spawn(move || {
            hub.background_index_with_options(&root, &cell, &options);
        });
        // Ready describes the completed index pass, not a bound on OS event
        // delivery. Signal changes explicitly and wait for each pass to finish.
        passes
            .recv_timeout(Duration::from_secs(20))
            .expect("initial index pass did not finish");
        assert!(matches!(server.status().unwrap(), Some(IndexStatus::Ready)));
        let r = search(&server, "lighthouse ships");
        assert_eq!(r.hits.len(), 1);
        assert_eq!(r.segments_embedded, 0, "searches read the cache only");

        let bread = dir.path().join("bread.md");
        std::fs::write(&bread, "grandmother bakes bread").unwrap();
        changes.send(bread.clone()).unwrap();
        passes
            .recv_timeout(Duration::from_secs(20))
            .expect("change index pass did not finish");
        assert!(matches!(server.status().unwrap(), Some(IndexStatus::Ready)));
        let r = search(&server, "grandmother bread");
        assert_eq!(r.hits.len(), 2);
        let bread_id = ResourceId::from_path(bread.canonicalize().unwrap()).unwrap();
        assert!(r.hits.iter().any(|hit| hit.resource == bread_id));
        assert_eq!(
            r.segments_embedded, 0,
            "searches read the updated cache only"
        );
        drop(changes);
        worker.join().unwrap();
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn failed_pdf_watcher_stays_ready_and_replays_diagnostics_separately() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("bad.pdf"),
            super::pdf_fixture::malformed_cmap(true),
        )
        .unwrap();
        std::fs::write(dir.path().join("good.txt"), "garden basil").unwrap();
        let hub = Hub::new(Box::new(fake_engine), true);
        let server = SeykServer::new(vec![dir.path().into()], hub.clone()).unwrap();
        wait_for(&server, |state| matches!(state, IndexStatus::Ready));
        let replay = hub
            .engine()
            .unwrap()
            .index()
            .paths([dir.path()])
            .execute()
            .unwrap();
        assert_eq!(replay.sync.resources_unchanged, 2);
        assert_eq!(replay.sync.resources_updated, 0);
        assert_eq!(replay.sync.page_diagnostics.len(), 1);
        let response = serde_json::to_value(search(&server, "basil")).unwrap();
        assert_eq!(response["index"]["state"], "ready");
        assert_eq!(response["index_diagnostics"]["pages_total"], 1);
        assert_eq!(response["index_diagnostics"]["skipped"][0]["count"], 1);
        assert_eq!(
            response["index_diagnostics"]["page_diagnostics"][0]["code"],
            "pdf_parser_panicked"
        );
        assert!(response["page_diagnostics"].as_array().unwrap().is_empty());
    }

    #[test]
    fn loading_model_search_returns_words_and_names_without_waiting() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("quartz.md"), "quartz lighthouse keeper").unwrap();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let hub = Hub::new(
            Box::new(move || {
                entered_tx.send(()).unwrap();
                release_rx
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
                fake_engine()
            }),
            true,
        );
        let server = SeykServer::new(vec![dir.path().into()], hub).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let started = Instant::now();
        let response = search(&server, "quartz");
        release_tx.send(()).unwrap();
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(response.hits.len(), 1);
        assert!(response.hits[0].matched.contains(&"words"));
        assert!(response.hits[0].matched.contains(&"name"));
        assert_eq!(
            response.degraded.unwrap().reason,
            seyk::DegradedReason::ModelLoading
        );
        assert_eq!(response.segments_embedded, 0);
        wait_for(&server, |s| matches!(s, IndexStatus::Ready));
        assert!(search(&server, "quartz").degraded.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn local_content_loading_fallback_keeps_policy_and_counts() {
        let dir = tempfile::Builder::new()
            .prefix("local-fallback-")
            .tempdir()
            .unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::write(root.join("quartz.md"), "quartz lighthouse keeper").unwrap();
        for name in [
            ".hidden.md",
            "Library/private.md",
            "Targets/gaze/Debug/Deps/out.rs",
            "id_rsa.txt",
        ] {
            let path = root.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "quartz excluded fixture").unwrap();
        }
        let provider = Arc::new(
            seyk::local_content::LocalContentProvider::new(std::slice::from_ref(&root)).unwrap(),
        );
        let engine_provider = provider.clone();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let hub = Hub::new(
            Box::new(move || {
                entered_tx.send(()).unwrap();
                release_rx
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
                Ok(SearchEngine::lexical_with_provider(
                    engine_provider.clone(),
                )?)
            }),
            true,
        );
        let server = local_content_server(provider, hub).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let response = server
            .run_search(SearchParams {
                queries: vec!["quartz".into()],
                kind_counts: true,
                paths: vec![],
                limit: None,
                min_score: None,
                include: vec![],
                exclude: vec![],
                mode: None,
            })
            .unwrap();
        release_tx.send(()).unwrap();
        assert_eq!(
            response.degraded.unwrap().reason,
            seyk::DegradedReason::ModelLoading
        );
        assert_eq!(response.hits.len(), 1);
        assert_eq!(
            response.hits[0].resource.to_file_path().unwrap(),
            root.join("quartz.md")
        );
        assert_eq!(response.kind_counts.unwrap()["text"], 1);
        assert_eq!(response.hits[0].r#match.matched_terms, ["quartz"]);
        wait_for(&server, |s| matches!(s, IndexStatus::Ready));
        let no_model = search(&server, "quartz");
        assert_eq!(
            no_model.degraded.unwrap().reason,
            seyk::DegradedReason::NoModel
        );
        assert_eq!(no_model.hits.len(), 1);
    }

    #[test]
    fn ready_model_mutex_contention_keeps_identical_ranked_hits() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("quartz.md"), "quartz lighthouse keeper").unwrap();
        let hub = Hub::new(Box::new(fake_engine), false);
        let server = SeykServer::new(vec![dir.path().into()], hub.clone()).unwrap();
        let baseline = search(&server, "quartz");
        let _held = hub.engine.lock().unwrap();
        let response = search(&server, "quartz");
        assert!(response.degraded.is_none());
        assert_eq!(
            serde_json::to_vec(&baseline.hits).unwrap(),
            serde_json::to_vec(&response.hits).unwrap()
        );
    }

    #[test]
    fn model_free_hub_reports_degradation_after_initial_index() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("quartz.md"), "quartz lighthouse keeper").unwrap();
        let hub = Hub::new(Box::new(|| Ok(SearchEngine::lexical()?)), true);
        let server = SeykServer::new(vec![dir.path().into()], hub).unwrap();
        wait_for(&server, |s| matches!(s, IndexStatus::Ready));
        let response = serde_json::to_value(search(&server, "quartz")).unwrap();
        assert_eq!(
            response["degraded"],
            serde_json::json!({"mode":"lexical_only","reason":"no_model"})
        );
        assert_eq!(response["hits"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn failed_model_load_stops_background_indexing() {
        let dir = tempfile::TempDir::new().unwrap();
        let hub = Hub::new(Box::new(|| Err(anyhow!("no weights"))), true);
        let server = SeykServer::new(vec![dir.path().into()], hub).unwrap();
        let s = wait_for(&server, |s| matches!(s, IndexStatus::Stopped { .. }));
        assert!(matches!(s, IndexStatus::Stopped { error } if error.contains("no weights")));
    }

    #[test]
    fn sessions_share_one_model_and_reuse_covering_watchers() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static LOADS: AtomicUsize = AtomicUsize::new(0);
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/a.md"), "lighthouse keeper").unwrap();
        let hub = Hub::new(
            Box::new(|| {
                LOADS.fetch_add(1, Ordering::SeqCst);
                fake_engine()
            }),
            true,
        );
        let outer = SeykServer::new(vec![dir.path().into()], hub.clone()).unwrap();
        let inner = SeykServer::new(vec![dir.path().join("sub")], hub.clone()).unwrap();
        wait_for(&outer, |s| matches!(s, IndexStatus::Ready));
        wait_for(&inner, |s| matches!(s, IndexStatus::Ready));
        assert_eq!(
            hub.watched().len(),
            1,
            "the sub folder reuses the outer watcher"
        );
        assert_eq!(search(&inner, "lighthouse").hits.len(), 1);
        assert_eq!(
            LOADS.load(Ordering::SeqCst),
            1,
            "one model for all sessions"
        );
    }

    #[test]
    fn combined_status_reports_the_least_settled_watcher() {
        use IndexStatus::*;
        let indexing = |d, t| Indexing {
            segments_done: d,
            segments_total: t,
        };
        assert!(combine(vec![]).is_none());
        assert!(matches!(combine(vec![Ready, Ready]), Some(Ready)));
        assert!(matches!(
            combine(vec![Ready, indexing(1, 4), indexing(2, 6)]),
            Some(Indexing {
                segments_done: 3,
                segments_total: 10
            })
        ));
        assert!(matches!(
            combine(vec![indexing(1, 4), Stopped { error: "x".into() }]),
            Some(Stopped { .. })
        ));
    }

    #[test]
    fn combined_diagnostics_merge_counts_and_cap_pages_without_changing_ready() {
        let diagnostic = || IndexDiagnostics {
            skipped: vec![seyk::SkippedCount {
                code: seyk::SkipCode::PdfParserPanicked,
                reason: "PDF parser panicked".into(),
                count: 2,
            }],
            page_diagnostics: (1..=200)
                .map(|page| seyk::PageDiagnostic {
                    code: seyk::SkipCode::PdfParserPanicked,
                    resource: "mem://pdf".parse().unwrap(),
                    page,
                    reason: "PDF parser panicked".into(),
                })
                .collect(),
            pages_total: 200,
        };
        let combined = combine_diagnostics([diagnostic(), diagnostic()]);
        assert_eq!(combined.skipped.len(), 1);
        assert_eq!(combined.skipped[0].count, 4);
        assert_eq!(combined.pages_total, 400);
        assert_eq!(combined.page_diagnostics.len(), MAX_PAGE_DIAGNOSTICS);
        assert_eq!(
            serde_json::to_value(combine(vec![IndexStatus::Ready, IndexStatus::Ready])).unwrap()["state"],
            "ready"
        );
    }

    #[test]
    fn watcher_diagnostics_are_confined_to_session_roots() {
        let dir = tempfile::tempdir().unwrap();
        let allowed = dir.path().join("allowed");
        let sibling = dir.path().join("sibling");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();
        let diagnostic = |path: PathBuf, reason: &str| seyk::PageDiagnostic {
            code: seyk::SkipCode::PdfParserPanicked,
            resource: ResourceId::from_path(path).unwrap(),
            page: 1,
            reason: reason.into(),
        };
        let pages = vec![
            diagnostic(allowed.canonicalize().unwrap().join("bad.pdf"), "allowed"),
            diagnostic(sibling.canonicalize().unwrap().join("secret.pdf"), "hidden"),
        ];
        let skipped = pages
            .iter()
            .map(|page| seyk::SkippedResource {
                code: page.code,
                resource: page.resource.clone(),
                path: page.resource.to_file_path(),
                reason: page.reason.clone(),
            })
            .collect();
        let snapshot = WatcherDiagnostics::from_sync(&seyk::SyncStats {
            skipped,
            page_diagnostics: pages,
            ..Default::default()
        });
        let confined = snapshot.confined(std::slice::from_ref(&allowed));
        assert_eq!(confined.pages_total, 1);
        assert_eq!(confined.page_diagnostics.len(), 1);
        assert_eq!(confined.skipped.len(), 1);
        assert!(!serde_json::to_string(&confined).unwrap().contains("hidden"));

        // A removed root must retain confined diagnostics through the fallback.
        let missing_root = allowed.canonicalize().unwrap();
        std::fs::remove_dir(&allowed).unwrap();
        assert!(!missing_root.exists());
        let missing = snapshot.confined(&[missing_root]);
        assert_eq!(
            serde_json::to_value(&missing).unwrap(),
            serde_json::to_value(&confined).unwrap()
        );
    }

    fn stopped() -> IndexStatus {
        IndexStatus::Stopped {
            error: "test stopped".into(),
        }
    }

    fn indexing() -> IndexStatus {
        IndexStatus::Indexing {
            segments_done: 0,
            segments_total: 1,
        }
    }

    fn params(paths: &[PathBuf]) -> SearchParams {
        SearchParams {
            kind_counts: false,
            queries: vec!["quartz".into()],
            paths: paths.iter().map(|p| p.to_str().unwrap().into()).collect(),
            limit: None,
            min_score: None,
            include: vec![],
            exclude: vec![],
            mode: None,
        }
    }

    #[test]
    fn root_progress_is_scoped_and_does_not_leak_parent_inventory() {
        let base = tempfile::tempdir().unwrap();
        let a = base.path().join("a");
        let b = base.path().join("b");
        let child = a.join("child");
        for path in [&a, &b, &child] {
            std::fs::create_dir_all(path).unwrap();
        }
        let a = a.canonicalize().unwrap();
        let b = b.canonicalize().unwrap();
        let child = child.canonicalize().unwrap();
        let (server, _) = ownership_server(
            vec![child.clone()],
            vec![
                (a.clone(), IndexStatus::Ready),
                (b.clone(), IndexStatus::Ready),
            ],
        );
        let progress = seyk::IndexProgress {
            by_kind: [(
                "text".into(),
                seyk::KindProgress {
                    indexed: 41,
                    waiting: 9,
                    skipped: 2,
                },
            )]
            .into(),
            eta_seconds: Some(3),
        };
        seed_progress(&server.hub, &a, &progress);
        seed_progress(&server.hub, &b, &progress);
        let confined = server.root_progress();
        assert_eq!(confined.len(), 1);
        assert_eq!(confined[0].root, ResourceId::from_path(&child).unwrap());
        assert!(confined[0].progress.is_none());
        let own = server
            .hub
            .root_progress(Some(&[PolicyRoot::new(&a).unwrap()]));
        assert_eq!(own.len(), 1);
        assert_eq!(own[0].progress.as_ref(), Some(&progress));
        assert_eq!(server.hub.root_progress(None).len(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn root_progress_preserves_non_utf8_folder_identity() {
        use std::os::unix::ffi::OsStringExt;
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().canonicalize().unwrap();
        // APFS rejects creating these names; serialization must still preserve
        // arbitrary Unix path bytes without consulting the filesystem.
        let root = parent.join(std::ffi::OsString::from_vec(b"folder-\xff".to_vec()));
        let (server, _) = ownership_server(vec![parent], vec![(root.clone(), IndexStatus::Ready)]);
        let progress = server.root_progress();
        let encoded = serde_json::to_string(&progress).unwrap();
        assert!(encoded.contains("%FF"));
        let decoded: Vec<RootProgress> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded[0].root.to_file_path().as_ref(), Some(&root));
    }

    #[cfg(unix)]
    #[test]
    fn root_progress_hides_watchers_reached_through_a_directory_symlink() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().canonicalize().unwrap();
        let (allowed, outside) = (base.join("allowed"), base.join("outside"));
        let plain = allowed.join("plain");
        for dir in [&plain, &outside] {
            std::fs::create_dir_all(dir).unwrap();
        }
        let linked = allowed.join("linked");
        std::os::unix::fs::symlink(&outside, &linked).unwrap();
        let hub = Hub::new(Box::new(fake_engine), false);
        let confined = SeykServer::new(vec![allowed.clone()], hub.clone()).unwrap();
        // Another session on the shared hub explicitly watches the symlink.
        let other = SeykServer::new(vec![linked.clone()], hub.clone()).unwrap();
        let progress = seyk::IndexProgress {
            by_kind: [(
                "text".into(),
                seyk::KindProgress {
                    indexed: 7,
                    waiting: 1,
                    skipped: 0,
                },
            )]
            .into(),
            eta_seconds: Some(2),
        };
        for root in [&linked, &plain] {
            hub.watched
                .lock()
                .unwrap()
                .push((root.clone(), Arc::new(Mutex::new(IndexStatus::Ready))));
            seed_progress(&hub, root, &progress);
        }
        let roots = |progress: Vec<RootProgress>| -> Vec<_> {
            progress
                .into_iter()
                .map(|p| (p.root.to_file_path().unwrap(), p.progress.is_some()))
                .collect()
        };
        assert!(!progress_visible(&allowed, &linked));
        assert_eq!(roots(confined.root_progress()), [(plain.clone(), true)]);
        let status = confined.index_status_json().unwrap();
        let shown: Vec<RootProgress> =
            serde_json::from_value(status["index_progress"].clone()).unwrap();
        assert_eq!(roots(shown), [(plain, true)]);
        assert_eq!(roots(other.root_progress()), [(linked, true)]);
    }

    #[cfg(unix)]
    #[test]
    fn cached_outside_progress_stays_hidden_after_its_symlink_changes() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().canonicalize().unwrap();
        let (allowed, outside) = (base.join("allowed"), base.join("outside"));
        let plain = allowed.join("plain");
        for dir in [&plain, &outside] {
            std::fs::create_dir_all(dir).unwrap();
        }
        let linked = allowed.join("linked");
        std::os::unix::fs::symlink(&outside, &linked).unwrap();
        let hub = Hub::new(Box::new(fake_engine), false);
        let confined = SeykServer::new(vec![allowed.clone()], hub.clone()).unwrap();
        let progress = seyk::IndexProgress {
            by_kind: [(
                "text".into(),
                seyk::KindProgress {
                    indexed: 7,
                    waiting: 1,
                    skipped: 0,
                },
            )]
            .into(),
            eta_seconds: Some(2),
        };
        // Snapshots as the watchers measured them; no callback runs afterwards.
        for root in [&linked, &plain] {
            hub.watched
                .lock()
                .unwrap()
                .push((root.clone(), Arc::new(Mutex::new(IndexStatus::Ready))));
            seed_progress(&hub, root, &progress);
        }
        let shown = || -> Vec<(PathBuf, bool)> {
            let status = confined.index_status_json().unwrap();
            let status: Vec<RootProgress> =
                serde_json::from_value(status["index_progress"].clone()).unwrap();
            let mut request = params(&[]);
            request.queries = vec!["anything".into()];
            let search = confined.run_search(request).unwrap().index_progress;
            let roots = |progress: Vec<RootProgress>| -> Vec<_> {
                progress
                    .into_iter()
                    .map(|p| (p.root.to_file_path().unwrap(), p.progress.is_some()))
                    .collect()
            };
            let (status, search) = (roots(status), roots(search));
            assert_eq!(status, search);
            status
        };
        let in_scope_only = [(plain.clone(), true)];
        assert_eq!(shown(), in_scope_only, "outside symlink present");
        std::fs::remove_file(&linked).unwrap();
        assert_eq!(shown(), in_scope_only, "outside symlink unlinked");
        std::fs::create_dir(&linked).unwrap();
        assert_eq!(
            shown(),
            in_scope_only,
            "symlink replaced by an in-scope dir"
        );
    }

    #[cfg(unix)]
    #[test]
    fn watcher_snapshots_carry_the_root_their_listing_walked() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().canonicalize().unwrap();
        let outside = base.join("outside");
        std::fs::create_dir(&outside).unwrap();
        let linked = base.join("linked");
        std::os::unix::fs::symlink(&outside, &linked).unwrap();
        let hub = Hub::new(Box::new(fake_engine), false);
        let identity = Mutex::new(PassIdentity::default());
        let record = |e: ProgressEvent| hub.record_progress(&linked, &identity, &e);
        let counts = || ProgressEvent::IndexProgress {
            progress: seyk::IndexProgress::default(),
        };
        let walked = |roots: &[&Path]| ProgressEvent::Walked {
            roots: roots
                .iter()
                .map(|r| ResourceId::from_path(r).unwrap())
                .collect(),
        };
        let recorded = || {
            hub.progress
                .lock()
                .unwrap()
                .get(&linked)
                .map(|s| s.identity.clone())
        };
        record(counts());
        assert_eq!(recorded(), None, "no identity before discovery");
        record(ProgressEvent::Discovered { resources: 1 });
        record(counts());
        assert_eq!(recorded(), None, "a pass without a carried root");
        record(walked(&[&outside]));
        record(ProgressEvent::Discovered { resources: 1 });
        record(counts());
        assert_eq!(recorded(), Some(outside.clone()));
        // The carried root is kept even once the alias names something else.
        std::fs::remove_file(&linked).unwrap();
        std::fs::create_dir(&linked).unwrap();
        record(counts());
        assert_eq!(recorded(), Some(outside.clone()));
        record(walked(&[&outside, &linked]));
        record(ProgressEvent::Discovered { resources: 1 });
        record(counts());
        assert_eq!(recorded(), None, "an ambiguous walk records nothing");
    }

    #[cfg(all(unix, debug_assertions))]
    #[test]
    fn outside_counts_stay_hidden_when_the_alias_is_swapped_during_the_walk() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().canonicalize().unwrap();
        let (allowed, outside) = (base.join("allowed"), base.join("outside"));
        let plain = allowed.join("plain");
        for dir in [&plain, &outside] {
            std::fs::create_dir_all(dir).unwrap();
        }
        for i in 0..3 {
            std::fs::write(outside.join(format!("{i}.txt")), "outside inventory").unwrap();
        }
        std::fs::write(plain.join("p.txt"), "plain inventory").unwrap();
        let linked = allowed.join("linked");
        std::os::unix::fs::symlink(&outside, &linked).unwrap();
        let hub = Hub::new(Box::new(fake_engine), false);
        let confined = SeykServer::new(vec![allowed.clone()], hub.clone()).unwrap();
        for root in [&linked, &plain] {
            hub.watched
                .lock()
                .unwrap()
                .push((root.clone(), Arc::new(Mutex::new(IndexStatus::Ready))));
        }
        // Discovery has resolved linked -> outside; before it walks, replace
        // the alias with an ordinary, empty, in-scope directory.
        let alias = linked.clone();
        seyk::discovery::test_hooks::after_root_resolved(linked.clone(), move || {
            std::fs::remove_file(&alias).unwrap();
            std::fs::create_dir(&alias).unwrap();
        });
        // A real watcher pass: core events into the hub's own recorder.
        let engine = hub.engine().unwrap();
        for root in [&linked, &plain] {
            let identity = Mutex::new(PassIdentity::default());
            let progress = |e: ProgressEvent| hub.record_progress(root, &identity, &e);
            engine
                .index()
                .paths([root])
                .progress(&progress)
                .execute()
                .unwrap();
        }
        assert!(linked.is_dir() && !linked.is_symlink(), "hook ran");
        let measured = hub.progress.lock().unwrap()[&linked].clone();
        assert_eq!(measured.identity, outside, "counts carry the walked root");
        assert_eq!(measured.progress.by_kind["text"].indexed, 3);
        let roots = |progress: Vec<RootProgress>| -> Vec<_> {
            progress
                .into_iter()
                .map(|p| (p.root.to_file_path().unwrap(), p.progress.is_some()))
                .collect()
        };
        let status = confined.index_status_json().unwrap();
        let status: Vec<RootProgress> =
            serde_json::from_value(status["index_progress"].clone()).unwrap();
        let mut request = params(&[]);
        request.queries = vec!["inventory".into()];
        let search = confined.run_search(request).unwrap().index_progress;
        assert_eq!(roots(status), [(plain.clone(), true)], "index_status");
        assert_eq!(roots(search), [(plain, true)], "search index_progress");
    }

    /// Record a snapshot the way a watcher pass does: under the root's
    /// canonical identity at measurement time.
    fn seed_progress(hub: &Hub, root: &Path, progress: &seyk::IndexProgress) {
        hub.progress.lock().unwrap().insert(
            root.to_path_buf(),
            MeasuredProgress {
                identity: root.canonicalize().unwrap(),
                progress: progress.clone(),
            },
        );
    }

    #[cfg(unix)]
    #[test]
    fn index_status_revalidates_retargeted_and_newly_denied_roots() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let store = temp.path().join("store");
        std::fs::create_dir(&store).unwrap();
        let alias = temp.path().join("alias-dir");
        symlink("store", &alias).unwrap();
        let hub = Hub::new(Box::new(fake_engine), false);
        let retargeted = SeykServer::new(vec![alias.clone()], hub.clone()).unwrap();
        let denied = SeykServer::new(vec![alias.clone()], hub).unwrap();
        retargeted.index_status_json().unwrap();
        denied.index_status_json().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        std::fs::remove_file(&alias).unwrap();
        symlink(elsewhere.path(), &alias).unwrap();
        let error = retargeted.index_status_json().unwrap_err();
        assert!(error.to_string().contains("target changed"), "{error:#}");
        // Same canonical target as at session open, reached through a secret chain.
        std::fs::remove_file(&alias).unwrap();
        symlink("store", temp.path().join("vault.opvault")).unwrap();
        symlink("vault.opvault", &alias).unwrap();
        assert_eq!(alias.canonicalize().unwrap(), store.canonicalize().unwrap());
        assert!(denied.index_status_json().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn root_revalidation_preserves_exposed_error_types() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let store = temp.path().join("store");
        std::fs::create_dir(&store).unwrap();
        let alias = temp.path().join("alias");
        symlink("store", &alias).unwrap();
        let server =
            SeykServer::new(vec![alias.clone()], Hub::new(Box::new(fake_engine), false)).unwrap();
        std::fs::remove_file(&alias).unwrap();
        let missing = server.validate_roots().unwrap_err();
        assert_eq!(
            missing.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::NotFound
        );
        symlink("store", temp.path().join("vault.opvault")).unwrap();
        symlink("vault.opvault", &alias).unwrap();
        let denied = server.validate_roots().unwrap_err();
        assert!(matches!(
            denied.downcast_ref::<seyk::Error>(),
            Some(seyk::Error::SecretFile { .. })
        ));
    }

    #[test]
    fn mcp_returns_the_same_opt_in_counts_as_the_library() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "quartz lighthouse").unwrap();
        std::fs::write(dir.path().join("b.txt"), "quartz harbor").unwrap();
        let (server, _) = ownership_server(vec![dir.path().to_path_buf()], Vec::new());
        let mut request = params(&[]);
        request.queries = vec!["quartz".into()];
        request.kind_counts = true;
        request.limit = Some(1);
        let response = server.run_search(request).unwrap();
        assert_eq!(response.kind_counts.unwrap()["text"], response.hits.len());
        assert_eq!(response.hits.len(), 1);
    }

    /// Seed statuses without live threads so only the search can embed documents.
    fn ownership_server(
        roots: Vec<PathBuf>,
        watched: Vec<(PathBuf, IndexStatus)>,
    ) -> (SeykServer, Arc<AtomicUsize>) {
        let documents = Arc::new(AtomicUsize::new(0));
        let count = documents.clone();
        let hub = Hub::new(Box::new(move || counted_engine(count.clone())), false);
        let mut server = SeykServer::new(roots, hub.clone()).unwrap();
        let watched: Vec<_> = watched
            .into_iter()
            .map(|(path, state)| (path, Arc::new(Mutex::new(state))))
            .collect();
        server.watchers = Arc::new(
            watched
                .iter()
                .filter(|(path, _)| {
                    server
                        .roots
                        .iter()
                        .any(|root| root.uri_path().starts_with(path))
                })
                .map(|(_, cell)| cell.clone())
                .collect(),
        );
        *hub.watched.lock().unwrap() = watched;
        (server, documents)
    }

    /// Reproducible synthetic workload, deliberately separate from CI timing assertions.
    /// Run with: cargo test --bin seyk measure_disjoint_watcher_inline_sync -- --ignored --nocapture
    #[test]
    #[ignore = "synthetic latency measurement; no timing threshold"]
    #[cfg(debug_assertions)]
    fn measure_disjoint_watcher_inline_sync() {
        use seyk_core::LeaseEvent;
        use std::sync::mpsc;

        struct SampleWatcher(Option<std::thread::JoinHandle<()>>);
        impl SampleWatcher {
            fn join(mut self) {
                if let Some(worker) = self.0.take() {
                    worker.join().unwrap();
                }
            }
        }
        impl Drop for SampleWatcher {
            fn drop(&mut self) {
                if let Some(worker) = self.0.take() {
                    let joined = worker.join();
                    if !std::thread::panicking() {
                        joined.unwrap();
                    }
                }
            }
        }

        struct TimedEmbedder {
            inner: WordHash,
            entered: mpsc::Sender<()>,
            release: Mutex<mpsc::Receiver<()>>,
            first: std::sync::atomic::AtomicBool,
        }
        impl Embedder for TimedEmbedder {
            fn identity(&self) -> &ModelIdentity {
                self.inner.identity()
            }
            fn supports(&self, modality: Modality) -> bool {
                self.inner.supports(modality)
            }
            fn embed_query(&self, query: &str) -> seyk_embed::Result<Embedding> {
                self.inner.embed_query(query)
            }
            fn embed_batch(
                &self,
                requests: &[seyk_embed::EmbeddingRequest<'_>],
            ) -> seyk_embed::Result<Vec<Embedding>> {
                if requests
                    .iter()
                    .all(|r| r.task == seyk_embed::EmbeddingTask::SearchDocument)
                {
                    let inputs: Vec<_> = requests.iter().map(|r| r.input.clone()).collect();
                    self.embed_resources(&inputs)
                } else {
                    self.inner.embed_batch(requests)
                }
            }
            fn embed_resources(
                &self,
                inputs: &[ResourceInput<'_>],
            ) -> seyk_embed::Result<Vec<Embedding>> {
                let background = inputs.iter().any(|input| {
                    matches!(input,
                    ResourceInput::Text { text, .. } if text.contains("watcher"))
                });
                if background && self.first.swap(false, Ordering::SeqCst) {
                    self.entered.send(()).unwrap();
                    self.release
                        .lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(20))
                        .unwrap();
                }
                // Fixed synthetic inference service time; no model or private data.
                std::thread::sleep(Duration::from_millis(2 * inputs.len() as u64));
                self.inner.embed_resources(inputs)
            }
        }

        for files in [0, 64, 256] {
            for sample in 1..=5 {
                let temp = tempfile::tempdir().unwrap();
                let a = temp.path().join("watched");
                let b = temp.path().join("inline");
                std::fs::create_dir(&a).unwrap();
                std::fs::create_dir(&b).unwrap();
                for i in 0..files {
                    std::fs::write(
                        a.join(format!("{i:04}.txt")),
                        format!("watcher synthetic document {i:04} quartz pebble"),
                    )
                    .unwrap();
                }
                std::fs::write(b.join("query.txt"), "inline synthetic quartz pebble").unwrap();
                let (entered_tx, entered_rx) = mpsc::channel();
                let (release_tx, release_rx) = mpsc::channel();
                let engine = Arc::new(
                    SearchEngine::builder()
                        .embedder(TimedEmbedder {
                            inner: WordHash(
                                ModelIdentity {
                                    model_id: "test/word-hash".into(),
                                    revision: "1".into(),
                                    variant: "test".into(),
                                    dimension: Dimension::D128,
                                    normalization: "l2".into(),
                                    query_prompt: String::new(),
                                    document_prompt: String::new(),
                                    max_tokens: 512,
                                },
                                Arc::new(AtomicUsize::new(0)),
                            ),
                            entered: entered_tx,
                            release: Mutex::new(release_rx),
                            first: std::sync::atomic::AtomicBool::new(true),
                        })
                        .cache(seyk::SqliteCache::open(temp.path().join("cache.db")).unwrap())
                        .batch_size(16)
                        .build()
                        .unwrap(),
                );
                // The inline root has no watcher. A real Hub watcher owns only A.
                let hub = Hub::new(Box::new(|| unreachable!()), false);
                *hub.engine.lock().unwrap() = Some(engine.clone());
                let server = SeykServer::new(vec![b.clone()], hub.clone()).unwrap();
                let watcher = if files > 0 {
                    let a = a.canonicalize().unwrap();
                    let cell = Arc::new(Mutex::new(IndexStatus::LoadingModel));
                    hub.watched.lock().unwrap().push((a.clone(), cell.clone()));
                    let watcher = std::thread::spawn(move || {
                        hub.background_index_with_options(
                            &a,
                            &cell,
                            &WatchOptions {
                                stop_after_first_pass: true,
                                ..WatchOptions::default()
                            },
                        );
                    });
                    // Keep the handle even if the arrival signal fails, so cleanup joins.
                    let arrival = entered_rx.recv_timeout(Duration::from_secs(20));
                    if arrival.is_err() {
                        let _ = release_tx.send(());
                        watcher.join().unwrap();
                        panic!("watcher did not enter its first batch: {arrival:?}");
                    }
                    Some(watcher)
                } else {
                    None
                };
                let watcher = SampleWatcher(watcher);
                let started = Instant::now();
                let observations = Arc::new(Mutex::new(Vec::new()));
                let recorded = observations.clone();
                let inline_thread = std::thread::current().id();
                engine.observe_indexing_lease(Arc::new(move |event| {
                    if std::thread::current().id() != inline_thread {
                        return;
                    }
                    recorded.lock().unwrap().push((event, started.elapsed()));
                    if event == LeaseEvent::Contended {
                        release_tx.send(()).unwrap();
                    }
                }));
                let response = server.run_search(params(&[b]));
                let elapsed = started.elapsed();
                // Join while the temp roots and SQLite cache still exist, including
                // if the search returned an error. No watcher survives a sample.
                watcher.join();
                let response = response.unwrap();
                assert_eq!(response.segments_embedded, 1);
                assert_eq!(response.hits.len(), 1);
                let observed = observations.lock().unwrap();
                let wait = if files == 0 {
                    assert_eq!(observed.len(), 1);
                    Duration::ZERO
                } else {
                    assert_eq!(observed.len(), 2);
                    assert_eq!(observed[0].0, LeaseEvent::Contended);
                    assert_eq!(observed[1].0, LeaseEvent::Acquired);
                    observed[1].1 - observed[0].1
                };
                println!(
                    "watcher_files={files} watcher_segments={files} inline_files=1 inline_segments=1 batch=16 synthetic_ms_per_segment=2 cache=sqlite sample={sample} inline_ms={:.3} lease_wait_ms={:.3}",
                    elapsed.as_secs_f64() * 1000.0,
                    wait.as_secs_f64() * 1000.0
                );
                drop(observed);
                engine.observe_indexing_lease(Arc::new(|_| {}));
            }
        }
    }

    #[test]
    fn inline_sync_uses_requested_coverage_not_session_display_status() {
        // Fresh caches per case prove positive inline cases actually embed too.
        for (states, requested, expected) in [
            (vec![stopped(), indexing()], vec![1], 0),
            (vec![stopped(), indexing()], vec![0, 1], 0),
            (vec![stopped(), indexing()], vec![0], 1),
            (vec![stopped(), stopped()], vec![0, 1], 2),
            (vec![], vec![0, 1], 2),
            (vec![indexing()], vec![1], 1),
            (
                vec![
                    stopped(),
                    IndexStatus::PassFailed {
                        error: "retry".into(),
                    },
                ],
                vec![0, 1],
                0,
            ),
            (vec![stopped(), IndexStatus::Ready], vec![0, 1], 0),
            (vec![stopped(), IndexStatus::LoadingModel], vec![0, 1], 0),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let roots: Vec<_> = ["a", "b"]
                .into_iter()
                .map(|name| {
                    let root = dir.path().join(name);
                    std::fs::create_dir(&root).unwrap();
                    std::fs::write(root.join("note.txt"), format!("quartz pebble {name}")).unwrap();
                    root.canonicalize().unwrap()
                })
                .collect();
            let has_watchers = !states.is_empty();
            let display = serde_json::to_value(combine(states.clone())).unwrap();
            let (server, documents) =
                ownership_server(roots.clone(), roots.iter().cloned().zip(states).collect());
            let paths: Vec<_> = requested.into_iter().map(|i| roots[i].clone()).collect();
            let response = server.run_search(params(&paths)).unwrap();
            assert_eq!(
                documents.load(Ordering::SeqCst),
                expected,
                "paths: {paths:?}"
            );
            assert_eq!(response.segments_embedded, expected);
            let json = serde_json::to_value(response).unwrap();
            if has_watchers {
                assert_eq!(json["index"], display);
            } else {
                assert!(json.get("index").is_none());
            }
        }
    }

    #[test]
    fn active_child_owns_search_of_parent_even_outside_session_cells() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let child = root.join("child");
        std::fs::create_dir(&child).unwrap();
        std::fs::write(child.join("note.txt"), "quartz pebble").unwrap();
        let (server, documents) = ownership_server(
            vec![root.clone()],
            vec![(root.clone(), stopped()), (child, indexing())],
        );
        assert_eq!(server.watchers.len(), 1);
        server.run_search(params(&[root])).unwrap();
        assert_eq!(documents.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn child_first_registration_checks_later_parent_watcher() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let child = root.join("child");
        std::fs::create_dir(&child).unwrap();
        std::fs::write(child.join("note.txt"), "quartz pebble").unwrap();
        // Failed startup gives real registration without leaving live indexers.
        let hub = Hub::new(Box::new(|| Err(anyhow!("test startup"))), true);
        let inner = SeykServer::new(vec![child.clone()], hub.clone()).unwrap();
        wait_for(&inner, |s| matches!(s, IndexStatus::Stopped { .. }));
        let outer = SeykServer::new(vec![root], hub.clone()).unwrap();
        wait_for(&outer, |s| matches!(s, IndexStatus::Stopped { .. }));
        assert_eq!(hub.watched().len(), 2);
        *outer.watchers[0].lock().unwrap() = indexing();
        let documents = Arc::new(AtomicUsize::new(0));
        *hub.engine.lock().unwrap() = Some(Arc::new(counted_engine(documents.clone()).unwrap()));
        let response = inner.run_search(params(&[child])).unwrap();
        assert!(matches!(response.index, Some(IndexStatus::Stopped { .. })));
        assert_eq!(documents.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn poisoned_overlapping_status_does_not_enable_inline_sync() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let child = root.join("child");
        std::fs::create_dir(&child).unwrap();
        std::fs::write(child.join("note.txt"), "quartz pebble").unwrap();
        let (server, documents) = ownership_server(
            vec![root.clone()],
            vec![(root.clone(), stopped()), (child, stopped())],
        );
        let cell = server.hub.watched.lock().unwrap()[1].1.clone();
        let _ = std::thread::spawn(move || {
            let _guard = cell.lock().unwrap();
            panic!("poison status for test");
        })
        .join();
        let error = server.run_search(params(&[root])).unwrap_err();
        assert!(error.to_string().contains("index status lock poisoned"));
        assert_eq!(documents.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn search_includes_cannot_admit_secrets() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::write(root.join("safe.txt"), "quartz pebble").unwrap();
        std::fs::write(root.join("id_rsa.txt"), "quartz secret").unwrap();
        let (server, documents) = ownership_server(vec![root.clone()], vec![]);
        let mut request = params(std::slice::from_ref(&root));
        request.include = vec!["*.txt".into()];
        let response = server.run_search(request).unwrap();
        assert_eq!(documents.load(Ordering::SeqCst), 1);
        assert!(!serde_json::to_string(&response).unwrap().contains("id_rsa"));
        let error = server
            .run_search(params(&[root.join("id_rsa.txt")]))
            .unwrap_err();
        assert!(error.to_string().contains("refusing secret file"));
    }

    #[cfg(unix)]
    #[test]
    fn confine_refuses_secret_alias_before_canonicalizing() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::write(root.join("safe.txt"), "safe").unwrap();
        std::os::unix::fs::symlink(root.join("safe.txt"), root.join("id_rsa")).unwrap();
        let root = PolicyRoot::new(&root).unwrap();
        let error = confine(Path::new("id_rsa"), &[root]).unwrap_err();
        assert!(error.to_string().contains("refusing secret file"));
    }

    #[test]
    fn confine_normalizes_parent_traversal_in_relative_and_absolute_requests() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir(root.join("child")).unwrap();
        let file = root.join("safe.txt");
        std::fs::write(&file, "safe").unwrap();
        let roots = vec![PolicyRoot::new(&root).unwrap()];
        for supplied in [
            PathBuf::from("child/../safe.txt"),
            root.join("child/../safe.txt"),
        ] {
            assert_eq!(confine(&supplied, &roots).unwrap(), file);
        }
    }

    #[cfg(unix)]
    #[test]
    fn confine_normalizes_parent_traversal_with_filesystem_symlink_semantics() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("target/child")).unwrap();
        std::os::unix::fs::symlink("target/child", root.join("alias")).unwrap();
        let file = root.join("target/safe.txt");
        std::fs::write(&file, "safe").unwrap();
        let roots = vec![PolicyRoot::new(&root).unwrap()];
        for supplied in [
            PathBuf::from("alias/../safe.txt"),
            root.join("alias/../safe.txt"),
        ] {
            assert_eq!(confine(&supplied, &roots).unwrap(), file);
        }
    }

    #[test]
    fn confine_rejects_paths_outside_roots() {
        let root = std::env::temp_dir().canonicalize().unwrap();
        let inside = root.join("seyk-confine-test");
        std::fs::create_dir_all(&inside).unwrap();
        let roots = vec![PolicyRoot::new(&inside).unwrap()];
        assert_eq!(confine(Path::new("."), &roots).unwrap(), inside);
        assert!(confine(Path::new(".."), &roots).is_err());
        assert!(confine(Path::new("/"), &roots).is_err());
    }
}
