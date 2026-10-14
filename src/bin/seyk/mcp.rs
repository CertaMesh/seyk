//! The MCP search tool: one [`Hub`] holding the model and the folder watchers,
//! and one [`SeykServer`] per agent session.
//!
//! The server only finds passages. The calling agent reads them and writes any
//! answer, so no generation happens here.
//!
//! A hub indexes each session's folders in the background and re-indexes on
//! change, and searches read only what is indexed. A search therefore never waits
//! for a large first index; it reports progress instead. `seyk mcp` normally runs
//! the hub inside the shared daemon (see `daemon.rs`), so every session shares
//! one model; `--standalone` runs it in-process.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
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
The server indexes its folders in the background; while `index.state` is \"indexing\", results cover only what is indexed so far.";

pub type EngineFactory = Box<dyn Fn() -> Result<SearchEngine> + Send + Sync>;

/// What every session shares: the model, loaded once, and one watcher per
/// folder. Sessions whose folders are already watched reuse that watcher.
pub struct Hub {
    engine: Mutex<Option<Arc<SearchEngine>>>,
    factory: EngineFactory,
    /// `false` turns background indexing off; searches then embed inline.
    background: bool,
    watched: Mutex<Vec<(PathBuf, Arc<Mutex<IndexStatus>>)>>,
    diagnostics: Mutex<std::collections::HashMap<PathBuf, WatcherDiagnostics>>,
    progress: Mutex<std::collections::BTreeMap<PathBuf, seyk::IndexProgress>>,
}

/// One agent session: the folders it may search, and the watchers covering them.
#[derive(Clone)]
pub struct SeykServer {
    #[cfg(unix)]
    local_content: Option<Arc<seyk::local_content::LocalContentProvider>>,
    roots: Arc<Vec<PathBuf>>,
    // Pin authorization even if a logical root symlink is later retargeted.
    resolved_roots: Arc<Vec<PathBuf>>,
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
                    path.strip_prefix(&real).is_ok_and(|relative| {
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
    pub fn root_progress(&self, allowed: Option<&[PathBuf]>) -> Vec<RootProgress> {
        let Ok(watched) = self.watched.lock() else {
            return Vec::new();
        };
        let Ok(progress) = self.progress.lock() else {
            return Vec::new();
        };
        let mut result = std::collections::BTreeMap::new();
        for (root, _) in watched.iter() {
            if allowed.is_none_or(|roots| roots.iter().any(|r| root.starts_with(r))) {
                result.insert(root.clone(), progress.get(root).cloned());
            } else if let Some(allowed) = allowed {
                for child in allowed.iter().filter(|r| r.starts_with(root)) {
                    result.entry(child.clone()).or_insert(None);
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
        let progress = |e: ProgressEvent| {
            if matches!(e, ProgressEvent::Discovered { .. }) {
                indexing(0, 0);
                if let Ok(mut roots) = self.progress.lock() {
                    roots.remove(root);
                }
            }
            if let ProgressEvent::IndexProgress { ref progress } = e
                && let Ok(mut roots) = self.progress.lock()
            {
                roots.insert(root.to_path_buf(), progress.clone());
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

impl SeykServer {
    pub fn new(roots: Vec<PathBuf>, hub: Arc<Hub>) -> Result<Self> {
        let roots = roots
            .into_iter()
            .map(|r| {
                seyk::discovery::ensure_allowed_path(&r)?;
                r.canonicalize()
                    .with_context(|| format!("root {}", r.display()))?;
                seyk::discovery::absolute_logical_path(&r)
                    .with_context(|| format!("root {}", r.display()))
            })
            .collect::<Result<Vec<_>>>()?;
        if roots.is_empty() {
            return Err(anyhow!("no folders to serve"));
        }
        let resolved_roots = roots
            .iter()
            .map(|r| r.canonicalize())
            .collect::<std::io::Result<Vec<_>>>()?;
        let watchers = hub.watch_roots(&roots)?;
        Ok(Self {
            #[cfg(unix)]
            local_content: None,
            roots: Arc::new(roots),
            resolved_roots: Arc::new(resolved_roots),
            hub,
            watchers: Arc::new(watchers),
            tool_router: Self::tool_router(),
        })
    }

    fn validate_roots(&self) -> Result<()> {
        for (logical, resolved) in self.roots.iter().zip(self.resolved_roots.iter()) {
            seyk::discovery::ensure_allowed_path(logical)?;
            if logical.canonicalize()? != *resolved {
                return Err(anyhow!("server root target changed: {}", logical.display()));
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
                    .any(|session| paths_overlap(session, root))
            })
            .collect();
        matching.sort_by_key(|(root, _)| *root);
        Ok(combine_diagnostics(
            matching
                .into_iter()
                .map(|(_, record)| record.confined(&self.roots)),
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
        let paths = if p.paths.is_empty() {
            self.roots.to_vec()
        } else {
            p.paths
                .iter()
                .map(|s| self.confine(Path::new(s)))
                .collect::<Result<Vec<_>>>()?
        };

        let status = self.status()?;
        // Display status is session-wide; ownership depends on the requested
        // paths and every overlapping hub watcher, including other sessions.
        // PassFailed still owns its root because it retries on the next change.
        let inline = self.hub.may_sync_inline(&paths)?;
        let engine = self.hub.engine()?;
        let mut queries = queries.into_iter();
        let first = queries.next().expect("checked non-empty");
        let mut req = queries
            .fold(engine.search(first), |r, q| r.query(q))
            .paths(&paths)
            .kind_counts(p.kind_counts)
            .limit(p.limit.unwrap_or(10).clamp(1, MAX_LIMIT))
            .sync(inline);
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
        resp.index = status;
        resp.index_progress = self.hub.root_progress(Some(&self.roots));
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
                self.roots[0].join(path)
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

// Compare filesystem identities for watcher ownership without discarding the
// logical roots used by discovery and policy checks.
fn paths_overlap(a: &Path, b: &Path) -> bool {
    let a = a.canonicalize().unwrap_or_else(|_| a.to_path_buf());
    let b = b.canonicalize().unwrap_or_else(|_| b.to_path_buf());
    a.starts_with(&b) || b.starts_with(&a)
}

/// Resolve `path` against the first root and refuse anything outside the roots,
/// so an agent cannot read previews of arbitrary files through this server.
fn confine(path: &Path, roots: &[PathBuf]) -> Result<PathBuf> {
    let path = seyk::expand_tilde(path);
    let joined = if path.is_absolute() {
        path
    } else {
        roots[0].join(path)
    };
    seyk::discovery::ensure_allowed_path(&joined)?;
    let real = joined
        .canonicalize()
        .with_context(|| format!("{} does not exist", joined.display()))?;
    let mut logical = None;
    for root in roots {
        let resolved = root
            .canonicalize()
            .with_context(|| format!("root {}", root.display()))?;
        if let Ok(relative) = real.strip_prefix(&resolved) {
            let candidate = root.join(relative);
            // Check every applicable logical root, including for absolute cached IDs.
            seyk::discovery::ensure_allowed_path(&candidate)?;
            let supplied = if joined.starts_with(root) {
                joined.clone()
            } else if let Ok(relative) = joined.strip_prefix(&resolved) {
                root.join(relative)
            } else {
                candidate
            };
            // Keep directory symlinks supplied below a broader server root too.
            logical.get_or_insert(supplied);
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
                .map(|r| r.display().to_string())
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
        description = "Current indexing state and per-root resource counts with an observed-throughput ETA. Counts describe the current or last completed pass; null progress is unavailable."
    )]
    async fn index_status(&self) -> Result<CallToolResult, McpError> {
        let index = self
            .status()
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let status = serde_json::json!({
            "schema_version": crate::JSON_SCHEMA_VERSION,
            "index": index,
            "index_progress": self.hub.root_progress(Some(&self.roots)),
        });
        Ok(CallToolResult::success(vec![ContentBlock::json(status)?]))
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
            .with_server_info(Implementation::new(name, env!("CARGO_PKG_VERSION")))
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
    let resolved_roots = roots
        .iter()
        .map(|root| root.canonicalize())
        .collect::<std::io::Result<Vec<_>>>()?;
    let watchers = hub.watch_roots(&roots)?;
    Ok(SeykServer {
        local_content: Some(provider),
        resolved_roots: Arc::new(resolved_roots),
        roots: Arc::new(roots),
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
        assert_eq!(server.resolved_roots.as_ref(), &[stored]);
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
        let valid = search(&server, "quartz");
        assert_eq!(valid.schema_version, 7);
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
        assert_eq!(response["schema_version"], 7);
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
        assert!(search(&server, "quartz").hits.is_empty());
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
        let roots = vec![temp.path().to_path_buf()];
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
        let roots = vec![inside.path().canonicalize().unwrap()];
        assert!(confine(&link, &roots).is_err());
        let server = SeykServer::new(roots, Hub::new(Box::new(fake_engine), false)).unwrap();
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

    #[test]
    fn background_index_reaches_ready_and_follows_changes() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("keeper.md"),
            "lighthouse keeper counts ships",
        )
        .unwrap();
        let hub = Hub::new(Box::new(fake_engine), true);
        let server = SeykServer::new(vec![dir.path().into()], hub).unwrap();

        wait_for(&server, |s| matches!(s, IndexStatus::Ready));
        let r = search(&server, "lighthouse ships");
        assert_eq!(r.hits.len(), 1);
        assert_eq!(r.segments_embedded, 0, "searches read the cache only");

        std::fs::write(dir.path().join("bread.md"), "grandmother bakes bread").unwrap();
        let started = Instant::now();
        while search(&server, "grandmother bread").hits.len() < 2 {
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "change never indexed"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
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
        let diagnostic = |path: PathBuf, reason: &str| seyk::PageDiagnostic {
            code: seyk::SkipCode::PdfParserPanicked,
            resource: ResourceId::from_path(path).unwrap(),
            page: 1,
            reason: reason.into(),
        };
        let pages = vec![
            diagnostic(allowed.join("bad.pdf"), "allowed"),
            diagnostic(sibling.join("secret.pdf"), "hidden"),
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
        let confined = snapshot.confined(&[allowed]);
        assert_eq!(confined.pages_total, 1);
        assert_eq!(confined.page_diagnostics.len(), 1);
        assert_eq!(confined.skipped.len(), 1);
        assert!(!serde_json::to_string(&confined).unwrap().contains("hidden"));
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
        server
            .hub
            .progress
            .lock()
            .unwrap()
            .insert(a.clone(), progress.clone());
        server
            .hub
            .progress
            .lock()
            .unwrap()
            .insert(b.clone(), progress.clone());
        let confined = server.hub.root_progress(Some(&server.roots));
        assert_eq!(confined.len(), 1);
        assert_eq!(confined[0].root.to_file_path().as_ref(), Some(&child));
        assert!(confined[0].progress.is_none());
        let own = server.hub.root_progress(Some(&[a]));
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
        let progress = server.hub.root_progress(Some(&server.roots));
        let encoded = serde_json::to_string(&progress).unwrap();
        assert!(encoded.contains("%FF"));
        let decoded: Vec<RootProgress> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded[0].root.to_file_path().as_ref(), Some(&root));
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
                .filter(|(path, _)| server.roots.iter().any(|root| root.starts_with(path)))
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
        let error = confine(Path::new("id_rsa"), &[root]).unwrap_err();
        assert!(error.to_string().contains("refusing secret file"));
    }

    #[test]
    fn confine_rejects_paths_outside_roots() {
        let root = std::env::temp_dir().canonicalize().unwrap();
        let inside = root.join("seyk-confine-test");
        std::fs::create_dir_all(&inside).unwrap();
        let roots = vec![inside.clone()];
        assert_eq!(confine(Path::new("."), &roots).unwrap(), inside);
        assert!(confine(Path::new(".."), &roots).is_err());
        assert!(confine(Path::new("/"), &roots).is_err());
    }
}
