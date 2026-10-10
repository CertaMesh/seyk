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
}

/// One agent session: the folders it may search, and the watchers covering them.
#[derive(Clone)]
pub struct SeykServer {
    roots: Arc<Vec<PathBuf>>,
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

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
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
    cache_status: Option<seyk::CacheStats>,
    segments_searched: usize,
    files: usize,
    segments_embedded: usize,
    took_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<IndexStatus>,
}

impl Hub {
    pub fn new(factory: EngineFactory, background: bool) -> Arc<Self> {
        Arc::new(Self {
            engine: Mutex::new(None),
            factory,
            background,
            watched: Mutex::new(Vec::new()),
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
            if let Some((_, cell)) = watched.iter().find(|(w, _)| root.starts_with(w)) {
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
            if paths
                .iter()
                .any(|path| path.starts_with(root) || root.starts_with(path))
            {
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
            if let ProgressEvent::Embedding { done, total } = e {
                indexing(done, total);
            }
        };
        let result = watch(
            &engine,
            &[root.to_path_buf()],
            &DiscoveryOptions::default(),
            &WatchOptions::default(),
            Some(&progress),
            |pass| {
                set(match pass {
                    Ok(_) => IndexStatus::Ready,
                    Err(e) => IndexStatus::PassFailed {
                        error: e.to_string(),
                    },
                })
            },
        );
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
                r.canonicalize()
                    .with_context(|| format!("root {}", r.display()))
            })
            .collect::<Result<Vec<_>>>()?;
        if roots.is_empty() {
            return Err(anyhow!("no folders to serve"));
        }
        let watchers = hub.watch_roots(&roots)?;
        Ok(Self {
            roots: Arc::new(roots),
            hub,
            watchers: Arc::new(watchers),
            tool_router: Self::tool_router(),
        })
    }

    fn run_extract(&self, params: ExtractParams) -> Result<seyk::ExtractedContent> {
        let hit: SelectedHit = serde_json::from_value(params.hit).map_err(|_| {
            anyhow!("invalid selected hit: resource, locator and metadata are required")
        })?;
        let path = hit
            .resource
            .to_file_path()
            .ok_or_else(|| anyhow!("only local resources may be extracted"))?;
        // Authorization is independent of the caller's claimed content hash.
        let path = confine(&path, &self.roots)?;
        let hit = seyk::SearchHit {
            resource: ResourceId::from_path(path)?,
            locator: hit.locator,
            metadata: hit.metadata,
            score: 0.0,
            signals: Default::default(),
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

    fn run_search(&self, p: SearchParams) -> Result<Response> {
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
                .map(|s| confine(Path::new(s), &self.roots))
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
        resp.cache_status = Some(engine.with_cache(|cache| cache.stats())?);
        Ok(resp)
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

/// Resolve `path` against the first root and refuse anything outside the roots,
/// so an agent cannot read previews of arbitrary files through this server.
fn confine(path: &Path, roots: &[PathBuf]) -> Result<PathBuf> {
    let path = seyk::expand_tilde(path);
    let joined = if path.is_absolute() {
        path
    } else {
        roots[0].join(path)
    };
    let real = joined
        .canonicalize()
        .with_context(|| format!("{} does not exist", joined.display()))?;
    if roots.iter().any(|r| real.starts_with(r)) {
        Ok(real)
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
        segments_searched: r.stats.segments_searched,
        files: r.stats.sync.resources_discovered,
        segments_embedded: r.stats.sync.segments_embedded,
        took_ms: r.stats.total_ms,
        index: None,
        cache_status: None,
    }
}

#[tool_router]
impl SeykServer {
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
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("seyk", env!("CARGO_PKG_VERSION")))
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
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let service = server.serve(rmcp::transport::stdio()).await?;
        service.waiting().await?;
        Ok(())
    })
}

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
        Ok(SearchEngine::builder()
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
            .cache(MemoryCache::new())
            .build()?)
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
        assert_eq!(response["schema_version"], 5);
        assert_eq!(response["cache_status"]["in_synced_folder"], false);
        assert!(response["cache_status"]["backup_excluded"].is_null());
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
            queries: vec!["quartz".into()],
            paths: paths.iter().map(|p| p.to_str().unwrap().into()).collect(),
            limit: None,
            min_score: None,
            include: vec![],
            exclude: vec![],
            mode: None,
        }
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
