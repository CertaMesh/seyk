//! `seyk mcp`: a stdio MCP server exposing semantic search to agents.
//!
//! The server only finds passages. The calling agent reads them and writes any
//! answer, so no generation happens here. The model stays loaded between calls.
//!
//! By default a background thread indexes the roots on start and re-indexes on
//! change, and searches read only what is indexed. A search therefore never waits
//! for a large first index; it reports progress instead.

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
use seyk::{DiscoveryOptions, ProgressEvent, ResourceLocator, SearchEngine, SearchResults};

const MAX_QUERIES: usize = 8;
const MAX_LIMIT: usize = 50;

const INSTRUCTIONS: &str = "Local semantic search over the user's files with EmbeddingGemma 2. \
It matches meaning, not keywords, so describe what the passage would say rather than asking \
the question. Pass 2-4 different phrasings in one call; each hit keeps its best score. Hits \
are pointers: read the file at the given lines before answering, and cite path:lines. \
Prefer grep for exact identifiers or strings. The server indexes its folders in the \
background; while `index.state` is \"indexing\", results cover only what is indexed so far.";

type EngineFactory = Box<dyn Fn() -> Result<SearchEngine> + Send + Sync>;

#[derive(Clone)]
pub struct SeykServer {
    roots: Arc<Vec<PathBuf>>,
    engine: Arc<Mutex<Option<Arc<SearchEngine>>>>,
    factory: Arc<EngineFactory>,
    /// `None` when background indexing is off; searches then embed inline.
    status: Option<Arc<Mutex<IndexStatus>>>,
    tool_router: ToolRouter<Self>,
}

/// Background indexing, as agents see it. Every transition replaces it whole,
/// so a stale phase cannot outlive the event that ended it.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum IndexStatus {
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
    /// Nothing keeps the cache fresh any more, so searches embed inline.
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
}

#[derive(Debug, Serialize)]
struct Hit {
    path: PathBuf,
    /// "start-end" line range, when the hit is a text segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    lines: Option<String>,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    heading: Option<String>,
    matched_query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    preview: Option<String>,
}

#[derive(Debug, Serialize)]
struct Response {
    hits: Vec<Hit>,
    segments_searched: usize,
    files: usize,
    segments_embedded: usize,
    took_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<IndexStatus>,
}

impl SeykServer {
    pub fn new(roots: Vec<PathBuf>, factory: EngineFactory, background: bool) -> Result<Self> {
        let roots = roots
            .into_iter()
            .map(|r| {
                r.canonicalize()
                    .with_context(|| format!("root {}", r.display()))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            roots: Arc::new(roots),
            engine: Arc::new(Mutex::new(None)),
            factory: Arc::new(factory),
            status: background.then(|| Arc::new(Mutex::new(IndexStatus::LoadingModel))),
            tool_router: Self::tool_router(),
        })
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

    fn set_status(&self, s: IndexStatus) {
        if let Some(m) = &self.status
            && let Ok(mut g) = m.lock()
        {
            *g = s;
        }
    }

    /// `None` means background indexing is off, never that the lock failed.
    fn status(&self) -> Result<Option<IndexStatus>> {
        self.status
            .as_ref()
            .map(|m| {
                m.lock()
                    .map(|g| g.clone())
                    .map_err(|_| anyhow!("index status lock poisoned"))
            })
            .transpose()
    }

    /// Index the roots, then keep them fresh. Runs on its own thread for the
    /// server's lifetime.
    fn background_index(&self) {
        let engine = match self.engine() {
            Ok(e) => e,
            Err(e) => {
                return self.set_status(IndexStatus::Stopped {
                    error: format!("{e:#}"),
                });
            }
        };
        let indexing = |done, total| {
            self.set_status(IndexStatus::Indexing {
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
            &self.roots,
            &DiscoveryOptions::default(),
            &WatchOptions::default(),
            Some(&progress),
            |pass| {
                self.set_status(match pass {
                    Ok(_) => IndexStatus::Ready,
                    Err(e) => IndexStatus::PassFailed {
                        error: e.to_string(),
                    },
                })
            },
        );
        self.set_status(IndexStatus::Stopped {
            error: match result {
                Ok(()) => "file watcher stopped".into(),
                Err(e) => e.to_string(),
            },
        });
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
        // Embed inline only when no background indexer keeps the cache fresh. A
        // failed pass does not count: the watcher retries, and embedding here too
        // would race it.
        let inline = matches!(status, None | Some(IndexStatus::Stopped { .. }));
        let engine = self.engine()?;
        let mut queries = queries.into_iter();
        let first = queries.next().expect("checked non-empty");
        let mut req = queries
            .fold(engine.search(first), |r, q| r.query(q))
            .paths(&paths)
            .limit(p.limit.unwrap_or(10).clamp(1, MAX_LIMIT))
            .sync(inline);
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
        Ok(resp)
    }
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
            path: h.resource.path().to_path_buf(),
            score: (h.score * 1000.0).round() / 1000.0,
            heading: h.heading,
            matched_query: r.queries.get(h.query_index).cloned().unwrap_or_default(),
            preview: h.preview,
        })
        .collect();
    Response {
        hits,
        segments_searched: r.stats.segments_searched,
        files: r.stats.sync.resources_discovered,
        segments_embedded: r.stats.sync.segments_embedded,
        took_ms: r.stats.total_ms,
        index: None,
    }
}

#[tool_router]
impl SeykServer {
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

pub fn serve(roots: Vec<PathBuf>, factory: EngineFactory, background: bool) -> Result<()> {
    let server = SeykServer::new(roots, factory, background)?;
    if background {
        let s = server.clone();
        std::thread::spawn(move || s.background_index());
    }
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
    use std::time::{Duration, Instant};

    /// Bag-of-words hashing, enough for lifecycle tests without model weights.
    struct WordHash(ModelIdentity);

    impl WordHash {
        fn vector(&self, text: &str) -> seyk::Result<Embedding> {
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
        fn embed_query(&self, q: &str) -> seyk::Result<Embedding> {
            self.vector(q)
        }
        fn embed_resources(&self, inputs: &[ResourceInput<'_>]) -> seyk::Result<Vec<Embedding>> {
            inputs
                .iter()
                .map(|i| match i {
                    ResourceInput::Text { text, .. } => self.vector(text),
                    _ => unreachable!(),
                })
                .collect()
        }
    }

    fn fake_engine() -> Result<SearchEngine> {
        Ok(SearchEngine::builder()
            .embedder(WordHash(ModelIdentity {
                model_id: "test/word-hash".into(),
                revision: "1".into(),
                variant: "test".into(),
                dimension: Dimension::D128,
                normalization: "l2".into(),
                query_prompt: String::new(),
                document_prompt: String::new(),
                max_tokens: 512,
            }))
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
            })
            .unwrap()
    }

    #[test]
    fn background_index_reaches_ready_and_follows_changes() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("keeper.md"),
            "lighthouse keeper counts ships",
        )
        .unwrap();
        let server = SeykServer::new(vec![dir.path().into()], Box::new(fake_engine), true).unwrap();
        let bg = server.clone();
        std::thread::spawn(move || bg.background_index());

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
        let server = SeykServer::new(
            vec![dir.path().into()],
            Box::new(|| Err(anyhow!("no weights"))),
            true,
        )
        .unwrap();
        server.background_index();
        assert!(matches!(
            server.status().unwrap(),
            Some(IndexStatus::Stopped { error }) if error.contains("no weights")
        ));
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
