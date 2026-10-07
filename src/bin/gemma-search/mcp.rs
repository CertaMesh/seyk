//! `gemma-search mcp`: a stdio MCP server exposing semantic search to agents.
//!
//! The server only finds passages. The calling agent reads them and writes any
//! answer, so no generation happens here. The model loads on the first search
//! and stays loaded, which is the point of running a server instead of the CLI.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use gemma_search::{ResourceLocator, SearchEngine, SearchResults};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const MAX_QUERIES: usize = 8;
const MAX_LIMIT: usize = 50;

const INSTRUCTIONS: &str = "Local semantic search over the user's files with EmbeddingGemma 2. \
It matches meaning, not keywords, so describe what the passage would say rather than asking \
the question. Pass 2-4 different phrasings in one call; each hit keeps its best score. Hits \
are pointers: read the file at the given lines before answering, and cite path:lines. \
Prefer grep for exact identifiers or strings. The first search of a large folder embeds \
every file and can take minutes; later searches take about a second.";

type EngineFactory = Box<dyn Fn() -> Result<SearchEngine> + Send + Sync>;

#[derive(Clone)]
pub struct GemmaSearchServer {
    roots: Arc<Vec<PathBuf>>,
    engine: Arc<Mutex<Option<Arc<SearchEngine>>>>,
    factory: Arc<EngineFactory>,
    tool_router: ToolRouter<Self>,
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
}

impl GemmaSearchServer {
    pub fn new(roots: Vec<PathBuf>, factory: EngineFactory) -> Result<Self> {
        let roots = roots
            .into_iter()
            .map(|r| r.canonicalize().with_context(|| format!("root {}", r.display())))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            roots: Arc::new(roots),
            engine: Arc::new(Mutex::new(None)),
            factory: Arc::new(factory),
            tool_router: Self::tool_router(),
        })
    }

    fn engine(&self) -> Result<Arc<SearchEngine>> {
        let mut slot = self.engine.lock().map_err(|_| anyhow!("engine lock poisoned"))?;
        if let Some(e) = slot.as_ref() {
            return Ok(e.clone());
        }
        let e = Arc::new((self.factory)()?);
        *slot = Some(e.clone());
        Ok(e)
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

        let engine = self.engine()?;
        let mut queries = queries.into_iter();
        let first = queries.next().expect("checked non-empty");
        let mut req = queries
            .fold(engine.search(first), |r, q| r.query(q))
            .paths(&paths)
            .limit(p.limit.unwrap_or(10).clamp(1, MAX_LIMIT));
        if let Some(s) = p.min_score {
            req = req.min_score(s);
        }
        for g in p.include {
            req = req.include(g);
        }
        for g in p.exclude {
            req = req.exclude(g);
        }
        Ok(to_response(req.execute()?))
    }
}

/// Resolve `path` against the first root and refuse anything outside the roots,
/// so an agent cannot read previews of arbitrary files through this server.
fn confine(path: &Path, roots: &[PathBuf]) -> Result<PathBuf> {
    let path = gemma_search::expand_tilde(path);
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
                } => Some(if a == b { a.to_string() } else { format!("{a}-{b}") }),
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
    }
}

#[tool_router]
impl GemmaSearchServer {
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
impl ServerHandler for GemmaSearchServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("gemma-search", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }
}

pub fn serve(roots: Vec<PathBuf>, factory: EngineFactory) -> Result<()> {
    let server = GemmaSearchServer::new(roots, factory)?;
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

    #[test]
    fn confine_rejects_paths_outside_roots() {
        let root = std::env::temp_dir().canonicalize().unwrap();
        let inside = root.join("gemma-search-confine-test");
        std::fs::create_dir_all(&inside).unwrap();
        let roots = vec![inside.clone()];
        assert_eq!(confine(Path::new("."), &roots).unwrap(), inside);
        assert!(confine(Path::new(".."), &roots).is_err());
        assert!(confine(Path::new("/"), &roots).is_err());
    }
}
