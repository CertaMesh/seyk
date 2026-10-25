# Rust library

Seyk is a Rust library first. The `seyk` CLI is a thin layer over it, so other
applications can embed the same engine.

The crates are **not on crates.io yet**. Use a Git dependency:

```toml
[dependencies]
seyk = { git = "https://github.com/CertaMesh/seyk" }
```

## Use Seyk as a search engine

```rust
use seyk::embedder::gemma::{GemmaConfig, GemmaEmbedder};
use seyk::{Dimension, ResourceLocator, SearchEngine, SqliteCache};

let embedder = GemmaEmbedder::load(GemmaConfig {
    dimension: Dimension::D256,
    ..Default::default()
})?;

let engine = SearchEngine::builder()
    .embedder(embedder)
    .cache(SqliteCache::open_default()?) // omit for an in-memory, ephemeral cache
    .build()?;

let results = engine
    .search("what did we decide about pricing?")
    .paths(["~/notes"])
    .exclude("drafts")
    .limit(10)
    .execute()?;

for hit in &results.hits {
    match &hit.locator {
        ResourceLocator::Text { line_start, line_end, .. } => {
            println!("{:.3} {}:{line_start:?}-{line_end:?}", hit.score, hit.resource)
        }
        ResourceLocator::WholeResource => println!("{:.3} {}", hit.score, hit.resource),
        other => println!("{:.3} {} {other:?}", hit.score, hit.resource),
    }
}

// Optional cache warming
engine.index().paths(["~/notes"]).execute()?;
```

Each `SearchHit` carries a resource URI, a modality-independent
`ResourceLocator` (`Text` byte and line range, `Page`, `Time`, `ImageRegion`, or
`WholeResource`),
the score, the resource metadata (modality, size, mtime, content hash), the
Markdown heading path and an optional short preview.

Explicit extraction in Rust:

```rust
let results = engine.search("planting basil")
    .paths(["./notes"])
    .previews(None)
    .execute()?;
if let Some(hit) = results.hits.first() {
    let content = engine.extract(hit)?;
    print!("{}", content.text);
}
```

`SearchEngine::builder().extractor(custom)` registers a synchronous
`Extractor: Send + Sync` before the default `TextExtractor`. Custom extractors
receive full, hash-verified bytes; the first supporting extractor wins and its
errors propagate. For model-free library extraction, use
`extract_resource(&provider, &TextExtractor, &hit.resource, &hit.locator,
&hit.metadata)`. Direct low-level `Extractor::extract` calls require the caller
to verify bytes first. Neither entry point adds an authorization layer.

Resource identity migration: replace `hit.resource.path()` with
`hit.resource.to_file_path()`, which returns `Option<PathBuf>`. Use the URI's
`Display` implementation when displaying any resource, as in the example above;
use the optional path when opening a local file. `ResourceId::from_path(path)?`
now requires an absolute path and returns a `Result`. Request `.paths(...)`
still accepts relative paths and `~/...`.

CLI JSON objects and MCP search responses include `schema_version: 8`. Version 7 added result-kind counts, per-root
indexing progress and `index_status`; version 8 adds optional lexical-only
`degraded` metadata and its reason.
Every hit also includes `match` (`SearchHit::r#match` in Rust): optional one-based
`semantic_rank`, `keyword_rank` (BM25), and `filename_rank`, plus `matched_terms`
containing only normalized query tokens confirmed by the keyword index. Ranks
precede fusion and the final limit; `null` means the leg did not contribute.
Match provenance leaves existing scores/signals and ranking unchanged; older
JSON defaults to empty provenance. See
[match provenance](mcp.md#per-hit-match-provenance) for token and compatibility details.
A search hit's `resource` is a URI (`file:///notes/my%20note.md`); MCP also keeps
an explicit decoded `path`. URI identity preserves non-UTF-8 Unix path bytes;
the JSON path is a display aid and can contain replacement characters.
See the [resource identity report](architecture.md#phase-2-report-resource-identity-and-providers)
for all API changes and cache upgrade details.

Library callers can use `SearchEngine::lexical()?` without an embedder, or opt into
`.sync(false).lexical_fallback(true)` while another thread indexes. The latter
uses a separate in-memory keyword index for new or changed resources, leaving
the semantic cache untouched. Ordinary `.sync(false)` remains cache-only. All
surfaces use the same optional `degraded` object; ready results omit it.

The engine uses `ResourceProvider` for enumeration, reads, freshness and deletion
checks. `FsProvider` is the default. `ResourceProvider`, `ResourceScope`,
`ProvidedResource`, `ResourceListing` and `ResourceVersion` may change before 1.0;
filesystem is the only supported provider. There are no remote connectors.

Other backends plug in through the `Embedder` trait:

```rust
pub trait Embedder: Send + Sync {
    fn identity(&self) -> &ModelIdentity;          // persisted with every cached vector
    fn supports(&self, modality: Modality) -> bool;
    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> seyk_embed::Result<Vec<Embedding>>;
}
```

`ResourceInput` already has `Text`, `Image`, `Audio` and `Video` variants. An
embedder declares what it supports, and the engine skips the rest with a reason
instead of guessing.

Cargo features on by default: `onnx` (EmbeddingGemma 2 backend), `image` (image
resources), `cli` (the binary), `mcp` (the MCP server), `watch` (background
re-indexing) and `pdf-render` (PDF text and page rendering). `audio` and
`audio-stt` are opt-in. A library user who only wants the engine with
their own embedder can use `default-features = false`.

## PDF parser logs and panics

Library integrations that install a `log` logger **must filter `pdf_extract`,
`hayro_interpret`, `hayro_syntax` and `hayro_font` to Off**: upstream
diagnostics can contain document data. Seyk core installs no logger. On the first
PDF parse, core installs one permanent panic dispatcher with `OnceLock`, capturing
the previous hook. Per-parse RAII guards change only thread-local suppression:
parser panics are recorded as sanitized diagnostics without printing, while
unrelated threads delegate to the captured hook. A component installing a later
hook replaces the dispatcher; parser panics are still caught but may then print.
Seyk never removes or overwrites that later hook. If the first parse starts
during unwinding, dispatcher installation is deferred until a normal parse;
that first parse can use the existing hook and may print. Neither mechanism contains
aborts, out-of-memory failures, hangs or hostile allocation behaviour.

## Use only the EmbeddingGemma 2 backend

The crates are **not on crates.io yet**. Use Git dependencies:

```toml
[dependencies]
seyk-embed = { git = "https://github.com/CertaMesh/seyk" }
seyk-embeddinggemma = { git = "https://github.com/CertaMesh/seyk" }
```

```rust
use seyk_embed::{Embedder, EmbeddingTask, cosine_similarity};
use seyk_embeddinggemma::EmbeddingGemma2;

// Uses already-provisioned local files; loading never downloads weights.
let model = EmbeddingGemma2::builder().dimension(256).build()?;
let a = model.embed_text("customer cannot log in", EmbeddingTask::SentenceSimilarity)?;
let b = model.embed_text("user is locked out of their account", EmbeddingTask::SentenceSimilarity)?;
println!("{}", cosine_similarity(&a, &b)?);
```

This entry point brings no search engine, SQLite, filesystem discovery, CLI, or
agent framework. It supports semantic similarity, clustering, recommendations,
retrieval embeddings, nearest-label classification/routing, and custom vector
applications. It does not perform search by itself.

See [standalone embedding usage and API migration](embedding.md) for batch
requests, offline model files, optional images, supported tasks, examples, and
migration from the old `Embedder` and cache write APIs.

For CPU scheduling controls in Rust, see
[CPU throughput controls](performance.md#cpu-throughput-controls).
