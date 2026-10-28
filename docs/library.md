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
`ResourceLocator` (`Text` byte and line range, `PageText` range within a PDF
page's extracted text, `Page`, `Time`, or `WholeResource`; `ImageRegion` is
reserved and not produced yet),
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

CLI JSON objects and MCP search responses include `schema_version: 11`. Version 7 added result-kind counts, per-root
indexing progress and `index_status`; version 8 added optional lexical-only
`degraded` metadata and its reason. Version 9 added cache image-code counts;
version 10 adds MCP `index_status.model_diagnostics` for optional model availability.
Every CLI/library hit and full MCP hit also includes `match` (`SearchHit::r#match`
in Rust): optional one-based
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
resources), `cli` (the binary, including `audio-stt` and `audio`), `mcp` (the MCP server), `watch` (background
re-indexing) and `pdf-render` (PDF text and page rendering). A library user
who only wants the engine with their own embedder can use
`default-features = false` and opt into `audio` or `audio-stt` separately.

## Rust source segmentation

The engine uses Rust item boundaries for local `.rs` resources classified as
code. Functions carry labels such as `mod worker > impl Endpoint > fn new`.
Modules, impls, traits and extern blocks are decomposed even when short.
Consecutive sibling items smaller than `min_chars` may share a chunk within
one container; that chunk carries the container label. Longer functions split
at line boundaries when possible, and every piece retains the function label.
Labels contain source identifiers, without name resolution or macro expansion.

Every chunk is an original UTF-8 source slice with exact byte and line ranges.
Rust chunks have no overlap and respect `max_chars`, including gaps. A budget
smaller than one UTF-8 scalar consumes that scalar to ensure progress. The
embedding title and hit heading keep the full label; the lexical heading field
contains identifiers only. Extraction still verifies the whole-resource hash.

Parsing is best effort. Invalid syntax, opaque macro bodies and unrecognized
items use bounded chunks without a heading. Python, TypeScript and Go retain
paragraph segmentation. The public `TextSegmenter::segment` method also retains
its existing language-unspecified behavior, including for `TextFormat::Code`.
Its `version()` identifies that legacy path. The engine versions Rust preparation
separately as `rust-symbols-v1`, without changing the cache schema or profile.
Changing the pinned parser or Rust preparation policy requires a new preparation
version, which also retries resources with older crash markers.

The first sync reparses Rust resources, retaining vectors only for byte-identical
model inputs under the same profile. Cached searches can serve the previous
valid chunks until sync replaces them atomically. After a binary downgrade, the
older binary can likewise serve the newer chunks until its own sync. Vector
input hashes and extraction checks remain in force in both directions.

Rust parsing uses pinned `syn` and `proc-macro2` with `span-locations`. Cargo
unifies that feature for library users too. Procedural macros continue to use
compiler-provided spans. Each engine parse runs on a short-lived thread with
an 8 MiB stack, releasing its source map at thread exit without changing the
caller's spans. Files larger than 1 MiB skip parsing. A conservative preflight
also skips delimiter nesting, prefix-operator runs or closure-bar runs beyond
128. It counts source bytes, including strings and comments, so some valid
files use fallback. These limits bound parser input and stack reservation;
they do not establish a fixed bound for all AST allocations.

An ordinary parser panic or thread creation failure uses heading-free fallback.
A hostile file can still overflow the parser stack and abort the process once.
SQLite caches write a private, versioned marker beside the cache before parsing
and remove it after returning. A surviving marker disables parsing of that
resource for this parser version and emits one warning per engine, preventing
a restart loop. A concurrent engine that sees an in-progress marker also uses
fallback. Custom persistent caches can supply `EmbeddingCache::parser_state_dir`;
the default in-memory cache has no restart recovery state.

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

Protocol 11 changes MCP search defaults. CLI payload fields remain unchanged apart
from the version label. See the [mixed-version compatibility matrix](mcp.md#protocol-compatibility).
