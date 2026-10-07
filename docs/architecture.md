# Architecture and roadmap notes

This file records the architecture as audited before each development phase, so
later phases can see what was deliberately left alone and why.

## Phase 0 audit (v0.1.0)

### What Seyk is today

| Layer | Where | Shape |
|---|---|---|
| Discovery | `src/discovery.rs` | Walks local paths with `walkdir`, applies globs and default excludes, classifies files into `ResourceKind::{Text(format), Image}` by extension. |
| Segmentation | `src/segment.rs` | Markdown-aware paragraphs for text; byte ranges and line numbers per segment. Images are one whole-resource segment. |
| Embedding | `src/embedder/` | `Embedder` trait with `ResourceInput::{Text, Image, Audio, Video}` and `supports(Modality)`. EmbeddingGemma 2 via ONNX is the only implementation. |
| Cache | `src/cache.rs` | `EmbeddingCache` trait; `SqliteCache` and `MemoryCache`. Vectors keyed by `(profile, input hash)`, resources by local path. No source content stored. |
| Ranking | `src/index.rs`, `src/engine.rs` | Exact brute-force cosine behind `VectorIndex`; several phrasings fused by best score. |
| Results | `src/engine.rs`, `src/resource.rs` | `SearchHit { resource, locator, score, metadata, heading, preview, query_index }`; `ResourceLocator::{Text, Page, Time, WholeResource}`. |
| Freshness | `src/watch.rs`, `src/bin/seyk/daemon.rs` | Incremental re-index on file changes; one shared daemon behind `seyk mcp`. |
| Front ends | `src/bin/seyk/` | CLI (`search`, `index`, `cache`, `model`, `mcp`, `daemon`), MCP tool `search`. |

### What it already does well

- **Correct reuse.** A vector is only reused for byte-identical model input
  under an identical profile, so stale vectors cannot leak into results.
- **Incremental and interruptible.** Size and mtime decide what to re-read,
  content hashes decide what to re-embed, and progress is saved per batch.
- **Structured evidence.** Hits carry a resource, an exact locator, metadata and
  a short preview. Nothing in the engine produces prose.
- **Modality-ready embedding.** The `Embedder` trait and `ResourceInput` already
  describe audio and video, and unsupported modalities are skipped with a reason.

### What is tightly coupled to the local filesystem

- **Identity is a path.** `ResourceId` wraps a `PathBuf`, and the cache keys
  resources by path bytes. A `drive://` or `meeting://` resource has nowhere to go.
- **Reading is `std::fs`.** Sync, previews and freshness checks call the
  filesystem directly; there is no provider to ask for bytes or versions.
- **Freshness is mtime.** `ResourceMetadata.modified_ns` is a filesystem notion;
  remote sources have ETags or revision ids instead.
- **Path confinement.** The MCP server confines sessions by path prefix, which is
  right for files but has no equivalent for other sources yet.

### What is coupled to text

- **Search is semantic only.** Exact names (`father.md`), identifiers
  (`INC-18372`, `Session::restore()`) and rare words depend on the embedding
  noticing them, which it often does not.
- **Previews are byte slices.** `PreviewReader` slices the source file at the
  locator; there is no notion of extracting readable content from a non-text
  resource.
- **Discovery's kind enum.** Adding a format means editing `ResourceKind`,
  discovery and the engine's `prepare` match together.

### What would block multimodal extraction

1. No extraction concept: the only way to read a hit is to open the file.
2. No provider abstraction: an extractor for a remote recording would have no way
   to fetch bytes.
3. No region locator for images (`ImageRegion`), so OCR results could not point
   at where on a page or photo they came from.

### What should not change yet

- The EmbeddingGemma 2 integration, its parity tests and the cache profile rules.
- The segmenter and its algorithm version: retrieval quality was evaluated on it.
- The vector key `(profile, input hash)`: it is what makes reuse safe.
- The CLI's simple form, `seyk search "…" <paths>`.
- The MCP server and daemon. They predate this roadmap and are kept as they are.
