# Architecture and roadmap notes

This file records the architecture as audited before each development phase, so
later phases can see what was deliberately left alone and why.

## Embedding crate split: pre-change audit (main @00c1038)
### Current embedding architecture
- One package, `seyk`, owns the library and binary. `src/embedder/mod.rs` defines a synchronous, object-safe `Embedder: Send + Sync`, `ModelIdentity`, `ResourceInput`, and `VideoFrame`; `src/embedding.rs` supplies `Dimension`, normalized vectors, dot products and cache byte conversion.
- `src/embedder/gemma.rs` implements text and images using tokenizers + mutex-protected ONNX sessions. Text batches sort by token length, run padded sub-batches, then restore input order. Images currently run individually through vision + text models; audio/video return UnsupportedModality.
- Loading is already offline: HF cache or local directory; explicit `download` uses hf-hub. Defaults: q8, 256 dimensions, 2048-token limit, batch 16, vision when image feature enabled. Local model fingerprinting and ONNX external-data hard-link/copy handling exist.
- ONNX returns pooled 768-wide sentence embeddings. Existing code truncates to 128/256/512/768 and then L2-normalizes. Zero vectors remain zero and must not be labelled normalized. There is no current general task API.
### Existing coupling
- The trait speaks only query/resources, uses search's Error/Result, and imports Modality from a module containing ResourceId/url. Even disabling CLI leaves SQLite, walkers and search in the package.
- Gemma's missing-weight error prescribes the Seyk CLI. hf-hub brings ureq 2 -> url: confirmed by `rustup run 1.99.0 cargo tree --locked --offline -e normal -i url`. Moving gemma.rs unchanged would fail the binding dependency constraint.
- Engine already stores `Arc<dyn Embedder>`; no concrete backend import is needed. Cache profile JSON includes ModelIdentity; input hashes use length-prefixed modality/title/model-text parts, not a newly reconstructed prompt.
- CachedSegment currently owns optional write-only lexical_text. Both caches map absent text to empty; writing a read result back can erase body keywords. No production failure is asserted.
### What can be extracted cleanly
- Traits/input types/Modality/vector math/errors and model identity move to a lightweight crate. Runtime, prompt formatting, tokenizer, image preprocessing, model location/fingerprint and inference move to the backend.
- Cache vector byte serialization remains in core. Segmentation, title selection, Markdown link-target stripping, providers, extraction, ranking and index/cache remain search concerns.
### Proposed crate boundaries
- Four packages: `crates/seyk-embed`, `crates/seyk-embeddinggemma`, `crates/seyk-core`, and root `seyk`. Root is the compatibility facade plus the existing thin CLI/MCP/daemon composition package (the seyk-cli role).
- Graph: `seyk -> seyk-core -> seyk-embed`; `seyk -> seyk-embeddinggemma -> seyk-embed`. Core has no backend dependency, including its mock-based tests. No fifth facade/CLI package solely to achieve a directory diagram.
### Migration impact
- Keep `use seyk::SearchEngine`, existing public search module paths, CLI commands/binary name, and root release/install behavior. Root re-exports core and embedding types; optional `seyk::embedder::gemma` re-exports backend types with GemmaEmbedder alias.
- Custom Embedder implementations migrate to task-aware batches and embedding-specific Result; cache implementors/struct literals migrate once to ResourceWrite/SegmentWrite. Document error matching changes, Dimension generalization and download ownership.

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

## Phase 1 report: hybrid text search

### What was implemented

- **Keyword search.** `src/lexical.rs` holds one tokenizer and query builder
  (FTS5's `unicode61 remove_diacritics 2`, with common
  function words removed from queries), used by both caches. An identifier such as
  `Session::restore()` becomes a group of tokens that must occur in the same field,
  in any order. `SqliteCache` keeps a contentless FTS5 table, `segments_fts`,
  with columns for file name, path, heading and body, ranked by BM25.
  `MemoryCache` runs the same BM25 in memory, so `--no-cache` behaves the same.
- **Schema v3.** Opening a v1 or v2 cache marks resources stale so each file is
  re-read once. Resource rows, segments and embeddings are kept; FTS5 `delete-all`
  empties the keyword index before it is rebuilt.
- **File-name matches.** A query that is a file's name, stem or trailing path
  (`father.md`, `father`, `people/father.md`) is its own signal, independent of
  both the model and the keyword index.
- **Fusion.** Semantic, keyword and file-name lists are merged with reciprocal
  rank fusion (k = 60). File names weigh 3. Keywords weigh 1.0 for names,
  identifiers and one- or two-word queries, and 0.3 for questions in prose
  (`keyword_weight`). Scores are normalised so 1.0 means first on every signal
  that found anything.
- **Explainable hits.** `SearchHit.signals` (`MatchSignals`) records the cosine,
  the BM25 score and whether the name matched. The CLI prints it as a tag
  (`[meaning 0.69+words+name]`); the MCP tool returns `matched`.
- **Modes.** `SearchMode::{Hybrid, Semantic, Lexical}`, `--mode` on the CLI,
  `mode` on the MCP tool. Semantic mode is the v0.1.0 behaviour, score included.

### Evaluation

A private local folder of long-form prose (not published), eight queries, each
with one known target passage. Queries and text stay local; only anonymized
ranks are reported. Rank of the target (lower is better):

| Query | Semantic | Hybrid (prose weight 0.3) |
|---|---:|---:|
| Q1 | 9 | 1 |
| Q2 | 1 | 7 |
| Q3 | 1 | 1 |
| Q4 | 1 | 2 (a duplicated passage took 1st) |
| Q5 | 1 | 1 |
| Q6 | 18 | 2 |
| Q7 | 19 | 25 |
| Q8 | 7 | 7 |

- Targets in the top 10 went from **6 of 8 to 7 of 8**.
- Two files that semantic search did not put in the top 3 when queried by
  their own names now rank **first**.
- Prose keyword weights of **1.0 and 0.15 were both worse overall** than 0.3.
  At 1.0, common words in long questions dominate; at 0.15, keywords barely
  break ties.

### What was deliberately not done

- **No learned or calibrated weights.** Eight queries are enough to reject bad
  settings, not to fit parameters. The constants are documented and overridable
  (`--lexical-weight`, hidden).
- **No user-facing query syntax.** Identifiers become token conjunctions
  automatically, but quotes, `AND`/`OR` and field filters (`name:`) are not
  supported.
- **No change to segmentation or the embedding cache key.** Retrieval quality
  numbers above rely on both staying as they were.
- **No query expansion or reranking model.** Both would add a second model.

### What we learned

- **The weight has to depend on the query.** One global keyword weight cannot
  serve both `INC-18372` and "where did I write about someone becoming like
  their father":
  the first is all keyword, the second is all meaning.
- **Rank fusion hides strength.** A hybrid score of 0.97 can be a weak match
  that happens to rank second of three. That is why hits carry their signals
  and the CLI shows the cosine.
- **A keyword index is not content-free.** Contentless FTS5 stores word
  frequencies. The privacy follow-up below removes source word order, but the
  cache remains sensitive.
- **Two regressions remain** (Q2 1 → 7, Q7 19 → 25): keyword hits on a common word push a purely thematic passage
  down. `--mode semantic` covers those cases for now.

### What should happen next

Phase 2: give resources an identity that is not a path (a URI-like
`ResourceId`), put reading behind a `ResourceProvider` with a filesystem
implementation, and add `ResourceLocator::ImageRegion`, without changing
behaviour. Phase 3 then builds `extract(&hit)` on top of the provider.

### Phase 1 follow-up: keyword index privacy

Schema v3 sorts each field's tokens before inserting them into contentless
FTS5. Duplicate tokens stay, preserving term frequency and field length; stored
positions describe alphabetical order only. Names, paths, headings and body
fields all take this path. Resource metadata and headings remain cache metadata.
Opening either v1 or v2 atomically clears the old keyword index and marks all
resources stale while keeping their segment references, so vectors survive
partial-root syncs and restarts until each resource is rebuilt without re-embedding
unchanged inputs.
Freed-page erasure (`VACUUM` / `secure_delete`) remains a separate follow-up;
this migration is not a forensic scrub of an existing cache file.

Reproduced with bundled SQLite in automated tests: contentless `detail=none`
and `detail=column` reject phrase queries and return zero from `bm25()`;
`detail=none` also rejects column filters. `detail=full` with sorted input
retains nonzero, field-weighted BM25. See
[SQLite's FTS5 documentation](https://sqlite.org/fts5.html#the_detail_option)
for detail modes and their restrictions.

Queries quote each token, AND identifier tokens within a field, and OR the
fields and query groups. Field filters prevent an identifier from matching
across unrelated metadata and body words. Source adjacency is intentionally
not required. Memory BM25 uses the same conjunctions, weighted frequencies,
total document-length normalization and IDF floor as FTS5; synthetic tests
compare scores and ranking with SQLite, including partial and repeated groups.

Both caches now reuse the bundled `unicode61 remove_diacritics 2` tokenizer
through a thread-local, memory-only scratch index that is rolled back after
each field. This avoids Rust/SQLite Unicode-category and case-fold differences
(for example Greek final sigma, Indic marks and digit-mixed Roman numerals).
Round-trip tests also cover accented Latin, Cyrillic and CJK tokens. No original
text or token order from this scratch index is written to disk.

The release build was evaluated with the same eight targets and a throwaway
`SEYK_CACHE`. Row indexes follow the original Phase 1 evaluation table above;
no additional source queries, filenames or passage text are recorded here.
Ranks are lower-is-better; `>30` means absent from the first 30 results.

| Row | Semantic | Weight 1.0 | Weight 0.3 | Weight 0.15 | Default | Prior weight 0.3 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 2 | 1 | 4 | 1 | 1 | 0 |
| 2 | 1 | 19 | 7 | 6 | 7 | 7 | 0 |
| 3 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| 4 | 1 | 2 | 2 | 3 | 2 | 2 | 0 |
| 5 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| 6 | 18 | 1 | 2 | 9 | 2 | 2 | 0 |
| 7 | 19 | >30 | 25 | 22 | 25 | 25 | 0 |
| 8 | 7 | 8 | 7 | 7 | 7 | 7 | 0 |

All eight weight-0.3 ranks are unchanged, and default ranks match them.


## Phase 2 report: resource identity and providers

### What was implemented

`ResourceId` is a validated URI serialized as a string. Local resources use
`file:///...`, with percent-encoded path bytes; Unix non-UTF-8 names round-trip.
The ID supports Display, parsing, scheme inspection and optional conversion to
a local path. Filesystem request paths resolve symlinks before URI normalization,
so a path such as `link/../note.md` keeps its filesystem meaning. Non-file IDs
are representable but the filesystem provider rejects them for access.

`ResourceLocator::ImageRegion { x, y, width, height }` reserves a rectangle in
original-image pixels: top-left origin, x increases right, y increases down.
Nothing produces it yet. Existing locator variants and fields are unchanged.

The engine now receives a synchronous `ResourceProvider: Send + Sync` for
listing, reading, version checks and confirmed in-scope deletion. `FsProvider`
retains discovery filters and the no-follow directory walk. Library requests
keep `.paths(...)`, a filesystem-only convenience that canonicalizes locally
regardless of provider. `.roots(...)` is the generic entry point for URI identities.
Known boundary leaks remain: ResourceScope carries filesystem DiscoveryOptions,
and ProvidedResource.kind imports discovery::ResourceKind. Move ResourceKind and
TextFormat into shared content types at the crate split.
The provider's descriptor supplies content kind, size, optional mtime, opaque
version and the title prefix used by embedding. All engine/cache identities
are ResourceId values; `discover()` remains a filesystem compatibility helper.

Known equal listing versions and descriptor fingerprints skip rereading and
avoid a second provider.version()/filesystem stat call. Version checks only
surround actual reads. Unknown versions always reread and hash;
identical content and model inputs still reuse vectors. Fs versions encode size
and mtime. Version checks around reads detect changes during indexing, and
previews still validate the hash of the bytes they actually read. The engine
delegates resource reads, stats and deletion to providers; request `.paths()`
still canonicalizes through ResourceScope locally.
Deletion requires a provider's positive confirmation inside the request scope;
filter exclusion or a permissions error does not establish deletion. The provider
resolves deletion roots once per pass; the engine filters cached IDs by those
root boundaries before calling `is_deleted`. Non-NotFound errors retain entries
and appear in the skipped report (and CLI/watch logs), without aborting the pass.

File titles, lexical name/path fields, model input hashes, cache profiles,
segmentation and file tie-break ordering are preserved. Watchers, daemon
transport and MCP confinement remain filesystem-specific. MCP continues to
canonicalize requested paths against session roots before searching; discovery
and per-session candidates do not expand that scope.

### Cache v4

The public cache API uses ResourceId, while file identities keep the existing
BLOB path keys in SQLite, including canonical Windows drive prefixes. Non-file IDs
use a NUL-prefixed UTF-8 URI key, which cannot collide with a valid local
filename. Legacy SQL column names remain internal details. The v4 migration adds
an opaque `version` and nullable `descriptor_hash` column. Legacy fingerprints are
unknown, forcing preparation on the next sync. The migration keeps resource keys,
segment rowids, contentless FTS rows, input hashes, profiles and vector bytes. A
v3 cache keeps its keyword index until sync and reuses byte-identical model inputs
without re-embedding. v1/v2 upgrades still perform the Phase 1 privacy
invalidation, keeping vectors across partial-scope rebuilds.

This is a one-way schema upgrade. Old binaries treat unfamiliar cache versions
as disposable and can clear v4; do not run old and new binaries against the
same cache. Separate cache files permit side-by-side versions. Existing
forensic-erasure limitations of old keyword indexes remain unchanged.

### JSON and public API migration

The old unversioned machine output is legacy v1. CLI JSON objects now have a
flattened `schema_version: 2` field: search, index/watch, cache stats and daemon
status. No-running-daemon output remains `null`. MCP's existing JSON content
block also has the version; no separate structuredContent channel was added.
After a document schema is released, any field addition, removal or rename
bumps its version, including additive fields, because consumers parse strictly.
A new command may introduce a new document type at the current version without
changing existing document schemas. Using an existing nullable field with null
does not change its schema. The field corrections here finalize unreleased v2.
Bare `null` for no daemon is the sole non-object exception.

CLI search retains queries/mode/hits/stats. A hit now contains, for example,
`"resource": "file:///notes/my%20note.md"` instead of a path string. MCP hits add
that resource field and retain `"path": "/notes/my note.md"` for opening files.
MCP paths and CLI skipped-record paths are omitted for non-file IDs. The URI is
the lossless identity; JSON display paths replace invalid UTF-8 bytes. Skipped resources expose a URI
and optional local path. Parse `resource` as a URI and percent-decode its file
path; never strip the `file://` prefix. Confinement checks must compare decoded,
canonical filesystem paths, not URI strings. Human CLI output for existing files
is unchanged.
The daemon endpoint hash includes the JSON protocol version, so new clients
do not reuse a daemon serving the old output shape.

Public Rust API breaks:

- `ResourceId::path()` is removed. Replace it with `to_file_path() -> Option<PathBuf>`;
  use `Display` or `as_str()` when a URI is wanted.
- `ResourceId::from_path` is fallible and requires an absolute local path
  (remote UNC hosts are rejected); its serde representation and parser use URIs, so legacy serialized path strings are
  rejected. File URIs normalize aliases; ordering retains local Path ordering
  for file/file comparisons and uses URI order for non-file identities.
- ResourceLocator adds ImageRegion; both ResourceLocator and Modality become
  `#[non_exhaustive]`, so downstream matches need a wildcard arm.
- `SkippedResource` adds `resource: ResourceId`; `path` becomes `Option<PathBuf>`.
- `EmbeddingCache::resource_index` returns
  `HashMap<ResourceId, CachedResourceState>` containing metadata and version.
  `resource`, `put_resource`, `touch_resource`, `remove_resources`, `vectors_for`
  and `lexical_search` take ResourceId keys/collections instead of Path keys.
  `touch_resource` now accepts full metadata plus a ResourceVersion instead of
  only an optional mtime.
- `CachedResource` gains required `version` and `descriptor_hash` fields;
  `CachedResourceState` also exposes the fingerprint. `StoredVector.path` and
  `LexicalHit.path` become `resource: ResourceId`.

EmbeddingCache, CachedResourceState, ResourceProvider, ResourceScope,
ProvidedResource and ResourceVersion are explicitly unstable: they may change before 1.0; filesystem is the only supported
provider. They are not an authorization/ACL framework. Direct library
SearchResults serialization includes URI hits but does not add the CLI envelope.

### Verification and demo

The required Rust 1.99 fmt and all-target Clippy checks pass. The default test
suite passes with the 9 existing weight/reference-dependent tests ignored and
two Windows-only follow-ups documented below; PR #9 records the per-OS results
for the final head.
The updated library doctest compiles. Automated evidence includes:

- A test-only memory provider driving indexing, semantic/lexical/hybrid retrieval,
  previews, updates, unknown versions, read races and scoped deletion without
  source files. FsProvider retains discovery and symlink path semantics.
- Both caches supporting opaque URI identities; a reconstructed v3 schema
  migrating with unchanged vector bytes, segment rowids, FTS bytes and ranked
  hits, zero embedding calls, and retained entries across reopening/partial scopes.
- Existing ranking, cache reuse, privacy migration, failure isolation, watcher,
  shared-hub and daemon relay tests. Human output golden tests are unchanged.
- Versioned CLI/MCP JSON, explicit MCP paths, shared-cache session isolation,
  symlink escape rejection, and daemon protocol separation.
- URI round-trips for spaces, Unicode, reserved characters and non-UTF-8 Unix
  bytes. This Mac filesystem rejects creating non-UTF-8 filenames, so the output
  test constructs that identity directly. Native Windows drive-path and
  legacy-key tests are included; CI now runs fmt, Clippy and the default suite on Ubuntu, macOS and
  Windows. Native Windows outcomes are recorded on PR #9.

A real local EmbeddingGemma 2 q8 run (128 dimensions, text-only, debug binary)
searched two synthetic Markdown notes, `garden.md` and `harbor.md`, using a
throwaway cache. The first describes growing tomatoes/basil; the second describes
ships and a lighthouse. No private corpus was used.

| Pass | Query / mode | Top hit | Segments embedded | Search time |
|---|---|---|---:|---:|
| Cold | `garden.md` / hybrid | garden.md | 2 | 2556 ms |
| Warm | `garden.md` / hybrid | garden.md | 0 | 668 ms |
| Semantic | `vegetables growing outside` / semantic | garden.md | 0 | 488 ms |

All responses carried schema version 2, file URIs and previews. Both later
passes reported two unchanged resources. Times are the engine's reported search
time, excluding model load; process wall times were 12307, 10804 and 8643 ms.
This is an end-to-end smoke demo, not a scale/performance benchmark or a rerun
of the private Phase 1 relevance evaluation.

### What was deliberately not implemented

No remote provider, provider registry, stable third-party connector API,
Extractor/extract(), OCR, transcription, ACL/tenancy, new model or ranking change.
Filesystem is the only real provider; the memory provider exists only in tests.

### What we learned

Native Windows testing passed all nine provider tests, including actual deletion
and vector pruning. Additional CLI/MCP, lexical-tie and failure-isolation URI
fixtures now compare portable absolute paths or canonicalize decoded paths.
Two unrelated pre-existing tests are ignored only on Windows pending follow-ups
(discovery fixture separators and read-only timestamp handle).
Provider/deletion regressions remain enabled on Windows.

The first Windows CI run exposed two previously uncompiled lint paths: a cache
fixture used only by Unix assertions and a daemon-only status method. The
fixture now checks its path on every platform; the method compiles for Unix
and tests. CI uses no-fail-fast to expose all platform test failures in one run.

A provider abstraction still has filesystem assumptions in `.paths()`, discovery
filters and content-kind ownership; they are now explicit unstable boundaries.
Listing tokens avoid a redundant stat on unchanged resources while read-race
checks remain. Non-exhaustive content enums allow later modalities/locators;
strict JSON consumers require version bumps even for additive fields.

Deletion must tolerate unavailable metadata: per-entry errors retain that entry;
a scope-resolution error retains all candidates. Canonicalizing roots once per
pass removes the former cached-IDs × roots realpath cost. Synthetic provider
errors and Unix permission-denied parents exercise retention and reporting;
out-of-scope cached IDs never reach the deletion provider.

Windows canonical paths use verbatim prefixes while decoded file URIs use disk
prefixes. Deletion scope comparisons normalize through the same URI codec; the
missing child is never canonicalized. Native CI covers actual file deletion,
cache removal and orphan-vector pruning, with portable URI/cache fixtures.

Content hashes alone cannot establish preparation freshness: title and kind/format
also affect model inputs. Both version and content shortcuts now require the
persisted descriptor fingerprint, including unknown-version providers. Tests
compare warm and cold caches after title-only and format-only changes.

File identity affects cache keys, deterministic ranking and model title inputs,
not just result display. Keeping the old file-key codec avoids a cache rebuild
and preserves tie ordering. Enumeration omission cannot prove deletion because
filters and errors also omit resources. URI normalization must happen after
filesystem path resolution when accepting CLI paths with symlinks and `..`.
The old daemon's output contract also needs an identity boundary during upgrades.

### What should happen next

Phase 3 can add the separate Extractor abstraction and a text-only implementation
on top of provider reads and current locators. Before broader providers, revisit
scope semantics and version guarantees with a real implementation. Size/mtime
blind spots and filesystem TOCTOU remain limitations; this refactor does not
claim atomic snapshots or introduce an ACL layer.

## Phase 3 report: explicit text extraction

### What was implemented

`SearchEngine::extract(&SearchHit)` returns `ExtractedContent { text, source,
locator, metadata }`. Text, Markdown and code use the exact half-open byte
range from the hit. Markdown syntax and link destinations, Unicode, whitespace
and line endings are preserved. The metadata retains the BLAKE3 hash of the
whole resource; it is not a hash of the extracted passage.

The synchronous `Extractor: Send + Sync` interface has `supports(source,
locator, metadata)` and `extract(bytes, source, locator, metadata)`. The engine
selects the first supporting custom extractor, in registration order, with
`TextExtractor` as the built-in fallback. An extractor error is returned without
trying another extractor. Search does not invoke the extractor registry.

The central `extract_resource(provider, extractor, source, locator, metadata)`
gateway reads through `ResourceProvider`, verifies the full BLAKE3 hash, then
passes those exact bytes to the extractor. A custom extractor cannot bypass
verification when invoked through the engine or gateway. Direct calls to the
low-level trait method require already-verified bytes from the caller.
The public gateway also permits extraction without constructing an engine or
loading a model. PreviewReader shares its read/hash/verify and strict text-range
helpers, retains one read per resource per search, and continues to suppress
failed optional previews. Explicit extraction returns typed errors instead:
`StaleResource`, `UnsupportedExtraction`, `InvalidLocator`, or `InvalidText`;
provider read errors propagate. Unsupported locators/modalities fail before a
read. Invalid UTF-8 is never replaced with guessed characters.

`seyk extract --from-json <FILE|-> [--hit INDEX] [--json]` accepts a complete
version-2 search envelope and selects one hit (zero-based index, default 0).
It rejects unknown/unversioned schemas, invalid hits and missing indexes before
reading the source. Raw output adds no newline; JSON output is a new document
type with `schema_version: 2`, `text`, `source`, `locator`, and `metadata`.
Extraction needs no model, cache, daemon or network. Errors produce stderr and
a nonzero exit status without extracted content on stdout.

`seyk search --no-preview` skips preview reads and retains `"preview": null` in
CLI JSON. Search still reads resources as needed for indexing. Default text
previews and every existing v2 document shape remain unchanged. The new extract
document does not require a version bump or a new daemon endpoint identity.

### What was deliberately not implemented

No PDF, OCR, audio/video, remote provider, LLM, automatic extraction, ranking or
segmentation changes. Cache schema remains v4 with no embedding invalidation.
No persisted media type: today's text modality plus text locator identifies
raw UTF-8. The source ID is available to `supports` for extension-based hints
when another format needs them. An extension is not proof of format, and opaque
IDs will eventually need authoritative provider metadata. Phase 4 should decide
that with a working PDF implementation rather than speculative cache fields.

No MCP extract tool: MCP search currently returns display line ranges but omits
the exact byte locator and content hash. Those cannot be reconstructed safely
from line numbers. A future MCP extraction protocol needs round-trip references
and session-root checks before reads. Existing MCP confinement is unchanged;
the CLI supplies the process boundary needed by downstream wrappers.

### Public API changes

- `Error` is now `#[non_exhaustive]` and gains `StaleResource`,
  `UnsupportedExtraction`, `InvalidLocator` and `InvalidText`. Downstream
  `match` statements need a wildcard arm.
- New: `Extractor`, `TextExtractor`, `ExtractedContent`, `extract_resource`,
  `SearchEngine::extract`, `SearchEngineBuilder::extractor`.
- `Extractor`, `ExtractedContent` and `extract_resource` are explicitly
  unstable: they may change before 1.0, and a range/streaming method is planned
  for audio and video.

### Audit 2026-10: public API changes

- `BruteForceIndex::new` and `BruteForceIndex::with_capacity` now return
  `Result<Self>`, with `Error::Config` for zero width ([PR #27](https://github.com/CertaMesh/seyk/pull/27)).
- Named Hugging Face revisions now record the resolved commit in `ModelIdentity`.
  Alias/branch users get a new cache profile and re-embed once; users of pinned
  commits are unaffected ([PR #20](https://github.com/CertaMesh/seyk/pull/20)).

### What we learned

Verification belongs before extractor dispatch, where every implementation
receives the same checked snapshot. It cannot depend on each plugin remembering
to perform a hash check. Whole-resource reads/hashes are suitable for this text
phase; future range/streaming extractors for audio/video need a separate method
and a corresponding hash contract. No such method is implemented yet.

The content hash proves agreement with the supplied hit, not authorization or
authenticity of caller-supplied metadata. Filesystem TOCTOU is still a provider
limitation; returned text always comes from the verified buffer, never a second
read. Source text is not persisted by extraction. Tests and demos use synthetic
content only.

### Verification and demo

The required Rust 1.99 fmt, all-target locked Clippy with `-D warnings`, and
locked default test suite pass. The suite includes exact Markdown/plain/code
ranges, multibyte boundary and malformed-range rejection, invalid UTF-8, empty
ranges, stale filesystem and memory resources, custom extractor ordering and
hash gating, preview read reuse, CLI stdin/file selection, v2/null-preview
compatibility, and model-free error handling. Existing model/reference-dependent
tests remain ignored by the default suite; no private evaluation was run.
The PR records Ubuntu, macOS and Windows CI results for its final head.

A real local EmbeddingGemma 2 q8/128-dimension CLI run searched two synthetic
notes, `garden.md` and `harbor.md`, with previews disabled and a throwaway cache.
The `garden.md` query returned bytes 0–110 (lines 1–4) from garden.md with
`preview: null`. Both JSON extraction through stdin and raw extraction from the
saved search envelope reproduced all 110 source bytes, including Markdown link
targets and Unicode. Replacing `basil` with same-length `thyme` then made extraction
fail with empty stdout. The full demo took 3.10 seconds on this machine; this is
a smoke check, not a performance benchmark.

### What should happen next

Evaluate the extraction API with downstream CLI consumers, then implement text
PDF extraction in Phase 4 behind the same explicit boundary. Resolve media
metadata and MCP hit references with their first real consumers. Keep search,
extraction and downstream reasoning separate.

## Embedding split implementation (0.1.0)

The workspace now separates `seyk-embed` (generic embedding inputs, tasks, values
and errors), `seyk-embeddinggemma` (local runtime/preprocessing), and `seyk-core`
(search, evidence, cache, provider and extraction). The root `seyk` package is the
compatibility facade plus thin CLI/MCP composition layer. There is no separate
`seyk-cli` package: a fifth package would only mirror the diagram while requiring
an extra facade to preserve `use seyk::SearchEngine` and source installation.

Core has no concrete backend dependency. It submits SearchQuery/SearchDocument
requests through the generic trait. Exact query/document formatting, ModelIdentity
serialization, cache profiles, input hashes, schema v4 and vector encoding remain
unchanged. Other standalone tasks never enter the search cache. Downloading remains
an explicit operation in the root package; the backend resolves only local files.

The cache write API now borrows explicit source bodies paired with stored segment
evidence. Readback types have no source-body field and cannot be used as complete
write input. Preparation retains original source text until keyword indexing;
only the processed text goes to embedding. Image bodies are explicitly empty.
No ranking, extraction, keyword tokenization, field weight or schema change is
part of this refactor. SQLite's registration requirement and MemoryCache's
pre-registration behavior are both retained.

The default test suite exercises both caches, unchanged vector reuse, typed error
compatibility, backend-free mocks and a golden profile captured from the pre-split
binary. On a synthetic two-note cache, the new binary reused both embeddings and
returned identical hits, profile rows, input hashes and vector bytes. Local q8
model tests passed token/vector parity at all four widths, batch ordering,
retrieval and generated-image search. CI covers Linux, macOS and Windows; the PR
records final-head results and dependency-tree evidence. No private corpus was
accessed or evaluated.

All packages and internal path dependencies retain version `0.1.0`, inherited
by packages from `workspace.package.version`. Release archive and binary names
remain unchanged. The new crates are not published to crates.io.
See [embedding usage and migration](embedding.md) for Git dependencies, offline
files, supported prompts/modalities, feature flags, examples and API changes.

Remaining limitations: image inference is sequential, session locks serialize
calls, the token default remains 2048, and audio/video are unsupported. Local
model identity still fingerprints graph bytes plus weight-file sizes; staging
still reuses equal-size targets. Changing those existing rules would require a
separate cache-identity migration.

## Phase 4a: embedded PDF text (0.1.0)

### Implemented

The default CLI discovers PDFs and converts usable embedded text into ordinary
SearchDocument text requests, independently for each physical page. `Document`
is container metadata, never a model input. A pinned pdf-extract 0.12.1 adapter
is behind core's optional `pdf` feature; embedding crates remain PDF-free.
The resource stem supplies the title. Ranking, text segmentation, input hashes,
cache profiles and INPUT_FORMAT_VERSION remain unchanged for text and images.

PDF evidence uses `PageText`: a 1-based physical page number, a half-open UTF-8
range within the complete decoded page, the decoding version, and a page-text
hash. Preparation, preview and explicit extraction share the decoder. Extraction
first verifies the full source hash through the provider, then the derived page
hash/version and boundaries. It returns exact decoded text with provenance;
it does not promise original layout or byte identity with the PDF stream.
Opaque provider identifiers work without filename inference. No-preview searches
perform no preview reads or decoding. Selected-page preview text is transient
and capped at 32 MiB per request.

Cache schema 5 adds nullable media_type transactionally. A copied v4 database
migration test preserves profile rows, segment rowids, FTS data and vector bytes,
then proves zero warm re-embedding. Source text remains transient; the existing
sorted-token privacy path is unchanged. JSON output is schema 3; extraction
accepts schema 2 and 3. Do not open a v5 database with an older binary: its
unknown-schema handling can recreate tables. Use a backup for rollback.

### Deliberately not implemented

No rendering, image-page retrieval, OCR, passwords, layout reconstruction,
remote services or automatic model downloads. Blank/scan pages have explicit
page diagnostics; all-scan documents fail explicitly. Parser failures reject
the whole resource, preserving prior cache state until replacement succeeds.
Missing results are preferable to a silently accepted prefix. No 4b dependency
or model is present; all packages remain 0.1.0.

### Learned and measured

The convenience upstream per-page extraction function can silently return a
prefix on errors, so the adapter enumerates/checks the page tree and propagates
each page error. Output caps are 512 pages, 1 MiB decoded text per page and
32 MiB per document, with explicit errors, never truncation. These bound adapter
output, **not** upstream decompression, allocations or CPU time.

Upstream log messages and panic payloads can contain content. The root binary
and core install no log logger. The binary's panic hook emits only a fixed message
and file:line, and every PDF parse catches unwinding panics into a sanitized
per-resource error. Library callers must filter the `pdf_extract` logger target
to Off. Core lazily installs one permanent dispatcher with `OnceLock`, capturing the
previous hook. Per-parse RAII guards change only thread-local suppression and
record sanitized panic reasons; unrelated threads delegate to the captured hook.
Guard drop restores thread-local state even during unwinding. First installation
is deferred if a parse starts while unwinding, because Rust forbids set_hook
then; the existing hook may print until the dispatcher is installed. A later
external hook replaces the dispatcher: catch_unwind still catches parser panics, but they
may print. Seyk never removes or overwrites a hook installed after its own.
This does not contain aborts, OOM or hangs. A non-UTF-8 Type-4 stream is rejected
explicitly: without a logger, the upstream log argument's UTF-8 unwrap is never
evaluated. A separate malformed operator exercises a real parser panic.
Real-binary index/preview/extract tests capture both channels and prove hidden
fixture markers never appear, with successful public text unchanged.

A generated 20-page corpus with 10 queries, distractors and identifiers produced
Recall@5=1.000 and MRR@10=1.000 for both PDF and equivalent text documents; all
20 extractions were exact. Local Gemma q8/128, one CPU thread, debug macOS ARM:
model load 3347 ms, cold PDF index 982 ms, warm query p50/p95 443/486 ms. This is a
small synthetic regression gate, not a claim about arbitrary PDFs. The macOS
ARM release executable is 40,919,520 bytes with PDF versus 39,427,232 without
(+1,492,288 bytes, 3.8%). Linked libraries remain system libraries/frameworks;
PDF adds no dynamic runtime. Unicode, ligature, malformed-page, stale-source,
provider, limit and cache tests complement that evaluation. Existing model parity
and profile goldens remain separate gates. Before the strict-validation review
fix, a direct debug limit-test run (no compiler/model) used 55.8 MiB maximum RSS and 38.74 s for
1 MiB/page rejection, 32 MiB/document rejection and 512 accepted pages. The
full synthetic PDF/text evaluation process used 811.9 MiB maximum RSS, including
the shared model and both engines; this is not incremental PDF memory. A second
evaluation run again passed all quality gates, with 2152 ms load, 711 ms cold
index and 436/897 ms warm p50/p95. Timings vary with concurrent local work.

### Next

Phase 4b needs its own approval and renderer/ORT compatibility spike before
scanned-page retrieval and on-demand OCR. Subprocess parser isolation is a
separate follow-up. Sparse embedded text, unusual font encodings, rotation and
multi-column reading order remain retrieval/evidence limitations; the current
adapter never claims semantic layout reconstruction.

### Phase 4a review fixes

Before accepting any page output, the adapter now resolves every page's Contents
(single stream or array), validates decompression and calls lopdf 0.42's public
Content::decode_strict API, which rejects unconsumed bytes. Missing objects,
non-stream values, invalid compression and unterminated syntax reject the whole
resource, even when preview/extract selects an earlier valid page. Missing
Contents, null and empty arrays still represent empty pages. Validated streams
are normalized in the transient document so pdf-extract cannot use its raw-byte
fallback. Nothing is written back to the source.

lopdf's Flate/LZW helpers can log errors and return partial decoded output.
Content-stream support is therefore limited to unfiltered bytes or complete,
checksum-verified FlateDecode streams without DecodeParms; other filters and
parameters fail explicitly. flate2 was already in the locked graph; core now
uses its checked decoder directly. No upstream fork or package update is needed.
The PDF-only preparation descriptor advances to strict-v2 so cached PDFs are
revalidated; existing text/image profiles and hashes remain unchanged.

The current PDF preparation path isolates decoder failures/panics per physical
page; invalid UTF-8 text is a page failure too. Document-wide validation and
output-cap failures still reject the whole resource. A deterministic whole-PDF
failure stores an empty binding for the verified source version; partial failures
store only valid pages, without shifting page numbers.

Unchanged deterministic PDF failures keep the normal preparation descriptor. An
engine-local map keyed by (resource, descriptor) replays page diagnostics and
whole-resource failures for watchers/MCP without parsing, put_resource, FTS rewrites
or new scrub markers. Changed bytes or preparation versions invalidate reuse.
Diagnostics are not persisted: a fresh process reports failures only on a parse,
and cannot replay an already-fresh persistent cache's previous failures.
MCP preserves the existing `index.state` values, including `ready`, and exposes
last-pass watcher diagnostics separately in `index_diagnostics`. Counts merge
by (code, reason); page details are capped at 128 with `pages_total` retaining the
total. Inline response page details have the same cap and a separate
`page_diagnostics_total`. CLI skip summaries group by code and show at most five
codes. Benchmark diagnostics print codes, counts and opaque resource hashes only.

Page diagnostics now have their own SyncStats field and are emitted only after
a successful resource commit. CLI summaries distinguish those pages from skipped
resources. PDF skip reasons no longer repeat the resource identity. Preparation
has one dispatcher, so diagnostics cannot be lost in a second PDF path.

After strict validation, the same cap regression passes in 37.51 s with
94.7 MiB maximum RSS (99,270,656 bytes), including the transient validated
streams. The repeated synthetic evaluation still gives 100% Recall@5, MRR@10
1.0 and 20/20 exact extractions; load 2196 ms, cold index 795 ms, warm p50/p95
290/386 ms on the same debug Mac ARM setup.

## Phase 4b retrieval — scanned PDF pages

### Implemented

Normal CLI/release builds enable `pdf-render`, which implies `pdf` and the
existing image embedding feature. Minimal core users can omit rendering or all
PDF support. `hayro =0.5.0` renders on the CPU with bundled Foxit fonts, no system
font lookup or GPU. Crop and rotation are respected, the background is opaque
white and the longest edge is 2048 pixels. The complete page appearance is the
image embedding input; no text prompt is fabricated from the raster.

Pages with usable embedded text keep their verified `PageText` segments. Other
pages become one `Page { page }` hit through Gemma's image `SearchDocument`
path. These hits have no preview and an empty lexical body; filename/path
signals remain available. Extract returns typed `PdfOcrUnavailable`, explaining
that readable scan text needs OCR. Text-only models/builds index embedded text
and report omitted scan pages as page diagnostics, including scan-only PDFs.
CLI diagnostics say how many pages were not indexed and give the reason,
separately from skipped resources.

The PDF descriptor includes renderer policy/version and text/image capabilities.
Only PDF freshness checks change; the model profile and non-PDF input goldens
stay identical. Image input hashes cover precisely the encoded PNG bytes sent
to the model and share the ordinary image-input identity. Cache schema stays v5.
No raster, source text or OCR output is persisted in the cache.

Each PDF is parsed and processed separately. At most
`min(configured_batch_size, 4)` page model inputs are retained in a batch.
Completed batches retain only locators and transient lexical text. Embedding
calls, including single-item retries, use `checked_embed_batch`, which checks
vector count and dimension before writes. Resource bindings are committed once,
after required jobs succeed; a later failure keeps the old resource binding.
Successfully computed vectors may remain reusable until ordinary orphan pruning.
These limits bound queued rasters, not hostile decoder allocation or CPU time.

### Deliberately not implemented

OCR is tracked as a follow-up. The compatibility spike found that paddle-ocr-rs 0.6.1 does not compile with the pinned single
ORT rc.13 runtime, and model conversion provenance was incomplete. No OCR code,
weights, download command, fork, MSRV increase or alternate inference runtime is
included. Search and extraction remain separate: a relevant page can be found
without readable text being available.

### Learned and limitations

Hayro omits pages with absent `/Contents`. The adapter therefore supplies an
in-memory serialization of the already strictly validated document with empty
streams normalized, preserving physical page numbering. Original provider bytes
and their content hash never change. All page content streams still undergo
whole-resource resolution, decompression and full-consumption validation before
any page is accepted.

Hayro can silently omit undecodable image data without notifying its warning
callback. To avoid blank scan hits, the adapter inspects image XObjects,
including inherited and nested form resources. A resulting near-uniform raster
gets the page diagnostic "scanned page could not be rendered" and no embedding.
The guard treats RGB ranges <=8, or <=0.01% pixels more than 8 below white, as
near-uniform. Unknown/cyclic resource references are conservative; nested
inspection stops at depth 16. These choices are encoded in the renderer policy
and can be retried after a descriptor change. Very faint or nearly blank scans
may be rejected. Partial image omissions on otherwise nonblank pages remain a
recall limitation, not an assertion of document fidelity.

No logger is installed. Library applications with logging must filter
`pdf_extract`, `hayro_interpret`, `hayro_syntax` and `hayro_font` to Off because
upstream diagnostics can contain source data. Renderer loading and rendering
catch unwinding panics into sanitized errors under the same thread-local
suppression guard as text parsing. The permanent dispatcher delegates unrelated
panics to its captured hook; a later external hook replaces it and may print
parser panics. Seyk never removes or overwrites that later hook. catch_unwind
does not contain aborts, OOM or hangs. Tests exercise an unwind inside Hayro through a
panicking font callback and real CLI output capture for hidden-marker render
fixtures, in addition to the parser leak regressions.

The committed synthetic scan evaluation has 20 image-only pages and 14 fixed
queries (semantic, identifier and title queries), default 256-dimensional q8
Gemma image embeddings, numeric filenames and semantic-only ranking. It gates
Recall@5 at 80%; it represents clean, short, single-column printed pages, not
arbitrary dense, damaged or multilingual scans. Render tests freeze encoded PNG
hashes, crop/rotation placement and opaque white pixels. Core tests cover mixed
PDFs, vision-off diagnostics, cache capabilities, image omission and malformed
model outputs with prior resource bindings preserved.

The release scan evaluation on Apple M5 Max / 64 GiB / macOS 26.5 measured
Recall@5=1.000 and MRR@20=1.000 (all 14 queries ranked the correct page first).
With q8 Gemma revision `daa72c51243991dfcaf9f9137d2c573d8f7790c0`,
256 dimensions and one CPU thread, end-to-end indexing per page was
6668 ms p50 / 7693 ms p95, 136896 ms total for 20 pages. Model loading took
498 ms; peak process RSS was 1,502,969,856 bytes, including fixture generation,
model and engine. These are shared-host measurements, not isolated performance
benchmarks. The earlier compatibility spike measured rendering alone at roughly
34 ms/page; image embedding accounts for seconds of CPU work per page. The
`--text-only` option avoids scan embedding while retaining embedded-text search.

### Next

Complete the compatible single-ORT OCR adapter and model-provenance follow-up
before adding scan text extraction. Process isolation remains separate.
While GitHub Actions was unavailable, the full local macOS merge gate supplied
implementation verification. Linux/Windows and all release targets must be
rerun in CI; this is tracked as a follow-up. The earlier
four-target compatibility spike established renderer build/hash/link feasibility, but does
not replace verification of this implementation. Platform-sensitive areas are
floating-point raster/PNG determinism, filesystem behavior and dependency
linking; production rendering adds no platform-specific branches. The optional
evaluation RSS measurement uses getrusage (bytes on macOS, KiB on other Unix;
unavailable on Windows).

## Phase 5a: audio retrieval

This is the 5a release record. Phase 5b-1 below supersedes its fixed input-size
and duration limits and whole-file encoded transport.

Opt-in `audio` adds semantic retrieval over WAV, FLAC, MP3 and AAC-LC/ALAC in
M4A. Default release features are unchanged. Symphonia is pinned to 0.5.5;
`deny.toml` permits MPL-2.0 only for the eleven approved package/version pairs,
with source/licence notices in `THIRD_PARTY.md`. No FFmpeg library or executable
is used at runtime (FFmpeg and macOS speech tools only generate test fixtures).

### Preparation and evidence

The core reads at most the existing 32 MiB encoded input limit. Symphonia emits
bounded packets with gapless MP3 enabled. A bounded MP4 box reader applies one
`elst` entry's `media_time` and `segment_duration`, converting the media/movie
timescales, clamping to the `mdhd` media end so millisecond rounding cannot
retain AAC block padding. Without an edit list it uses iTunSMPB delay/length;
without either it uses the decoded-sample timeline. Empty, multi-entry, non-unit-rate edits and
multiple tracks are rejected rather than assigned speculative offsets. AAC-LC
AudioSpecificConfig is checked before decoding: Symphonia can otherwise ignore
SBR while decoding the LC core of HE-AAC.

Channels are averaged with f64 accumulation, then a versioned 64-tap Blackman
windowed-sinc filter (0.94 Nyquist guard, zero-extended edges) produces mono
16 kHz f32. Decoding never retains a full PCM file. The packet buffer, resampler
history and 10-second rolling window are bounded independently of duration.
Window starts advance by 8 seconds; the first short window ends the sequence,
without emitting further windows contained in that tail. Tails shorter than 320
samples are zero-padded only for embedding. Time locators use
floor(start sample × 1000 / 16000), ceil(end sample × 1000 / 16000), with the real
post-trim end, never the padded end. Intervals are half-open and non-empty.

The decoded cap is **16,777,216 samples at 16 kHz = 1,048.576 seconds**, derived
from 32 MiB of 16-bit mono PCM. Known exact lengths can fail before decoding;
the streaming counter also enforces it. Recordings over about 17.5 minutes are
skipped entirely, even if windows were processed before the limit was detected.
This bounds work for highly compressed inputs while keeping the existing
encoded-file policy. Supported source rates
are 8–192 kHz, with up to eight channels (AAC-LC is mono/stereo); format changes,
corrupt frames and unsupported profiles return `UnsupportedAudio`.

ALAC cookies must contain 24 or 48 bytes, 1–16,384 samples per frame (four times
Apple’s usual 4096), 1–8 channels and a supported 16/20/24/32-bit depth before
decoder construction. This caps each ALAC output/tail buffer at 512 KiB.
Symphonia probing, construction, packet reads and decoding use a shared
`catch_unwind` boundary returning payload-free `UnsupportedAudio` errors.
Library callers must also use a payload-free panic hook; aborts, OOM and hangs
are not contained.

### Embedding, cache and failure boundaries

The optional backend ports the pinned Gemma 4 feature extractor: periodic Hann,
160 zero samples on the left, 321-sample frame indexing/320-sample window,
160-sample hop, 512-point FFT **magnitude**, 128 HTK mel filters and
`ln(mel + 0.001)`. The mask sampled every four frames determines the audio token
count; a full 10-second window has 999 frames and 250 tokens. The q8 encoder
feeds `[tokens,512]` audio features into the existing text graph between audio
begin/end tokens, with no task prefix. Output truncation/normalization and text
query behavior are unchanged.

`seyk model download --audio` provisions only
`onnx/audio_encoder_quantized.onnx` and its external data file at pinned ONNX
revision `daa72c51243991dfcaf9f9137d2c573d8f7790c0`. Together they are
340,307,954 bytes (324.54 MiB). The encoder loads lazily. Missing/corrupt optional
weights fail individual audio files while text, image and PDF indexing proceeds.
Inference has no download path.

API changes: the opt-in `pub mod audio` exposes preparation helpers/constants;
`ResourceKind::Audio` requires downstream exhaustive matches to add an arm.
`Embedder::input_identity(Modality)` is a new defaulted generic seam, forwarded by
Box/Arc. The backend's audio identity hashes both audio files and the feature
policy. The core includes this identity, the preparation policy and exact f32
sample bytes in **`audio-input-v1`**, and includes the policy/identity in the
resource descriptor. Neither changes the global cache profile, text model
identity or existing text input hashes. Tests cover local audio installation
without changing text identity/vectors and byte-identical files being re-prepared
with corrected locators after a timing-policy change.

At most `min(batch_size,4)` windows reach embedding together. The existing media
batch flush caches validated vectors and drops PCM immediately; only compact
locators remain until one atomic resource binding. Late decode/model failure
leaves any previous binding intact. Cache failures still propagate. No PCM,
mel arrays, audio tokens, ordered source text or transcripts are stored in the
cache. Tests inspect the database and cover failure after a successful batch.

CLI/JSON/MCP use the existing `Time` locator and schema; no MCP tool or option
was added. Search creates no previews or transcripts. Explicit audio extraction
uses the existing `UnsupportedExtraction` variant and says STT is not available
until 5b. STT, VAD, diarization, previews and a larger input-size policy remain
outside this phase.

### Measurements and reproduction

Measured locally on an Apple M5 Max, 64 GiB RAM, macOS 26.5, Rust 1.99.0,
CPU/q8, one ONNX thread. Evaluations use the production engine with fresh
in-memory caches and semantic-only queries. A hit is correct only when the
resource matches and its Time locator contains the **entire** annotated phrase.
Each corpus has 20 × 21-second synthetic recordings and 40 queries; 20 phrases
cross the first window boundary. The frozen manifests contain 1.87–3.43-second
phrases placed fully inside at least one window. Phrases longer than the
two-second overlap that straddle windows without fitting in either are
unmeasured; the boundary subset does not test that case.
No private corpus or recognizer is used.

The original spike set uses Samantha/Daniel. The separately frozen held-out set
uses new scripts and Karen, a voice unused by the spike. Its script SHA256 is
`36afa09b0063fd2c76db1575af68e953b5cd24129675aa837db5e948458a6591`.
No held-out tuning was performed. These synthetic results establish compatibility,
not a claim about noisy, multilingual or real-world speech.

| Set | Dimensions | Recall@5 | MRR@20 | Boundary Recall@5 / MRR@20 | Index s/audio-minute |
|---|---:|---:|---:|---:|---:|
| Spike | 256 | 1.000 | 0.9750 | 1.000 / 0.950 | 20.173 |
| Spike | 768 | 1.000 | 0.9875 | 1.000 / 0.975 | 17.089 |
| Held-out | 256 | 1.000 | 0.9875 | 1.000 / 1.000 | 16.600 |
| Held-out | 768 | 1.000 | 0.9875 | 1.000 / 1.000 | 15.838 |

Every format has Recall@5 1.0. Spike per-format MRR at 256/768: WAV 1/1,
FLAC .95/1, AAC 1/1, MP3 .95/.95 (10 queries each). Held-out per-format MRR is
identical at both widths: WAV/FLAC/AAC/ALAC 1.0, MP3 .9375 (8 queries each).
Elapsed measurements include decoding/resampling and run on a shared machine;
they are not isolated comparative dimension benchmarks. These post-fix timings
were collected under heavy host contention (observed load averages 32–52);
comparison with the earlier roughly 6.7 s/audio-minute run does not isolate
the code change.

A one-second Rust-generated chirp plus tones supplies the committed parity
signal. Pinned local Transformers 5.19.0 / Torch 2.14.1 FP32 generated the feature,
mask/token and 768-dimensional embedding goldens. Feature maximum absolute error
is **4.7683716e-7** (normal test, gate 1e-5); embedding cosine is **0.9998887**
(ignored model-backed test, gate .99). Exact reference revisions/source hashes
are recorded in `crates/seyk-embeddinggemma/tests/fixtures/audio/reference.json`.
Additional goldens repeat the same Rust signal to 16,001 samples (partial-frame
padding, 100 frames / 25 tokens) and 160,000 samples (full window, 999 frames /
250 tokens). Both features/masks and 768-dimensional embeddings use the same
pinned reference and parity gates. Their feature maximum errors are both
4.7683716e-7; embedding cosines are 0.9999343 and 0.9997173 respectively.
No additional source audio is committed.

Ten generated timing fixtures have zero cross-correlation lag and exact output
lengths at 16 kHz: WAV, FLAC, ALAC, FFmpeg AAC (1024 priming samples), Apple AAC
(2112), MP3/LAME, a nonzero-media-time trimmed AAC, MP3 without LAME metadata and
AAC without `edts`, plus an AAC file ending between whole milliseconds. The two
metadata-absent cases compare to their decoded-timeline WAV reference, so no
hidden constant is used to guess missing encoder delay. HE-AAC and
multiple-track fixtures produce typed rejections.

The largest allowed WAV is exactly **33,554,432 bytes**, representing
1,048.574625 seconds. It indexed 131 windows in **302.202 seconds**
(**17.292 s/audio-minute**); `/usr/bin/time -l` measured peak RSS
**1,257,816,064 bytes (1.171 GiB)** for the process, including ONNX Runtime.

Stripped (`-x`) Rust 1.99.0 release binaries: default **34,908,632 bytes**,
`--features audio` **36,368,200 bytes**; delta **1,459,568 bytes (1.392 MiB,
4.18%)**. No probe harness is included. Default dependency-tree verification
contains no Symphonia.

The local merge gate passed: default/audio workspace tests and clippy, core
feature matrix, ignored cache/PDF/audio goldens, timing fixtures, standalone
checks, Rust 1.89 checks and builds with/without audio, release smoke, licenses
(default/all features) and dependency boundaries. The fractional-duration fix
was followed by the affected audio gates again. The ALAC/panic-containment
and tail-window fixes passed the full local gate, including the expanded
feature/embedding goldens and the ten timing fixtures. Linux and Windows
validation remains for the CI matrix; this report covers macOS ARM64 only.

Reproduce (local weights only; ignored tests never provision models):

```sh
# Fixture generator: macOS say/afconvert, ffmpeg, Python numpy/soundfile.
python scripts/audio/generate.py spike /tmp/seyk-audio-spike
python scripts/audio/generate.py heldout /tmp/seyk-audio-heldout
SEYK_AUDIO_MANIFEST=/tmp/seyk-audio-heldout/manifest.json \
  cargo test --release --features audio --test audio_eval retrieval_gate -- --ignored --nocapture
# Repeat with the spike manifest; SEYK_AUDIO_REPORT optionally writes metrics JSON.
python scripts/audio/timing.py /tmp/seyk-audio-timing
SEYK_AUDIO_TIMING=/tmp/seyk-audio-timing \
  cargo test -p seyk-core --release --features audio --test audio_timing -- --ignored --nocapture
cargo test -p seyk-embeddinggemma --features audio --test audio_golden -- --include-ignored --nocapture
cargo test --release --features audio --test audio_eval largest_allowed_file -- --ignored --nocapture
# Regenerate parity input with the Rust audio_signal example, then run
# scripts/audio/reference.py using the pinned locally cached Transformers model.
```

Next: Phase 5b requires a separate STT/extraction decision and a user decision
before changing the 32 MiB input-size policy.

## P11 report: per-profile BM25 and live deletion statistics

### Implemented

Schema v6 gives every registered profile a regular FTS5 table named
`segments_fts_p_<hex(profile UTF-8)>`. Its name, path, heading and body columns
hold sorted normalized tokens, including duplicates, never source word order.
This stores the same information recoverable from the old full-detail postings
in a more directly readable form, with an additional copy on disk. The index
and vectors remain sensitive; OS encryption remains the at-rest boundary.

BM25 now uses only the selected profile's live indexed segments. Other profiles
cannot affect its scores, order or top-k results. Ordinary FTS deletes let SQLite
update document and token totals immediately, including replacement writes;
results match a fresh surviving-document index without periodic rebuilds.
The scoring corpus is still the whole profile, even when querying a path subset.
Field weights, query construction, tie ordering, CLI JSON and MemoryCache stay
unchanged.

The v3/v4/v5 migration reconstructs sorted fields from live `fts5vocab(instance)`
postings. It retains token multiplicities, field boundaries and zero-token rows,
checks full occurrence equivalence, and ignores deleted/orphan rows. Known
legacy-invalidated segments stay unindexed until their next sync. Unexpected
missing keyword rows abort the migration. Embedding bytes, input hashes, resource
state, profile IDs and segment rowids survive; no source reads or re-embedding
are required for a valid v5 cache. The same transaction pins segment rowids as
`id INTEGER PRIMARY KEY`. Any migration failure rolls back to the old schema.
The v1/v2 privacy reset still keeps segment/vector references and marks resources
stale, including their provider version and preparation descriptor.

Registration creates its table transactionally. Unknown-profile searches return
empty; a registered profile with a missing table produces an error. Clear drops
all per-profile FTS virtual tables, including orphans, then retains the existing
empty-cache checkpoint/VACUUM. Many profiles can exceed rusqlite's default
statement-cache capacity; this affects preparation cost, not correctness.

### Deliberately excluded

No automatic migration VACUUM, secure deletion, vector pruning, new profile
identity, or extraction change. Migration replaces the active index; it does not
scrub freed pages, old WAL frames, backups or snapshots. Physical erasure remains
separate work. Migration needs space for both indexes and the WAL, plus memory
for reconstructed tokens and SQLite's verification queries.

### Upgrading

This is a **one-way cache upgrade**. Stop older Seyk processes before opening an
existing cache with schema v6. Older binaries treat an unknown schema as disposable
and **delete the cache, including embeddings**. Use separate `--cache-path` files
when running mixed versions. New binaries reject unknown/future versions and
nonempty databases missing version metadata without dropping their tables.
The next release notes must repeat this warning; the crate version remains 0.1.0.

### Measurements and checks

Public synthetic release-mode benchmark on macOS ARM with bundled SQLite 3.53.2:
32 profiles, 20,031 documents and 4,046,200 postings (20,000 documents in the
largest profile). `dbstat` allocated keyword pages grew from 22,441,984 to
68,923,392 bytes, **3.071×** on this corpus; this is larger than the design's
approximate 2× estimate. Every synthetic document has the identical 200-token
body: postings compress especially well while the stored content repeats those
tokens in each row, making this corpus near worst case for the ratio. On a fresh
copy of the local default cache, migration grew allocated keyword pages from
16,384 to 20,480 bytes, **1.250×**. The ratio depends on token distribution and
corpus size; neither measurement is a general storage bound. These are active
keyword pages, excluding relational tables, freed pages and WAL.

In the final run, single-document replacement p50/p95 was 0.607/1.613 ms;
deletion was 0.180/0.717 ms; a 100-resource batch delete took 13.61 ms.
Stored document/token totals and BM25 matched a fresh live index exactly after
100 replacements and 200 deletions. Migration took 29.71 s with a 73,039,392-byte
WAL; the full benchmark process peaked at 359,481,344 bytes RSS. Earlier runs
on the same busy machine migrated in 50.94–99.04 s. The migration's occurrence
verification scans the old vocabulary per profile, so this one-time cost grows
with both postings and profile count. These are local observations, not latency
bounds. Reproduce with:

```sh
cargo test -p seyk-core --release --locked --test cache measure_p11 -- --ignored --nocapture
```

The public asymmetric ranking regression keeps A's top two documents and scores
(3.928078, 3.868289) unchanged across B registration, addition, replacement,
deletion, pruning and reopen. Previously, B reversed A's top two results and its
deletion left inflated scores. Additional bundled-SQLite tests cover 24 edit/delete
steps against fresh SQLite and MemoryCache, content integrity checks, Unicode and
duplicate postings, empty-token documents, sparse IDs across VACUUM, transactional
rollback, reconstructed v1–v5 layouts, warm v5 reuse with a provider that forbids reads,
unknown schemas, arbitrary profile IDs, and sorted persisted fields.

The existing eight-target evaluation was run with main and P11 release binaries
against separate copies of the same cache. All **40 ranks were unchanged** across
semantic, hybrid weights 1.0/0.3/0.15, and default search. Default top-10 coverage
remains **7/8**. Only anonymized ranks are reported:

| Target | Default before → after | Semantic before → after |
| --- | --- | --- |
| Q1 | 1 → 1 | 9 → 9 |
| Q2 | 7 → 7 | 1 → 1 |
| Q3 | 1 → 1 | 1 → 1 |
| Q4 | 2 → 2 | 1 → 1 |
| Q5 | 1 → 1 | 1 → 1 |
| Q6 | 2 → 2 | 18 → 18 |
| Q7 | 25 → 25 | 19 → 19 |
| Q8 | 7 → 7 | 7 → 7 |

All 2,195 embedding blobs and all 2,195 segment rows (including rowids) were
identical across the evaluated v5 and v6 copies. The evaluation harness decodes
CLI resource URIs before matching the existing path-based targets; queries,
paths and source text remain local.

The local merge gate passed: pinned Rust 1.99 formatting and workspace Clippy
with warnings denied; workspace, core minimal, PDF and PDF-render tests; both
model-backed cache-profile and PDF CLI goldens; Rust 1.89 all-target checks;
release CLI smoke; license checks; and embedding dependency boundaries. Linux
and Windows execution still requires a CI run.

## Phase 5b-1: verified streaming audio and duration policy

The optional `audio` feature now admits a canonical 16 kHz presentation of up to
7,200 seconds by default. `SearchEngineBuilder::max_audio_duration_secs` and CLI
`--max-audio-duration-secs` set one positive, inclusive, whole-second limit.
Daemon startup arguments always include the value (including the default), so
its endpoint and engine use the same policy; request discovery options cannot override the engine.
Audio discovery derives its byte ceiling as
`192000 * 8 * 4 * seconds + 64 MiB`; other kinds retain their 32 MiB default.
This does not add codecs or container variants. Manual edit/priming skip is
independently limited to 30 seconds. Unknown presentation lengths receive a
packet-duration demux pass before the first embedding; canonical sample counting
remains the final authority. Over-limit resources are rejected in full.

`ResourceProvider::open` supplies a `Read + Seek + Send + Sync` source, with an
unsupported default and a filesystem implementation. Audio no longer calls the
whole-file `read` method. One pass computes the existing full-source BLAKE3 and
an in-memory table of BLAKE3 hashes for fixed 1 MiB chunks. `VerifiedReader`
fetches and verifies a whole chunk before serving any bytes to either MP4 timing
inspection or Symphonia. Its two-chunk cache serves only already-verified bytes;
a refetch checks again. Every parser/decoder byte is therefore equal to the
corresponding byte committed by the source hash, even across seeks or transient
ABA writes. This is a commitment to the first pass's byte sequence, not a claim
that a mutable file was atomically snapshotted at a single instant. Unconsumed
changes need not be detected; they cannot introduce different decoder input.
Mismatch/truncation poisons the reader and becomes `ChangedWhileReading`, even
if an upstream parser translates the I/O error. Other refetch failures return
an ordinary I/O error from `Read` without setting the content-change poison flag.
No source snapshot is written.

MP4 is admitted only with `ftyp` at offset 4. Preflight and the directly
constructed isomp4 reader both start at byte zero; the non-MP4 probes never
register isomp4, including the second pass after a duration scan. Prefixed MP4
is rejected. MP3 ID3 prefixes are skipped by checked extents without allocating
or interpreting their payload, so embedded tag bytes cannot select a probe origin.
Inspection traverses checked, parent-bounded ranges, skips `mdat`, and handles
`moov` after media and extended sizes. Size-zero is accepted for
opaque media/padding only; parsed metadata with that encoding is rejected
because Symphonia gives it a zero data length. Original edit-list conversion,
mdhd-end clamping and iTunSMPB precedence are retained.

Before demux construction, stsz/stts/stsc/stco/co64 counts are checked against
box payload lengths and an aggregate **64 MiB live parsed-table allocation**
budget; stsc entries are charged at their larger in-memory size. Every constant
and variable stsz sample size is bounded to the **1 MiB packet limit**, including
before the demux-only duration pass. A fixed scratch buffer scans variable sizes.
A separate 64 MiB cumulative top-level metadata-byte limit bounds metadata
(excluding media and free padding). Parent bounds are checked throughout metadata
containers, every ilst item (including cover art and unknown tags), sample
entries and wave codec containers. Parsed leaves/items are capped at **64 KiB**;
minimum prefixes prevent upstream length subtraction from underflowing. ESDS
nested descriptor lengths are checked against their enclosing bytes as well.
Unsupported sample entries, ES optional flags and fragmented MP4 structures
(moof/mvex/sidx/mfra) are rejected before upstream construction. These limits do
not scale with duration. AAC/ALAC sample entries retain their existing codec,
packet, sample-rate/channel and ALAC-cookie checks.

PCM remains a bounded packet, resampler history, rolling 10 s window and at most
four pending windows. The chunk hash table costs 32 bytes per MiB (64 KiB for
2 GiB). At the default 44,303,908,864-byte ceiling, the table has at most
42,252 entries / 1,352,064 hash bytes (about 1.29 MiB, excluding Vec capacity).
The first pass hashes the full admitted byte extent **before** checking duration;
a source near that byte ceiling is fully read even if its duration is then
rejected. The small measured over-cap fixtures do not represent that worst-case
rejection cost. Compact locators grow with the configured duration: exactly two
hours produce 899 full windows and one 8 s tail, 900 windows total. Source hashing and
verification add I/O; unknown-length files additionally require demux traversal.
No PCM, readable content or chunk commitments are persisted in the cache.

The `PREPARATION` string, including historical `cap16777216`, is deliberately
frozen as an input-identity compatibility token. The suffix no longer describes
the effective limit. Raising a limit changes neither model-input hashes nor the
cache profile and does not re-embed already-valid windows. There is no schema
migration. Both cache reuse paths and result selection enforce eligibility using
cached Time locators: `max(end_ms) <= 1000 * limit_seconds` is exact for the
inclusive canonical-sample limit because `end_ms = ceil(N / 16)`. Cached-only
search also excludes unknown/changed versions and preparation descriptors.
Failed syncs preserve previous bindings but exclude them from that request.

`audio::extract_samples` / `SearchEngine::extract_audio_samples` verify the full
source hash before returning a selected, at-most-30-second PCM interval.
`slice_verified` decodes sequentially from the presentation origin using the
same resampler, discards earlier output and stops at the requested end with its
normal right support when a reliable length is available. Unknown lengths are
fully validated before returning. It never resets phase at an interior boundary.
Half-open millisecond bounds select `[16*start_ms, min(16*end_ms, N))`; only the
legitimate rounded EOF endpoint is clamped. The model-only short-tail padding is
removed. No per-codec fast seek ships. Readable audio extraction still returns
STT unavailable; search previews remain null. STT, Whisper, model downloads and
new transcript provenance are outside this phase.

Tests cover changed chunks before decoding, after a successful window/batch and
on an evicted-chunk reread; no partial resource binding or slice escapes. Limit
lowering/re-raising exercises both cache fast paths and cached-only search,
retaining identical vector input hashes. The ignored input-parity test compares
against hashes captured before the refactor using the same generated timing
fixtures, and checks bit-exact extraction versus embedding samples, including
one-sample, short-tail and non-16k cases. Reproduce fixtures with
`uv run --no-project --with numpy python scripts/audio/timing.py tmp/timing`;
run `audio_timing` and `audio_input_parity` with `SEYK_AUDIO_TIMING` pointing there.
The fixture source hashes retain baseline provenance. The gate compares model
inputs and locators, allowing container creation timestamps to change on
regeneration; different codec tooling can still change the canonical PCM.
`audio_input_capture.rs` is a standalone ignored capture helper using only APIs
available at `7cab7ee`; its header gives the command for a clean historical
checkout. Use the same generated files for historical and current captures to
compare complete JSON. CoreAudio AAC encoding can change the actual audio
between fixture generations, not just timestamps. For regenerated fixtures,
pass `SEYK_AUDIO_BASELINE=/path/to/7cab7ee-capture.json` to `audio_input_parity`;
the default remains the committed capture. Do not capture that override with the
implementation under test. This review's same-file control using unmodified
`7cab7ee` audio modules produced identical complete JSON for all 18 files and
20 windows; only the regenerated Apple AAC differed from the older capture.
The legacy `windows(Vec<u8>)` helper always uses the default duration; configured
callers use `windows_verified` or the engine builder instead.

Measurements use `scripts/audio/long.py tmp/long` and the release
`bench_audio` example, separately per source and mode. The generator streams
changing synthetic chirps to WAV/MP3 without whole-file PCM allocation. The
`prepare` mode reports first-pass hash throughput, complete verified-read
throughput, decode-plus-verification time and peak process RSS without a model.
The `index` mode uses fresh in-memory caches, the pinned q8 model at 256 dimensions
and one ORT thread; counts must show every unique window actually embedded.
Every measurement records the three load averages before and after it. Long
fixtures live under ignored `tmp` and are deleted after verification.

### Phase 5b-1 measured evidence

The cost and held-out recall observations below were collected at `c969667`,
before review hardening; they were not remeasured in the fix round.

Measured on Apple M5 Max, 64 GiB, macOS 26.5, Rust 1.99.0 release. Indexing uses CPU/q8, 256 dimensions, one ORT thread, a fresh model/process and empty in-memory cache per file. OS caches were not flushed. Up to two indexing measurements ran concurrently; other host work also ran. Load columns show **(1, 5, 15 minute averages), before → after**, not a run average. The host one-minute load was also observed above 82 during the long runs; these are shared-host costs, not isolated performance comparisons.

#### Preparation (no embedding model)

Decode time includes verified reads and, where required, the demux duration pass. RSS is the per-process peak, including libraries. The hash and complete verified-read stages are measured separately. MB/s uses decimal MB; RSS uses MiB.

| Source | Hash MB/s | Verify MB/s | Decode + verify s | Windows/s | s/audio-min | Peak MiB | Load before → after |
|---|---:|---:|---:|---:|---:|---:|---|
| 10min.wav | 548.7 | 1141.1 | 0.200 | 374.8 | 0.0200 | 16.91 | (30.5, 24.4, 20.9) → (30.5, 24.4, 20.9) |
| 10min.mp3 | 455.1 | 950.4 | 0.329 | 228.2 | 0.0329 | 16.44 | (30.5, 24.4, 20.9) → (30.5, 24.4, 20.9) |
| old-cap.wav | 599.4 | 1209.1 | 0.386 | 339.6 | 0.0221 | 18.55 | (30.5, 24.4, 20.9) → (30.5, 24.4, 20.9) |
| old-cap.mp3 | 592.9 | 1159.0 | 0.508 | 258.0 | 0.0290 | 18.58 | (30.5, 24.4, 20.9) → (30.5, 24.4, 20.9) |
| 2h.wav | 533.7 | 1125.0 | 2.376 | 378.7 | 0.0198 | 21.25 | (28.8, 24.2, 20.8) → (28.8, 24.2, 20.8) |
| 2h.mp3 | 622.7 | 1194.6 | 3.273 | 274.9 | 0.0273 | 19.61 | (28.8, 24.2, 20.8) → (28.3, 24.2, 20.9) |
| over-cap.wav | 612.7 | 1045.0 | 0.001 | 0.0 | 0.0000 | 15.06 | (28.3, 24.2, 20.9) → (28.3, 24.2, 20.9) |
| over-cap.mp3 | 507.3 | 1020.1 | 0.072 | 0.0 | 0.0006 | 12.61 | (28.3, 24.2, 20.9) → (28.3, 24.2, 20.9) |

Over-cap preparation timings are **rejection latency**, not successful decode throughput; both emitted zero windows. Durations: 10 min = 600 s; old cap = 1,048.576 s; 2 h = 7,200 s; over cap = 7,260 s. Here old cap means the prior decoded-duration ceiling; the WAV header puts that synthetic source slightly above the former encoded-byte ceiling.

#### Full indexing

Index time excludes initial text-model construction but includes lazy audio-model load, source verification, decoding, embedding and resource binding. Embedding throughput includes the first lazy audio load. Total s also includes initial model construction; it starts inside the benchmark process.

| Source | Total s | Index s | Embedded windows | Embedding windows/s | Index windows/s | Index s/audio-min | Peak GiB | Load before → after |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 10min.wav | 246.415 | 244.358 | 75 | 0.308 | 0.307 | 24.436 | 1.150 | (12.9, 24.4, 30.6) → (43.8, 36.9, 34.6) |
| 10min.mp3 | 605.480 | 602.335 | 75 | 0.125 | 0.125 | 60.234 | 1.151 | (19.6, 25.1, 30.7) → (28.2, 40.5, 38.0) |
| old-cap.wav | 350.712 | 345.317 | 131 | 0.381 | 0.379 | 19.759 | 1.144 | (43.3, 40.5, 36.6) → (19.6, 25.1, 30.7) |
| old-cap.mp3 | 325.285 | 319.794 | 131 | 0.411 | 0.410 | 18.299 | 1.144 | (43.3, 40.5, 36.6) → (12.9, 24.4, 30.6) |
| 2h.wav | 2590.337 | 2588.832 | 900 | 0.349 | 0.348 | 21.574 | 1.159 | (40.7, 30.7, 24.5) → (43.3, 40.5, 36.6) |
| 2h.mp3 | 2589.060 | 2587.555 | 900 | 0.349 | 0.348 | 21.563 | 1.123 | (40.7, 30.7, 24.5) → (43.3, 40.5, 36.6) |
| over-cap.wav | 10.184 | 1.830 | 0 | — | 0.000 | 0.015 | 0.422 | (43.8, 36.9, 34.6) → (47.6, 37.9, 35.0) |
| over-cap.mp3 | 9.768 | 0.885 | 0 | — | 0.000 | 0.007 | 0.405 | (47.3, 38.0, 35.1) → (46.3, 38.1, 35.1) |

Over-cap index timings are rejection latency. The harness asserts zero embedder calls for both over-cap sources. The two-hour admitted cases each embedded 900 windows; source windows contain 8,998 audio seconds including overlap. No STT runs in any measurement. Over-cap RSS excludes the lazy audio model because embedding is never attempted. Admitted-case RSS stays in a narrow band across durations. The ten-minute MP3 observation is much slower than the other admitted cases; these unisolated runs do not establish a codec performance difference.

#### Retrieval compatibility

The frozen held-out synthetic corpus retains its earlier results at both widths. Load is the whole recall gate’s before/after (1, 5, 15 minute) averages, including compilation; it is not a per-width average. The two widths ran sequentially while host contention changed, so their elapsed costs are not a dimension comparison.

| Dimensions | Recall@5 | MRR@20 | Boundary Recall@5 / MRR@20 | Index s/audio-min | Gate load before → after |
|---|---:|---:|---:|---:|---|
| 256 | 1.000 | 0.9875 | 1.000 / 1.000 | 58.742 | (73.8, 47.5, 36.9) → (13.3, 25.4, 34.2) |
| 768 | 1.000 | 0.9875 | 1.000 / 1.000 | 27.613 | (73.8, 47.5, 36.9) → (13.3, 25.4, 34.2) |

The input-identity gate retained every recorded window hash and locator across all 18 timing/edge fixtures (20 windows). Slice samples are bit-exact after removing model-only short-tail padding. The audio embedding and feature goldens also pass their existing gates.

Local validation passed on the integrated branch: Rust 1.99.0 formatting,
default/audio workspace tests and warning-denied Clippy; generated timing and
input-hash/slice parity; model-backed audio and cache-profile goldens; held-out
retrieval recall; and Rust 1.89.0 default/audio all-target workspace checks.
All-feature license validation passes. The complete cargo-deny audit reports
pre-existing unmaintained packages number_prefix 0.4.0 (RUSTSEC-2025-0119),
paste 1.0.15 (RUSTSEC-2024-0436), and ttf-parser 0.25.1 (RUSTSEC-2026-0192).
Dependency versions, the lockfile and advisory policy are unchanged.

The review round reran formatting, default/audio workspace Clippy with warnings
denied, both workspace test configurations, audio goldens, generated timing
fixtures, same-file historical input-hash/slice parity, and both Rust 1.89
all-target checks; all passed. The benchmark usage-error smoke check passed.

## Phase 5b-2: on-demand speech-to-text extraction

The optional root `audio-stt` feature implies `audio` and adds `seyk-stt`, the
only production implementation of the core `audio::SpeechToText` trait.
`seyk-core`, including its audio feature, has no ORT, tokenizer or download
dependency. Default facade builds do not enable STT. The embedding input
identity, profile and vector cache are unchanged.

`AudioExtractor` is a separate provider-backed extractor, deliberately absent
from byte-extractor/preview dispatch. An explicitly selected audio hit passes
through 5b-1's full-source BLAKE3 check and chunk-verified slice before calling
STT or loading its model. Half-open millisecond bounds select canonical samples
`[16*start_ms, min(16*end_ms, N))`; only rounded EOF is clamped. This retains
sequential decoding from the presentation origin and its exact resampler phase;
it adds no fast seek. Requests longer than 30 seconds return the typed
`AudioIntervalTooLong` error. Each request is independent; there is no prior
transcript conditioning, stitching, transcript cache, or content logging.
Indexing, search (including previews) and model info never load Whisper.

CLI extraction uses `seyk extract --from-json hits.json --hit 0`, optionally
`--stt-model-dir DIR` (or `SEYK_STT_MODEL_DIR`) and `--stt-language en`.
Omitting the language requests detection from that interval alone. Both STT
options are rejected on `model download`; they configure extraction, not
provisioning. The same settings configure audio transcription in the explicitly
invoked MCP `extract` tool. Text/PDF extraction is available in MCP without STT.
MCP hits now carry locator and metadata; extraction confines the local path to
the session roots independently of the caller's supplied hash.
CLI/MCP JSON output advances to schema 4; CLI input accepts search schemas
2, 3 and 4. Raw text and PDF extraction retain their existing behavior.

Human audio output starts with **Machine transcript (may contain errors)**.
JSON provenance records `method=speech_to_text`, `implementation`,
`fidelity=machine_transcription`, requested locator, effective canonical sample
bounds/rate, manifest SHA-256, requested/detected language and `no_speech`.
These are input bounds, not word timestamps or a claim of verbatim accuracy.
`no_speech` returns empty text and no detected language, even for a requested
language. `ExtractedContent` Debug redacts text. Error messages contain no
PCM, token IDs, transcripts or underlying inference diagnostics.

### Provisioning and offline runtime

The artifact is `Xenova/whisper-tiny` at
`5332fcc35e32a33b86612b9a57a89be7906102b1`: separate FP32 encoder, decoder and
decoder-with-past graphs. The compiled
[`manifest.json`](../crates/seyk-stt/assets/manifest.json) pins every asset's
length and SHA-256, complete graph input/output contracts, decoding policy,
mel filters and notices. Its compact JSON has SHA-256
`3cefb2a90f9cee1983933dcaee20a25689c8b35a3c9bedfd94ae3c38ffb1a254`;
a unit test pins those bytes. Reformatting changes the STT installation directory
and provenance identity, but neither model semantics nor existing audio vectors.
The graphs use opset 11, IR 6, no external data, and
no separate attention-mask or position inputs. `ort =2.0.0-rc.13` uses CPU
sessions with one intra/inter thread by default.

`seyk model download --stt` is the sole network boundary. It downloads only
this immutable manifest through HF Hub, copies bounded assets to a sibling
staging directory, verifies the complete tree and publishes it by directory
rename. An install lock serializes writers; failed staging never publishes
partial files. Valid installations are reused; an installation carrying our
manifest marker can be repaired with backup/restore on publication failure.
Unmanaged destinations are preserved. A crash can leave a staging directory
or lock requiring operator cleanup. Model storage is the platform cache under
`seyk/stt/<manifest SHA-256>`, independent of Gemma's model/cache identity.

Runtime uses only local files. User directories receive the same exact asset
and layout checks; extra files and directory symlinks are rejected. Each graph
is loaded from the same bytes just length/hash-verified, avoiding a second
open between verification and ORT parsing. File symlinks do not bypass that
check. Assets and marker files must resolve to regular files before opening;
Unix opens also use `O_NONBLOCK` and recheck the opened descriptor, so a FIFO
substitution cannot hang verification. Direct FIFOs and symlinks to FIFOs have a
two-second regression deadline and fail with the typed model error.
Missing/corrupt/tampered assets return `SttModel` with
`seyk model download --stt` in the error. Sessions retain verified weights in
memory, so later disk changes affect the next session load, not already loaded
weights. No inference path invokes HF Hub or repairs an installation.

Weights are downloaded by the operator, not bundled. The pinned Xenova card
states Apache-2.0 and links Transformers.js; that exact tree has no LICENSE or
NOTICE file. The original OpenAI weights and reference mel implementation are
MIT. The card, Apache text, OpenAI MIT notice and derived pinned mel bank are
retained under `crates/seyk-stt/assets`; see `THIRD_PARTY.md`. Cargo-deny checks
Rust dependency licences separately from these pinned model notices.

### Mel and decoding contract

Canonical mono 16 kHz input is zero-padded to 30 seconds. Rust computes the
periodic-Hann STFT (400 samples, hop 160, reflected center padding), power,
pinned 80-band Slaney mel projection, log clamp and normalization, yielding
`[1,80,3000]`. The independent pinned OpenAI/PyTorch fixture compares every
value with maximum absolute tolerance `1e-4`, including reflected edges and
the padded tail. Its regeneration command and source hashes are in
`scripts/audio/stt_fixture.py` and the fixture provenance JSON.

The encoder returns only `last_hidden_state [1,1500,384]`. An initial SOT
`50258` pass requests only raw logits (no returned KV tensors);
`p_no_speech = softmax(raw)[50362]` before suppression (`50362` is `<|nocaptions|>` in this export). Automatic language
selection takes the largest language-token logit (lowest ID wins ties), or
uses the validated explicit language. A fresh initial decoder pass uses
`[50258, language, 50359, 50363]` (transcribe, no timestamps). Greedy decoding
then feeds one token with all four layers' cached self/cross key/value tensors.
Self KV grows; encoder KV is reused. Each step fetches only logits and required
`present.*`; **attention outputs are never requested** from any graph.

The pinned 88-token always-suppress list applies throughout; IDs above EOT
`50257` are suppressed after the prefix. Blank `220` and EOT are additionally
suppressed at the first generation step only. The total context is capped at
448 tokens, including the four-token prefix; exhaustion returns a typed
inference error rather than a silently truncated transcript. Average log
probability sums generated tokens including EOT and divides by generated
count plus one. `p_no_speech > 0.6 && avg_logprob <= -1.0` returns `no_speech`.
Exact-zero PCM short-circuits to `no_speech` after validating/loading the model;
it never invents a language. Tiny Whisper may still hallucinate on noise or
misrecognize speech; the transcript label is mandatory, not a confidence claim.

### Validation and reproduction

Ordinary tests cover full-tensor mel parity, token suppression, missing,
corrupt and equal-length tampering, failed installation, the 30-second cap,
hash mismatch before STT, no STT during indexing/search/preview, provenance
bounds, redacted Debug and no transcript bytes in SQLite. Tests also pin bundled
notice/mel hashes and policy constants against the compiled manifest, and reject
extraction-only options on `model download`. Ordinary CLI tests generate a Rust
chirp WAV at runtime; they require no speech recording. MCP tests exercise
explicit extraction, stale hashes and session-root confinement. Dependency
checks enforce no ORT in standalone core (with and without audio).

The local model-backed gate exercises 10/30-second English text/token goldens,
cached-versus-uncached logits at every 30-second decode step (maximum absolute
tolerance `1e-4` and identical allowed argmax), silence without detected
language, installation/repair, missing graphs, equal-length graph tampering
and unexpected files. It copies only manifest assets into temporary storage;
it never edits the supplied model source or downloads anything:

Generate speech locally first (macOS `say`, FFmpeg and the pinned OpenAI reference
files are required). **System Voice output must never be committed or
redistributed**. All speech WAVs are generated under ignored `tmp/`; only the
public text, token goldens and mathematical chirp/mel fixtures are committed.
The generator also rebuilds `mel-80.f32le` from the pinned reference bank and
asserts its manifest SHA-256. Voice/OS changes may change exact speech goldens.

```sh
uv run --no-project --with numpy==2.5.3 --with torch==2.14.1 python scripts/audio/stt_fixture.py /path/to/pinned/reference
SEYK_STT_SPEECH_WAV=$PWD/tmp/stt-speech.wav SEYK_STT_MODEL_DIR=/path/to/pinned/assets cargo test -p seyk-stt --release -- --include-ignored
SEYK_STT_SPEECH_WAV=$PWD/tmp/stt-speech.wav SEYK_STT_MODEL_DIR=/path/to/pinned/assets cargo test --features audio-stt --test stt_cli -- --include-ignored
```

These gates are ignored by default and read the local speech path from
`SEYK_STT_SPEECH_WAV`. CI runs non-model tests and warning-denied Clippy with
`audio-stt`, along with the Rust 1.89 all-target check and existing licence checks.

For cost measurements, generate the public 10/30-second WAVs with the fixture
script, then run each in a fresh release process:

```sh
cargo run --release --no-default-features --features audio-stt --example bench_stt -- /path/to/exact/model-dir tmp/stt-10.wav 10
cargo run --release --no-default-features --features audio-stt --example bench_stt -- /path/to/exact/model-dir tmp/stt-30.wav 30
```

This reports full manifest-verified cold session load, first extraction, warm
nearest-rank p50/p95 across 20 extractions, cost per audio minute, process peak
RSS and before/after load averages. Warm measurements include source hashing,
verified decoding/slicing, mel, encoder, greedy decode and provenance, but not
model load. Cold means fresh sessions/process, not a purged OS file cache.

### Local measurements (shared host)

Measured on 2026-10-09 with Rust 1.99.0 release, native ORT 1.28.0,
macOS 26.5, Apple M5 Max (18 logical CPUs, 64 GiB RAM), CPU provider with
one intra/inter thread. Each short duration ran in a fresh process, followed by
one first extraction and 20 measured warm requests. The first extraction is excluded
from warm percentiles. This was a busy shared host; every measured value below
carries its load range and is an observation, not a quiet-host capacity claim.

| Selected interval | Measurement | Result | Load before → after (1 / 5 / 15 min) |
|---|---|---:|---|
| 10 s | Cold verified session load | 0.934 s | 6.85 / 8.31 / 13.02 → 6.85 / 8.31 / 13.02 |
| 10 s | Warm p50 extraction | 0.254 s | 6.85 / 8.31 / 13.02 → 6.54 / 8.22 / 12.96 |
| 10 s | Warm p95 extraction | 0.255 s | 6.85 / 8.31 / 13.02 → 6.54 / 8.22 / 12.96 |
| 10 s | Warm p50 cost per audio minute | 1.525 s | 6.85 / 8.31 / 13.02 → 6.54 / 8.22 / 12.96 |
| 10 s | Peak process RSS | 1127.41 MiB | 6.85 / 8.31 / 13.02 → 6.54 / 8.22 / 12.96 |
| 30 s | Cold verified session load | 0.924 s | 6.54 / 8.22 / 12.96 → 6.26 / 8.14 / 12.90 |
| 30 s | Warm p50 extraction | 0.397 s | 6.26 / 8.14 / 12.90 → 6.32 / 8.12 / 12.86 |
| 30 s | Warm p95 extraction | 0.410 s | 6.26 / 8.14 / 12.90 → 6.32 / 8.12 / 12.86 |
| 30 s | Warm p50 cost per audio minute | 0.794 s | 6.26 / 8.14 / 12.90 → 6.32 / 8.12 / 12.86 |
| 30 s | Peak process RSS | 1131.83 MiB | 6.26 / 8.14 / 12.90 → 6.32 / 8.12 / 12.86 |
| [7190,7200) s in 2 h WAV | Cold verified session load | 0.942 s | 6.32 / 8.12 / 12.86 → 6.21 / 8.07 / 12.82 |
| [7190,7200) s in 2 h WAV | End-to-end first extraction | 1.998 s | 6.21 / 8.07 / 12.82 → 6.21 / 8.07 / 12.82 |
| [7190,7200) s in 2 h WAV | Peak process RSS | 1121.83 MiB | 6.21 / 8.07 / 12.82 → 6.21 / 8.07 / 12.82 |

The short WAVs begin at zero. The late-hit measurement selects the final ten
seconds of a generated two-hour, 230,400,044-byte mono PCM16 WAV: streamed
silence followed by the local ten-second speech golden. Its end-to-end time
includes full-source hash verification, sequential decode to the interval,
verified slicing, mel, encoder, greedy decode and provenance; session load is
reported separately. Fixture metadata hashing runs before timing, warming the
OS file cache, but extraction still rehashes the whole source. This is one
late-hit observation, not a percentile or a seek-speed claim. Peak RSS is the
Rust process high-water mark including native ORT and all three FP32 sessions.
The release binaries ran successfully; the host emitted a rust-objcopy/libLLVM
strip warning.

#### Small local accuracy evaluation

The generator creates 24 public-text clips at test time only: eight authored
English texts in both Karen (en_AU) and Samantha (en_US), plus eight German
texts in Anna (de_DE), using `say -r 150`. Audio remains in ignored `tmp/` and
must not be redistributed. The Rust evaluation uses the same verified
`AudioExtractor` with automatic language detection and emits only aggregate
scores, never transcripts. It lowercases Unicode, replaces non-alphanumeric
characters with spaces, and collapses whitespace; WER is total word edit
distance divided by reference words, and CER is total Unicode-character edit
distance divided by reference characters (including normalized spaces).

| Language / voices | Clips | WER (errors / words) | CER (errors / characters) | Load before → after (1 / 5 / 15 min) |
|---|---:|---:|---:|---|
| English / Karen + Samantha | 16 | 3.48% (11 / 316) | 0.76% (13 / 1716) | 7.36 / 8.46 / 13.12 → 6.85 / 8.31 / 13.02 |
| German / Anna | 8 | 14.38% (23 / 160) | 2.53% (25 / 989) | 7.36 / 8.46 / 13.12 → 6.85 / 8.31 / 13.02 |

Every clip selected the expected language and none returned `no_speech`
(load 7.36 / 8.46 / 13.12 → 6.85 / 8.31 / 13.02). This small clean synthetic
sample is a local smoke evaluation, not evidence of general transcription
quality. **Noise, music and broader multilingual quality remain an open
pre-release gate**, including evaluation on real recordings and selected hits.

```sh
python3 scripts/audio/stt_eval_fixture.py
cargo run --release --no-default-features --features audio-stt --example eval_stt -- /path/to/exact/model-dir tmp/stt-eval/manifest.json
cargo run --release --no-default-features --features audio-stt --example bench_stt -- /path/to/exact/model-dir tmp/stt-two-hours.wav 10 7190 0
# After gates and measurements, remove generated System Voice output:
rm -f tmp/stt-speech.wav tmp/stt-10.wav tmp/stt-30.wav tmp/stt-two-hours.wav tmp/stt-eval/*.wav
```

Local checks passed: default/audio/audio-stt workspace tests and warning-denied
Clippy, the standalone dependency boundaries, the release model gate and
real-binary CLI gate, regenerated audio timing fixtures, existing audio
feature/embedding goldens, and Rust 1.89 default/audio/audio-stt all-target checks.
Both default and all-feature cargo-deny audits pass licenses, bans and sources;
the advisory check retains the three pre-existing unmaintained findings:
number_prefix 0.4.0 (RUSTSEC-2025-0119), paste 1.0.15 (RUSTSEC-2024-0436), and
ttf-parser 0.25.1 (RUSTSEC-2026-0192). No external dependency version or advisory
exception changed. Linux/Windows execution remains the CI matrix's responsibility.


## Verified cache import for re-staged paths

`SqliteCache::import_from(source, destination, &profile, &mappings)` is separate
from the ingestion trait. `CacheImportMapping::read_jsonl` parses strict JSONL v1;
`SqliteCache::import_profile` selects and validates a source profile without a
writable cache open. The facade re-exports the mapping and report types. CLI
`seyk cache import --from OLD --map FILE --profile ID --cache-path NEW [--json]`
requires an explicit destination flag, even when `SEYK_CACHE` is set. No model
initialization, extraction or download occurs.

Import accepts schema v6 and v7. Derived `image_codes` are ignored; only retained f32 vectors are imported. It opens a `mode=ro` URI connection and keeps one
read transaction for a consistent snapshot, including committed WAL frames. It
never migrates or copies the source database. Older sources must first be opened
normally with this version; future and unversioned sources fail. Read-only WAL
access needs existing WAL/SHM sidecars or directory write access; failures are
surfaced. Canonical paths and, on Unix, device/inode pairs reject same-file aliases.
Profile JSON, derived profile ID and stored dimension must agree exactly.

Each mapping is one-to-one between local absolute file URIs and includes the
caller’s full-file BLAKE3. Only the new file is opened, using the filesystem
provider’s existing policy and no-follow handle access. A bounded 64 KiB buffer
hashes the complete handle once; fstat metadata before/after must agree, and the
path identity is checked after reading. Size/mtime never replace hashing. The
observed hash must equal the old cache content hash; the supplied hash is an
extra guard. Any mismatch is a `hash_mismatch` drop. Missing hashes/files,
unstable or disallowed files and absent mappings omit all resource evidence.
Metadata and freshness tokens come from the verified handle. No re-verification
occurs before commit; later ordinary extraction still checks the content hash.
Unix handles are pinned; Windows remains path-based, not pinned. This is not an
atomic filesystem snapshot.

Preparation fingerprints compare kind and filename-derived title without opening
the old path. Text descriptors must match the new fingerprint directly. NULL
descriptors drop. PDF/audio descriptors are composite and cannot be recomputed
without runtime preparation identities: import preserves them and normal runtime
checks decide compatibility on reuse. Parent-directory moves preserve inputs;
changes to kind or title drop conservatively. No ranking, chunking or profile
format changes are involved.

The destination is absent or a completely empty compatible v6 cache. Existing
private-cache guards, foreign keys and secure_delete apply. Schema initialization,
profile registration and every accepted write share one IMMEDIATE transaction.
Referenced vectors are copied byte-for-byte once per input hash; unrelated vectors
and profiles are excluded. Missing vectors, bad widths/locators/ordinals and
missing or non-order-free FTS fields abort. Source FTS heading/body columns are
read by segment rowid and checked for `order_free_text` idempotence. New segment
IDs receive those fields plus freshly tokenized name/path labels; old path/name
postings and cleanup metadata are never copied. Foreign-key, vector-reference,
FTS cardinality and destination FTS integrity checks precede the single commit.
Fatal errors roll back schema and rows; row-level drops are counted outcomes.
`CacheImportReport::drops` always lists all five reasons, including zero counts:
`hash_mismatch`, `missing_hash`, `preparation_changed`, `unavailable_or_unstable`,
and `unmapped`. Key presence alone does not indicate a drop.
A failed fresh destination may remain an empty shell suitable for retry.

Synthetic FakeEmbedder tests cover unchanged remaps, exact vector bits, zero new
resource embedding calls, extraction verification, metadata, token frequencies,
old-path/source-order exclusion, mismatches, descriptor changes, missing paths,
non-UTF-8 identities, mapping validation, schemas/profiles, corrupt evidence,
WAL reads, shared inputs, isolation, aliases and populated destinations. Unit
failure checkpoints exercise rollback after resource/vector/segment/FTS writes
and before commit, and deterministic post-read mutations test handle verification.
CLI tests use synthetic fixtures and no model weights or private data.

## Photo Anchor image scan (v7)

The v6-to-v7 migration adds `image_codes(profile,input_hash,codec_id,codes)` and updates schema metadata in one transaction. It does not rebuild segment IDs or FTS postings and does not re-embed. A composite foreign key cascades embedding deletion into derived-code deletion. Exact f32 vectors and CacheProfile IDs remain unchanged.

Every cache open upgrades to v7, even when q4 is off. The immediately preceding v6 binary rejects v7; older binaries that predate fail-closed schema handling may discard unknown caches. Use separate caches when running mixed versions.

An opted-in sync derives missing standalone-image codes within its write phase. Cached-only queries never write codes. The q4 path reads codes and exact fallback rows, builds one LUT per query, selects the fixed image shortlist and fetches only its retained f32 vectors. It reranks with the same exact dot kernel before thresholds and fusion. SQLite is required; custom caches default to unsupported. Stats count derived image rows separately from f32 embeddings.
