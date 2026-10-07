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

## Phase 1 report: hybrid text search

### What was implemented

- **Keyword search.** `src/lexical.rs` holds one tokenizer and query builder
  (FTS5's `unicode61 remove_diacritics 2`, with common
  function words removed from queries), used by both caches. An identifier such as
  `Session::restore()` becomes a group of tokens that must occur in the same field, in any order. `SqliteCache` keeps a contentless FTS5 table, `segments_fts`,
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

A real folder of book drafts, eight queries, each with one known target passage. Rank of the target (lower is better):

| Query | Semantic | Hybrid (prose weight 0.3) |
|---|---:|---:|
| detached from reality | 9 | 1 |
| grief as something physical | 1 | 7 |
| toy / wet roof | 1 | 1 |
| fanatics | 1 | 2 (a duplicated passage took 1st) |
| tattoos | 1 | 1 |
| loss you can't feel | 18 | 2 |
| ink on the skin | 19 | 25 |
| dream city | 7 | 7 |

- Targets in the top 10 went from **6 of 8 to 7 of 8**.
- Two character files that semantic search did not put in the top 3 for their
  own names now rank **first**.
- Prose keyword weights of **1.0 and 0.15 were both worse overall** than 0.3.
  At 1.0, common words in long questions dominate; at 0.15, keywords barely
  break ties.

### What was deliberately not done

- **No learned or calibrated weights.** Eight queries are enough to reject bad
  settings, not to fit parameters. The constants are documented and overridable
  (`--lexical-weight`, hidden).
- **No user-facing query syntax.** Identifiers become token conjunctions automatically, but
  quotes, `AND`/`OR` and field filters (`name:`) are not supported.
- **No change to segmentation or the embedding cache key.** Retrieval quality
  numbers above rely on both staying as they were.
- **No query expansion or reranking model.** Both would add a second model.

### What we learned

- **The weight has to depend on the query.** One global keyword weight cannot
  serve both `INC-18372` and "where did I write about loss you can't feel":
  the first is all keyword, the second is all meaning.
- **Rank fusion hides strength.** A hybrid score of 0.97 can be a weak match
  that happens to rank second of three. That is why hits carry their signals
  and the CLI shows the cosine.
- **A keyword index is not content-free.** Contentless FTS5 stores word frequencies. The privacy
  follow-up below removes source word order, but the cache remains sensitive.
- **Two regressions remain** ("grief as something physical" 1 → 7, "ink on the
  skin" 19 → 25): keyword hits on a common word push a purely thematic passage
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
retains nonzero, field-weighted BM25. See [SQLite's FTS5 documentation](https://sqlite.org/fts5.html#the_detail_option)
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

The public cache API uses ResourceId, while file identities retain the existing
BLOB path keys in SQLite, including canonical Windows drive prefixes. Non-file IDs use a NUL-prefixed UTF-8 URI key, which
cannot collide with a valid local filename. Legacy SQL column names remain
internal details. The v4 migration adds an opaque `version` and nullable `descriptor_hash`
column. Legacy fingerprints are unknown, forcing preparation on the next sync. It retains resource keys, segment rowids, contentless FTS rows, input
hashes, profiles and vector bytes. A v3 cache retains its keyword index until sync and reuses byte-identical
model inputs without re-embedding. v1/v2 upgrades still perform the Phase 1 privacy invalidation,
retaining vectors across partial-scope rebuilds.

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
does not change its schema. The field
corrections here finalize unreleased v2. Bare `null` for no daemon is the sole
non-object exception.

CLI search retains queries/mode/hits/stats. A hit now contains, for example,
`"resource": "file:///notes/my%20note.md"` instead of a path string. MCP hits add
that resource field and retain `"path": "/notes/my note.md"` for opening files.
MCP paths and CLI skipped-record paths are omitted for non-file IDs. The URI is
the lossless identity;
JSON display paths replace invalid UTF-8 bytes. Skipped resources expose a URI
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
  (remote UNC hosts are rejected); its serde
  representation and parser use URIs, so legacy serialized path strings are
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
ProvidedResource and ResourceVersion are
explicitly unstable: they may change before 1.0; filesystem is the only supported
provider. They are not an authorization/ACL framework. Direct library
SearchResults serialization includes URI hits but does not add the CLI envelope.

### Verification and demo

The required Rust 1.99 fmt and all-target Clippy checks pass. The default test
suite passes with the 9 existing weight/reference-dependent tests ignored and
two Windows-only follow-ups documented below; PR #9
records the per-OS results for the final head.
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
  test constructs that identity directly. Native Windows drive-path and legacy-key tests are
  included; CI now runs fmt, Clippy and the default suite on Ubuntu, macOS and
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
Two unrelated pre-existing tests are ignored only on Windows pending Solo
#4348 (discovery fixture separators) and #4349 (read-only timestamp handle).
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
