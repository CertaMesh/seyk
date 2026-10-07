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
