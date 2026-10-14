# MCP indexing and inline search latency

A search covered by a running watcher uses the existing index. With no active
overlapping watcher, a search may sync inline, but it still shares the
engine-wide indexing lease with watchers on disjoint roots. It can wait for an
entire remaining pass, including resource commits and deletion/pruning. The
mutex provides no fairness or maximum-wait guarantee. Cached-only searches do
not acquire this lease.

A synthetic SQLite probe (debug build, five samples, one segment per file,
batches of 16, fake inference at 2 ms per document segment) measured median
inline latency for a one-file/one-segment request: **7.2 ms with no watcher
load**, **268.1 ms with 64 watcher files/segments**, and **841.5 ms with 256
watcher files/segments**. All data lived in fresh temporary directories. Each
watcher stopped after its first pass and was joined before the next sample. The
request arrived at the watcher's first embedding batch. These measure the MCP
handler and a real Hub watcher, excluding model loading, ONNX inference and
transport; they do not establish production bounds.

Reproduce with:

```sh
cargo test --bin seyk measure_disjoint_watcher_inline_sync -- --ignored --nocapture
```

Every sample prints its load beside latency and lease wait. Searches that sync
inline can wait behind indexing on another root. A maximum wait time would
require a separately evaluated finer GC/lease design.

## Per-hit match provenance

Every MCP and CLI JSON search hit includes an additive `match` object:

```json
{"semantic_rank": 2, "keyword_rank": 1, "filename_rank": null, "matched_terms": ["cafe", "restore"]}
```

The library exposes this as `SearchHit::r#match` (`MatchReason`). Ranks are
one-based positions in the contributing candidate lists before fusion and final
truncation, so they can exceed the requested result limit. `keyword_rank` is the
BM25 leg. A `null` rank means that leg did not contribute (including a disabled
keyword weight); semantic-only searches have no keyword or filename ranks.
Semantic ranks combine all query phrasings using their best score. Filename ranks
follow the engine's filename candidate order, not relevance within the file.

`matched_terms` contains normalized query tokens from complete matching BM25
groups, deduplicated in query order across all phrasings. Case and accents follow
the index's Unicode tokenizer; standalone stopwords are excluded. An identifier
such as `INC-18372` contributes its tokens only when the entire group matches one
indexed field. Matches can come from name, path, heading or body. Terms are
confirmed against the existing keyword index, never recovered from source text,
and no new text is persisted in the cache. Without a contributing keyword leg,
the list is empty.

Existing `matched`, `matched_query`, scores and CLI/library `signals` keep their
meaning. Current JSON `schema_version` is **7**, including optional result-kind
counts and indexing progress. Match provenance does not change existing fields
or extraction requirements. Older search
JSON without `match` deserializes to empty provenance. No cache migration is
needed. Rust callers constructing `SearchHit` literals add `r#match:
Default::default()`; custom cache implementations implement
`EmbeddingCache::lexical_matched_terms`, returning one query-token list per key
in request order. The engine calls this batch method only for final returned
hits ranked by the keyword leg. SQLite checks each query group in one statement
over the selected rowids; candidate ranking does no term-membership probes.
