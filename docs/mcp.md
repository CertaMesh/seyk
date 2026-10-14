# MCP indexing and inline search latency

A search covered by a running watcher uses the existing index once all discovered
supported resources are current. While the model loads, it uses an independent
in-memory keyword index and returns `degraded: {"mode": "lexical_only", "reason":
"model_loading"}`. New or changed resources awaiting embedding trigger the same
fallback with `reason: "not_yet_embedded"`. Missing weights use `reason: "no_model"`.
The keyword index covers text and PDF text, respects request filters and root
confinement, and creates no vectors. Preparing it still takes discovery and text
parsing time; no hard latency bound is promised. Ready-state ranking is unchanged.

CLI JSON, MCP and library results share this optional marker (protocol 7). With no active
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
