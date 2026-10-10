# MCP indexing and inline search latency

A search covered by a running watcher uses the existing index. With no active
overlapping watcher, a search may sync inline, but it still shares the
engine-wide indexing lease with watchers on disjoint roots. It can wait for an
entire remaining pass, including resource commits and deletion/pruning. The
mutex provides no fairness or maximum-wait guarantee. Cached-only searches do
not acquire this lease.

A synthetic SQLite probe (debug build, five samples, one segment per file,
batches of 16, fake inference at 2 ms per document segment) measured median
inline latency for a one-file/one-segment request: **5.2 ms with no watcher
load**, **240.5 ms with 64 watcher files/segments**, and **1,379.1 ms with 256
watcher files/segments**. All data lived in fresh temporary directories. The
request arrived at the watcher's first embedding batch. These measure the MCP
handler and a real Hub watcher, excluding model loading, ONNX inference and
transport; they do not establish production bounds.

Reproduce with:

```sh
cargo test --bin seyk measure_disjoint_watcher_inline_sync -- --ignored --nocapture
```

Every sample prints its
load beside latency and lease wait. Document this behavior and keep the lease
intact; bounded cross-root latency would require a separately evaluated finer
GC/lease design.
