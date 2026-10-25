# Performance

See [Benchmarks](benchmarks.md) for measured numbers and how to reproduce them.

## Retrieval quality

![Seyk hybrid, semantic and lexical retrieval next to QMD, DigUp and LEANN measured on Seyk's harness, and sourced published model scores; higher nDCG@10 is better](assets/benchmarks.svg)

nDCG@10 measures how well the first ten results rank relevant documents, giving
more credit to relevant results near the top. Scores range from 0 to 1.
Purple bars are other local search tools (QMD, DigUp, LEANN) run on Seyk's
harness: same corpora, queries and scorer, but each tool uses its own models and
chunking. Dashed bars are published scores from different models and
whole-document setups, so they provide context rather than a controlled
comparison with Seyk's segmented-document retrieval.

Seyk 0.1.0 at `a4a0eea63f74` · EmbeddingGemma 2 · q8 · 256 dimensions · CPU ·
Apple M5 Max (18 cores, 64 GiB), macOS 26.5 · 2 inference threads.
Hybrid (default, RRF k=5) still trails semantic-only on SciFact.
See [full results, sources and reproduction steps](benchmarks.md#published-retrieval-comparison)
and the [local competitor runs](benchmarks.md#local-competitor-runs).

## Speed

Speed limits such as first-search time, start-up cost and image embedding are
listed in [Known limitations](limitations.md).

## CPU throughput controls

Defaults are one CPU session, ONNX Runtime's automatic intra-op thread count,
16 inputs per batch and q8 weights. For measured tuning, the CLI accepts:

```bash
seyk index --text-only --sessions 2 --threads 4 --batch-size 16 --embedding-window 32 ./notes
```

`--threads` is per session. `--batch-size` limits one model call;
`--embedding-window` controls how many inputs the engine sends at once and defaults
to `sessions × batch-size`. A window no larger than one batch cannot use two
sessions concurrently; use at least two full batches to keep both busy. Each
additional text session has its own weights/allocator arena (several GiB each;
measure memory use). Avoid automatic thread counts in a multi-session experiment,
because several full-machine thread teams can be slower.
Larger windows also increase the work repeated after an interruption or batch
failure; the engine still writes cache/progress updates serially.
Image and audio requests stay serial. The same controls work with search and MCP,
and nondefault controls select a separate daemon endpoint, not a new vector cache.

Library callers use `EmbeddingGemma2::builder().sessions(n).threads(m).batch_size(b)`
and `SearchEngine::builder().batch_size(window)`. These scheduling controls preserve
existing CPU cache identities; q8 and fp32 continue to have distinct identities.
The pool bounds concurrent inference, not total queued memory across concurrent
library callers. Each caller prepares its inputs before entering the pool, so
callers must limit their own concurrency and input window sizes.
Use the [paired benchmark protocol](benchmarks.md#paired-cpu-throughput-harness)
to choose settings for your own machine. The busy-machine evidence is indicative;
it does not establish a faster default.
