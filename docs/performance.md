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
at most 16 inputs per batch, and q8 weights. Text calls also have budgets of
5120 padded tokens and 4194304 attention cells. On the full SciFact and NFCorpus
benchmarks, the measured peak physical footprint was about 2.8 GB with these
budgets and about 5.7 GB with row-count-only batching. These are workload-specific
measurements, not a universal 2 GB cap. Five paired prose runs had a 5.2% higher
median time with the budgets; the timing ranges overlapped.
See [memory measurements](benchmarks.md#text-batch-memory-budgets) for conditions
and retrieval results. For measured tuning, the CLI accepts:

```bash
seyk index --text-only --sessions 2 --threads 4 --batch-size 16 --embedding-window 32 ./notes
```

`--threads` is per session. `--batch-size` limits one model call;
`--embedding-window` controls how many inputs the engine sends at once and defaults
to `sessions × batch-size`. A dispatch that fits one inference call uses one
session. Budgets can split a dispatch into several calls; larger windows also
provide work for several sessions. Each
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
