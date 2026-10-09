# Benchmarks

> **Caveat.** Every number here was measured on an Apple M5 Max (18 cores) **while
> other, unrelated builds and benchmarks were running** (1-minute load average 23–80,
> recorded per run). Treat absolute timings as pessimistic and approximate. Ratios
> measured in the same run are more reliable. Rerun on a quiet machine before
> quoting them.

## Reproduce

```bash
cargo bench --bench similarity                                   # no model needed
cargo run --release --example bench_model -- --precision q8      # needs weights
cargo run --release --example bench_model -- --precision fp32
```

`bench_model` prints a table with peak RSS and the load average next to each row.

## Similarity search (brute force, top 10)

`BruteForceIndex::search` over synthetic unit vectors, criterion medians. Load
average about 45–52.

| Segments | 256 dims | 768 dims |
|---:|---:|---:|
| 100 | 4.6 µs | 12.2 µs |
| 1,000 | 42 µs | 98 µs |
| 10,000 | 0.36 ms | 1.16 ms |
| 100,000 | 3.5 ms | 11.3 ms |

The dot product uses eight independent accumulators so it vectorises. In the same
run, that was 5.2x (256 dims) and 5.8x (768 dims) faster than a plain
`iter().sum()`. Brute force stays well below the cost of embedding a single query
even at 100,000 segments, so there is no need for an approximate index yet.

## Loading cached vectors (SQLite, 256 dims)

`SqliteCache::vectors_for`, load average about 23–29:

| Segments | Median |
|---:|---:|
| 1,000 | 1.05 ms |
| 10,000 | 20.7 ms |
| 100,000 | 170 ms |

## Model

EmbeddingGemma 2 via ONNX Runtime on the CPU, 256 dims, ~241-token segments
(including the document prompt). Load average 30–50.

| Measurement | q8 | fp32 |
|---|---:|---:|
| Model load | 0.9 s | 4.5 s |
| Text embedding, batch 16 | 5.2 segments/s | 6.1 segments/s |
| Text embedding, batch 32 | 5.9 segments/s | 6.2 segments/s |
| Query embedding | 101 ms | 39 ms |
| First search, 200 segments (uncached) | 38.7 s | 35.3 s |
| First search, 1,000 segments (uncached) | 201.7 s | 175.3 s |
| Repeat search, 1,000 segments (cached) | 100 ms | 46 ms |
| Peak RSS, batch 16 | 2.8 GiB | 3.1 GiB |
| Peak RSS, batch 32 | 4.8 GiB | 5.1 GiB |

What this shows, with the caveat above:

- **The first search of a folder is dominated by embedding.** Under this load, that
  was about 5–6 segments per second, roughly 30 minutes for 10,000 segments. Later
  searches only embed the query and changed segments, which took well under a
  second here. Run `seyk index` once on a large folder.
- **q8 was not faster than fp32 here.** Its advantages are a 3.5x smaller download
  and faster loading. Compare speed again on a quiet machine before choosing a
  default for speed reasons.
- **Memory grows by about 2 GiB from batch 16 to batch 32**, probably ONNX Runtime's
  allocator arena (not verified). The engine's default batch size is 16.
- **Images** take several seconds each on the CPU. Every image uses the full
  2,520-patch budget, whatever its size.

## Retrieval smoke test

`examples/eval_corpus.rs` on a 13-segment hand-written corpus. Each target passage
avoids the query's words, and keyword-bait distractors were planted ("memory card",
"Father Christmas", "detached house"). This is a smoke test, not a quality benchmark.

| Query | Target rank (q8/256) | Keyword baseline |
|---|---:|---|
| becoming like his father | 1 | distractors only |
| memory becoming unreliable | 9 | distractor first |
| grief as something physical | 1 | nothing found |
| repeating parents' behaviour | 1 | distractors only |
| feeling detached from reality | 2 (after "detached house") | distractor first |

q8 and fp32 ranked identically, and 768 dims did not fix the misses. The misses are
cases where the passage *shows* the theme (a disputed kitchen colour, a story that
shifts) without naming it, while a distractor shares a literal word. Expect that
kind of miss: semantic search surfaces material keyword search cannot, but it
still favours literal matches.


## Paired CPU throughput harness

`bench_cpu_pool` compares **this run's** baseline and candidate with alternating
A/B then B/A pairs (A/B/B/A across two pairs). Every cell is a fresh process, loads
only its own text sessions, warms the requested input shapes and measures that
configuration. Peak RSS therefore does not inherit a larger configuration's arena.
Weights must already be local; the harness never downloads them.

```bash
cargo build --release --example bench_cpu_pool --locked
# Config = precision:sessions:intra-threads|auto:microbatch:engine-window
./target/release/examples/bench_cpu_pool \
  --baseline q8:1:8:16:32 --candidate q8:2:4:16:32 \
  --pairs 5 --segments 64 > pool.jsonl
# Add --mixed for heterogeneous lengths, --corpus 32 for cold/cached engine search
# and query contention, or --model-dir DIR for provisioned local weights.
```

JSONL contains environment metadata, all cell measurements and the paired median,
minimum and maximum candidate/baseline throughput ratios. Each timing carries the
1/5/15-minute load before and after and process high-water RSS (`null` when unavailable).
Compute segments/s as `segments / warmed.seconds` and real/padded tokens/s from the
corresponding counts. Query samples support median/tail summaries. `stage_seconds`
contains prompt construction, tokenization (including ID conversion), padding and
tensor construction, ORT execution, and extraction/normalization. Worker stage
elapsed times overlap in parallel runs; their sum is **not** end-to-end wall time.
The concurrent-document interval includes joining the query/document workload.
`cold_index` times a first search including indexing and query embedding;
`cached_search` repeats it with the same in-memory cache.

A quiet sweep must be coordinated with the user. Compile before the window, use
stable AC/power settings, record hardware/core topology, OS, Rust and ONNX Runtime
versions, model revision, memory pressure and swap before/after, then wait for idle
CPU and a stable low load (for example 1-minute load below 2 for two minutes on the
18-core machine). Do not terminate unrelated work. Load is context, not a divisor
for correcting timings. Baseline drift or competing work invalidates a quiet claim.

Sweep one session first, then small pools at equal thread budgets: 1×18, 2×9, 3×6;
also 1×8, 2×4, 4×2 and auto for a single session. Sweep batch 1/4/8/16/32 separately
for q8 and fp32, with normal, mixed and long inputs. Compare equal per-session batch
and equal aggregate in-flight counts. Increase to 64 only with memory headroom.
Record startup, first/warmed embedding, query p50/p95, cold indexing and cached
search separately. Stop growing configurations on swap/memory pressure. Do not
compare quiet results with the old busy baseline to claim an optimization ratio.

CPU defaults and all cache/profile fields remain unchanged. Optional tuning is
for explicit experiments; selecting q8 versus fp32 still uses separate profiles.
The pool bounds concurrent inference only. It does not bound total queued memory
across library callers, which prepare inputs before entering the pool. Callers
must limit their own concurrency and input window sizes.

### Busy-machine CPU controls measurement

**Busy-machine, indicative only — 2026-10-09.** Apple M5 Max, 18 cores,
64 GiB, macOS 26.5; Rust 1.99.0; ORT 1.28.0 build da9b5e3; model
`onnx-community/embeddinggemma-2-ONNX@daa72c51243991dfcaf9f9137d2c573d8f7790c0`. All correctness tests and our builds finished before
these timings, but unrelated work continued. Memory pressure stayed normal and
pre-existing swap usage stayed at 23,102 MiB (load before/after the set: 21.8/81.2). These results cannot choose a default.

Each row is five alternating A/B or B/A pairs. Configurations are
`precision:sessions:threads:microbatch:window`. Throughput and RSS are medians of
fresh-process cells; the ratio is the median of **within-pair ratios**, not the
ratio of aggregate medians. Do not compare absolute rates between rows: they ran
at different times. The load columns show the range of 1-minute load samples
around timed embedding; JSONL emitted by the harness records all 1/5/15 values.

| Comparison (baseline → candidate) | Paired median ratio [range] | Baseline / candidate segments/s | Baseline / candidate peak RSS MiB | 1-min load baseline / candidate |
|---|---:|---:|---:|---|
| pool: `q8:1:8:16:32` → `q8:2:4:16:32` | 0.919× [0.500–1.700] | 3.27 / 2.36 | 2397 / 4417 | 21.9–72.5 / 54.5–79.6 |
| batch: `q8:1:8:16:32` → `q8:1:8:32:32` | 1.125× [0.387–1.621] | 3.74 / 2.92 | 2397 / 4280 | 57.6–96.6 / 56.6–111.2 |
| precision: `q8:1:8:16:32` → `fp32:1:8:16:32` | 1.074× [0.767–1.854] | 2.63 / 3.75 | 2398 / 2691 | 41.2–81.6 / 51.4–72.5 |
| window: `q8:1:8:4:4` → `q8:1:8:4:16` | 2.296× [1.316–3.915] | 1.11 / 2.49 | 2514 / 1466 | 48.6–68.5 / 42.2–72.0 |

The batch row's paired ratio points up while its aggregate median throughput
points down; the precision row's aggregate medians point up. These directional
disagreements under changing load are why neither summary should select a
default. The precision row's paired and aggregate summaries agree in direction,
but their magnitudes also differ substantially.

The first three rows use `--pairs 5 --segments 32` (about 278 tokens/input).
The window row uses `--pairs 5 --segments 16 --mixed` and microbatch 4 to bound
memory for longer inputs. Pass each row's configurations to `--baseline` and
`--candidate` in the command above to reproduce it.

Pooling and batch/precision changes had wide, conflicting pair ratios. More
sessions consumed much more memory; no default changed. The mixed-window result
was positive in all five pairs and reduced padded tokens from 12,888 to 4,888 for
the same 4,878 real tokens (baseline load 48.6–68.5; candidate 42.2–72.0). This uses
the existing stable length sort over the explicitly selected dispatch window;
there is no new bucketing policy or change to tokenization/pooling.

**Not adopted:** buffer reuse, output-copy removal, I/O binding, tokenizer
overlap, a different bucketing policy, or a spinning-policy change. The pool
comparison's stage medians per 32 inputs were:

| Stage | One session | Two sessions | 1-min load, one / two |
|---|---:|---:|---|
| Tokenization + ID conversion | 10.959 ms | 13.946 ms | 21.9–72.5 / 54.5–79.6 |
| Padding + tensor construction | 0.044 ms | 0.049 ms | 21.9–72.5 / 54.5–79.6 |
| Extraction + normalization | 0.039 ms | 0.038 ms | 21.9–72.5 / 54.5–79.6 |

Those are observed stage budgets, **not** optimized-variant speedup measurements.
No paired gain has been established for the unimplemented candidates, so they
remain unadopted. The owned input tensors already wrap contiguous buffers.
Worker stage times overlap with multiple sessions and must not be summed as
end-to-end wall time. The explicit window control is the only padding-related
lever retained; the existing sort itself is unchanged.

### Query contention diagnostic

A separate same-run pair used `--pairs 1 --segments 32 --corpus 16 --queries 20`.
This is a diagnostic sample, not a latency guarantee. Query samples follow one
warm query; query shape warmup is not pinned to every pool session. These are
**queries during a workload started concurrently**: document embedding may finish
before all query samples, and per-sample overlap was not recorded. Cold search includes indexing the
16 generated files; cached search reuses the same in-memory engine.

| Configuration | Idle query p50/p95 | 1-min load | Queries during a workload started concurrently, p50/p95 | 1-min load |
|---|---:|---|---:|---|
| q8:1:8:16:32 | 257 / 700 ms | 68.9–71.1 | 599 / 10,412 ms | 71.1–84.9 |
| q8:2:4:16:32 | 437 / 1,203 ms | 81.1–81.8 | 630 / 1,542 ms | 80.5–83.7 |

| Configuration | Cold first search | Load before → after (1/5/15) | Cached search | Load before/after (1/5/15) |
|---|---:|---|---:|---|
| q8:1:8:16:32 | 10.464 s | 84.9/66.8/59.1 → 80.5/66.5/59.0 | 0.415 s | 80.5/66.5/59.0 |
| q8:2:4:16:32 | 7.159 s | 80.8/69.8/60.9 → 81.2/70.1/61.0 | 0.245 s | 81.2/70.1/61.0 |

Queries can still queue behind document batches. Do not infer a pool throughput
or tail-latency guarantee from this single, heavily contended pair.
