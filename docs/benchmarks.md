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
