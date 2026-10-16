# Benchmarks

> **Timing caveat.** The historical timing measurements below used an Apple M5 Max (18 cores) **while
> other, unrelated builds and benchmarks were running** (1-minute load average 23–80,
> recorded per run). Treat absolute timings as pessimistic and approximate. Ratios
> measured in the same run are more reliable. Rerun on a quiet machine before
> quoting them.

## Published retrieval comparison

Measured on 2026-10-14 with Seyk 0.1.0, clean commit
`28814a851ded4d1b3d5e0def1d112094a852e334` (the current `origin/main` when these runs started).
The chart records that measured commit; later documentation commits do not change
the search engine. Machine: Apple M5 Max · 18 cores · 64 GiB · macOS 26.5 · 2
inference threads. ONNX Runtime CPU inference, q8,
256-dimensional vectors, default segmentation and fusion, 200-segment retrieval
depth, all hybrid/semantic/lexical queries. Model:
`onnx-community/embeddinggemma-2-ONNX@daa72c51243991dfcaf9f9137d2c573d8f7790c0`.
Only provisioned weights were used; SHA-256 hashes were checked against the pinned
model files. Compatible document embedding caches were reused; every query was
rerun. This measures ranking quality, not cold indexing speed.

The full test splits contain 5,183 documents / 300 queries (SciFact),
3,633 / 323 (NFCorpus), and 3,226 nonempty pages / 180 queries (IRPAPERS text).
All 180 IRPAPERS questions remain in the denominator, including the question
whose relevant page has an empty transcription. Resource-level metrics use the
first returned segment per resource; IRPAPERS text resources are individual pages.
One-minute load samples across these runs were 25.4–35.5 (SciFact), 17.9–22.4
(NFCorpus) and 17.2–23.9 (IRPAPERS text). These runs make no timing or
quiet-machine performance claim.

| Dataset | Seyk mode | nDCG@10 | MRR@10 | Recall@10 |
|---|---|---:|---:|---:|
| SciFact | hybrid | 0.76764 | 0.73208 | 0.89189 |
| SciFact | semantic | 0.82192 | 0.80290 | 0.90122 |
| SciFact | lexical | 0.64764 | 0.61227 | 0.77867 |
| NFCorpus | hybrid | 0.35968 | 0.56807 | 0.17379 |
| NFCorpus | semantic | 0.33831 | 0.54093 | 0.16152 |
| NFCorpus | lexical | 0.30220 | 0.50526 | 0.14644 |
| IRPAPERS text | hybrid | 0.63204 | 0.56024 | 0.85556 |
| IRPAPERS text | semantic | 0.58941 | 0.51834 | 0.81111 |
| IRPAPERS text | lexical | 0.62941 | 0.56413 | 0.83333 |

Hybrid trails semantic-only on SciFact. This known fusion limitation is a
planned follow-up.
nDCG@10 rewards relevant documents near the top of the first ten results, with
linear graded relevance and a 0–1 scale (higher is better).

### Published systems and sources

These are published results from **different setups**, not systems rerun on this
machine. BEIR/MTEB retrieval embeds or indexes whole documents without Seyk's
segmentation; neural models can truncate inputs at their context limit. Prompts,
model precision, dimensions, training data, scoring implementations and index
configuration can differ. Do not interpret the bars as a controlled comparison or
as a measurement of Seyk's underlying embedding model in isolation.
Model-card nDCG@10 percentages are divided by 100. Sources retrieved 2026-10-14.

| Published system | SciFact nDCG@10 | NFCorpus nDCG@10 | Protocol |
|---|---:|---:|---|
| [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5/blob/d4aa6901d3a41ba39fb536a557fa166f842b0e09/README.md) | 0.74607 | 0.38129 | MTEB/BEIR test; whole documents, no chunking; 1024 dimensions; model-specific prompts; inputs truncated at model context limit |
| [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5/blob/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/README.md) | 0.74039 | 0.37389 | MTEB/BEIR test; whole documents, no chunking; 768 dimensions; model-specific prompts; inputs truncated at model context limit |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2/blob/f52bf8ec8c7124536f0efb74aca902b2995e5bcd/README.md) | 0.71944 | 0.35389 | MTEB/BEIR test; whole documents, no chunking; 768 dimensions; model-specific prompts; inputs truncated at model context limit |
| [BM25 (BEIR)](https://arxiv.org/html/2104.08663v4#S5.T2) | 0.665 | 0.325 | BEIR test; whole documents, no chunking; Anserini/Lucene BM25 k=0.9, b=0.4; separate title and passage fields |

No per-dataset SciFact/NFCorpus nDCG@10 scores were found for EmbeddingGemma 2 in
[Google's model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2)
or the [MTEB GitHub results](https://github.com/embeddings-benchmark/results)
and [MTEB aggregate archive](https://huggingface.co/datasets/mteb/results/tree/2246f8f15ca638d26179d396c71f4fceae4f80da)
checked on that date; the card gives aggregate
MTEB scores, which cannot substitute for dataset-level scores. The same check
found no English SciFact/NFCorpus rows for the older EmbeddingGemma-300M variant.
The chart therefore has no published EmbeddingGemma bar. IRPAPERS' original
[dataset card](https://huggingface.co/datasets/weaviate/IRPAPERS/blob/7d8ca2f6dd9efded3e27013d15782d584f93e9da/README.md)
reports recall, not nDCG@10, so it has no published comparison bar either.

### Reproduce the chart

The [comparison manifest](benchmarks/comparisons.toml) records each source title,
URL, retrieval date, protocol and numeric source score. The small
[aggregate snapshots](benchmarks/results) record measured provenance and metrics;
no dataset text, document maps, per-query rows or per-document hash lists are
committed. This run's per-query rows and uv environments were deleted after scoring.
When datasets are shared with concurrent runs, retain the raw inputs and
materialized corpora until every consumer finishes. Preserve provisioned model
files.

Run the public retrieval pipeline below for `scifact`, `nfcorpus` and `irpapers`,
with `--dimension 256 --precision q8 --threads 2 --depth 200
--modes hybrid,semantic,lexical`. Avoid running other heavy jobs on the machine
at the same time. Score the three files for each dataset together. The chart command defaults
to `~/.cache/seyk-bench/results`; it picks the newest matching aggregate per
dataset by modification time, with filename as the tie-breaker, for an explicit
full commit SHA. It ignores other dimensions, precisions and ranking/depth
settings, and rejects dirty, incomplete or inconsistent mode provenance.
Use a dedicated result directory to reproduce an exact archived selection:

```bash
python3.12 scripts/bench/chart.py \
  --results docs/benchmarks/results \
  --commit 28814a851ded4d1b3d5e0def1d112094a852e334 \
  --version 0.1.0 \
  --machine 'Apple M5 Max · 18 cores · 64 GiB · macOS 26.5 · 2 inference threads'
python3.12 -m unittest discover -s scripts/bench -p test_chart.py
```

Rendering uses Python 3.11+ and its standard library only. The SVG uses a fixed
0–1 axis, embeds light/dark styles and contains no timestamp or host-dependent
content. Given the same selected inputs and caption arguments, output bytes are
identical.

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

A quiet sweep needs a reserved window on an otherwise idle machine. Compile before the window, use
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

The tuning controls leave CPU defaults and all cache/profile fields unchanged.
They are for explicit experiments; selecting q8 versus fp32 still uses separate profiles.
The pool bounds concurrent inference only. It does not bound total queued memory
across library callers, which prepare inputs before entering the pool. Callers
must limit their own concurrency and input window sizes.

### Busy-machine CPU controls measurement

**Busy-machine, indicative only, 2026-10-09.** Apple M5 Max, 18 cores,
64 GiB, macOS 26.5; Rust 1.99.0; ORT 1.28.0 build da9b5e3; model
`onnx-community/embeddinggemma-2-ONNX@daa72c51243991dfcaf9f9137d2c573d8f7790c0`.
All correctness tests and our builds finished before these timings, but unrelated
work continued. Memory pressure stayed normal and pre-existing swap usage stayed
at 23,102 MiB (load before/after the set: 21.8/81.2). These results cannot choose
a default.

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
before all query samples, and per-sample overlap was not recorded. Cold search
includes indexing the 16 generated files; cached search reuses the same in-memory
engine.

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

## Public retrieval harness

The harness is evaluation tooling; it does not change ranking, segmentation, the
cache schema or the public API. Dataset contents, document maps, runs and per-query
diagnostics stay under `~/.cache/seyk-bench`. Do not commit or paste them into
issues or PRs. Commit only aggregate metrics and provenance. NanoSciFact is a
pipeline **smoke** check, not evidence of general retrieval quality.

The inputs are pinned in `scripts/bench/datasets.toml` by URL, revision and SHA-256.
The fetch command verifies every file, including existing downloads. Materialization
writes opaque `blake3(doc_id)[:16].txt` names to avoid ID-based filename matches;
BEIR and IRPAPERS text use `title\n\nbody`. NanoSciFact supplies only `text`, which
is preserved. Document IDs and qrels live outside the indexed corpus. Materialized
files and metadata have integrity hashes, checked before indexing and scoring.
Repeat materialization preserves unchanged file mtimes. Unexpected stale files,
duplicate IDs, missing qrels targets and incomplete runs fail explicitly, except
for the known empty-transcription IRPAPERS pages described below.

### Sources and tracks

| Dataset | Pinned split | Resources / queries | Licence / source |
|---|---|---:|---|
| NanoSciFact | train (the published 50-query subset) | 2,919 / 50 | [HF card](https://huggingface.co/datasets/zeta-alpha-ai/NanoSciFact/blob/309f1d1ae3ae2e092444a8a0c25bed59b82318bc/README.md) says CC BY 4.0; upstream abstracts are ODC-By 1.0 |
| SciFact | BEIR test | 5,183 / 300 | [Upstream](https://github.com/allenai/scifact/blob/master/LICENSE.md): claims CC BY 4.0, abstracts ODC-By 1.0 |
| NFCorpus | BEIR test, graded qrels | 3,633 / 323 | CC BY-SA 4.0 per [BEIR card](https://huggingface.co/datasets/BeIR/nfcorpus) |
| IRPAPERS text | original 180 questions | 3,230 raw pages (3,226 with text) / 180 | [MIT dataset card](https://huggingface.co/datasets/weaviate/IRPAPERS/blob/7d8ca2f6dd9efded3e27013d15782d584f93e9da/README.md); source-paper rights remain with their authors |
| IRPAPERS PDF | same questions | 166 PDFs / 180 | [Upstream PDFs and MIT repository licence](https://github.com/weaviate/IRPAPERS/tree/7fce2967a94ade0e62c4efee9376d58164cf32f8/data/PDFs); source-paper rights remain with their authors |

SciFact licence labels conflict across Hugging Face cards (CC BY-NC 2.0,
CC BY-SA 4.0, and NanoSciFact CC BY 4.0). The table records the upstream
component licences instead of treating a mirror card as permission to redistribute
the corpus. This harness stores data locally and publishes only aggregates.

IRPAPERS was inspected at the pinned sources. The document CSV uses `pdf_id_x`,
`page_number`, `pdf_title`, and `transcription`; `pdf_id_y` is an optional OCR join
field. The query CSV uses `dataset_id`, `pdf_id`, `page_number`, `question`, and
`answer`. `dataset_id` identifies a page and is not unique across questions
(179 IDs for 180 rows), so query IDs append the stable source row number. Page
numbers are one-based physical pages, as in the upstream PDF conversion notebook.
Text resources hash `paper_id#page`; PDF resources hash the numbered PDF's paper ID.
The text adapter streams the 3.78 GB CSV and discards base64 image fields. It never
decodes images. The PDF adapter reads only numbered PDFs from the pinned
source archive (318 MB of uncompressed PDFs). PDF inference disables vision and
audio and records omitted-page diagnostics; it measures embedded PDF text, not the OCR text track. The image
track is deferred.

IRPAPERS text skips pages whose transcription is empty or whitespace-only, after
validating their row structure, paper/page identity and PDF join. The declared
`documents = 3230` validates raw rows, including skipped pages; duplicate IDs,
bad joins, invalid page numbers and other malformed rows still fail loudly.
The dataset manifest's `empty_transcriptions` provenance records `raw_rows`,
`materialized_pages`, `skipped_page_keys` (opaque hashes only), `skipped_pages`
and `qrels_targeting_skipped_pages`. Skipped pages are absent from the corpus and
document map. Queries and qrels targeting them are retained, so those targets
count as misses with the original metric denominators. Score JSON, score output
and the report include both skip and affected-qrel counts.

### Run and score

Use Python 3.11–3.13 and uv. The committed `scripts/bench/bench.py.lock` pins
all Python dependencies, including the metric engine and NumPy; `--locked`
rejects a stale lockfile. Result scoring metadata records the installed
`ir_measures` and `pytrec_eval` (`pytrec-eval-terrier`) versions. Rust uses the
repository's pinned 1.99.0 toolchain.
Keep a single Cargo target directory when disk is tight. Model weights must be
provisioned inside the benchmark root too; the runner refuses the default HF
cache, symlink escapes and arbitrary corpus paths. It never downloads weights.

```bash
export PATH="$HOME/.rustup/toolchains/1.99.0-aarch64-apple-darwin/bin:$PATH" # macOS ARM
export UV_CACHE_DIR="$HOME/.cache/seyk-bench/uv"
export HF_HOME="$HOME/.cache/seyk-bench/huggingface"
# Explicit, optional model provisioning (once per precision):
cargo run --release -- model download --precision q8 --text-only

uv run --locked --python 3.12 scripts/bench/bench.py fetch nanoscifact
uv run --locked --python 3.12 scripts/bench/bench.py materialize nanoscifact
cargo run --release --example bench_retrieval -- \
  --ds nanoscifact \
  --dimension 256 --precision q8 --threads 2 \
  --modes hybrid,semantic,lexical --depth 200 --label smoke
```

The runner opens `cache/<dataset>/<profile-id>.sqlite`, indexes once and runs every
query with previews disabled. Progress reports segments needing embedding before
inference. Model weights, precision, dimension, prompts and segmentation determine
the engine's profile; a different dimension uses a separate cache. Environment
rows record the commit, dirty flag, profile, model revision, thread count, platform,
CPU count, Rust version, load averages, index timing and cache statistics before
and after indexing. Completion rows record query counts and final load. A run is
written to `.partial` until complete; a repeated label cannot overwrite a run.

```bash
# Use the exact three filenames printed by the runner; do not mix labels/profiles.
uv run --locked scripts/bench/bench.py score nanoscifact \
  ~/.cache/seyk-bench/runs/nanoscifact/*-smoke-hybrid.jsonl \
  ~/.cache/seyk-bench/runs/nanoscifact/*-smoke-semantic.jsonl \
  ~/.cache/seyk-bench/runs/nanoscifact/*-smoke-lexical.jsonl
uv run --locked scripts/bench/bench.py report ~/.cache/seyk-bench/results/nanoscifact/*-smoke.json
```

Scoring uses pinned `ir_measures` with nDCG@10 (graded, linear gain), MRR@10 and
Recall@1/5/10/20/100. Each query contributes, including empty rankings. Segments
collapse to the first occurrence of each resource, preserving engine tie order;
scoring uses strictly descending rank-derived scores instead of re-sorting raw
fusion scores. The PDF track also collapses `(resource, PageText.page)` separately:
its `page` metrics measure locator correctness and its `resource` metrics measure
paper retrieval. Neither is substituted for the other.

Depth is a **segment** budget. Two hundred segments do not guarantee 100 distinct
resources; results report the number of queries with fewer than 100 resources.
Recall@100 is recall within that returned budget, and a deeper run is a separate
configuration. Per-query local logs join every positive gold document's rank in
each mode, top-1 keys and non-relevance flags. “Top-result false-evidence rate”
means top-1 absent from positive qrels, not proof that the retrieved text is false;
unjudged hits count as non-relevant. Its denominator is queries with a result;
empty-result counts are reported separately.

### Harness checks and later full runs

```bash
uv run --locked --python 3.12 scripts/bench/bench.py test
cargo test --example bench_retrieval --locked
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked --no-fail-fast
```

The Python tests check hand-computed graded nDCG, MRR, recall, empty rankings,
ID/page mapping, empty-transcription skips, retained qrels scored as misses and
rejection paths. Rust tests check duplicate collapse and
byte-identical rankings across a reopened warm dedicated SQLite cache with a
deterministic test embedder. For a real-model determinism check, repeat the runner
with a new `--label warm` and compare each mode's original and warm file:

```bash
uv run --locked scripts/bench/bench.py determinism ORIGINAL.jsonl WARM.jsonl
```

This compares ranked keys, ranks and locators, excluding timing/environment noise.
Run paths must remain inside the benchmark root.

For full runs, use `scifact`, `nfcorpus`, `irpapers`, or
`irpapers-pdf` in the same pipeline. Run 768 dimensions on SciFact separately.
`--lexical-weight FLOAT` is an explicit diagnostic override recorded in provenance,
not a proposed default. On full SciFact scoring, lexical nDCG@10 outside [.60,.68]
is an error requiring investigation before any ranking conclusion. This band is
not applied to NanoSciFact. The published full SciFact run scores lexical
nDCG@10 0.64764, inside the band.
The pinned full IRPAPERS CSV contains four empty-transcription pages among its
3,230 raw rows. Materialization retains the 180 queries and writes 3,226 text
documents. One qrel targets a skipped page and remains a miss; the manifest and
score output record that count.

See [the current published comparison](#published-retrieval-comparison) for
full measured results, source links and setup differences.
