# Benchmarks

> **Timing caveat.** The historical timing measurements below used an Apple M5 Max (18 cores) **while
> other, unrelated builds and benchmarks were running** (1-minute load average 23–80,
> recorded per run). Treat absolute timings as pessimistic and approximate. Ratios
> measured in the same run are more reliable. Rerun on a quiet machine before
> quoting them.

## Published retrieval comparison

Measured on 2026-10-15–16 with Seyk 0.1.0, clean commit
`a4a0eea63f747b0acf4e3a5fa844f79fc5bef39c`, with default RRF k=5.
The chart records the measured source commit, before later main integration.
Machine: Apple M5 Max · 18 cores · 64 GiB · macOS 26.5 · 2 inference threads.
ONNX Runtime CPU inference, q8, 256-dimensional vectors, default segmentation,
adaptive lexical weighting, filename weight 3, candidate depth 800 and
200-segment output depth; all hybrid/semantic/lexical queries. Model:
`onnx-community/embeddinggemma-2-ONNX@daa72c51243991dfcaf9f9137d2c573d8f7790c0`.
Only provisioned weights were used; SHA-256 hashes were checked against the pinned
model files. Main populated the `;text-literals-v1` embedding profile across
bounded runs; the k=5 branch reused those embeddings and embedded zero new
segments. Every query was rerun. This measures ranking quality, not indexing speed.

The full test splits contain 5,183 documents / 300 queries (SciFact),
3,633 / 323 (NFCorpus), and 3,226 nonempty pages / 180 queries (IRPAPERS text).
All 180 IRPAPERS questions remain in the denominator, including the question
whose relevant page has an empty transcription. Resource-level metrics use the
first returned segment per resource; IRPAPERS text resources are individual pages.
Machine load varied and the baseline required cold indexing while the branch
used warm caches. These runs make no timing or quiet-machine performance claim.

| Dataset | Seyk mode | nDCG@10 | MRR@10 | Recall@10 |
|---|---|---:|---:|---:|
| SciFact | hybrid | 0.80700 | 0.77649 | 0.92056 |
| SciFact | semantic | 0.82192 | 0.80290 | 0.90122 |
| SciFact | lexical | 0.64764 | 0.61227 | 0.77867 |
| NFCorpus | hybrid | 0.35972 | 0.55989 | 0.17545 |
| NFCorpus | semantic | 0.33831 | 0.54093 | 0.16152 |
| NFCorpus | lexical | 0.30220 | 0.50526 | 0.14644 |
| IRPAPERS text | hybrid | 0.64951 | 0.57637 | 0.87778 |
| IRPAPERS text | semantic | 0.58941 | 0.51834 | 0.81111 |
| IRPAPERS text | lexical | 0.62941 | 0.56413 | 0.83333 |

The same-profile baseline was clean main commit
`4b3ab33b54affa38f2bc3cc43781e4624fe9e4fb` with RRF k=60. Separate release
builds supplied the main and branch binaries. Changing only k to 5 improved
hybrid nDCG@10 in all three datasets:

| Dataset | k=60 | k=5 | Delta |
|---|---:|---:|---:|
| SciFact | 0.767638 | 0.807000 | +0.039362 |
| NFCorpus | 0.359682 | 0.359716 | +0.000034 |
| IRPAPERS text | 0.632042 | 0.649511 | +0.017469 |

Semantic and lexical rankings were unchanged across all 803 queries. The NFCorpus
increase is small; the result meets the 0.005 maximum-drop gate for both NFCorpus
and SciFact. Hybrid still trails semantic-only on SciFact. These datasets also
informed the choice of k, so the comparison is not an unseen-dataset evaluation.
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
committed. Raw captures stay outside the repository; shared scoring environments
are retained for concurrent work.
When datasets are shared with concurrent runs, retain the raw inputs and
materialized corpora until every consumer finishes. Preserve provisioned model
files.

Run the public retrieval pipeline below for `scifact`, `nfcorpus` and `irpapers`,
with `--dimension 256 --precision q8 --threads 2 --depth 200
--modes hybrid,semantic,lexical`. Avoid running other heavy jobs on the machine
at the same time. Score the three files for each dataset together. The chart command defaults
to `~/.cache/seyk-bench/results` and selects one aggregate per dataset for an
explicit full commit SHA. Selection never uses file timestamps: several matching
files are accepted only when their charted scores agree (the first by relative
path is used); disagreeing duplicates are an error. It skips other dimensions,
precisions and ranking/depth settings and older-schema files that lack those
fields, and rejects dirty, incomplete or inconsistent mode provenance.
The chart also draws the headline rows of the
[competitor aggregate](benchmarks/head-to-head.toml) as a separate group labelled
"measured on Seyk's harness"; diagnostic rows such as QMD's native AND lexical run
appear as footnotes, not bars. Published scores use dashed bars.
Use a dedicated result directory to reproduce an exact archived selection:

```bash
python3.12 scripts/bench/chart.py \
  --results docs/benchmarks/results \
  --commit a4a0eea63f747b0acf4e3a5fa844f79fc5bef39c \
  --version 0.1.0 \
  --machine 'Apple M5 Max · 18 cores · 64 GiB · macOS 26.5 · 2 inference threads'
python3.12 -m unittest discover -s scripts/bench -p test_chart.py
```

Rendering uses Python 3.11+ and its standard library only. The SVG uses a fixed
0–1 axis, embeds light/dark styles and contains no timestamp or host-dependent
content. It is 480 units wide so labels stay legible on a phone-width README.
Given the same selected inputs and caption arguments, output bytes are
identical; the test suite regenerates the committed chart and compares bytes.

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

The historical local prose smoke test overlapped private evaluation material.
Its queries and passages are omitted. These results are not a reproducible
public benchmark. Use the public retrieval harness for quality comparisons.

`tests/retrieval.rs` now checks independent, invented examples about astronomy,
cooking, caching, gardening, and pottery. Its model-backed tests cover semantic
retrieval, section locators, and every embedding dimension. They are regression
checks, not a retrieval benchmark.

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

Verified PDF runs keep all 166 resources in the manifest and all 180 original
questions and qrels in both metric denominators. Only encoded files strictly
larger than the unchanged 33,554,432-byte discovery default count as policy
exclusions. The runner observes length and BLAKE3 together and proves the exact
eligible set through public discovery. The scorer checks that evidence against
the manifest and fixed `pdf-policy-miss-v2` contract, without remeasuring length.
Excluded targets and empty results remain misses. Excluded PDFs have unknown
physical-page counts. Paper-resource and physical-page scores stay separate;
page retrieval does not establish extraction byte-span validity.

PDF runs require Unix task locking, `--pdf-cache-session <opaque-id>`, and all three modes. A new
session exclusively creates `cache/irpapers-pdf/tasks/<opaque-id>/cache.sqlite`
with a private ownership receipt and a held process lock. Explicit
`--resume-pdf-cache` accepts only that session's matching manifest, profile, and
immutable v1 ownership protocol. This receipt is not a coverage certificate;
matching original receipts remain unchanged. Only new v2 evaluation envelopes
are accepted. Each phase starts a fresh engine, removes only its eligible
resource bindings through the public cache API, and prepares every eligible PDF
before querying all modes in that engine. Committed input-keyed vectors can be
reused, while page diagnostics are freshly recovered. Eligible positive physical-page outcomes are classified as the exact configured
no-usable-embedded-text reason, `pdf_parse_failed`, or `pdf_parser_panicked`.
Parser-failed pages are explicit misses with every original query and qrel
retained. Reports list safe opaque keys/physical pages, per-reason failure totals,
and affected positive qrels and distinct questions. Healthy pages in the same
paper remain eligible: resource success cannot satisfy a failed-page qrel.
Hits claiming any omitted/failed page, duplicate/conflicting outcomes, unknown
reasons, whole-resource skips (including all-pages-failed PDFs), incomplete
preparation and replay mismatches abort. Dependency error/panic payloads are
replaced by fixed parser codes in evaluation provenance. Interrupted phases cannot certify results. Resume preserves committed
SQLite work, without promising retention of uncommitted work or inputs pruned
after a failed full pass. Shared and historical caches are never adopted or reset.
Corpus hashes, identities, and metadata are rechecked after indexing and before
each mode completes. These checks detect persistent boundary changes and normal
request-observed changes, without claiming an atomic filesystem snapshot.

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

## Photo Anchor quality gate (P2, 2026-10-16)

This offline experiment compares the P1 Photo Anchor codec with exact f32 image
vectors. It does not change the engine, cache, schema, ranking, or f32 default.
The q4 scan optimization remains separate work.

### Reproduce

The committed `scripts/fixtures/photo-manifest.json` pins 280 public CC0 photos
from Wikimedia Commons by URL and SHA-256, with each item's source page and
licence URL. Photos, prepared vectors and results live in
`~/.cache/seyk-bench/photo-fixture/`, outside worktrees and protected from cleanup.
Set `SEYK_PHOTO_FIXTURE` to override this directory for both the fetcher and evaluator.
The Rust evaluator, latency example and P2 test use `HOME`, then `USERPROFILE`,
for the home directory, including `~/` overrides. A direct override needs neither.
The collection began with 300 candidates across 30 subjects; visual review removed two artwork
reproductions and 18 repeated views before embedding. Photographs of objects,
including museum artifacts, remain as distractors. No photos are distributed.
The 60 hand-written queries in `scripts/fixtures/photo-queries.json` contain
explicit image-ID relevance labels: 2 = directly relevant, 1 = incidental or
partial match, absent = not relevant. Labels were set before scoring, from
contact-sheet inspection; source search terms are not used as relevance labels.
These are single-reviewer judgments, with residual near-duplicates and subject
imbalance, not an independent benchmark or a representative personal library.

```sh
python3 scripts/fetch_photo_fixture.py
python3 scripts/fetch_photo_fixture.py --verify
# SEYK_MODEL must name a directory of already-provisioned q8 text + vision files.
export SEYK_MODEL=/path/to/provisioned/model
export CARGO_NET_OFFLINE=true
export ORT_LIB_PATH=/path/to/provisioned/onnxruntime-library-directory
TC=$(rustup run 1.99.0 rustc --print sysroot)
export PATH="$TC/bin:$PATH" RUSTC="$TC/bin/rustc" RUSTDOC="$TC/bin/rustdoc"
rustup run 1.99.0 cargo run --release --locked --example image_quant_eval
/usr/bin/time -l target/release/examples/image_quant_eval --score-only
```

The fetch script checks all manifest entries before fetching and verifies every
image, including existing files. A checksum mismatch fails without replacing the
file. `--verify` never fetches. The evaluator also runs this verification before
reading images. Vector cache keys include the model identity, modality identity,
task and input bytes. Each completed 768-dimensional embedding is cached
atomically; 256-dimensional vectors are derived by truncating and normalizing.
An interrupted inference run resumes from completed vectors. The 35-minute guard
is cooperative: it stops between photos once the budget is reached, without
interrupting an in-flight embedding. The text-query loop is not time-checked.
Local runs additionally need the shared gate lock and a separate external
40-minute timeout for the **whole phase**, including compilation.

`--score-only` checks the current fixture/queries against the prepared cache and
runs without a model. Output `~/.cache/seyk-bench/photo-fixture/results.json` includes per-query
rankings, all per-dimension clipping rates, model/input identity, and the fixture
fingerprint. Retain this protected fixture and vector cache for later phases; worktree cleanup
must not remove them.

### P3 correctness and timing commands

`image_quant_eval --prepare-only` prepares resumable f32 image/query vectors without scoring.
`image_quant_eval --score-only` reports untimed diagnostics, clipping counts/rates and quantization error versus original f32.
Independent decode-score references and actual engine acceptance are checked separately:

```sh
cargo test -p seyk-embed --locked quant::tests
cargo test -p seyk-core --test image_scan --locked -- --ignored
```

The engine gate checks all 60 queries at d256/d768, semantic and hybrid, and
k=1/5/10/20/50 plus zero and k>N. It writes paired rankings, independent numerical
errors, nDCG deltas and matched SQLite snapshot sizes under the protected fixture.
A parity failure is a stop, with the failing queries reported. R is not tuned.

Before the reserved slot, build `similarity` with `--no-run` and the
`photo_anchor_latency` release example.
The single Criterion `photo_anchor_scan` run keeps seed 4517, N=1000,
30 samples, 500 ms warmup and 2 s measurement. Compare medians independently at
each dimension against `q4 <= f32 + 2 ms per 1000 vectors`.
The guarded latency example requires `SEYK_Q4_SLOT_GO=4517-p3` and a dimension
argument. It uses fixture query 0, k=10, five repetitions per mode/scan and rotated
exact/q4 order. Its end-to-end requests include real text-query inference, SQLite
reads, scanning, reranking and fusion. Model loading, image preparation and warmup
are excluded. The cached query runs use semantic R=50 and hybrid R=200.
Clipping rates are reported per coordinate without a pass/fail threshold.

NanoSciFact `bench_retrieval --rank-only --cache-session <opaque-id>` omits latency
measurements and uses a task-owned cache under `cache/nanoscifact/tasks/<opaque-id>`.
Compare the same fresh queries and ordered rankings on base and candidate, using
the same warm f32 embeddings. Shared caches stay unchanged.

### P3 fixture refresh (2026-10-24)

Wikimedia re-rendered one pinned thumbnail after P2: `apple-fruit-06`.
The explicit re-pin mode accepted only HTTP 200 `image/jpeg` bytes from the same URL,
with a successful JPEG decode and the URL's 960-pixel width. The source page and
CC0 licence stayed unchanged; the manifest records the old hash and re-pin date.
All 280 current files pass offline SHA-256 verification.

The historical P2 numbers below used the previous bytes for that thumbnail.
P3 uses a fresh full-f32 baseline and q4 comparison on the current fixture;
the historical P2 rankings are not its acceptance evidence. The current manifest
SHA-256 is `2fdd0ba135334c13d0bc028189fff34348c766692b589c3377e6cddee278035c`.
`python3 scripts/fetch_photo_fixture.py --repin` requires explicit operator
approval to update pins. Normal fetch and `--verify` fail on a mismatch.

### Historical P2 definitions and measurement boundaries

nDCG@10 uses gain `2^grade - 1` and log2 rank discount, averaged across queries.
Recall@1 is the number of relevant images in the first result divided by all
relevant images for that query (not hit rate). Each quality share is the ratio
of macro averages to f32. Top-10 overlap ignores order; the pass bar's identical
top-10 fraction requires identical order and uses manifest order to break ties.
Re-ranking sorts the q4 top 20 or 50 by exact f32 scores, excluding the approximate
tail. Error is absolute score error over all query/image pairs before re-ranking.
Clipping counts values outside the calibration interval before rounding; the
3% check is applied to **every dimension**, with mean/max also reported.

Image-vector memory means payload bytes: `N*d*4` for f32 and `N*d/2` for flat q4.
Query LUTs and optional exact re-ranking scratch are reported separately. The
quality harness retains f32 ground truth, so its RSS cannot establish a deployed
q4 memory reduction. Scoring-only RSS excludes model loading/inference but still
includes ground truth, both query sets, and reporting allocations. Cached-query
retrieval timing includes LUT creation, scanning, sorting and exact shortlist
scoring, averaged over 20 repetitions per query, with variant order rotated between
queries. It excludes query embedding,
SQLite I/O and RRF; it is not end-to-end engine search latency.

The P1 Criterion scan medians are reused without rerunning or claiming a speedup:
256: f32 33.631 µs / q4 101.190 µs per 1,000 vectors;
768: f32 115.110 µs / q4 383.410 µs. They are medians of five busy-machine runs
(30 samples, 500 ms warm-up, 2 s measurement; Apple M5 Max, 18 logical CPUs).
The original ranges were 31.741–41.676 / 99.470–114.510 µs at 256 and
98.970–127.690 / 344.050–462.150 µs at 768. Setup, encoding and LUT creation were
excluded from those scan timings. P2 cached-query timings have a different scope.

SQLite page counts, production end-to-end search latency, text-ranking parity
and NanoSciFact before/after belong to the P3 search-path experiment. P2 neither
creates a SQLite image-code table nor changes search behavior, so it cannot
supply those measurements. This small fixture cannot predict 10k–100k-image
quality, particularly for the default dimension of 256.

### Historical P2 measured results and decision

**NO-GO for P3 adoption at both 256 and 768 dimensions.** The table below applies the original pass bar to the vendored Pastel tables only. No local calibration was fitted.

Run: 280 photos, 60 queries, 317 graded labels; fixture/query BLAKE3 `9cb18b087d0b30e06792472c86e8a7ef07a8b19e0811473ef864f9daa824766d`. q8 ONNX model revision `daa72c51243991dfcaf9f9137d2c573d8f7790c0`, four CPU threads; Rust 1.99.0 release, ONNX Runtime 1.28, macOS 26.5 arm64, Apple M5 Max (18 logical CPUs). All four provisioned model graph/data SHA-256s matched the revision.

| Dim | Variant | nDCG@10 | Share of f32 | Recall@1 | Share of f32 | Top-10 overlap | Identical ordered top-10 |
|---|---|---:|---:|---:|---:|---:|---:|
| 256 | f32 | 0.956234 | 1.000000 | 0.377415 | 1.000000 | 100.0000% | 60/60 (100.00%) |
| 256 | q4 | 0.953261 | 0.996891 | 0.388526 | 1.029440 | 95.5000% | 4/60 (6.67%) |
| 256 | q4_rerank20 | 0.956234 | 1.000000 | 0.377415 | 1.000000 | 100.0000% | 60/60 (100.00%) |
| 256 | q4_rerank50 | 0.956234 | 1.000000 | 0.377415 | 1.000000 | 100.0000% | 60/60 (100.00%) |
| 768 | f32 | 0.959800 | 1.000000 | 0.390749 | 1.000000 | 100.0000% | 60/60 (100.00%) |
| 768 | q4 | 0.960649 | 1.000884 | 0.394082 | 1.008531 | 96.5000% | 4/60 (6.67%) |
| 768 | q4_rerank20 | 0.959800 | 1.000000 | 0.390749 | 1.000000 | 100.0000% | 60/60 (100.00%) |
| 768 | q4_rerank50 | 0.959800 | 1.000000 | 0.390749 | 1.000000 | 100.0000% | 60/60 (100.00%) |

| Dim | Absolute score error mean / max | Clip mean / max | Dimensions above 3% | Fixture mean–anchor cosine |
|---|---:|---:|---:|---:|
| 256 | 0.003132 / 0.026925 | 1.0240% / 8.5714% | 14 | 0.973864 |
| 768 | 0.002106 / 0.021545 | 1.0649% / 7.8571% | 35 | 0.972840 |

The quality bar requires rerank-50 nDCG share ≥ 0.999 and identical ordered top-10 for ≥ 95% of queries, plus stored-only nDCG share ≥ 0.98. The other bars require every dimension’s clipping ≤ 3%, image payload ≤ 1/7 of f32, and scan latency ≤ f32.

| Dim | Quality bar | Every-dimension clipping bar | Image payload bar | P1 scan bar | Decision |
|---|---|---|---|---|---|
| 256 | PASS | FAIL | PASS | FAIL | NO-GO |
| 768 | PASS | FAIL | PASS | FAIL | NO-GO |

| Dim | f32 image payload | q4 image payload / capacity | Query LUT payload | Top-50 exact scratch payload |
|---|---:|---:|---:|---:|
| 256 | 286,720 B | 35,840 B | 131,076 B | 51,200 B |
| 768 | 860,160 B | 107,520 B | 393,220 B | 153,600 B |

The image payload is exactly 1/8 of f32 (87.5% fewer bytes), passing the 1/7 bar. This excludes LUTs and exact re-ranking buffers; with only 280 images these fixed costs matter. It does not imply an eightfold reduction in process RSS.

| Dim | Variant | Cached-query retrieval, median µs | Range across five runs, µs |
|---|---|---:|---:|
| 256 | f32 | 13.953 | 13.416–15.255 |
| 256 | q4 | 124.109 | 118.141–139.886 |
| 256 | q4_rerank20 | 126.529 | 118.652–160.797 |
| 256 | q4_rerank50 | 130.887 | 120.870–144.556 |
| 768 | f32 | 31.842 | 30.945–55.515 |
| 768 | q4 | 304.278 | 280.271–389.500 |
| 768 | q4_rerank20 | 302.499 | 281.738–348.875 |
| 768 | q4_rerank50 | 298.263 | 287.893–364.082 |

Five fresh scoring-only processes, 20 repetitions per query/variant/dimension per process. Peak RSS (`/usr/bin/time -l`, bytes on macOS): median 31,326,208 B, range 30,834,688–32,423,936 B. These runs retain f32 reference vectors and measure the whole harness, not isolated q4 search.

The machine was shared with unrelated builds/benchmarks (load averages 16.54/20.91/23.16 at gate acquisition; 18 logical CPUs). Timing ranges are descriptive; no concurrency or compiler tuning was attempted. A separate one-second inference profile located work inside ONNX Runtime attention/matrix operations; it was taken before scoring timings. The measured scoring path scans all 280 images and sorts candidates; no engine I/O is involved.

All 480 query/variant/dimension metric rows and aggregate shares were independently recomputed from committed labels and result rankings. All five scoring runs produced identical rankings. Each scoring run rechecked every photo hash. The fetcher was also checked for valid cached input, corrupt bytes, missing files and unapproved licences, including zero network access in verification mode.

The P1 scan bar and the per-dimension clipping bar fail at both dimensions. Quality passes on this fixture: re-ranking 20 or 50 reproduces every baseline top-10 list. Keep f32 as the default; this experiment does not authorize P3 integration. Scan speed needs further work. Any later adoption decision needs representative library sizes and the deferred engine measurements; this small, single-reviewer fixture provides no large-library guarantee.

### Local competitor runs

`bench.py compete` adapts pinned QMD, DigUp and LEANN installs to the same
materialized `.txt` corpus, query IDs, first-document-occurrence collapse and
`ir_measures` scoring described above. It never provisions software or models.
The aggregate/provenance file is `docs/benchmarks/head-to-head.toml`; only completed,
scored runs belong there as **measured on Seyk's harness**. Published vendor or
model scores are a separate comparison.

#### Measured results

Measured on 2026-10-14–16 with clean adapter commit
`b2c13ec5eb5d90befc65b3bffc3a9ea0b3ea73c7`. The table compares complete tool
configurations, including their different models and native chunking. All use
the same materialized corpora, query sets and scoring. Each configuration has
one evaluation pass; QMD's stochastic expansion can vary on another pass.

A later cleanup-only fix terminates surviving process-group children even when
their leader has exited. The 18 measurements retain the commit above; this fix
does not change retrieval, ranking, scoring or their recorded provenance.

| Tool / mode | SciFact nDCG@10 | NFCorpus nDCG@10 | IRPAPERS text nDCG@10 |
|---|---:|---:|---:|
| QMD vector | 0.718541 | 0.376507 | 0.447891 |
| QMD hybrid | 0.679850 | 0.368419 | 0.445633 |
| QMD full (reranker) | 0.695797 | 0.380972 | 0.465203 |
| DigUp hybrid | 0.800734 | 0.370653 | 0.682180 |
| LEANN Contriever | 0.651517 | 0.318308 | 0.570307 |

QMD's native lexical run is reported separately as a diagnostic footnote.[^qmd-native-and]

The [aggregate manifest](benchmarks/head-to-head.toml) records all 18 completed
configurations, query/result-depth counts, model and install hashes, dataset
pins, hardware paths and timings. Timings are operational context from single
passes and bounded restarts, not replicated speed comparisons. Comparative speed
is inconclusive. No dataset text, query text, ranked rows or document maps are
included.

The adapter currently requires macOS. Every indexer and query worker, including
children, runs under `sandbox-exec` with `scripts/bench/offline.sb`. External
network access is denied; localhost is allowed for LEANN's local ZMQ embedding
process. `offline-check` proves that numeric external TCP fails with EPERM/EACCES
and localhost succeeds. Cloud API keys, proxy settings and service endpoints are
not inherited. HF offline flags are additional guards, not the isolation proof.

Installs live in `~/.cache/seyk-bench/competitors`: QMD 2.8.3 in `qmd-install`
(`npm install --prefix qmd-install @tobilu/qmd@2.8.3`); LEANN 0.3.8 in `leann-env`
(`uv venv --python 3.12 leann-env`, then
`uv pip install --python leann-env/bin/python leann==0.3.8`); and the DigUp source
at commit `f509c329e0032fbee4d945e1edbdbca7d94af50f` in `digup`.
DigUp uses its `scripts/build-llama.sh` (build parallelism limited to two) followed
by `swift build -c release --product digup -j 2`. Its llama.cpp pin is
`4d756bc72bf00a4aacf410ae15a2d315f3db400d` plus the upstream no-logits patch.
All three tools' code is MIT; model terms are recorded separately.

Provision the files listed in the aggregate file under `qmd-models`, `digup-models`
and `leann-models`. The local `competitors/model-provenance.json` records each
`tool`, `repository`, `revision`, `file`, `bytes`, `sha256`, and `license`.
Keep QMD's npm lockfile and `uv pip freeze` output in
`competitors/leann-requirements.txt`. Before each phase, the adapter checks the
installed version and every model's size/hash, and fingerprints the install lock
or DigUp binary. Results also record platform, CPU, load, hardware path, indexing
duration and per-query wall time. Model provisioning is explicitly separate from
running this adapter.

```bash
export UV_CACHE_DIR="$HOME/.cache/seyk-bench/competitors/harness-uv"
uv run --locked --python 3.12 scripts/bench/bench.py offline-check
uv run --locked --python 3.12 scripts/bench/bench.py fetch scifact
uv run --locked --python 3.12 scripts/bench/bench.py materialize scifact
# Hold seyk-local-gate for each bounded index/query phase.
uv run --locked --python 3.12 scripts/bench/bench.py compete qmd scifact \
  --mode vector --phase index --label comparison
uv run --locked --python 3.12 scripts/bench/bench.py compete qmd scifact \
  --mode vector --phase query --label comparison
# Repeat the identical query command until it reports the completed .jsonl.
# Then score the exact output filename:
uv run --locked --python 3.12 scripts/bench/bench.py score scifact \
  ~/.cache/seyk-bench/runs/scifact/qmd-vector-comparison.jsonl
```

Queries checkpoint after 50 by default (`--max-queries`), or at the phase time
budget (at most 3,000 seconds plus one bounded query). Resumption requires the
same commit, dirty state, models, install fingerprint and dataset manifest, and
an exact prefix of the query list. A `.partial` file is never scored. Use a new
label for a new configuration. Keep the code unchanged while a run is in progress.

QMD's modes are `bm25` (SDK `searchLex`), `vector` (`searchVector`), `hybrid`
(native query expansion and RRF with reranking disabled), and `full` (native
query expansion, RRF and reranking). The latter two request 200 candidates/results;
the pinned engine's internal 20-hit retrieval per signal remains unchanged.
Short rankings are retained and reported, not padded or refilled. Vector indexing
uses QMD's regex chunker and Metal. BM25 uses CPU and should get its own label if
measuring lexical-only index time. Vector, hybrid and full can share the vector
index, with separate run files. Full and hybrid use the scorer's `hybrid` slot;
`competitor_mode` and the run ID distinguish them. Score these files separately.
QMD's native query expansion samples with temperature 0.7, top-p 0.8 and top-k
20, so repeat runs can vary. When hybrid and full share an index, run hybrid
first: full then reuses the cached expansions. Their query times consequently
reflect different cache states; they are not an isolated reranker latency test.

DigUp's `hybrid` mode uses its stock `index --include-datasets` and persistent
`search-service`, with `limit=200`, meaning search enabled and prefix completion
disabled. Its index must contain every corpus file, with no pending/unfinished
files. A silent keyword fallback is an error. LEANN's `contriever` mode explicitly
selects local Contriever with its stock HNSW CLI ingestion/chunking and persistent
search API with PyTorch MPS embedding; Contriever is not the current macOS CLI
default (MiniLM is). HNSW traversal remains on CPU.
Its embedding process is started inside the sandbox, without reusing a daemon.
LEANN receives `--include-hidden --file-types .txt` because its directory reader
otherwise filters files under the harness's `.cache` ancestor. The adapter first
checks that the corpus contains exactly the manifest's files, then verifies that
every file appears in the built passage metadata. The reported index time excludes
this post-build integrity check. LEANN uses one thread for HNSW and CPU
OpenMP/BLAS pools: the initial multi-thread CPU configuration crashed in `libomp` during model
initialization, matching the stack in this [upstream PyTorch/Faiss issue](https://github.com/pytorch/pytorch/issues/149201).
Both the one-thread CPU fallback and the selected MPS configuration passed
offline indexing and retrieval. Reported runs use MPS with the native FP16 default;
the timings describe this configuration, not maximum LEANN throughput.

Mapping uses the opaque corpus filename, never QMD's content-derived docid.
Unexpected paths, unknown document keys, invalid scores or incomplete indexing
fail the run. The scoring formula is unchanged. The Seyk-specific SciFact BM25
sanity interval does not apply to another engine's lexical score.

After scoring and preserving aggregates/provenance, delete your generated query
rows, run rows, competitor indexes and harness uv cache. When datasets are shared,
retain raw downloads, corpora and document maps until all consumers are finished. Keep removable tool/model installs until their removal
is authorized. Do not delete another worker's artifacts.

[^qmd-native-and]: **Native AND lexical diagnostic:** scifact: 0.045448, nfcorpus: 0.202307, irpapers: 0.011111. Empty rankings: 283/300, 136/323, 175/180 queries, respectively. Pinned QMD 2.8.3 joins positive query terms with `AND`; these scores are not a headline comparison of lexical quality. Its [documented lexical grammar](https://github.com/tobi/qmd/blob/facd35e01359e59d938bc9418e93fb9318addee3/docs/SYNTAX.md) has no any-term option. The documented natural-query pipeline combines expansion with lexical/vector retrieval and is represented by hybrid/full above. Query terms were not rewritten or removed.
