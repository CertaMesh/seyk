# Photo Anchor P3 evidence

Opt-in q4 with retained f32 and exact reranking. All correctness and local CI/NanoSciFact gates passed. The coordinator-authorized slot passed the scan speed contract at both dimensions. q4 scanning is slower than f32 here, within the allowed +2 ms per 1,000 vectors.

## Fixture

280 CC0 photos, 60 queries, 317 graded labels. Exactly one re-rendered thumbnail (`apple-fruit-06`) was re-pinned after HTTP/content-type/JPEG/960-pixel-width checks. Source page and licence are unchanged. Historical P2 numbers used its old bytes; every P3 comparison below uses a fresh full-f32 baseline on the current fixture.

Manifest SHA-256: `2fdd0ba135334c13d0bc028189fff34348c766692b589c3377e6cddee278035c`.
Fixture/query BLAKE3: `c2bbf14d318d9cc812941d8216d703f7cba7f2c9d4e067ed09f5460d5d4ce575`.

## Correctness

Each dimension checks 16,800 query/image score pairs against independent sequential f32 decode-dot and f64 dequantized-dot references parsed directly from the assets. The separate codec suite also covers every byte, mixed nibbles, sparse/sign/random queries, Python goldens and the unchanged 4714 witness.

| Width | Max error vs sequential f32 | Max error vs f64 | Tolerance |
|---:|---:|---:|---:|
| 256 | 1.13248825e-06 | 5.35883399e-07 | 5e-05 |
| 768 | 1.7285347e-06 | 6.52776846e-07 | 0.00015 |

The actual engine produced identical ordered hits and exact scores for every query at k=1/5/10/20/50, zero, and k=281, in semantic and hybrid mode at both widths. Cached-only fallback, warm no-inference backfill and reopening were exercised. R stayed fixed at `min(image_count,max(50,4*take))`. Independent Python readback verified all 240 paired top-10 lists and the four macro nDCG values.

| Width | Mode | f32 nDCG@10 | q4 + exact nDCG@10 | Drop |
|---:|---|---:|---:|---:|
| 256 | Semantic | 0.956234038 | 0.956234038 | 0.000000000 |
| 256 | Hybrid | 0.960910761 | 0.960910761 | 0.000000000 |
| 768 | Semantic | 0.959800118 | 0.959800118 | 0.000000000 |
| 768 | Hybrid | 0.959599265 | 0.959599265 | 0.000000000 |

Hybrid depth gives R=200 at k=10, nearly exhaustive on 280 images. Semantic k=1/5/10 uses R=50 and is stronger evidence. These finite-fixture results do not establish universal recall.

## Clipping

[Every coordinate’s clipping count and rate](photo-anchor-p3-clipping.md) is reported without a pass/fail threshold.

## Disk

Matched compacted SQLite snapshots include schema v7 in both variants. f32 stays on disk. These are database bytes after `VACUUM INTO`, without active WAL/SHM sidecars.

| Width | f32 only | f32 + q4 | Added bytes |
|---:|---:|---:|---:|
| 256 | 774144 | 917504 | 143360 |
| 768 | 1536000 | 1757184 | 221184 |

## Model provenance

Provisioned ONNX q8 text/vision graphs, four inference threads. No models, tables or packages were downloaded.

| File | Bytes | SHA-256 |
|---|---:|---|
| `tokenizer.json` | 32170510 | `4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4` |
| `onnx/model_quantized.onnx` | 495165 | `d06edd601f851c633a2519304cbeb8dc6170d7ceb61b436625c17fb9b6e74953` |
| `onnx/model_quantized.onnx_data` | 313724928 | `278a7ff1248c3618e4bd11a607fc54f7bdc7778854230f3956d3f86bd9db4f3b` |
| `onnx/vision_encoder_quantized.onnx` | 162495 | `bb0de2df53a2448a32dc7908a187c168c8afd514d4d6f674f7f46024875fa4e3` |
| `onnx/vision_encoder_quantized.onnx_data` | 195228672 | `3dabd69c0a36e9a8771ad82030dde74daa5a0e02b7047a5d3f3382b1137bab89` |

## Authorized slot measurements

One slot batch on 2026-10-24, 17:40:31–17:40:55 UTC, source `2d3604a724efcbf77d410bc99d00581c3537ac17`. Start/end load1 were **4.07 / 5.04**; memory free was **70% / 70%** (70% before each model run). Gaze #773’s leak-only capture (~1 core) remained running with coordinator approval; other projects were paused. All three prebuilt release commands ran sequentially at nice 10; no builds ran alongside. All exited zero.

Criterion 0.8.2 medians below are from `estimates.json` (not the stdout slope estimate): 30 samples per side, 0.5 s warmup and 2 s measurement per function. Each iteration scans 1,000 deterministic vectors and consumes every score. Setup, allocation, encoding and query LUT construction are outside the samples. The comparison is a CPU scan microbenchmark, not an end-to-end speedup claim; f32 precedes q4 within each width. Raw samples, median confidence intervals and request samples are in [the slot artifact](photo-anchor-p3-slot.json).

| Width | f32 median ms / 1k | q4 median ms / 1k | q4 − f32 ms / 1k | +2 ms bar |
|---:|---:|---:|---:|---|
| 256 | 0.023400304 | 0.068473554 | +0.045073250 | Pass |
| 768 | 0.073009686 | 0.273220423 | +0.200210737 | Pass |

Warm end-to-end cached engine requests include actual text-query inference, SQLite reads, scanning, exact rerank and fusion. Model loading, preparation, warmup and previews are excluded. Five repetitions per mode/scan alternate order; query index 0 only, k=10, all 280 images searched. All 40 timed requests returned 10 hits identical to their f32 reference, with zero errors. This narrow request sample does not establish an end-to-end improvement; small differences and the d768 hybrid variation should not be generalized. Query inference and cache work dominate the request relative to these sub-millisecond scan costs.

| Width | Mode | f32 median ms (range) | q4 median ms (range) |
|---:|---|---:|---:|
| 256 | Semantic | 85.523 (84.724–89.241) | 85.510 (84.891–86.882) |
| 256 | Hybrid | 86.874 (85.246–87.874) | 87.505 (86.754–87.726) |
| 768 | Semantic | 83.273 (81.747–84.118) | 84.227 (83.835–85.096) |
| 768 | Hybrid | 84.278 (82.508–97.975) | 90.731 (84.045–95.729) |

## Local verification

All 24 sequential stages passed on `007c7fe3bbc55458bde20b53b352c3b26507bef3`: the CI verification commands for default/audio/audio-STT, standalone dependencies/library/CLI/core/PDF, Python chart/metric adapters, the exact `cargo deny --locked --all-features check licenses` form, all Rust 1.89 checks/builds, the requested Windows GNU no-default core all-target clippy check, and fetcher tests. Toolchains and the GNU target were already installed; no toolchain or package downloads were needed. These are native macOS checks plus Windows cross-target linting, not a Windows runtime test.

The final P2 gate was repeated on that implementation with f64 error/tolerance comparisons throughout. It passed without changing R. The preserved 4714 witness also passes the amended bound. Timed runs started only after explicit `ROOT: SLOT-GO`.

## Review follow-up

Three review findings were addressed separately: a non-ignored 60-code cutoff test at both widths and semantic k=1/5, the actual UTC re-pin date, and one shared Rust fixture resolver using HOME then USERPROFILE with the existing override. The cutoff case ties every approximate score, including positions 50 and 51; the ten excluded rows have better exact scores. The asserted resource-order result therefore proves R=50 truncation, rather than exhaustive reranking. This synthetic collision case is separate from the full P2 parity contract, which passed unchanged.

After these changes, affected core tests passed (7 passed, P2 provisioning test ignored in this focused run), both example test binaries passed (5 and 3 tests), and all 6 offline fetcher tests passed. The re-pin regression first failed against the hardcoded date, then passed with a mocked future UTC date. `cargo fmt --all --check`, workspace/all-target clippy with warnings denied, and the requested no-default core Windows GNU/all-target clippy all passed. Cargo commands ran sequentially at nice 10 with four build jobs and 33–34% free memory at launch. The prior full local gate, P2 correctness and slot measurements remain identified by their original source heads above; search/codec production logic did not change in this follow-up.

## NanoSciFact default-off regression

Base `515e943eedadb2f3da2a9bd0ff76bb9963af84a1` and candidate `007c7fe3bbc55458bde20b53b352c3b26507bef3` ran the same rank-only harness and task-owned copies of the same warm cache. The base had only the test-harness patch; its production code stayed pinned. All 50 queries in each of semantic, hybrid and lexical mode returned identical ordered rankings and scores. Zero document embeddings were recomputed. All 3,544 retained f32 vectors were byte-identical after v6-to-v7 migration; the candidate had zero image-code rows. No latencies were recorded.

| Mode | Before nDCG@10 | After nDCG@10 |
|---|---:|---:|
| Hybrid | 0.891716331 | 0.891716331 |
| Semantic | 0.896350587 | 0.896350587 |
| Lexical | 0.714471114 | 0.714471114 |

Environment: Apple M5 Max, 18 logical CPUs, macOS 26.5 (25F71), Rust 1.99.0 (`b940084d7`). The provisioned ORT static library is identified by its path and SHA-256 in the retained raw evidence bundle. Local build/test commands used four build jobs and nice 10, with at least 30% free memory at launch.
