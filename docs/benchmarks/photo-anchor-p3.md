# Photo Anchor P3 evidence

Opt-in q4 with retained f32 and exact reranking. Timing and the complete local CI/NanoSciFact gate are pending; this report does not yet establish the speed contract.

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
