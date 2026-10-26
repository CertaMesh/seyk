# Multimodal support: status and what is missing

EmbeddingGemma 2 maps text, images, audio and video into one 768-dimensional
space. This crate implements text, images, PDF retrieval and audio retrieval.
The default CLI includes audio; library consumers can enable it separately.
This note records current support (including on-demand speech-to-text extraction)
and what video and scanned-page text extraction still need, based on the model's processor configuration
(`processor_config.json` at the pinned revision) and the Transformers reference
implementation (`EmbeddingGemma2Processor`, `Gemma4ImageProcessor`,
`Gemma4AudioFeatureExtractor`).

## What is already in place

- **Types.** `ResourceInput` has `Text`, `Image`, `Audio { samples, sample_rate }`
  and `Video { frames }` variants. `ResourceLocator` has `Text`, `PageText`, `Page`,
  `Time { start_ms, end_ms }` and `WholeResource`. PDF uses `PageText` for embedded
  text and `Page` for scanned pages.
- **Capability check.** `Embedder::supports(Modality)` gates every resource. The
  engine reports unsupported files as skipped, with a reason, instead of guessing.
- **Model plumbing.** The ONNX text model takes `image_features`,
  `video_features` and `audio_features` inputs (`[n_tokens, 512]`). Video features
  stay empty. The optional audio encoder supplies audio features.
- **Cache.** Vectors are keyed by a hash of the exact model input, so time windows
  include the audio preparation and weights policy in their input hash and resource
  descriptor. Audio changes do not invalidate text vectors or the global profile.

## Images (done)

`embedder::gemma::vision` ports the Gemma 4 image processor:

- aspect-preserving bicubic resize to multiples of 48 px within 280 × 9 patches;
- rescale to [0, 1];
- 16 × 16 patches;
- (x, y) positions, padded to 2,520 patches with (-1, -1).

The vision encoder's 512-d tokens replace `<|image|>` placeholders in
`<|image>` + N × `<|image|>` + `<image|>`. Images get no task prefix.

Measured against Python fp32 on four generated images (square, wide, tall,
landscape):

- resize sizes, soft-token counts, positions and token ids are identical;
- pixel values differ by at most 0.063, from the resize kernel;
- embedding cosine is ≥ 0.99999 (fp32) and ≥ 0.9979 (q8).

Possible improvements:

- **Vision token budget.** The processor supports 70, 140, 280 (default), 560 and
  1120 soft tokens. A lower budget would make large photo libraries much cheaper.
  It must become part of the cache profile.
- **Lazy vision loading.** The encoder loads at start-up even for text-only
  folders. That costs about 0.1 s, plus the memory for its weights (the q8 weight
  file is 195 MB).

## Audio retrieval (done, default CLI)

See [Audio in one command](model-setup.md#audio-in-one-command) for provisioning.

The `audio` feature adds Symphonia 0.5.5 and a Rust port of the pinned Gemma 4
feature extractor. The CLI enables `audio-stt` (and therefore `audio`) by default;
`--no-default-features` library builds keep audio dependencies disabled. The core streams
WAV/FLAC/MP3 and AAC-LC/ALAC M4A, applies presentation edits/gapless metadata,
means channels and resamples with a deterministic 64-tap Blackman-windowed sinc
filter (0.94 Nyquist guard). Windows are 10 seconds with an 8-second stride,
including the final short tail. `Time` evidence describes the post-trim timeline.

Features use magnitude, not power: periodic Hann, 160-sample left padding,
321-sample frame indexing, 512-point FFT, 128 HTK mel bins and `ln(mel + 0.001)`.
The q8 encoder's 512-dimensional features replace audio placeholders between
`<|audio>` and `<audio|>` without a task prefix. Valid mask positions sampled at
stride four determine token count. A full window produces 999 frames/250 tokens.

`seyk model download --audio` explicitly provisions only the two pinned q8
encoder files (324.54 MiB); search never downloads. Missing weights fail each
audio file independently. Audio weights/policy live in `audio-input-v1` and the
preparation descriptor, leaving the text cache profile unchanged. No PCM, mel
features, audio tokens or transcript are written to the cache.

Audio identity preparation also retains about 324.54 MiB of encoder bytes per
model, including cached-only audio search. Session initialization temporarily
clones another 324.31 MiB of weights, plus ONNX Runtime allocations. Text-only
use allocates no audio snapshot. See [audio snapshot memory](embedding.md#audio-snapshot-memory)
for reload and retry semantics and the scoped release memory measurement.

The encoded limit remains 32 MiB; the derived decoded-duration limit is
1,048.576 seconds. HE-AAC/SBR, multiple tracks and complex edit lists are typed
unsupported inputs. Search never transcribes. Explicit extraction of a `time`
hit transcribes that range with `audio-stt`, included in the default CLI (Phase 5b-2);
without it, extraction returns `UnsupportedExtraction`. Previews, VAD and
diarization are not implemented.
See the [implementation report](architecture.md#phase-5a-audio-retrieval) for
reproducible synthetic evaluations and limitations.

## Video (not implemented)

What is needed:

1. **Frame decoding.** Realistically `ffmpeg` (a library binding or the CLI as a
   subprocess); pure-Rust decoders do not cover common codecs. This is the main
   new dependency decision.
2. **Sampling.** The processor samples 1 frame per second, caps an input at 32
   frames (uniformly subsampled beyond that) and uses 140 soft tokens per frame.
3. **Token layout.** Without timestamps (the default), each frame becomes
   `<|image>` + 140 × `<|video|>` + `<image|>`. The features go into the text model's
   `video_features` input, not `image_features`. The per-frame preprocessing is the
   image processor at a 140-token budget, so most of `vision` can be reused.
4. **Windowing.** For example 30 s windows at 1 fps (30 frames, under the 32-frame
   cap, about 4,200 tokens), each returned as a `Time` locator such as
   `meeting.mp4 @ 00:42:10–00:43:05`.
5. **Audio track.** The simplest design embeds the soundtrack as audio windows of
   the same resource, so a file can return both visual and audio hits. Embedding
   frames and audio together in one interleaved input is supported by the model,
   but has not been evaluated here.

## PDF (retrieval done, OCR deferred)

Normal builds support both embedded text and scanned pages. The `pdf` feature
extracts embedded text per physical page and reuses the text segmenter. Hits
carry `PageText` locators; extraction verifies the PDF content hash, extraction
version and page-text hash before returning readable evidence.

The optional `pdf-render` feature, enabled in normal CLI/release builds, renders
pages without usable embedded text through pinned Hayro 0.5.0 and the existing
Gemma image embedding path. It preserves crop and rotation, uses bundled fonts
and a white background, and caps the longest edge at 2048 pixels. Each scanned
page produces a `Page` hit with no text preview. Mixed PDFs keep both kinds of
hit. Image-XObject pages that render nearly uniform get a page diagnostic
instead of an embedding; faint scans and partially omitted corrupt images
remain recall limitations.

Scanned-page text extraction needs OCR, which is tracked as a follow-up.
`extract` on a `Page` hit returns the typed `PdfOcrUnavailable` error; no OCR
model is shipped or downloaded. Text-only models, `--text-only`, and builds
without `pdf-render` keep embedded-text search and report omitted image pages
as page diagnostics. Rendering alone measured roughly 34 ms/page in the spike;
image embedding takes seconds per page on CPU. See the
[Phase 4b architecture report](architecture.md#phase-4b-retrieval--scanned-pdf-pages)
for the synthetic quality and end-to-end timing measurements, batching limits,
cache behavior and parser containment contract.

## Verification standard

For every new modality, before it is enabled:

1. a reference dump from the official Python implementation in float32;
2. Rust preprocessing compared tensor by tensor, and token ids compared exactly;
3. final vectors with cosine ≥ 0.999 (fp32) and ≥ 0.995 (q8), or a documented
   reason why not;
4. a ranking test with a text query against generated fixtures, as
   `tests/multimodal.rs` does for images.

## Experimental Photo Anchor scan

`seyk search "red bicycle" ./photos --image-scan q4` opts into Photo Anchor candidate scanning for standalone images with an EmbeddingGemma 2 d256 or d768 profile. Rust callers use `SearchRequest::image_scan(ImageScan::PhotoAnchorQ4)`. A persistent SQLite cache is required. Lexical-only mode and `--no-cache` reject q4.

Exact f32 remains the default. Opted-in sync derives missing codes from retained f32 embeddings without inference. Cached-only searches never backfill: missing or unknown-codec representations use f32. A malformed current-codec row is an error. PDF pages, text and audio continue to use f32.

For each query the image shortlist is `min(image_count, max(50, 4*take))`, with `take=limit` for semantic and `take=max(4*limit,50)` for hybrid. The engine fetches only selected f32 image vectors, reranks with its exact dot kernel, applies the semantic threshold, and merges with exact candidates. Approximate scores never reach results. This bounded shortlist cannot guarantee recall on arbitrary libraries. On the 280-photo fixture hybrid depth is near exhaustive; semantic k=1/5/10 is the stronger test.

Both representations remain on disk, so q4 adds cache storage. Clear, pruning and vector replacement invalidate derived codes through foreign-key cascades. Schema v7 upgrades every cache, even when q4 is off; older binaries cannot reopen v7. Import from v6/v7 copies f32 only and leaves sources unchanged.
