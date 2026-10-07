# Multimodal support: status and what is missing

EmbeddingGemma 2 maps text, images, audio and video into one 768-dimensional
space. This crate implements text and images. This note records what audio, video
and PDF still need, based on the model's processor configuration
(`processor_config.json` at the pinned revision) and the Transformers reference
implementation (`EmbeddingGemma2Processor`, `Gemma4ImageProcessor`,
`Gemma4AudioFeatureExtractor`).

## What is already in place

- **Types.** `ResourceInput` has `Text`, `Image`, `Audio { samples, sample_rate }`
  and `Video { frames }` variants. `ResourceLocator` has `Text`, `Page`,
  `Time { start_ms, end_ms }` and `WholeResource`. Neither needs to change for
  audio, video or PDF.
- **Capability check.** `Embedder::supports(Modality)` gates every resource. The
  engine reports unsupported files as skipped, with a reason, instead of guessing.
- **Model plumbing.** The ONNX text model already takes `image_features`,
  `video_features` and `audio_features` inputs (`[n_tokens, 512]`). Today the last
  two are always empty.
- **Cache.** Vectors are keyed by a hash of the exact model input, so time windows
  only need their window parameters included in that hash. Window length and hop
  must also go into the segmenter version, so changing them invalidates old vectors.

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
  folders. That costs about 0.1 s, plus the memory for its weights (the q8 weight file is 195 MB).

## Audio (not implemented)

What is needed:

1. **Decoding and resampling** to 16 kHz mono `f32`. A pure-Rust option is
   `symphonia` for decoding plus `rubato` for resampling; otherwise use an `ffmpeg`
   dependency.
2. **Port `Gemma4AudioFeatureExtractor`.** It produces log-mel features with
   128 mel bins, 0–8000 Hz, FFT size 512, frame length 320 (20 ms), hop 160
   (10 ms), mel floor 0.001, no pre-emphasis, no dither, and semicausal padding of
   `frame_length / 2` on the left. Verify the window function and log/scaling
   details against the Python reference before trusting it, using the same parity
   method as images (dump features and final vectors, compare).
3. **Run `onnx/audio_encoder*.onnx`.** It takes `input_features [b, T, 128]` and
   `input_features_mask [b, T]` and returns `audio_features [n, 512]`.
4. **Token layout.** `<|audio>` + N × `<|audio|>` + `<audio|>`. N comes from the
   mel frame count after two stride-2 convolutions (kernel 3, padding 1), which is
   about 25 tokens per second, capped at `audio_seq_length` (280 in this
   checkpoint's processor config).
5. **Windowing.** One embedding per overlapping window, for example 120 s windows
   with a 90 s hop (00:00–02:00, 01:30–03:30, …). Each becomes a
   `ResourceLocator::Time`. Note that a 280-token cap is about 11 s of audio. If
   that cap is real for this checkpoint, windows must be about 10 s, and window
   length should be benchmarked on spoken-word retrieval before choosing a
   default. The model card's "~327 seconds" maximum assumes the full 8K context.
6. **Precision.** The ONNX export notes that 4-bit quantisation hurts audio most.
   Use q8 or fp32.

No transcription is needed or wanted: the encoder embeds the audio directly.

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

## PDF (not implemented)

`ResourceLocator::Page` is reserved. There are two options, and they can coexist:

- extract text per page and reuse the text segmenter;
- render pages to images and use the vision path. EmbeddingGemma 2 reports
  MMEB-VisDoc NDCG@5 of 67.84, so page images are a real option for scanned or
  layout-heavy documents.

## Verification standard

For every new modality, before it is enabled:

1. a reference dump from the official Python implementation in float32;
2. Rust preprocessing compared tensor by tensor, and token ids compared exactly;
3. final vectors with cosine ≥ 0.999 (fp32) and ≥ 0.995 (q8), or a documented
   reason why not;
4. a ranking test with a text query against generated fixtures, as
   `tests/multimodal.rs` does for images.
