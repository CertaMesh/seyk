# Known limitations

- **Literal matches still pull.** A passage that *shows* a theme without naming it
  (a story whose details keep shifting, for "memory becoming unreliable") can rank
  below an unrelated note that shares a word ("memory card … unreliable"). See
  [the retrieval smoke test](benchmarks.md#retrieval-smoke-test).
- **Hybrid can push a good meaning match down.** On a private eight-query test
  folder, hybrid search moved 7 of 8 target passages into the top 10 (semantic
  alone: 6 of 8), but one purely thematic query fell from 19th to 25th because keyword
  hits crowded it out. `--mode semantic` is there for those cases.
- **First search of a large folder is slow.** In a historical measurement on a
  busy Apple M5 Max (q8, 256 dimensions, segments of about 241 tokens), CPU
  embedding managed roughly 5–6 segments per second, about 30 minutes for 10,000
  segments. See [the model benchmark](benchmarks.md#model) for its method.
  After that, only changes are embedded. Run `index --watch` or the MCP
  server so this happens in the background.
- **Video and scanned-page OCR are not implemented.** Audio transcription exists
  only as opt-in, on-demand extraction (`audio-stt`). See
  [the multimodal status](multimodal.md) for current support and remaining
  work, including [PDF extraction limits](multimodal.md#pdf-retrieval-done-ocr-deferred).
- **Indexing can still use a few GB of memory.** Since 0.2.0, text embedding
  batches default to a budget of 5120 padded tokens and 4194304 attention cells,
  with a 16-row cap. On the full SciFact and NFCorpus runs, peak physical
  footprint was about 2.8 GB, down from about 5.7 GB. This is not a fixed cap:
  other corpora can use more, and ONNX Runtime keeps its peak allocation, so a
  long-running daemon stays at its highest batch so far. See
  [the measurements](benchmarks.md#text-batch-memory-budgets).
- **CPU only.** ONNX Runtime runs on the CPU. No GPU or Neural Engine provider is
  enabled yet.
- **Start-up cost is per process**, not per file: mostly ONNX graph optimisation
  and parsing the 32 MB tokenizer. It was about 1.5 s in a historical,
  unrepeated observation; [the model benchmark](benchmarks.md#model) has the
  measured model load times.
- **Images are slow to embed** compared with text, because each image always uses
  the full 2,520-patch budget. Large photo collections are better indexed ahead of
  time.
- **Exact brute-force search by default.** By default every query is compared
  with every cached f32 vector. That is exact and simple, and the `VectorIndex`
  trait leaves room for an approximate index if collections outgrow it. An
  experimental, opt-in scan for standalone images (`--image-scan q4`) shortlists
  candidates from compressed codes and reranks them with the f32 vectors. It
  needs a persistent SQLite cache and a d256 or d768 profile, cannot guarantee
  recall on arbitrary libraries, and stores both representations, so it adds
  cache storage. See [the Photo Anchor scan](multimodal.md#experimental-photo-anchor-scan).
- **Filesystem size and mtime are cheap change hints.** They do not prove that
  bytes are unchanged. An equal-size rewrite with a restored mtime is not re-read
  on a warm pass when preparation is unchanged. Ranking can then use stale
  vectors and keyword rows. Extraction still checks the content hash and
  refuses changed text. To force a re-read, run `seyk cache clear` for the cache
  in use, then search or index again. `seyk index` alone does not force a re-read.
- **Single-writer cache.** Two processes writing the same cache wait on SQLite's
  lock rather than run in parallel. MCP sessions avoid this by sharing the daemon;
  a separate `seyk index --watch` on the same folder still duplicates its work.
- **Searches can wait for an indexing batch.** The daemon runs one model, so a
  query embedded while a large first index is running waits for the current
  batch (a few seconds on a busy CPU).
- **Setext headings** (underlined with `===`/`---`) are not recognised; ATX
  headings (`#`) are.
