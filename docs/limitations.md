# Known limitations

- **Literal matches still pull.** A passage that *shows* a theme without naming it
  (a story whose details keep shifting, for "memory becoming unreliable") can rank
  below an unrelated note that shares a word ("memory card … unreliable"). See
  [the retrieval smoke test](benchmarks.md#retrieval-smoke-test).
- **Hybrid can push a good meaning match down.** On a private eight-query test
  folder, hybrid search moved 7 of 8 target passages into the top 10 (semantic
  alone: 6 of 8), but one purely thematic query fell from 19th to 25th because keyword
  hits crowded it out. `--mode semantic` is there for those cases.
- **First search of a large folder is slow.** Embedding on the CPU managed roughly
  5–6 segments per second on a busy machine, about 30 minutes for 10,000
  segments. After that, only changes are embedded. Run `index --watch` or the MCP
  server so this happens in the background.
- **Video and scanned-page OCR are not implemented.** Audio transcription exists
  only as opt-in, on-demand extraction (`audio-stt`). See
  [the multimodal status](multimodal.md) for current support and remaining
  work, including [PDF extraction limits](multimodal.md#pdf-retrieval-done-ocr-deferred).
- **Indexing can use several GB of memory.** Embedding batches have no token
  budget, so files that tokenize densely (numeric JSON, tables) fill a batch with
  long inputs, and ONNX Runtime keeps its peak allocation. A long-running daemon
  stays at its highest batch so far. A fix is in progress
  ([#95](https://github.com/CertaMesh/seyk/pull/95)).
- **CPU only.** ONNX Runtime runs on the CPU. No GPU or Neural Engine provider is
  enabled yet.
- **Start-up costs about 1.5 s**, mostly ONNX graph optimisation and parsing the
  32 MB tokenizer. This is per process, not per file.
- **Images are slow to embed** compared with text, because each image always uses
  the full 2,520-patch budget. Large photo collections are better indexed ahead of
  time.
- **Brute-force search.** Every query is compared with every cached vector. That is
  exact and simple, and the `VectorIndex` trait leaves room for an approximate index
  if collections outgrow it.
- **Change detection uses size and mtime first.** A file rewritten with identical
  size and modification time to the nanosecond is not re-read until `index` or a
  cache clear. Vectors are still never reused for different text.
- **Single-writer cache.** Two processes writing the same cache wait on SQLite's
  lock rather than run in parallel. MCP sessions avoid this by sharing the daemon;
  a separate `seyk index --watch` on the same folder still duplicates its work.
- **Searches can wait for an indexing batch.** The daemon runs one model, so a
  query embedded while a large first index is running waits for the current
  batch (a few seconds on a busy CPU).
- **Setext headings** (underlined with `===`/`---`) are not recognised; ATX
  headings (`#`) are.
