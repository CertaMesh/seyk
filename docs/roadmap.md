# Roadmap

Outcomes Seyk is working towards, in priority order. There are no dates. Each
item says what a user gets, what is missing today and how we will know it is
done. Every change that affects ranking or segmentation still has to pass the
retrieval benchmark before it merges.

## Now

### Indexing does not take over the machine

- **For:** anyone running Seyk in the background, especially the shared MCP
  daemon.
- **Gap:** embedding batches have no token budget. Files that tokenize densely
  (for example numeric JSON) can push indexing memory to several GB, and the
  ONNX Runtime arena keeps that peak. Work in progress:
  [#95](https://github.com/CertaMesh/seyk/pull/95).
- **Done when:** peak memory (measured as process footprint, not RSS) stays
  under a documented bound on a corpus that includes dense files, across
  repeated daemon starts, with no loss in benchmark quality.

### Know whether Seyk helps an agent

- **For:** agent developers deciding whether to add Seyk.
- **Gap:** the [first agent evaluation](benchmarks.md#agent-evaluation)
  makes no claim: all answers were correct and cited in both setups, and where
  Claude chose Seyk it was faster but used 5% more tokens. It used one small
  corpus and predates compact results and code chunking; a rerun is planned.
- **Done when:** an evaluation compares agents with their built-in tools alone
  against the same agents with Seyk added, on correct and cited answers and on
  tokens per task, and the result is published either way. Seyk makes a public
  claim about agents only if it wins on one of those without losing on the
  other.

## Next

### Agent results fit the context budget

- **Gap:** each MCP hit carries full resource, locator and metadata objects
  plus diagnostics, which costs tokens.
- **Done when:** tokens per search are measured, and a more compact output
  reduces them without lowering the rate of correct, cited answers.

### Embed Seyk from crates.io with a stable contract

- **For:** Rust developers building Seyk into an application.
- **Gap:** the crates are on crates.io at 0.2.0, but several public types are
  marked unstable and the extraction output has no stability promise yet.
- **Done when:** a stability and migration policy is documented, and the [embedding example](../examples/embed_search.rs) runs in
  CI.

### Model files are checked by content

- **Gap:** for a local model directory, external weight files are identified
  by size only.
- **Done when:** any byte change to a weight file changes the model identity,
  so cached vectors are never reused for a different model.

### Code search finds the right function

- **Gap:** code is split at blank lines, not by function, and hits carry no
  symbol names.
- **Done when:** a code-location evaluation shows whether syntax-aware
  splitting improves the share of questions whose answer lands in the top
  results, and any change passes the retrieval benchmark.

## Later

- **Scanned PDFs become readable evidence:** OCR for scanned pages, labelled as
  derived text like audio transcripts, with quality measured on a public set.
- **Faster indexing on Macs:** a hardware-accelerated backend, kept in parity
  with the CPU results.
- **Retrieval research:** other embedding models and learned sparse retrieval,
  adopted only if the public benchmark improves.
- **PDF parser isolation:** run the PDF parser in a separate process once a
  library integration needs it.

## Done

- **Install in one step (v0.2.0):** release archives for macOS (Apple
  Silicon), Linux (x86_64, ARM64) and Windows (x86_64) ship with SHA-256 files
  and build provenance attestations. `scripts/install.sh` installs them without
  a Rust toolchain, and `cargo install --locked seyk` works from crates.io.

## Not planned

Generating answers or summaries, remote connectors, HTTP APIs, hosted search
and enterprise access control. See [What Seyk is for](product.md#non-goals).
