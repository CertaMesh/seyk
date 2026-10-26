<p align="center">
  <img src="https://raw.githubusercontent.com/CertaMesh/seyk/main/docs/assets/logo.png" width="600" alt="Seyk">
</p>

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://github.com/CertaMesh/seyk/blob/main/Cargo.toml"><img src="https://img.shields.io/badge/rust-1.89%2B-orange.svg" alt="Rust 1.89+"></a>
</p>

Seyk is local hybrid search over your own files and code. It combines semantic
meaning, BM25 keywords and file names. Semantic retrieval runs on Google's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).
Use the CLI, embed the Rust library, or connect an agent through MCP.

Ask a question in your own words, or type an exact name, and get back the
passages that match. Meaning, keywords and file names are searched together:

```console
$ seyk search "keeping tomatoes alive in a heatwave" ~/notes -n 1
1.000  garden.md:1-3  § Garden  [meaning 0.67]
       Water the vegetable beds at dawn, before the sun climbs. A thick layer of straw mulch holds moisture in the soil through the hottest afternoons.

$ seyk search "INC-18372" ~/notes -n 1
1.000  work.md:1-3  § Work  [meaning 0.76+words]
       Ship the pricing page before the conference. INC-18372 is still open: the old page lists the beta plan.

$ seyk search "garden.md" ~/notes -n 1
1.000  garden.md:1-3  § Garden  [meaning 0.73+words+name]
       Water the vegetable beds at dawn, before the sun climbs. A thick layer of straw mulch holds moisture in the soil through the hottest afternoons.
```

This is real output from a three-note folder. The first match shares no words
with the question. The tag in brackets shows which signals found each hit.

Seyk is for developers, applications and AI agents that need to search their
own files without sending those files anywhere. It returns relevant resources
and exact locations inside them. It does not generate answers or summaries.

## Why Seyk

- **Local and private.** Seyk sends no files, queries or embeddings over the
  network. The model runs on a laptop CPU. Only explicit model setup commands
  use the network.
- **Meaning, keywords and file names together.** The default hybrid mode
  combines semantic similarity, BM25 keyword scores and file-name matches. A
  question can find a passage that shares none of its words. An identifier such
  as `INC-18372` or `Session::restore` still matches exactly.
- **Evidence you can verify.** Hits point at exact locations, such as a line
  range, a PDF page, or an audio time window. `seyk extract` checks the full
  file hash before it returns the passage, and fails instead of returning
  different text if the file changed.
- **CLI, Rust library and MCP.** The `seyk` CLI is a thin layer over the
  library, so your application can embed the same engine. `seyk mcp` gives AI
  agents the same search. On Unix, all agent sessions share one loaded model.

## Quickstart

Build and install from source (Rust 1.89 or newer):

```bash
cargo install --locked --git https://github.com/CertaMesh/seyk
```

Prebuilt binaries for macOS (Apple Silicon), Linux (x86_64 and ARM64) and
Windows (x86_64) return with the next release. Linux needs glibc 2.38 or newer,
and there is no Intel Mac build.

Download the model weights once (about 0.5 GB):

```bash
seyk model download
```

Search a folder (recursively). Embeddings are cached as you go:

```bash
seyk search "how I fixed my sleep" ~/notes
```

Searching never downloads anything. Without the model weights, search still
returns keyword and file-name matches. For every option, see
[Install Seyk](https://github.com/CertaMesh/seyk/blob/main/docs/installation.md), [Model setup](https://github.com/CertaMesh/seyk/blob/main/docs/model-setup.md) and the
[CLI reference](https://github.com/CertaMesh/seyk/blob/main/docs/cli.md).

## Use Seyk from an agent (MCP)

`seyk mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server
over stdio. Agents get a `search` tool that returns ranked passages with path,
line range, heading, score and preview. The agent then reads those lines and
writes the answer.

```bash
# Claude Code: make your notes searchable in every session
claude mcp add --scope user seyk -- seyk mcp ~/notes

# Or per project, for a codebase
claude mcp add seyk -- seyk mcp .
```

On Unix, a shared background daemon loads the model once for all sessions.
The server indexes your folders in the background and re-indexes when files
change. The `search` tool refuses paths outside the folders the server was
started with.

**Privacy.** Seyk itself sends nothing anywhere, but a cloud-hosted agent sends
the passages it reads to its provider. Point the server only at folders you are
happy for that agent to read. See [Use Seyk from an agent (MCP)](https://github.com/CertaMesh/seyk/blob/main/docs/mcp.md).

## Use Seyk as a Rust library

The crates are **not on crates.io yet**. Use a Git dependency:

```toml
[dependencies]
seyk = { git = "https://github.com/CertaMesh/seyk" }
```

```rust
use seyk::embedder::gemma::{GemmaConfig, GemmaEmbedder};
use seyk::{Dimension, SearchEngine, SqliteCache};

let embedder = GemmaEmbedder::load(GemmaConfig {
    dimension: Dimension::D256,
    ..Default::default()
})?;

let engine = SearchEngine::builder()
    .embedder(embedder)
    .cache(SqliteCache::open_default()?) // omit for an in-memory, ephemeral cache
    .build()?;

let results = engine
    .search("what did we decide about pricing?")
    .paths(["~/notes"])
    .limit(10)
    .execute()?;
```

You can also use only the EmbeddingGemma 2 backend, without the search engine.
See the [Rust library guide](https://github.com/CertaMesh/seyk/blob/main/docs/library.md).

## Supported content

| Kind | Status |
|---|---|
| Markdown, plain text, source code | ✅ heading-, paragraph- or block-aware segments |
| Images | ✅ found by meaning, one vector per image |
| PDF | ✅ embedded text; scanned pages found by meaning, text extraction needs OCR |
| Audio | Included in the default CLI; `seyk setup audio` installs audio search and speech-to-text models |
| Video | ❌ not implemented |

Seyk never indexes keys, password stores, `.env` files or credentials, and
there is no override. See [Supported content](https://github.com/CertaMesh/seyk/blob/main/docs/supported-content.md).

## Status and limitations

Seyk is at version 0.1.0. Know these before you adopt it:

- **CPU only.** No GPU or Neural Engine provider is enabled yet.
- **The first search of a large folder is slow.** Expect roughly 30 minutes for
  10,000 segments on a busy machine. After that, Seyk embeds only what changed.
- **Hybrid can push a good meaning match down.** Use `--mode semantic` for
  purely thematic queries.
- **Video and scanned-page OCR are not implemented.**
- **The library API can still change.** Resource provider types may change
  before 1.0, and the filesystem is the only supported provider.

See all [Known limitations](https://github.com/CertaMesh/seyk/blob/main/docs/limitations.md).

## Benchmarks

![Seyk hybrid, semantic and lexical retrieval next to QMD, DigUp and LEANN measured on Seyk's harness, and sourced published model scores; higher nDCG@10 is better](https://raw.githubusercontent.com/CertaMesh/seyk/main/docs/assets/benchmarks.svg)

nDCG@10 measures how well the first ten results rank relevant documents. Higher
is better. The default hybrid mode still trails semantic-only search on SciFact. See
[Performance](https://github.com/CertaMesh/seyk/blob/main/docs/performance.md) for the full setup and
[Benchmarks](https://github.com/CertaMesh/seyk/blob/main/docs/benchmarks.md) for sources and reproduction steps.

## Documentation

- [Install Seyk](https://github.com/CertaMesh/seyk/blob/main/docs/installation.md) and [Model setup](https://github.com/CertaMesh/seyk/blob/main/docs/model-setup.md)
- [CLI reference](https://github.com/CertaMesh/seyk/blob/main/docs/cli.md): search modes, scores, extraction, audio, progress
- [Use Seyk from an agent (MCP)](https://github.com/CertaMesh/seyk/blob/main/docs/mcp.md)
- [Rust library](https://github.com/CertaMesh/seyk/blob/main/docs/library.md)
- [Supported content](https://github.com/CertaMesh/seyk/blob/main/docs/supported-content.md)
- [Caching and storage](https://github.com/CertaMesh/seyk/blob/main/docs/caching-and-storage.md): what is cached, where, and how to erase it
- [Performance](https://github.com/CertaMesh/seyk/blob/main/docs/performance.md) and [Known limitations](https://github.com/CertaMesh/seyk/blob/main/docs/limitations.md)
- [Development](https://github.com/CertaMesh/seyk/blob/main/docs/development.md) and [Architecture](https://github.com/CertaMesh/seyk/blob/main/docs/architecture.md)
- [All documentation](https://github.com/CertaMesh/seyk/blob/main/docs/README.md)

## License

Apache-2.0. EmbeddingGemma 2 is released by Google under Apache 2.0.
