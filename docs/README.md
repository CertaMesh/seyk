# Seyk documentation

Start with the [project README](../README.md) for what Seyk is and a quickstart.

## Use Seyk

- [Install Seyk](installation.md): prebuilt binaries, building from source, offline builds.
- [Model setup](model-setup.md): downloading weights, local model directories, index profiles, why EmbeddingGemma 2.
- [CLI reference](cli.md): commands, search flags, keyword search, extraction, audio, speech-to-text, scores and tags, result counts and progress.
- [Use Seyk from an agent (MCP)](mcp.md): setup, the shared daemon, local-content mode, inline search latency, match provenance.
- [Supported content](supported-content.md): file types, files Seyk never indexes, segmentation, images, PDFs.
- [Caching and storage](caching-and-storage.md): how caching works, where files live, privacy and scrubbing, schema versions, cache import.
- [Known limitations](limitations.md)

## Build on Seyk

- [Rust library](library.md): the search engine API, extraction, the EmbeddingGemma 2 backend on its own.
- [Standalone embeddings and API migration](embedding.md)
- [Index progress protocol](index-progress.md): the JSONL schema for `--progress jsonl`.

## Measure and contribute

- [Performance](performance.md): retrieval quality summary and CPU throughput controls.
- [Benchmarks](benchmarks.md): full results, sources and reproduction steps.
- [Multimodal support](multimodal.md): status of images, audio, video and PDF.
- [Development](development.md): correctness checks, tests, evaluating on your own notes.
- [Architecture](architecture.md): pipeline overview and design reports.
- [Changelog](CHANGELOG.md)
