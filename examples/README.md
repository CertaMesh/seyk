# Examples

## Usage examples

- [agent-decisions/](agent-decisions/): a folder of invented decision records,
  with CLI and MCP walk-throughs for search and extraction.
- [embed_search.rs](embed_search.rs): a Rust program that embeds Seyk, searches
  a folder and extracts the top hit. It uses `SearchEngine::lexical()`, so it
  needs no model weights.

```console
$ cargo run --example embed_search
query: how do background jobs write to the database?
1.000  0007-storage-sqlite.md:20-25  0007. Store tickets in SQLite, not Postgres > Consequences
0.857  0007-storage-sqlite.md:13-18  0007. Store tickets in SQLite, not Postgres > Decision
0.750  0012-attachments-object-storage.md:1-4  0012. Keep attachments out of the database

extracted 0007-storage-sqlite.md:20-25 (file BLAKE3 2d2a36a95aa4…):
## Consequences

Installation is one binary and one file. Only one process can write at a
time, so background jobs queue their writes through the main service instead
of opening their own connections. Full-text search uses SQLite FTS5 rather
than an external search server.
```

## Benchmark and evaluation tools

The other files here are harnesses for [Benchmarks](../docs/benchmarks.md) and
[Development](../docs/development.md), not usage examples: `bench_*`,
`eval_*`, `image_quant_eval`, `photo_anchor_latency`,
`photo_calibration_export` and `queries.example.txt`.
