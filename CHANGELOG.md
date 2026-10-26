# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

Before 1.0, a minor release (0.x) may contain breaking changes. They are marked
**Breaking** below.

## [Unreleased]

### Added

- Hybrid search: one query now combines meaning, keywords (BM25) and file names,
  so both a question and `garden.md` find the right file. Each hit shows which
  signals matched, as CLI tags and the `matched` MCP field.
  ([#5](https://github.com/CertaMesh/seyk/pull/5),
  [#7](https://github.com/CertaMesh/seyk/pull/7))
- `seyk extract` returns the exact readable text behind a search hit, verified
  against the source hash; `seyk search --no-preview` skips preview reads.
  ([#10](https://github.com/CertaMesh/seyk/pull/10))
- PDF search: text PDFs are searchable page by page with verified previews and
  extraction, and scanned pages are found through image embeddings
  (`--text-only` skips them). ([#12](https://github.com/CertaMesh/seyk/pull/12),
  [#17](https://github.com/CertaMesh/seyk/pull/17))
- Audio search for WAV, FLAC, MP3 and M4A recordings, returning time ranges as
  evidence. Recordings up to two hours are streamed and verified chunk by chunk;
  `--max-audio-duration-secs` changes the limit.
  ([#32](https://github.com/CertaMesh/seyk/pull/32),
  [#37](https://github.com/CertaMesh/seyk/pull/37))
- Offline speech-to-text: `seyk extract` can turn a selected audio hit into a
  verified machine transcript. Search and indexing never run speech-to-text.
  ([#38](https://github.com/CertaMesh/seyk/pull/38))
- `seyk setup audio` downloads and SHA-256 verifies the pinned audio and
  speech-to-text models, reusing valid files and resuming interrupted downloads.
  `seyk setup` shows offline model status; `--dry-run` and `--json` print the
  plan. ([#88](https://github.com/CertaMesh/seyk/pull/88))
- Lexical fallback: while the model loads, files wait for embeddings, or weights
  are missing, search still returns keyword and file-name hits, marked with
  `degraded` metadata. Library callers can use `SearchEngine::lexical()`.
  ([#55](https://github.com/CertaMesh/seyk/pull/55))
- Each hit carries a `match` field with its semantic, keyword and file-name
  ranks and the query terms the keyword index confirmed.
  ([#53](https://github.com/CertaMesh/seyk/pull/53))
- Result-kind counts in search output, per-root indexing progress with an ETA in
  daemon status, and a new MCP `index_status` tool.
  ([#57](https://github.com/CertaMesh/seyk/pull/57))
- `seyk index <root> --json --progress jsonl` streams live progress and ETA on
  stderr for tools that run Seyk as a child process. See [the progress
  protocol](docs/index-progress.md).
  ([#79](https://github.com/CertaMesh/seyk/pull/79))
- `seyk mcp --local-content`: an opt-in confined mode for automatic document
  collections that skips hidden, private, cache and generated paths and never
  joins the shared daemon. ([#50](https://github.com/CertaMesh/seyk/pull/50),
  [#51](https://github.com/CertaMesh/seyk/pull/51))
- `seyk cache import` seeds a fresh cache from an existing one when an unchanged
  collection moves to new paths, after verifying every file's hash.
  ([#77](https://github.com/CertaMesh/seyk/pull/77))
- `seyk model profile --json` reports the exact index profile without loading
  weights or opening a cache. New library APIs: `gemma::model_identity`,
  `gemma::missing_model_files` and `SearchEngineBuilder::cache_profile`.
  ([#82](https://github.com/CertaMesh/seyk/pull/82))
- `seyk extract --typed-errors` opts into stable exit codes (1 and 3 to 11) with
  one `error: <stable_code>` line on stderr. Default behaviour is unchanged. See
  the [exit-code table](docs/cli.md#typed-extraction-errors).
  ([#85](https://github.com/CertaMesh/seyk/pull/85))
- Experimental `--image-scan q4` (and `ImageScan::PhotoAnchorQ4`) shortlists
  standalone images with compact codes, then rescores them exactly. Exact f32
  scoring stays the default. ([#64](https://github.com/CertaMesh/seyk/pull/64),
  [#78](https://github.com/CertaMesh/seyk/pull/78))
- Opt-in CPU tuning for text indexing: `--sessions`, `--threads`, `--batch-size`
  and `--embedding-window`. Defaults are unchanged.
  ([#34](https://github.com/CertaMesh/seyk/pull/34))
- `seyk --version` marks non-release builds with `+g<short-sha>` (and `.dirty`
  for modified trees); `seyk cache stats` shows the cache schema version.
  ([#80](https://github.com/CertaMesh/seyk/pull/80))
- Reusable embedding crates `seyk-embed` and `seyk-embeddinggemma` let Rust
  projects use EmbeddingGemma 2 without the search engine. They are available as
  Git dependencies, not on crates.io.
  ([#11](https://github.com/CertaMesh/seyk/pull/11))
- Release archives include `THIRD_PARTY_LICENSES.html` with all third-party
  licence notices, including ONNX Runtime's.
  ([#86](https://github.com/CertaMesh/seyk/pull/86))

### Changed

- **Breaking:** search defaults to hybrid mode; `--mode semantic` gives the
  0.1.0 behaviour. Hybrid fusion now uses k=5, so a strong meaning match is no
  longer overtaken by a weak hit that only shares a keyword.
  ([#5](https://github.com/CertaMesh/seyk/pull/5),
  [#69](https://github.com/CertaMesh/seyk/pull/69))
- **Breaking:** JSON output carries a `schema_version`, now 9, and identifies
  resources by file URI instead of a local path. Later versions added PDF fields
  (3), speech-to-text (4), cache backup status (5), typed skip reasons (6),
  result counts and indexing progress (7), lexical fallback metadata (8) and
  image-code counts (9). `seyk extract` still accepts older search output.
  ([#9](https://github.com/CertaMesh/seyk/pull/9),
  [#12](https://github.com/CertaMesh/seyk/pull/12),
  [#38](https://github.com/CertaMesh/seyk/pull/38),
  [#43](https://github.com/CertaMesh/seyk/pull/43),
  [#47](https://github.com/CertaMesh/seyk/pull/47),
  [#55](https://github.com/CertaMesh/seyk/pull/55),
  [#57](https://github.com/CertaMesh/seyk/pull/57),
  [#78](https://github.com/CertaMesh/seyk/pull/78))
- **Breaking:** the cache schema is now v7. Opening an older cache upgrades it
  in place and keeps existing embeddings; some upgrades re-read files once
  without re-embedding them. The v7 upgrade is one-way: older binaries refuse a
  v7 cache, so give them a separate cache path.
  ([#5](https://github.com/CertaMesh/seyk/pull/5),
  [#9](https://github.com/CertaMesh/seyk/pull/9),
  [#12](https://github.com/CertaMesh/seyk/pull/12),
  [#33](https://github.com/CertaMesh/seyk/pull/33),
  [#78](https://github.com/CertaMesh/seyk/pull/78))
- **Breaking (library):** `BruteForceIndex::new` and `with_capacity` return a
  `Result`, `Embedder::input_identity` returns a `Result`, and the tokenizer
  helper `token_ids` is split into `text_token_ids` and `template_token_ids`.
  ([#27](https://github.com/CertaMesh/seyk/pull/27),
  [#44](https://github.com/CertaMesh/seyk/pull/44),
  [#62](https://github.com/CertaMesh/seyk/pull/62))
- A `--model REPO` without a pinned commit is now tied to the snapshot it
  resolves to, so moving the alias re-embeds instead of mixing vectors. Expect
  one re-embed after upgrading.
  ([#20](https://github.com/CertaMesh/seyk/pull/20))
- Every CLI build includes audio search and speech-to-text. Weights are still
  downloaded only by explicit setup commands, and missing-model messages point
  to `seyk setup audio`. ([#88](https://github.com/CertaMesh/seyk/pull/88))
- Clients from different builds or cache schemas start their own shared daemon
  instead of reusing a mismatched one. Daemon status, logs and MCP server
  metadata report the build version.
  ([#81](https://github.com/CertaMesh/seyk/pull/81))
- Loaded audio models keep their bytes in memory, so a replaced model file
  cannot change a running session. This adds about 650 MiB of peak memory while
  audio is in use. ([#71](https://github.com/CertaMesh/seyk/pull/71))
- Faster searches and startup: text-only collections no longer hash the audio
  encoder, semantic searches skip unneeded PDF loads, keyword hits load each
  file once, and cached local-content queries skip folder walks.
  ([#18](https://github.com/CertaMesh/seyk/pull/18),
  [#26](https://github.com/CertaMesh/seyk/pull/26),
  [#28](https://github.com/CertaMesh/seyk/pull/28),
  [#44](https://github.com/CertaMesh/seyk/pull/44),
  [#51](https://github.com/CertaMesh/seyk/pull/51))
- Installation points to `cargo install --git` until prebuilt binaries return
  with the next release. The README is a short overview, and the details live in
  [docs/](docs/README.md). ([#52](https://github.com/CertaMesh/seyk/pull/52),
  [#65](https://github.com/CertaMesh/seyk/pull/65),
  [#84](https://github.com/CertaMesh/seyk/pull/84),
  [#87](https://github.com/CertaMesh/seyk/pull/87))

### Removed

- **Breaking:** the text-only CLI build without audio is gone. Library users can
  still build without audio by selecting the `onnx` feature.
  ([#88](https://github.com/CertaMesh/seyk/pull/88))

### Fixed

- Concurrent indexing passes could drop each other's vectors, leaving files
  missing from semantic results.
  ([#21](https://github.com/CertaMesh/seyk/pull/21))
- Indexing recovers after the file watcher loses events, such as a Linux inotify
  queue overflow. ([#23](https://github.com/CertaMesh/seyk/pull/23))
- MCP searches no longer index inline while another watcher is already indexing
  the same paths. ([#19](https://github.com/CertaMesh/seyk/pull/19))
- The shared daemon no longer shuts down while a session is starting, and late
  clients get a clear reply. ([#22](https://github.com/CertaMesh/seyk/pull/22))
- `seyk mcp` and the daemon start under long HOME or cache paths: an overlong
  socket moves to a private per-user directory under TMPDIR or `/tmp`.
  ([#83](https://github.com/CertaMesh/seyk/pull/83))
- Keyword scores in one cache profile are no longer changed by other profiles or
  by deleted files. ([#33](https://github.com/CertaMesh/seyk/pull/33))
- Malformed PDF font maps no longer discard the readable pages or print panic
  messages; skipped files are reported with typed reasons.
  ([#47](https://github.com/CertaMesh/seyk/pull/47))
- Text containing Gemma control strings such as `<|image|>` or `<bos>` is
  embedded as plain text instead of failing or embedding wrongly.
  ([#59](https://github.com/CertaMesh/seyk/pull/59))
- The segmenter no longer panics on multi-byte characters at a paragraph
  boundary. ([#25](https://github.com/CertaMesh/seyk/pull/25))
- Corrupt cached vectors and malformed output from custom embedding backends are
  rejected instead of being cached or silently truncated.
  ([#16](https://github.com/CertaMesh/seyk/pull/16),
  [#27](https://github.com/CertaMesh/seyk/pull/27))
- Concurrent model setup can no longer truncate the cached model weights.
  ([#48](https://github.com/CertaMesh/seyk/pull/48))
- Intermittent `unknown exception in Initialize()` failures on macOS when two
  models load at once. ([#56](https://github.com/CertaMesh/seyk/pull/56))
- Local-content mode works on cold Linux automounts.
  ([#74](https://github.com/CertaMesh/seyk/pull/74))

### Security

- Secret files are never indexed: SSH keys, `.env` files, keychains, password
  stores and credential paths are denied everywhere, including explicit paths,
  and old cache entries for them are hidden. See [Files Seyk never
  indexes](docs/supported-content.md#files-seyk-never-indexes).
  ([#54](https://github.com/CertaMesh/seyk/pull/54))
- Deleted cache content is erased from SQLite free pages, keyword postings and
  old WAL frames. ([#42](https://github.com/CertaMesh/seyk/pull/42))
- The keyword index stores words without their original order.
  ([#6](https://github.com/CertaMesh/seyk/pull/6))
- Seyk-owned cache directories get a `CACHEDIR.TAG` and, on macOS, a Time
  Machine exclusion; Seyk warns when a cache sits in a synced folder.
  ([#43](https://github.com/CertaMesh/seyk/pull/43))
- Folder walks and file reads go through pinned directory handles without
  following symlinks, so swapping a folder for a link mid-walk cannot expose
  files outside the allowed roots. Confined MCP paths are normalized and
  revalidated. ([#60](https://github.com/CertaMesh/seyk/pull/60),
  [#61](https://github.com/CertaMesh/seyk/pull/61),
  [#63](https://github.com/CertaMesh/seyk/pull/63),
  [#73](https://github.com/CertaMesh/seyk/pull/73))
- Content reads refuse iCloud placeholders, dataless files and FIFOs before any
  parser sees them. ([#49](https://github.com/CertaMesh/seyk/pull/49))
- The daemon socket fallback directory is refused unless it is a real directory
  owned by the user with mode 0700.
  ([#83](https://github.com/CertaMesh/seyk/pull/83))

### Development

- The toolchain is pinned to Rust 1.99.0, with a monthly check against the
  newest stable; the minimum supported Rust stays 1.89.
  ([#4](https://github.com/CertaMesh/seyk/pull/4))
- A public retrieval benchmark harness (SciFact, NFCorpus, IRPAPERS) with
  competitor comparisons, documented in
  [docs/benchmarks.md](docs/benchmarks.md).
  ([#36](https://github.com/CertaMesh/seyk/pull/36),
  [#58](https://github.com/CertaMesh/seyk/pull/58),
  [#68](https://github.com/CertaMesh/seyk/pull/68))
- Contributor guide, code of conduct, security policy and project north star
  added for the open-source release.
  ([#8](https://github.com/CertaMesh/seyk/pull/8),
  [#66](https://github.com/CertaMesh/seyk/pull/66))

## [0.1.0] - 2026-10-07

First release. Its prebuilt binaries were later withdrawn because they lacked
third-party licence notices; build this version from source.

- Local semantic search over text files and images with EmbeddingGemma 2, as a
  Rust library and the `seyk` CLI.
- Asymmetric query and document prompts, paragraph-level segments, Markdown link
  text, and 768/512/256/128 dimensions (256 by default).
- `seyk search` with several phrasings per query (`-q`), include/exclude globs,
  `--min-score`, `--json` and `--no-cache`.
- A SQLite embedding cache, `seyk index` with `--watch`, and `seyk cache stats`
  / `clear`.
- `seyk mcp`, a stdio MCP server for agents with background indexing, sharing
  one daemon across sessions.
- `seyk model download` fetches pinned q8 or fp32 weights; search never uses the
  network.
- Prebuilt binaries for macOS (Apple Silicon), Linux and Windows, with an
  install script.

[Unreleased]: https://github.com/CertaMesh/seyk/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CertaMesh/seyk/releases/tag/v0.1.0
