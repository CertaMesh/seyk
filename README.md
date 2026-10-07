<p align="center">
  <img src="docs/assets/logo.png" width="128" alt="Seyk logo">
</p>

# Seyk

Local semantic search over your own files, built on Google's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).

Ask a question in your own words, point it at a folder, and get back the passages
that mean the same thing, even when they share no words with the question:

```console
$ seyk search "grief as something physical" ~/Writing/Book -n 2
0.766  ideas/inheritance.txt:3
       Grief: after the funeral her chest felt packed with wet sand. She could not lift her arms to hang up her coat.

0.665  ideas/inheritance.txt:1
       Things passed down without anyone choosing: a temper, a way of folding shirts, the silence at dinner.

Searched 6 segments: 3 files (3 unchanged, 0 new or changed), 0 segments embedded in 0.3s
```

(Real output from a small test folder. The best match shares no words with the
question except "grief".)

It is a Rust library first. The `seyk` CLI is a thin layer over it, so other
applications can embed the same engine.

What it is **not**: there is no chat, no answer generation, no summaries, no OCR and
no transcription. It finds relevant resources and points at the exact place inside
them. What happens next is up to the application.

## Why EmbeddingGemma 2

- **Small and local.** The text part is 270M parameters. It runs on a laptop CPU,
  so your notes never leave the machine.
- **Asymmetric retrieval.** Queries and documents get different task prompts
  (`task: search result | query: …` and `title: … | text: …`). That is what lets a
  question match a passage that answers it.
- **Matryoshka dimensions.** Vectors can be cut to 512, 256 or 128 numbers with
  little quality loss. The default is 256 (MTEB multilingual 60.41 vs 61.36 at 768,
  per the model card), a third of the storage.
- **One space for text, code, images, audio and video.** A text query can find a
  photo of a whiteboard. Text and images work today; see
  [Supported resources](#supported-resources).

## Installation

Prebuilt binaries for macOS (Apple Silicon), Linux (x86_64 and ARM64) and
Windows (x86_64) are attached to each
[release](https://github.com/CertaMesh/seyk/releases). While the repository is
private, install with the GitHub CLI:

```bash
gh api repos/CertaMesh/seyk/contents/scripts/install.sh \
  -H "Accept: application/vnd.github.raw" | sh
```

The script picks your platform, checks the SHA-256 checksum and installs to
`~/.local/bin` (set `SEYK_INSTALL_DIR` to change that). There is no Intel Mac
build, because ONNX Runtime publishes no prebuilt library for it.

From source (Rust 1.88 or newer):

```bash
cargo install --locked --git https://github.com/CertaMesh/seyk
```

The binary is self-contained. ONNX Runtime is linked statically; its prebuilt
library is downloaded by the `ort` crate **at build time**. For a fully offline
build, set `ORT_SKIP_DOWNLOAD=1` and point `ORT_LIB_LOCATION` at a local ONNX
Runtime build.

## Model setup

Download the weights once (about 0.5 GB for the default 8-bit text and vision models):

```bash
seyk model download            # 8-bit weights (default)
seyk model download --precision fp32   # full precision, ~1.8 GB
seyk model info                # where they are, whether they are present
```

This is the only command that uses the network. It fetches the ONNX export
`onnx-community/embeddinggemma-2-ONNX`, pinned to commit `daa72c5`, into the
standard Hugging Face cache. Searching never downloads anything: if the weights are
missing it stops and tells you to run `model download`.

To use weights you already have, pass a directory with the same layout
(`tokenizer.json`, `onnx/model_quantized.onnx` + `.onnx_data`, …):

```bash
seyk search "…" ~/notes --model /path/to/embeddinggemma-2-ONNX
```

Precision: `q8` (default) or `fp32`. There is deliberately no `fp16`: the model card
warns that EmbeddingGemma 2 overflows float16 and silently returns NaN or degraded
vectors.

## CLI

```bash
# Search a folder (recursively). Embeddings are cached as you go.
seyk search "fear of becoming like his father" ~/Writing/Book

# Several paths, files or folders
seyk search "ideas involving memory and identity" notes.md characters/

# Several phrasings of the same need; each result keeps its best score
seyk search "who is the main character" -q "protagonist of the story" ~/Writing/Book

# Only Markdown, skip drafts, top 5, hide weak matches
seyk search "grief as something physical" ~/Writing \
  --include '*.md' --exclude drafts --limit 5 --min-score 0.55

# Nothing written to disk
seyk search "query" ./notes --no-cache

# Machine-readable output (mirrors the library's SearchResults type)
seyk search "query" ./notes --json

# Optional: embed everything up front so later searches are instant
seyk index ~/Writing/Book

# Keep the index fresh: index once, then re-index whatever changes
seyk index --watch ~/Writing/Book

# Serve search to AI agents over MCP (see "Agents" below)
seyk mcp ~/Writing/Book

# Inspect or wipe the cache
seyk cache stats
seyk cache clear
```

| Flag | Meaning |
|---|---|
| `-n, --limit` | Number of results (default 10) |
| `-q, --also` | Another phrasing to search for too (repeatable) |
| `--min-score` | Hide results below this cosine similarity |
| `--dimension` | 768, 512, 256 (default) or 128 |
| `--precision` | `q8` (default) or `fp32` |
| `--include` / `--exclude` | Globs; a pattern without `/` matches the file name |
| `--hidden` | Also search dotfiles and dot-directories |
| `--no-cache` | Embed in memory only |
| `--json` | JSON output |
| `--text-only` | Skip loading the vision encoder; images are skipped |
| `--model` | Model directory or `REPO[@REVISION]` |
| `--cache-path` | Use a different cache file |

Scores are cosine similarities. Use them to rank results within one search. They
are not calibrated probabilities, and unrelated text rarely scores near zero, so
pick a `--min-score` by looking at your own results.

## Agents (MCP)

`seyk mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server
over stdio. Agents get one tool, `search`, that takes one to eight phrasings and
returns ranked passages as path, line range, heading, score and preview. The agent
then reads those lines itself and writes the answer. Seyk never generates text.

```bash
# Claude Code: make your notes searchable in every session
claude mcp add --scope user seyk -- seyk mcp ~/Writing/Book

# Or per project, for a codebase
claude mcp add seyk -- seyk mcp .
```

- **The model stays loaded.** After the first call, a search takes well under a
  second on a warm cache.
- **Indexing runs in the background.** On start the server indexes its folders,
  then re-indexes on change. Searches never wait for it: they answer from what is
  already indexed and report `index.state` (`loading_model`, `indexing` with
  progress, `ready`, `pass_failed` or `stopped`).
- **Confined to its folders.** The tool refuses paths outside the folders the
  server was started with.
- **For code, it complements grep.** Use Seyk for "where do we retry failed
  payments?" and grep for `retryPayment(`.

**Privacy.** Seyk itself sends nothing anywhere, but a cloud-hosted agent sends
the passages it reads to its provider. Point the server only at folders you are
happy for that agent to read.

## Library

```rust
use seyk::embedder::gemma::{GemmaConfig, GemmaEmbedder};
use seyk::{Dimension, ResourceLocator, SearchEngine, SqliteCache};

let embedder = GemmaEmbedder::load(GemmaConfig {
    dimension: Dimension::D256,
    ..Default::default()
})?;

let engine = SearchEngine::builder()
    .embedder(embedder)
    .cache(SqliteCache::open_default()?) // omit for an in-memory, ephemeral cache
    .build()?;

let results = engine
    .search("where did I discuss inherited behaviour?")
    .paths(["~/Writing/Book"])
    .exclude("drafts")
    .limit(10)
    .execute()?;

for hit in &results.hits {
    match &hit.locator {
        ResourceLocator::Text { line_start, line_end, .. } => {
            println!("{:.3} {}:{line_start:?}-{line_end:?}", hit.score, hit.resource.path().display())
        }
        ResourceLocator::WholeResource => println!("{:.3} {}", hit.score, hit.resource.path().display()),
        other => println!("{:.3} {} {other:?}", hit.score, hit.resource.path().display()),
    }
}

// Optional cache warming
engine.index().paths(["~/Writing/Book"]).execute()?;
```

Each `SearchHit` carries the resource path, a modality-independent
`ResourceLocator` (`Text` byte and line range, `Page`, `Time`, or `WholeResource`),
the score, the resource metadata (modality, size, mtime, content hash), the
Markdown heading path and an optional short preview.

Other backends plug in through the `Embedder` trait:

```rust
pub trait Embedder: Send + Sync {
    fn identity(&self) -> &ModelIdentity;          // persisted with every cached vector
    fn supports(&self, modality: Modality) -> bool;
    fn embed_query(&self, query: &str) -> Result<Embedding>;
    fn embed_resources(&self, inputs: &[ResourceInput<'_>]) -> Result<Vec<Embedding>>;
}
```

`ResourceInput` already has `Text`, `Image`, `Audio` and `Video` variants. An
embedder declares what it supports, and the engine skips the rest with a reason
instead of guessing.

Cargo features: `onnx` (EmbeddingGemma 2 backend), `image` (image resources) and
`cli` (the binary) are on by default. A library user who only wants the engine with
their own embedder can use `default-features = false`.

## How caching works

The original files stay the source of truth. Nothing is copied: the cache holds
only file metadata, segment locations and vectors.

There are three ways to search:

1. **Lazy cache (default).** For each file: if size and modification time match
   the cache, its vectors are reused without reading it. Otherwise the file is read
   and hashed. If the content hash is unchanged only the mtime is updated.
   Otherwise it is re-segmented, and only segments whose exact model input is new
   are sent to the model. Vectors are saved after every batch, so an interrupted
   first search keeps its progress.
2. **Ephemeral (`--no-cache`).** Everything happens in memory and is dropped.
3. **Explicit indexing (`index`).** Runs the same update ahead of time and also
   removes cache entries for files deleted under the indexed folders. It is an
   optimisation, never a requirement.
4. **Watching (`index --watch`, and `mcp`).** Indexes once, then repeats the same
   incremental update about 1.5 s after files stop changing. Changes inside
   `.git/`, `target/`, `node_modules/` and similar folders are ignored.

Stale vectors cannot be reused. Every vector is stored under a *profile*: a hash of
the model id, exact weights revision, precision, dimension, normalisation, both
prompt templates, maximum input length and segmenter version. Within a profile it
is keyed by a hash of the exact text sent to the model. Change any of these and
the old vectors are simply never looked up. `cache stats` lists the profiles.

## Where things are stored

| What | Where (macOS) | Where (Linux) |
|---|---|---|
| Embedding cache | `~/Library/Caches/seyk/cache.sqlite3` | `~/.cache/seyk/cache.sqlite3` |
| Model weights | `~/.cache/huggingface/hub/` (or `$HF_HOME/hub`) | same |
| Hard links to the ONNX files (a copy only if the cache is on another disk) | `~/Library/Caches/seyk/models/` | `~/.cache/seyk/models/` |
| Configuration | none: flags plus `SEYK_MODEL`, `SEYK_CACHE` | same |

The hard links exist because ONNX Runtime refuses weight files that live behind the
Hugging Face cache's symlinks.

**Privacy.** Files, queries and embeddings never leave the machine. Treat the cache
as sensitive: embeddings can leak information about the text they came from. The
cache directory is created `0700` and the file `0600`. `seyk cache clear`
deletes every vector.

## Supported resources

| Kind | Extensions | Status |
|---|---|---|
| Markdown | `.md` `.markdown` `.mdown` `.mkd` | ✅ heading-aware segments |
| Plain text | `.txt` `.text` `.rst` `.org` `.adoc` `.tex` | ✅ paragraph segments |
| Source code | `.rs` `.py` `.js` `.ts` `.go` `.java` `.c` `.cpp` `.rb` `.php` `.sh` `.sql` `.toml` `.yaml` and more | ✅ blank-line blocks |
| Images | `.png` `.jpg` `.jpeg` `.webp` `.gif` `.bmp` `.tiff` | ✅ one vector per image (`WholeResource`) |
| Audio | — | ❌ not implemented |
| Video | — | ❌ not implemented |
| PDF | — | ❌ not implemented (`Page` locator reserved) |

A file passed explicitly on the command line is searched as plain text even with
an unknown extension. Directories skip `.git`, `target`, `node_modules`, `.hg`,
`.svn`, `__pycache__`, `.venv` and hidden entries by default.

### How text is segmented

Embedding a whole file would average all its ideas into one vector, and a query
matching one of them would score poorly. So text is split along its own structure:

- Markdown headings start a new segment and stay with their content. The heading
  path (`Part One > The Kitchen`) is sent to the model as the document title, as the
  model card recommends. Lines inside fenced code blocks are never taken for
  headings.
- Every paragraph of 60 characters or more is its own segment. Shorter fragments
  (labels, one-line notes) merge with their neighbours, up to about 1,000
  characters (~250 tokens).
- Paragraphs longer than 1,800 characters (~450 tokens) are cut at sentence ends,
  and the next piece repeats the last sentence. There is no overlap across paragraph
  breaks you chose yourself. Fenced code blocks count as one paragraph.

Sizes are counted in bytes, which equals characters for English text. Accented
and non-Latin text therefore gets somewhat shorter segments.

This uses a small fraction of the model's 8K context on purpose: shorter, focused
segments retrieve better and give precise locations. Each segment keeps its exact
byte range and line numbers.

### Images

Images go through the EmbeddingGemma 2 vision encoder with the default 280-token
budget and come back as `WholeResource` hits, so a text query like
"diagram of boxes connected by lines" finds the right picture. The Rust
preprocessing is a port of the Gemma 4 image processor. See
[Correctness checks](#correctness-checks).

## Correctness checks

The Rust/ONNX path is compared against the official `sentence-transformers`
implementation in float32 (`tests/parity.rs`, `tests/multimodal.rs`):

| Check | Result |
|---|---|
| Text token ids (with prompts) | identical |
| Text vectors, fp32, at 768/512/256/128 | worst cosine 1.00000 |
| Text vectors, q8 | worst cosine 0.99994 |
| Image resize size, patch positions, token ids | identical |
| Image vectors, fp32 | worst cosine 0.99999 |
| Image vectors, q8 | worst cosine 0.9979 |

The remaining image difference comes from the resize filter (a different bicubic
implementation than torchvision's). It changes individual pixels by at most 0.06.

## Testing

```bash
cargo test                         # no model needed: segmentation, cache, CLI, ranking
cargo test --release -- --ignored  # also the model-backed parity and retrieval tests
```

Model-backed tests read `SEYK_MODEL` (local model dir) and
`SEYK_TEST_PRECISION` (`q8` or `fp32`). The image parity tests also need a
reference dump from `scripts/reference_image_embeddings.py` (see the test for the
environment variables).

## Evaluating on your own notes

`examples/eval_corpus.rs` runs a list of questions against a folder and prints the
semantic hits next to a naive keyword baseline. It also flags hits that share no
content words with the question, which is the material only semantic search can find:

```bash
cargo run --release --example eval_corpus -- \
  --queries examples/queries.example.txt ~/Writing/Book
```

Everything runs locally.

## Performance

See [docs/benchmarks.md](docs/benchmarks.md) for measured numbers and how to reproduce them.

## Known limitations

- **Literal matches still pull.** A passage that *shows* a theme without naming it
  (a story whose details keep shifting, for "memory becoming unreliable") can rank
  below an unrelated note that shares a word ("memory card … unreliable"). See
  [docs/benchmarks.md](docs/benchmarks.md#retrieval-smoke-test).
- **First search of a large folder is slow.** Embedding on the CPU managed roughly
  5–6 segments per second on a busy machine, about 30 minutes for 10,000
  segments. After that, only changes are embedded. Run `index --watch` or the MCP
  server so this happens in the background.

- **Audio, video and PDF are not implemented.** See
  [docs/multimodal.md](docs/multimodal.md) for what is missing.
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
  lock rather than run in parallel. Two MCP servers started on the same new
  folder may both embed it during the first index; identical inputs are stored
  once.
- **Setext headings** (underlined with `===`/`---`) are not recognised; ATX
  headings (`#`) are.

## Architecture

```text
paths ─► discovery ─► segmentation ─► Embedder (EmbeddingGemma 2 / ONNX) ─► cache ─► similarity ─► SearchHit
         (walkdir,      (Markdown,       query: SearchQuery prompt            (SQLite or  (brute-force   (path, locator,
          globs)         paragraphs,     docs:  Document prompt + title        memory)     cosine,        score, metadata,
                         images)         truncate → L2-normalise                           VectorIndex)   preview)
```

| Module | Responsibility |
|---|---|
| `discovery` | Walk paths, apply excludes and globs, classify files |
| `segment` | Turn text into segments with exact byte and line ranges |
| `embedder` | `Embedder` trait and `ResourceInput`; `embedder::gemma` is the EmbeddingGemma 2 backend |
| `cache` | `EmbeddingCache` trait, `SqliteCache`, `MemoryCache`, profiles |
| `index` | `VectorIndex` trait and `BruteForceIndex` |
| `engine` | `SearchEngine`: lazy sync, ranking, previews, `index` |
| `watch` | Re-index on file changes (`index --watch`, MCP background indexing) |
| `src/bin/seyk/main.rs` | Argument parsing and formatting only |
| `src/bin/seyk/mcp.rs` | The MCP server: tool schema, root confinement, background indexing |

## License

Apache-2.0. EmbeddingGemma 2 is released by Google under Apache 2.0.
