<p align="center">
  <img src="docs/assets/logo.png" width="128" alt="Seyk logo">
</p>

# Seyk

Local semantic search over your own files, built on Google's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).

Ask a question in your own words, or type an exact name, and get back the
passages that match. Meaning, keywords and file names are searched together:

```console
$ seyk search "turning into my dad" ~/notes -n 1
1.000  father.md:1-3  § Father  [meaning 0.70]
       He used to slam the door when he was angry. Last week I caught myself doing exactly the same thing, and my son flinched the way I used to.

$ seyk search "INC-18372" ~/notes -n 1
1.000  work.md:1-3  § Work  [meaning 0.76+words]
       Ship the pricing page before the conference. INC-18372 is still open: the old page lists the beta plan.

$ seyk search "father.md" ~/notes -n 1
1.000  father.md:1-3  § Father  [meaning 0.74+words+name]
       He used to slam the door when he was angry. Last week I caught myself doing exactly the same thing, and my son flinched the way I used to.
```

(Real output from a three-note folder. The first match shares no words with the
question; the tag in brackets says what found each hit.)

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
`~/.local/bin` (set `SEYK_INSTALL_DIR` to change that).

- **Linux needs glibc 2.38 or newer** (Ubuntu 24.04, Debian 13, Fedora 39 and
  later), because ONNX Runtime's prebuilt library does.
- **There is no Intel Mac build**, because ONNX Runtime publishes no prebuilt
  library for it.

From source (Rust 1.89 or newer):

```bash
cargo install --locked --git https://github.com/CertaMesh/seyk
```

The binary is self-contained. ONNX Runtime is linked statically; its prebuilt
library is downloaded by the `ort` crate **at build time**. For a fully offline
build, set `ORT_SKIP_DOWNLOAD=1` and point `ORT_LIB_LOCATION` at a local ONNX
Runtime build.

Older Seyk binaries reset caches written by newer versions. When running old and
new binaries side by side, use a separate `--cache-path` for each.

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
seyk search "how I fixed my sleep" ~/notes

# Several paths, files or folders
seyk search "ideas for the pricing page" ideas.md projects/

# Several phrasings of the same need; each result keeps its best score
seyk search "where do we retry failed payments" -q "payment retry logic" ~/code/shop

# Only Markdown, skip drafts, top 5, hide weak matches
seyk search "advice I got about hiring" ~/notes \
  --include '*.md' --exclude drafts --limit 5 --min-score 0.55

# Nothing written to disk
seyk search "query" ./notes --no-cache

# Machine-readable output (mirrors the library's SearchResults type)
seyk search "query" ./notes --json

# Optional: embed everything up front so later searches are instant
seyk index ~/notes

# Keep the index fresh: index once, then re-index whatever changes
seyk index --watch ~/notes

# Serve search to AI agents over MCP (see "Agents" below)
seyk mcp ~/notes

# Inspect or wipe the cache
seyk cache stats
seyk cache clear
```

| Flag | Meaning |
|---|---|
| `-n, --limit` | Number of results (default 10) |
| `-q, --also` | Another phrasing to search for too (repeatable) |
| `--mode` | `hybrid` (default: meaning + keywords + file names), `semantic` or `lexical` |
| `--min-score` | Hide meaning matches below this cosine similarity (keyword and name matches are kept) |
| `--dimension` | 768, 512, 256 (default) or 128 |
| `--precision` | `q8` (default) or `fp32` |
| `--include` / `--exclude` | Globs; a pattern without `/` matches the file name |
| `--hidden` | Also search dotfiles and dot-directories |
| `--no-cache` | Embed in memory only |
| `--no-preview` | Skip preview reads; JSON hits retain `"preview": null` |
| `--json` | JSON output |
| `--text-only` | Skip loading the vision encoder; images are skipped |
| `--model` | Model directory or `REPO[@REVISION]` |
| `--cache-path` | Use a different cache file |

### Extract a search hit

Search locates evidence; extraction reads a chosen hit and verifies its full
resource hash before returning the exact UTF-8 byte range. It preserves Markdown,
code, whitespace and line endings. If the source changed, extraction fails with
an error instead of returning a different passage.

```sh
seyk search "planting basil" ./notes --json --no-preview > hits.json
seyk extract --from-json hits.json --hit 0 --json

# Or pipe search output directly; the default hit index is zero.
seyk search "planting basil" ./notes --json --no-preview -n 1 |
  seyk extract --from-json - --json
```

Without `--json`, extract writes only the exact text, with no added newline.
It needs no model, cache or network. Input must be a complete search JSON envelope
at `schema_version: 2`, not a bare hit. The new extraction document also uses
version 2 and contains `text`, `source`, `locator` and `metadata`;
`metadata.content_hash` is the hash of the **whole source**, not just the passage.
Existing search documents keep their v2 shape: `--no-preview` sets `preview` to
null. This flag skips preview reads; indexing still reads source content.

Only text ranges are supported. Images, whole-resource and other locators return
an unsupported-extraction error. MCP extraction is deferred until MCP hits can
carry exact locators and hashes with session-scoped read checks; MCP search and
its root confinement are unchanged.

### Scores and tags

In the default **hybrid** mode, three lists are ranked separately and merged by
position (reciprocal rank fusion):

- **meaning**: cosine similarity between the question and each passage;
- **words**: keyword relevance (BM25) over the text, heading, file name and path;
- **name**: the query names the file (`father.md`, `father` or `notes/father.md`).

The hybrid score only depends on **where a hit ranked in each list**, scaled so
1.000 means first everywhere something was found. It says nothing about how
good the match is in absolute terms: with three unrelated notes, the third one
still scores close to 1. **The tag shows the strength**: `meaning 0.70` is the
cosine, `words` means the keywords matched and `name` means the file name did.

Keywords count fully for names, identifiers (`INC-18372`, `Session::restore()`)
and one- or two-word queries. In a question written in prose they count less
(0.3), so incidental shared words only break ties between passages of similar
meaning.

`--mode semantic` gives the old behaviour: the score is the cosine itself. Use
it to rank within one search; it is not a calibrated probability, and unrelated
text rarely scores near zero, so pick a `--min-score` by looking at your own
results. `--mode lexical` never embeds the query. The CLI still loads the model
and uses it to index new or changed files before searching.

## Agents (MCP)

`seyk mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server
over stdio. Agents get one tool, `search`, that takes one to eight phrasings and
returns ranked passages as path, line range, heading, score, preview and
`matched` (`meaning`, `words`, `name`). An optional `mode` switches between
`hybrid` (default), `semantic` and `lexical`. The agent
then reads those lines itself and writes the answer. Seyk never generates text.

```bash
# Claude Code: make your notes searchable in every session
claude mcp add --scope user seyk -- seyk mcp ~/notes

# Or per project, for a codebase
claude mcp add seyk -- seyk mcp .
```

- **One model for every session.** `seyk mcp` is a small relay (about 9 MB).
  The first one starts a shared background daemon that loads the model once,
  owns the cache and watches every folder any session asked for. Five agent
  sessions cost one model's memory, not five. The daemon exits after 15 minutes
  without sessions, and the next `seyk mcp` starts it again.
- **The model stays loaded.** After the first call, a search takes well under a
  second on a warm cache.
- **Indexing runs in the background.** On start the server indexes its folders,
  then re-indexes on change. Searches never wait for it: they answer from what is
  already indexed and report `index.state` (`loading_model`, `indexing` with
  progress, `ready`, `pass_failed` or `stopped`).
- **Confined to its folders.** The tool refuses paths outside the folders the
  server was started with.
- **For code, it complements grep.** Hybrid search finds `retryPayment` by name
  too, but grep is still the right tool for every exact occurrence of `retryPayment(`.

```bash
seyk daemon status    # pid, sessions, and each watched folder's index state
seyk daemon stop      # connected sessions lose the tool until they restart
seyk mcp --standalone ~/notes   # no daemon: one process per session
```

Sessions only share a daemon when they use the same model options
(`--model`, `--dimension`, `--precision`, `--text-only`, `--cache-path`) and the
same Seyk version, so an upgraded binary never talks to an old daemon. The
daemon is Unix-only; on Windows `seyk mcp` always runs standalone.

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
    .search("what did we decide about pricing?")
    .paths(["~/notes"])
    .exclude("drafts")
    .limit(10)
    .execute()?;

for hit in &results.hits {
    match &hit.locator {
        ResourceLocator::Text { line_start, line_end, .. } => {
            println!("{:.3} {}:{line_start:?}-{line_end:?}", hit.score, hit.resource)
        }
        ResourceLocator::WholeResource => println!("{:.3} {}", hit.score, hit.resource),
        other => println!("{:.3} {} {other:?}", hit.score, hit.resource),
    }
}

// Optional cache warming
engine.index().paths(["~/notes"]).execute()?;
```

Each `SearchHit` carries a resource URI, a modality-independent
`ResourceLocator` (`Text` byte and line range, `Page`, `Time`, `ImageRegion`, or
`WholeResource`),
the score, the resource metadata (modality, size, mtime, content hash), the
Markdown heading path and an optional short preview.

Explicit extraction in Rust:

```rust
let results = engine.search("planting basil")
    .paths(["./notes"])
    .previews(None)
    .execute()?;
if let Some(hit) = results.hits.first() {
    let content = engine.extract(hit)?;
    print!("{}", content.text);
}
```

`SearchEngine::builder().extractor(custom)` registers a synchronous
`Extractor: Send + Sync` before the default `TextExtractor`. Custom extractors
receive full, hash-verified bytes; the first supporting extractor wins and its
errors propagate. For model-free library extraction, use
`extract_resource(&provider, &TextExtractor, &hit.resource, &hit.locator,
&hit.metadata)`. Direct low-level `Extractor::extract` calls require the caller
to verify bytes first. Neither entry point adds an authorization layer.

Resource identity migration (Phase 2): replace `hit.resource.path()` with
`hit.resource.to_file_path()`, which returns `Option<PathBuf>`. Use the URI's
`Display` implementation when displaying any resource, as in the example above;
use the optional path when opening a local file. `ResourceId::from_path(path)?`
now requires an absolute path and returns a `Result`. Request `.paths(...)`
still accepts relative paths and `~/...`.

CLI JSON objects and MCP search responses now include `schema_version: 2`.
A search hit's `resource` is a URI (`file:///notes/my%20note.md`); MCP also keeps
an explicit decoded `path`. URI identity preserves non-UTF-8 Unix path bytes;
the JSON path is a display aid and can contain replacement characters.
See the [Phase 2 report](docs/architecture.md#phase-2-report-resource-identity-and-providers)
for all API changes and cache upgrade details.

The engine uses `ResourceProvider` for enumeration, reads, freshness and deletion
checks. `FsProvider` is the default. `ResourceProvider`, `ResourceScope`,
`ProvidedResource` and `ResourceVersion` may change before 1.0; filesystem is the
only supported provider. There are no remote connectors.

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

The original files stay the source of truth. The cache holds file metadata,
segment locations, vectors and a keyword index (see the privacy note below).

There are three ways to search:

1. **Lazy cache (default).** For each file: if size and modification time match
   the cache, its vectors are reused without reading it. Otherwise the file is read
   and hashed. If the content hash is unchanged only its metadata and version are updated.
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
| Daemon socket and log (`seyk mcp`) | `~/Library/Caches/seyk/daemon-*.sock`, `daemon.log` | `~/.cache/seyk/` |
| Configuration | none: flags plus `SEYK_MODEL`, `SEYK_CACHE` | same |

The hard links exist because ONNX Runtime refuses weight files that live behind the
Hugging Face cache's symlinks.

**Privacy.** Files, queries and embeddings never leave the machine. Treat the cache
as sensitive. Embeddings can leak information about the text they came from, and
the keyword index (a SQLite FTS5 table) stores **which words occur in each passage
and how often, not their order**. Each field is tokenized and sorted before it
is indexed. These word frequencies are still sensitive, like embeddings. The
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
  path (`Projects > Pricing`) is sent to the model as the document title, as the
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
  --queries examples/queries.example.txt ~/notes
```

Everything runs locally.

## Performance

See [docs/benchmarks.md](docs/benchmarks.md) for measured numbers and how to reproduce them.

## Known limitations

- **Literal matches still pull.** A passage that *shows* a theme without naming it
  (a story whose details keep shifting, for "memory becoming unreliable") can rank
  below an unrelated note that shares a word ("memory card … unreliable"). See
  [docs/benchmarks.md](docs/benchmarks.md#retrieval-smoke-test).
- **Hybrid can push a good meaning match down.** On a real folder of book drafts,
  hybrid search moved 7 of 8 target passages into the top 10 (semantic
  alone: 6 of 8), but one poetic query fell from 19th to 25th because keyword
  hits crowded it out. `--mode semantic` is there for those cases.
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
  lock rather than run in parallel. MCP sessions avoid this by sharing the daemon;
  a separate `seyk index --watch` on the same folder still duplicates its work.
- **Searches can wait for an indexing batch.** The daemon runs one model, so a
  query embedded while a large first index is running waits for the current
  batch (a few seconds on a busy CPU).
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
| `cache` | `EmbeddingCache` trait, `SqliteCache`, `MemoryCache`, profiles, keyword index |
| `lexical` | Tokenizer and keyword query (BM25) shared by both caches |
| `index` | `VectorIndex` trait and `BruteForceIndex` |
| `engine` | `SearchEngine`: lazy sync, hybrid ranking (rank fusion), previews, `index` |
| `watch` | Re-index on file changes (`index --watch`, MCP background indexing) |
| `src/bin/seyk/main.rs` | Argument parsing and formatting only |
| `src/bin/seyk/mcp.rs` | The MCP server: tool schema, root confinement, background indexing |

## License

Apache-2.0. EmbeddingGemma 2 is released by Google under Apache 2.0.
