# Standalone embeddings and the 0.2 API migration

Seyk's embedding layer is reusable independently of search. The crates are not
on crates.io yet; all workspace packages currently have version `0.2.0-dev`.

```toml
[dependencies]
seyk-embed = { git = "https://github.com/CertaMesh/seyk" }
seyk-embeddinggemma = { git = "https://github.com/CertaMesh/seyk" }
```

For reproducible applications, pin both dependencies to the same Git `rev`.
`seyk-embed` owns the model-independent interface. `seyk-embeddinggemma` implements
it using synchronous local ONNX CPU inference. Neither depends on Seyk search.
There is no network access during model loading or embedding. Async applications
can run inference in their executor's blocking pool; no Tokio runtime is required.

## Loading a model offline

```rust
use seyk_embeddinggemma::EmbeddingGemma2;

let model = EmbeddingGemma2::load_default()?;
// Or choose a local snapshot and output width explicitly:
let model = EmbeddingGemma2::builder()
    .model_path("/models/embeddinggemma-2-ONNX")
    .dimension(256)
    .vision(false)
    .build()?;
```

The default source is `onnx-community/embeddinggemma-2-ONNX`, revision
`daa72c51243991dfcaf9f9137d2c573d8f7790c0`. Cached snapshots resolve under
`$HF_HOME/hub`, or `~/.cache/huggingface/hub` when `HF_HOME` is unset, including
`refs/` resolution for named revisions. A local directory has the same layout:

```text
model-directory/
  tokenizer.json
  onnx/
    model_quantized.onnx
    model_quantized.onnx_data
    vision_encoder_quantized.onnx       # only when vision is enabled
    vision_encoder_quantized.onnx_data  # only when vision is enabled
```

Use the unsuffixed filenames for `Precision::Fp32`. Defaults remain q8, 256
output dimensions, 2048 tokens, and batches of 16. The builder also accepts
`precision`, `max_tokens`, `batch_size` and `threads`. Gemma accepts only
128/256/512/768 dimensions; truncation is followed by L2 normalization.

Provision files before loading, through the Hugging Face client or the explicit
`seyk model download` command. The compatibility facade also retains
`seyk::embedder::gemma::download(&config)`. The standalone backend deliberately
has no downloader: keeping hf-hub outside it avoids HTTP/url dependencies.
The model's terms and license apply to the weights separately from this code.

ONNX Runtime's prebuilt native library may be fetched **at build time** by `ort`.
For an offline build, provision that library too and set `ORT_SKIP_DOWNLOAD=1`
and `ORT_LIB_LOCATION` to its location. This is separate from offline inference.
Runtime-specific types are internal. Existing HF external-data staging uses
hard links, with a copy fallback across filesystems.

## Tasks, inputs and batches

```rust
use seyk_embed::{Embedder, EmbeddingRequest, EmbeddingTask, ResourceInput};

let query = model.embed_text("account access", EmbeddingTask::SearchQuery)?;
let requests = [
    EmbeddingRequest {
        input: ResourceInput::Text { title: Some("Help"), text: "Reset a password." },
        task: EmbeddingTask::SearchDocument,
    },
    EmbeddingRequest {
        input: ResourceInput::Text { title: None, text: "A billing question." },
        task: EmbeddingTask::Classification,
    },
];
let vectors = model.embed_batch(&requests)?;
```

Successful batches return one vector per request in the original order. Empty
batches return no vectors. Text tokenization and inference batch internally,
sorting by length and restoring order afterward. Image inference is currently
one image at a time. Session mutexes serialize inference; this API does not
promise parallel calls increase throughput.

The backend implements the eight distinct prompt meanings in the
[pinned official prompt configuration](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/blob/daa72c51243991dfcaf9f9137d2c573d8f7790c0/config_sentence_transformers.json):

| Task | Exact prefix |
|---|---|
| `SearchQuery` | `task: search result \| query: ` |
| `SearchDocument` | `title: none \| text: ` |
| `Classification` | `task: classification \| query: ` |
| `Clustering` | `task: clustering \| query: ` |
| `CodeRetrieval` | `task: code retrieval \| query: ` |
| `FactChecking` | `task: fact checking \| query: ` |
| `QuestionAnswering` | `task: question answering \| query: ` |
| `SentenceSimilarity` | `task: sentence similarity \| query: ` |

For documents, a supplied title replaces `none`, with the existing CR/LF
sanitization and empty-title fallback. Other text tasks ignore the title.
Rustdoc on `EmbeddingGemma2::text_input` lists the official alias names.
`QuestionAnswering` and aliases such as Summarization select embeddings; they
never generate prose. There is no arbitrary Custom prompt variant.

`ResourceInput` borrows text, encoded image bytes, PCM audio or decoded video
frames. Audio/video return `UnsupportedModality`; no implementation is implied.
For images, opt into preprocessing and provision the vision weights:

```toml
seyk-embeddinggemma = { git = "https://github.com/CertaMesh/seyk", features = ["image"] }
```

`image` is the backend's only optional feature. Without it the default is text
only; with it the default config enables vision, which can be disabled with
`.vision(false)`. Images use `SearchDocument` and the existing unprefixed media
path. Other image tasks return `UnsupportedTask`. Text/image matching uses
`SearchQuery` text and `SearchDocument` images in the shared vector space.

## Vectors and helpers

`Embedding` keeps values private, with `as_slice()`, `into_vec()`, `dim()` /
`dimension()` and `normalized()`. Its dimension is derived from its values.
`from_raw` truncates and normalizes, rejecting short or nonfinite output. A zero
vector remains zero and reports `normalized() == false`; cosine similarity for
zero vectors is undefined and the checked helper returns an error.

`dot_product`, `cosine_similarity` and `normalize` are small helpers, not a
numerical framework. `dot_product` rejects dimension mismatch and nonfinite
results. `Embedding::cosine` retains the existing fast dot-product convenience
for same-width normalized embeddings. Model metadata lives once in
`ModelIdentity`, not redundantly on every vector; tasks/modalities are on requests.

`Dimension` now represents any nonzero width, with the existing D128/D256/D512/
D768 constants, `ALL`, `get()`, default and numeric serialization. A generic
backend can use other widths. The Gemma builder validates its narrower support.
Matches over the old enum may need a fallback; the generic parser now accepts
other nonzero dimensions, while Gemma still rejects them before loading.

## Examples without search

Run after provisioning weights:

```bash
cargo run -p seyk-embeddinggemma --example similarity
cargo run -p seyk-embeddinggemma --example clustering
cargo run -p seyk-embeddinggemma --example classification
cargo run -p seyk-embeddinggemma --features image --example image_text -- picture.png 'a red circle'
```

Clustering demonstrates a small spherical k-means routine over batched notes.
Routing chooses the nearest embedded label using Classification; label wording
and the application domain affect quality. These examples need no search engine,
SQLite, resource discovery or agent integration.

## Migrating existing library code

`use seyk::SearchEngine` and search module paths remain supported. The root
`seyk` package is the compatibility facade and CLI/MCP composition package.
`seyk::embedder::gemma::{GemmaConfig, GemmaEmbedder, ...}` remains available under
`onnx`; `GemmaEmbedder` is an alias for `EmbeddingGemma2`. Source installation
still uses `cargo install --locked --git https://github.com/CertaMesh/seyk seyk`.
The binary remains `seyk`, and the existing command without the package name
still resolves the repository's only binary package.

Custom embedders now implement `embed_batch(&[EmbeddingRequest])` returning
`seyk_embed::Result<Vec<Embedding>>`, along with `identity()` and `supports()`.
The required trait method is object-safe and forwarded by Box/Arc. Its default
`embed`, `embed_text`, `embed_query`, `embed_resources` and `embed_text_document`
methods supply requests; custom implementations should not call these adapters
recursively from `embed_batch`. Search uses SearchQuery/SearchDocument explicitly.

When embedding errors enter core, existing variants remain `Error::Embedding`,
`ModelNotFound`, `UnsupportedModality`, `DimensionMismatch`, `Config`, and `Io`.
Only the new unsupported task case adds `UnsupportedTask`. Direct backend calls
return `seyk_embed::Error`; `?` converts it into `seyk::Error` in existing functions
returning `seyk::Result`. Enums are non-exhaustive; downstream matches need a
fallback for future variants.

### Cache writes require explicit lexical input

`CachedSegment` now has only `locator`, `heading` and `input_hash`.
`CachedResource` and cache reads remain evidence-only. To replace a resource,
provide `ResourceWrite` containing paired `SegmentWrite` inputs:

```rust
use seyk::cache::{ResourceWrite, SegmentWrite};

let write = ResourceWrite {
    metadata: &metadata,
    version: &version,
    descriptor_hash: Some(&descriptor_hash),
    segments: vec![SegmentWrite {
        segment: &segment,
        lexical_body: original_source_text,
    }],
};
cache.put_resource(profile_id, &resource_id, &write)?;
```

Text bodies are the original source, including link targets; image/name-only
writes explicitly use `""`. A cache read cannot be passed as complete ingestion
input, and no optional body silently defaults to empty. Each cache consumes the
borrowed body for the existing order-free keyword index and retains only evidence.
MemoryCache continues to permit writes before profile registration; SQLite
continues to reject them atomically through its foreign key constraint.

This split keeps schema v4, profile JSON/hash, model identity values, input hashes,
vector bytes, tokenization, field weights, ranking and extraction behavior.
Existing valid vectors require no migration or re-embedding. Older Seyk binaries
can reset caches written by newer ones; use separate `--cache-path` values when
running both versions.

## Dependency boundaries

`seyk-embed` has only serde and thiserror as normal direct dependencies.
`seyk-embeddinggemma` adds ort, tokenizers (onig, no HTTP feature), dirs, blake3,
and optional image/fast_image_resize. The heavy components are ONNX Runtime,
tokenizer/regex code, and optional image codecs. Core depends only on the
embedding abstraction; its tests use mock embedders without a backend dependency.

CI checks `cargo tree -p seyk-embed` and
`cargo tree -p seyk-embeddinggemma -e normal`, including the image feature.
Neither tree may contain clap, rusqlite, walkdir, globset, notify, rmcp, tokio,
schemars or url. `scripts/check_embedding_dependencies.py` also ensures core
has no Gemma, ort, tokenizers or hf-hub dependency.
