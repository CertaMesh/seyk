# Model setup

Seyk needs local EmbeddingGemma 2 weights for semantic search. This page covers
downloading them, using weights you already have, and inspecting the active
index profile.

Download the weights once (about 0.5 GB for the default 8-bit text and vision models):

```bash
seyk model download            # 8-bit weights (default)
seyk model download --precision fp32   # full precision, ~1.8 GB
seyk model info                # where they are, whether they are present
seyk model profile --text-only --json  # active index profile, without loading weights
```

`model download` is the only command that uses the network. It fetches the ONNX export
`onnx-community/embeddinggemma-2-ONNX`, pinned to commit `daa72c5`, into the
standard Hugging Face cache. Searching never downloads anything: if the weights are
missing, search returns keyword and filename matches with a `no_model` degradation marker
and CLI exit status 0. Run `model download` to enable semantic search. Indexing still
requires weights.

To use weights you already have, pass a directory with the same layout
(`tokenizer.json`, `onnx/model_quantized.onnx` + `.onnx_data`, …):

```bash
seyk search "…" ~/notes --model /path/to/embeddinggemma-2-ONNX
```

Precision: `q8` (default) or `fp32`. There is deliberately no `fp16`: the model card
warns that EmbeddingGemma 2 overflows float16 and silently returns NaN or degraded
vectors.

`seyk model profile --json` reports the profile that `index` uses with the same
model options (`--model`, `--dimension`, `--precision`, and `--text-only`). It does
not load an inference session, download files, or open or create an embedding cache.
`--cache-path` and `SEYK_CACHE` are accepted but ignored. Without `--json`, it
prints the ID and descriptor.

The JSON object uses the current CLI `schema_version` (9), with `profile_id`
(the exact `CacheProfile::id()` string) and `profile` (the full descriptor):

```json
{
  "schema_version": 9,
  "profile_id": "<24-character hash>",
  "profile": {
    "model": {
      "model_id": "google/embeddinggemma-2",
      "revision": "<repo@resolved-commit or local:fingerprint>",
      "variant": "onnx-q8;text-literals-v1",
      "dimension": 256,
      "normalization": "mean-pool;truncate;l2",
      "query_prompt": "task: search result | query: ",
      "document_prompt": "title: {title} | text: {text}",
      "max_tokens": 2048
    },
    "segmenter_version": "<segmenter and input-format version>"
  },
  "model_present": true,
  "missing_files": []
}
```

`model_present` reports whether the required files checked by model resolution
exist. `missing_files` lists absent relative file names. These checks use
filesystem metadata only. They do not validate model contents or run inference.
The ID is available before download, even when `model_present` is false.

Hugging Face sources resolve local refs when present, otherwise use the requested
revision. Use a pinned commit for an ID that stays stable across downloads.
Local directories must contain the selected model files: identity reads graph and
tokenizer bytes and external weight file sizes, as indexing does.

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
  [Supported content](supported-content.md).

Audio and speech-to-text need extra model files. See
[Audio](cli.md#audio-opt-in) and
[Speech-to-text extraction](cli.md#speech-to-text-extraction-opt-in).
