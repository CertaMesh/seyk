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

Only explicit model setup commands use the network. `seyk model download` fetches the ONNX export
`onnx-community/embeddinggemma-2-ONNX`, pinned to commit `daa72c5`, into the
standard Hugging Face cache. Searching never downloads anything: if the text weights are
missing, search returns keyword and filename matches with a `no_model` degradation marker
and CLI exit status 0. Run `model download` to enable semantic search. Indexing still
requires text weights.

If only vision encoder files are missing, `search`, `index`, `index --watch`, and
`mcp` keep semantic text search working without `--text-only`. Images and scanned
PDF pages are skipped with diagnostics listing the missing files and
`seyk model download`. PDF pages with usable embedded text remain searchable.
The CLI warns once when loading the model; MCP exposes the warning in
`index_status.model_diagnostics` after the model loads. No `no_model` marker is
reported for this condition. Explicit `--text-only` disables images without a warning.

In this mode, the active profile ID matches `--text-only` for the same text model.
Cached text vectors remain reusable only for identical embedding inputs under that
profile. `model profile` reports this effective profile while `model_present` and
`missing_files` still describe all files requested by the original options.

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

The JSON object uses the current CLI `schema_version` (10), with `profile_id`
(the exact `CacheProfile::id()` string) and `profile` (the full descriptor):

```json
{
  "schema_version": 10,
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

## Audio in one command

```sh
seyk setup audio                   # audio encoder + Whisper tiny STT
seyk search "repairing a bicycle" ./recordings
```

The default CLI includes audio decoding, audio search and speech-to-text.
Weights stay outside the binary: nothing downloads at build time, startup or
search time. The ordinary text model is still provisioned separately with
`seyk model download` and is needed for semantic queries.

`seyk setup audio` first prints the pinned repositories, revisions, sizes and
maximum remaining download size. It explicitly fetches the q8 audio encoder
(340,307,954 bytes) and Whisper tiny STT (267,743,552 bytes): 608,051,506 bytes
in total, about 580 MiB. Existing files are SHA-256 verified and valid files are
reused. Downloads are staged privately and all fetched files pass size and SHA-256
checks before publication. Audio uses a fixed private stage, so hf-hub can resume
partial downloads after interruption; real setup removes older random audio
stages. The stage is protected by an OS lock and is deleted after a successful
installation. Newly published audio and STT asset files use mode 0644 on Unix;
STT file modes previously depended on umask. Staging remains private.
Corrupt or incomplete managed installations are repaired; unmanaged STT
directories are preserved. Verification streams files without loading inference
sessions. The audio hashes and lengths come from the HF LFS metadata at
`daa72c51243991dfcaf9f9137d2c573d8f7790c0`; STT uses its existing checked manifest.

```sh
seyk setup                         # offline text, audio and STT status
seyk setup --json
seyk setup audio --dry-run --json   # offline plan; no downloads
seyk setup audio --json             # install; versioned completion on stdout
```

JSON uses the existing schema v9 envelope, with `action` (`status`, `plan` or
`complete`), `models`, `total_bytes`, `download_bytes` and `next_step`. Each
model reports `present`, `verification`, `bytes`, `repo` and `revision`.
Text status checks file presence only (`bytes: null`); audio and STT status check
SHA-256. The size fields cover the audio bundle; `download_bytes` is an upper
bound because partially verified models can reuse files. Download plans and
progress go to stderr so JSON stdout remains parseable. Failures exit nonzero.
`seyk model download --audio` and `--stt` remain available for separate installs.
Library consumers can use `default-features = false` and opt into `audio` or
`audio-stt` explicitly; enabling `cli` includes both.

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
[Audio](cli.md#audio) and
[Speech-to-text extraction](cli.md#speech-to-text-extraction).
