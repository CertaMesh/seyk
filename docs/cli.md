# CLI reference

The `seyk` binary is a thin layer over the Rust library. Run `seyk --help` or
`seyk <command> --help` for the full option list of each command.

## Common commands

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

# Live resource progress on stderr, final JSON report on stdout.
seyk index ~/notes --json --progress jsonl

# Keep the index fresh: index once, then re-index whatever changes
seyk index --watch ~/notes

# Serve search to AI agents over MCP (see mcp.md)
seyk mcp ~/notes

# Inspect or wipe the cache
seyk cache stats
seyk cache scrub  # erase deleted postings/WAL frames, preserving live records
seyk cache clear

# Seed a fresh epoch cache without loading a model
seyk cache import --from old.sqlite --map mapping.jsonl --profile PROFILE_ID --cache-path new.sqlite --json
```

See [the standalone progress protocol](index-progress.md) for the JSONL schema.

To move an indexed collection into a new directory, see
[Move an index with cache import](caching-and-storage.md#move-an-index-with-cache-import).

## Search flags

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

## Keyword search

The default `hybrid` mode combines meaning, BM25 keyword relevance, and file-name
matches. Use `lexical` to rank exact terms, identifiers, and names without
embedding the query:

```sh
seyk search "INC-18372" ~/notes --mode lexical
seyk search "Session::restore" ~/code --mode lexical
```

The CLI still loads a model to index new or changed files; lexical query ranking
does not make this a model-free indexing mode. See [scores and tags](#scores-and-tags)
for how the three match types are combined.

## Extract a search hit

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

For stable runtime failure codes, add `--typed-errors` with either raw output or
`--json`. Without the flag, failures retain exit 1 and prose. See
[Typed extraction errors](#typed-extraction-errors) for the code table and stream contract.

For raw text, extract writes the exact bytes without an added newline. Text and PDF
text extraction need no model. Audio speech-to-text (STT) is on-demand and labelled (see below).
Extraction needs no cache or network. Input must be a complete search JSON envelope
at `schema_version: 2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` or `10`, not a bare hit. The extraction document uses
version 10 and contains `text`, `source`, `locator`, `metadata`, and optional extraction provenance;
`metadata.content_hash` is the hash of the **whole source**, not just the passage.
`--no-preview` skips preview reads and sets `preview` to null; indexing still reads
source content.

Raw text ranges and PDF page-text ranges are supported. Audio `time` ranges are
supported in the default CLI (see
[Speech-to-text extraction](#speech-to-text-extraction)). Images,
whole-resource and other locators return an unsupported-extraction error. MCP has an explicit `extract`
tool that takes a selected MCP hit, verifies the hash and confines reads to the
session roots. Search never transcribes.

### Typed extraction errors

`seyk extract --typed-errors --from-json hits.json [--hit INDEX] [--json]`
enables stable exit codes for runtime failures. Without `--typed-errors`,
returned failures retain exit code 1 and the existing prose diagnostics.
Success output is identical in both modes, including raw text, the audio
transcript label, and versioned JSON. The flag does not change the JSON schema.
The current schema version is 9, and extract accepts search envelopes from
versions 2 through 9.

In typed mode, a returned runtime failure writes exactly one stderr line:

```text
error: <stable_code>
```

The code is a fixed string from the table below. The line contains no paths,
filenames, source bytes, hashes, model locations, parser payloads, or free-form
reasons. There is no JSON error document on either stream.

| Exit | Stable code | Meaning |
| --- | --- | --- |
| 0 | None | Success. Help and version also exit 0 but are informational. |
| 1 | `unknown_failure` | Any unlisted error, including future core variants, configuration, cache, embedding, and general model errors. |
| 2 | Clap diagnostic | Invalid arguments or usage. Clap retains its own diagnostic format. |
| 3 | `invalid_input` | Invalid JSON syntax or search results, absent or unsupported schema version, noninteger schema version, or missing selected hit. |
| 4 | `resource_changed` | `StaleResource` or `ChangedWhileReading`. |
| 5 | `unsupported_extraction` | `UnsupportedExtraction` or `PdfOcrUnavailable`. |
| 6 | `invalid_evidence` | `InvalidLocator`, `InvalidText`, or `AudioIntervalTooLong`. |
| 7 | `io_failed` | Core I/O, input-envelope open/read, or stdout write/flush failure, including broken pipes. |
| 8 | `policy_denied` | `SecretFile`. |
| 9 | `invalid_media` | `UnsupportedAudio`, `PdfParse`, `PdfPageDecode`, `ParserPanicked`, or `PdfRender`. |
| 10 | `stt_model_unavailable` | `SttModel`. |
| 11 | `transcription_failed` | `SpeechToText`. |

Classification uses typed errors, never their displayed text. An enclosing
input or core error takes priority over its lower-level I/O source. A PDF
page-text hash or extraction-version mismatch is an `InvalidLocator`, so it
exits 6 rather than 4.

Input and extraction failures leave stdout empty. A stdout write or flush
failure can leave partial output. Consumers must require exit 0 and valid
expected output before using any bytes.

Clap usage diagnostics may echo arguments and are outside the fixed runtime
line contract. Signals, aborts, uncaught panics, and failure to write stderr
are also outside that contract. Consumers must reject nonzero exits or missing
valid output even when no typed line arrives. No model is downloaded to
classify an error.

## Audio

Audio search and speech-to-text are included in the default CLI. Provision both
pinned models in one command:

```sh
seyk setup audio
```

The command prints the download plan first, verifies checksums and skips valid
files. `seyk setup` shows offline status for every model; use
`seyk setup audio --dry-run --json` for a machine-readable plan. See
[Audio in one command](model-setup.md#audio-in-one-command) for sizes and details.
Provision the ordinary text model separately with `seyk model download`.

`seyk search "repairing a bicycle" ./recordings` embeds sound directly, with no
transcription. WAV, FLAC, MP3 and AAC-LC/ALAC in M4A are decoded locally into
10-second windows every 8 seconds. Results use the existing half-open `time`
locator (`start_ms`, `end_ms`) and have no transcript or preview. Explicit audio
extraction requires `audio-stt` (see below); search needs no STT model.
Missing audio weights skip individual audio files; text, images and PDFs still
work. Search never downloads weights.

The default duration ceiling is two hours, configurable with
`--max-audio-duration-secs`. Decoding and windowing use bounded memory and never
hold whole-file PCM in RAM. Over-limit sources are skipped entirely. Complex MP4
edits, multiple tracks, HE-AAC/SBR and changing sample formats are rejected. MP4 edit
lists or iTunSMPB remove encoder priming/padding; files without timing metadata
use their decoded-sample timeline. See the [audio report](architecture.md#phase-5a-audio-retrieval)
for timing, parity and retrieval measurements.

## Speech-to-text extraction

`seyk setup audio` installs the pinned Xenova Whisper tiny FP32 model
(about 255 MiB); `seyk model download --stt` still installs it separately. Inference stays offline. A local exact asset directory
can be selected with `--stt-model-dir` or `SEYK_STT_MODEL_DIR`; it receives the
same manifest check, including rejection of unexpected files.

Extract an audio hit with `seyk extract --from-json hits.json --hit 0 --json`.
Each half-open interval is capped at 30 seconds; longer intervals return a
typed error. Language is detected from that hit, or set with `--stt-language en`.
Human output says **Machine transcript (may contain errors)**. JSON includes
machine-transcription provenance, verified sample bounds, model manifest hash,
language and `no_speech`; it does not claim word-level timestamps. Silence has
no detected language. Transcripts are never cached or logged, and search,
indexing and previews never invoke STT. See the
[implementation and model gate](architecture.md#phase-5b-2-on-demand-speech-to-text-extraction).

## PDFs

PDF text and scanned-page behaviour is described in
[Supported content: PDFs](supported-content.md#pdfs).

## Scores and tags

In the default **hybrid** mode, three lists are ranked separately and merged by
position (reciprocal rank fusion):

- **meaning**: cosine similarity between the question and each passage;
- **words**: keyword relevance (BM25) over the text, heading, file name and path;
- **name**: the query names the file (`garden.md`, `garden` or `notes/garden.md`).

The hybrid score depends only on **where a hit ranked in each list**, scaled so
1.000 means first everywhere something was found. It says nothing about how
good the match is in absolute terms: with three unrelated notes, the third one
still scores close to 1. **The tag shows the strength**: `meaning 0.70` is the
cosine, `words` means the keywords matched and `name` means the file name did.

Keywords count fully for names, identifiers (`INC-18372`, `Session::restore()`)
and one- or two-word queries. In a question written in prose they count less
(0.3), so incidental shared words only break ties between passages of similar
meaning.

With `--mode semantic`, the score is the cosine itself (the old behaviour). Use
it to rank within one search; it is not a calibrated probability, and unrelated
text rarely scores near zero, so pick a `--min-score` by looking at your own
results. `--mode lexical` never embeds the query. The CLI still loads the model
and uses it to index new or changed files before searching when weights are available.
Without text weights, it builds an in-memory keyword index of text and PDF text without
embedding. Search responses mark this as
`"degraded": {"mode": "lexical_only", "reason": "no_model"}`. A requested semantic
search also falls back and carries this marker; cosine thresholds do not filter
keyword matches. Missing vision files alone preserve semantic text search, warn
once on stderr, and skip images and scanned PDF pages with diagnostics naming the
missing files and `seyk model download`. The effective cache profile matches
`--text-only`; cached text vectors remain reusable for identical inputs under that
profile. Audio still requires its normal models.

## Result counts and indexing progress

Opt in with `seyk search "query" ./folder --json --kind-counts`, MCP search
`kind_counts: true`, or the Rust builder `.kind_counts(true)`. The optional
`kind_counts` map counts **returned segment hits**, after filtering, ranking and
limit. Keys are the existing locator kinds: `text`, `page_text` (PDF text),
`page` (rendered/scanned PDF page), `time` (audio window), `whole_resource`
(e.g. an image), and reserved `image_region`. Absent kinds have zero hits;
an empty result has `{}`. These are not unique-file counts or corpus totals.
Counting adds no candidate-total query and leaves ranking and candidate depth unchanged.

Daemon status (including `--json`), MCP `index_status`, and MCP search expose
`index_progress`: one entry per watched root, with `root` (a lossless file URI)
and nullable `progress`.
The `folders` and `index.state` fields are unchanged. Library callers get
`ProgressEvent::IndexProgress` during an indexing pass and its final snapshot in
`SyncStats::progress`. A library request spanning several roots gets one combined
pass snapshot; daemon watchers each index one root. Before `Discovered`, a pass
also sends `ProgressEvent::Walked` with the roots its listing actually walked,
resolved once before the walk, when the provider reports them through
`ResourceProvider::list_walked` (`FsProvider` does; the default reports none).
MCP sessions see a watcher's counts only when that walked root lies inside the
session's pinned root; a pass without one exposes no counts.

`progress.by_kind` uses resource kinds `text`, `pdf`, `image`, and `audio`.
Each entry counts resources `indexed`, `waiting`, and `skipped` in the current
or last completed discovery pass, **not a persistent cache inventory**. All
resources initially wait; successful cache reuse or a committed update moves
them to indexed, and observed whole-resource failures move them to skipped.
An empty/partially searchable PDF committed successfully counts as indexed;
individual failed pages remain in page diagnostics. Counts reset on rediscovery,
so removed files disappear. Skips describe this pass only; skipped-version counts
are not stored and do not survive a process restart.

`eta_seconds` is `ceil(waiting_resources * elapsed_pass_seconds /
newly_indexed_resources)`, using this process's current pass only. It stays null
until at least two newly indexed resources complete and at least 1 ms elapses,
and returns to null when nothing waits. Cache reuse and skips do not train the
rate. This rough resource-throughput estimate includes discovery-following
preparation/embedding/commit time and assumes the remaining resource mix has
similar cost; it is not a segment-based prediction or a countdown. Snapshots
update at resource completions. A large first PDF/audio resource can therefore
keep ETA null until later resources finish. Failed/stopped watchers invalidate
the snapshot. Loading/unavailable progress is null. A session confined to a child
of a watched root receives null progress for its own root rather than revealing
the parent's inventory. Status never starts indexing or loads a model.
