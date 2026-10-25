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

For raw text, extract writes the exact bytes without an added newline. Text and PDF
text extraction need no model. Audio speech-to-text (STT) is opt-in and labelled (see below).
Extraction needs no cache or network. Input must be a complete search JSON envelope
at `schema_version: 2`, `3`, `4`, `5`, `6`, `7` or `8`, not a bare hit. The extraction document uses
version 8 and contains `text`, `source`, `locator`, `metadata`, and optional extraction provenance;
`metadata.content_hash` is the hash of the **whole source**, not just the passage.
`--no-preview` skips preview reads and sets `preview` to null; indexing still reads
source content.

Raw text ranges and PDF page-text ranges are supported. Audio `time` ranges are
supported in builds with `audio-stt` (see
[Speech-to-text extraction](#speech-to-text-extraction-opt-in)). Images,
whole-resource and other locators return an unsupported-extraction error. MCP has an explicit `extract`
tool that takes a selected MCP hit, verifies the hash and confines reads to the
session roots. Search never transcribes.

## Audio (opt-in)

Build with `cargo build --release --features audio`, then provision the extra
q8 encoder with `seyk model download --audio` (about 325 MiB). This command only
downloads the two pinned audio files; provision the ordinary model separately.
Default release builds do not enable `audio`.

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

## Speech-to-text extraction (opt-in)

Build with `cargo build --release --features audio-stt`, then explicitly run
`seyk model download --stt` to install the pinned Xenova Whisper tiny FP32
model (about 255 MiB). Inference stays offline. A local exact asset directory
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
- **name**: the query names the file (`father.md`, `father` or `notes/father.md`).

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
Without weights, it builds an in-memory keyword index of text and PDF text without
embedding. Search responses mark this as
`"degraded": {"mode": "lexical_only", "reason": "no_model"}`. A requested semantic
search also falls back and carries this marker; cosine thresholds do not filter
keyword matches. Images, scanned PDF pages and audio require their normal models.

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
