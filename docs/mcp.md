# Use Seyk from an agent (MCP)

`seyk mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server
over stdio. Agents get three tools:

- `search` takes one to eight phrasings and returns ranked passages with path,
  line range, heading, score, preview and `matched` (`meaning`, `words`, `name`).
  An optional `mode` selects `hybrid` (default), `semantic` or `lexical` search.
- `extract` reads verified evidence from a selected hit. Pass `{"hit": hit}` or
  `{"ref": hit.ref, "path": hit.path}`. A reference that omits its URI needs `path`.
- `index_status` reports indexing progress, per-root counts and model diagnostics.

The agent reads the evidence and writes the answer. Seyk never generates text.

## Search results and extraction

JSON protocol 11 returns compact hits by default. `detail` accepts `"slim"`
(default) or `"full"`; null and other values are invalid. For example:

```json
{"queries":["keeping plants alive in heat"],"detail":"full"}
```

A slim hit has required `ref`, `score` and `matched`, plus optional `path`,
`lines`, `page`, `time`, `heading` and `preview`. Absent optional fields are
omitted. `lines` is a 1-based inclusive string such as `"12"` or `"12-19"`;
`page` is a 1-based physical page number. Audio `time` contains integer
`start_ms` and `end_ms`, not word alignment. Image hits can have none of these
location display fields. `matched` lists `meaning`, `words` and/or `name`.

lines reflect the indexed version; call extract for verified text

The opaque `ref` retains the exact locator, resource identity and original
metadata. Search never calls OCR or speech-to-text. The reference is
self-contained and survives process restarts and cache deletion; extraction
still checks current access policy and the indexed whole-resource hash.
Encoding is not encryption, authentication or proof of a previous search.

`detail:"full"` adds `resource`, `locator`, `metadata`, `match` and
`matched_query` with their existing semantics. Full hits also contain `ref`
and the audio `time` display field. Added fields can break strict parsers.

The default response contains `schema_version`, `hits`, requested
`kind_counts`, and `degraded`, `index` and `index_warnings` when present.
`kind_counts` describes returned hits after the limit. Full responses also
retain `index_progress`, `segments_searched`, `files`, `skipped`,
`page_diagnostics`, `page_diagnostics_total`, `index_diagnostics`,
`segments_embedded` and `took_ms`.

`degraded` reports semantic fallback. `index` retains loading, indexing,
ready, failed-pass and stopped states; combined state reflects the least
settled authorized root. An omitted `index` means no active watcher, not
proof of freshness.

`index_warnings`, when present, is exactly
`{"skipped_resources":N,"unindexed_pages":N}`. Counts are distinct resource
identities and distinct resource/page pairs from this query's synchronization,
confined before counting. Repeated reasons or overlapping watcher histories
do not increase these counts. Both members are present; the object is omitted
when both are zero. Unknown page inventories do not contribute invented pages.
Cached-only queries can have no sync diagnostics even while the index is
partial. Zero warnings does not certify completeness. Watcher history stays in
full results; `index_status` retains its existing fields. Last successful-pass
diagnostics can outlive a failed pass, whose `index` state reports the failure.
There is no new model-readiness flag in compact results.

### Extract selection

The advertised input schema uses `oneOf`: exactly one top-level `hit`, or a
`ref` with optional `path`. Unknown top-level fields, null selections and
combined forms are invalid. Nested hits follow this table:

| Descriptor fields | `ref` | Result |
|---|---|---|
| All of `resource`, `locator`, `metadata`, valid | Absent | Legacy full selection |
| None | Valid | Decode reference; require path if URI omitted |
| All three, valid | Valid | Require equality with decoded descriptor |
| Partial, malformed or null descriptor | Any | Reject |
| None | Absent, null, empty or invalid | Reject |
| Complete descriptor | Present but invalid | Reject |

Clients that round 64-bit metadata integers can pass only `ref` and `path`.
Mixed full/ref inputs must preserve every descriptor field exactly.

There is no fallback from malformed full fields to a reference. A mixed full
hit whose reference omits its URI still requires `path`. Current root checks,
local-content restrictions and source-hash verification apply to every form.
Non-file resources remain unsupported by MCP extraction.

A typical compact hit omits the URI from `ref` because its absolute `path`
reconstructs the resource identity exactly. Copy both fields:

```json
{"ref":"seyk-ref-v1:U1IBAQUBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","path":"/fixture/a"}
```

This fails selection parsing because that same reference requires a path:

```json
{"ref":"seyk-ref-v1:U1IBAQUBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}
```

An explicit-URI reference carries its own identity. The edited display path
below is ignored; this selection still identifies `file:///fixture/a`:

```json
{"ref":"seyk-ref-v1:U1IBABFmaWxlOi8vL2ZpeHR1cmUvYQUBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","path":"/edited/ignored"}
```

These invented examples use a zero hash to illustrate decoding, not successful
extraction. A real reference comes from search and must match the source bytes.
In omitted-URI mode, editing `path` changes the asserted resource identity,
subject to the same access and hash checks as an edited legacy `resource`.
`lines`, `page`, `time` and other display fields never change the decoded span.
The [reference encoding specification](mcp-reference.md) defines flags, bounds
and the complete binary format, including lossless non-UTF8 paths.

### Protocol compatibility

Protocol 11 changes MCP defaults and the shared version label. CLI search,
index and extraction payload fields are unchanged apart from `schema_version`;
Rust library types and the cache schema are unchanged.

| Producer and consumer | Behavior |
|---|---|
| CLI full search v2–10 into v11 CLI extract | Accepted |
| CLI full search v11 into v11 CLI extract | Accepted |
| CLI full search v11 into installed v10 CLI extract | Rejected before source reads, despite identical payload fields |
| v12, missing, null, malformed or unsupported version into v11 CLI extract | Rejected |
| Slim or full MCP response into CLI `extract --from-json` | Rejected; MCP is not CLI `SearchResults` |
| Legacy full MCP hit into new MCP extract | Accepted without a version wrapper |
| New full hit into old permissive MCP `SelectedHit` parser | Old descriptor fields work; the parser ignores added fields |
| New slim hit into old MCP extract | Rejected; descriptor fields are absent |
| Old MCP search caller against new default | Breaking; request `detail:"full"` or adopt slim plus ref |
| Consumer requiring `schema_version == 10` | Must explicitly accept protocol 11 for all machine output |
| Consumer using `deny_unknown_fields` against new full output | Must update its schema for ref/time/warnings; full is not exact-object compatibility |

MCP full output has `matched`, not the CLI `signals` representation, and lacks
the CLI `queries`, `mode` and `stats` envelope. The full opt-in restores field
availability, not version equality or document interchangeability. A new JSON
protocol selects a separate shared daemon endpoint; an older daemon can remain
running for its compatible clients. This change does not freeze extraction
output or introduce a new MCP typed-error contract.

## Set up an agent

Use an absolute directory. `seyk mcp .` indexes the agent's start folder, which
can be `$HOME`, rather than the project you intended.

GUI-launched agents may not have `seyk` on `PATH`. Run `command -v seyk` in your
terminal and use that absolute binary path in place of `seyk` after `--` below.
For example, a Cargo install normally uses `/Users/you/.cargo/bin/seyk` on macOS.
Replace all example paths with your own.

### Claude Code

```bash
# Make your notes searchable in every session
claude mcp add --scope user seyk -- seyk mcp /absolute/path/to/notes

# Or per project, for a codebase
claude mcp add seyk -- seyk mcp /absolute/path/to/codebase
```

### Codex

```bash
codex mcp add seyk -- seyk mcp /absolute/path/to/codebase
```

Codex defaults to 10 s for MCP startup and 60 s per tool call. These limits can
be too short for a cold first search. Raise them in the existing
`[mcp_servers.seyk]` table in `~/.codex/config.toml`, for example:

```toml
[mcp_servers.seyk]
command = "/Users/you/.cargo/bin/seyk"
args = ["mcp", "/absolute/path/to/codebase"]
startup_timeout_sec = 60
tool_timeout_sec = 300
```

Use your absolute binary and directory paths. These example timeouts are not a
latency guarantee; large folders can need longer. See the
[Codex MCP configuration reference](https://developers.openai.com/codex/mcp/#other-configuration-options)
for the timeout settings.

## How sessions work

- **One model for every session.** `seyk mcp` is a small relay (about 9 MB).
  The first one starts a shared background daemon that loads the model once,
  owns the cache and watches every folder any session asked for. Five agent
  sessions cost one model's memory, not five. The daemon exits after 15 minutes
  without sessions, and the next `seyk mcp` starts it again.
- **The model stays loaded.** After the first call, a search takes well under a
  second on a warm cache.
- **Indexing runs in the background.** On start the server indexes its folders,
  then re-indexes on change. While the model loads or discovered files await
  embedding, searches return keyword and filename matches from text and PDF text.
  They report `degraded.reason` as `model_loading` or `not_yet_embedded`; the marker
  disappears once semantic indexing catches up. They also report `index.state`
  (`loading_model`, `indexing` with progress, `ready`, `pass_failed` or `stopped`).
- **Confined to its folders.** The tool refuses paths outside the folders the
  server was started with.
- **For code, it complements grep.** Hybrid search finds `retryPayment` by name
  too, but grep is still the right tool for every exact occurrence of `retryPayment(`.

```bash
seyk daemon status    # pid, sessions, and each watched folder's index state
seyk daemon stop      # connected sessions lose the tool until they restart
seyk mcp --standalone ~/notes   # no daemon: one process per session
```

See [Indexing and inline search latency](#indexing-and-inline-search-latency) for how inline sync
can wait behind a watcher indexing another root.

## Daemon and sessions

Sessions share a daemon only when they use the same model options
(`--model`, `--dimension`, `--precision`, `--text-only`, `--cache-path`) and the
same package and JSON schema versions. Builds with the same values share an
endpoint, even when their implementation differs. The daemon is Unix-only; on
Windows `seyk mcp` always runs standalone.

Daemon sockets normally live at `<cache_dir>/seyk/daemon-<id>.sock`. If that
path reaches the Unix socket limit (104 bytes on macOS, 108 on Linux, including
the terminating NUL), only the socket moves to `$TMPDIR/seyk-<uid>/d-<hash>.sock`,
or `/tmp/seyk-<uid>/d-<hash>.sock` if TMPDIR is unset or not absolute, or its
socket path is too long. The fallback directory must be a real directory owned by your uid with mode `0700`. Seyk
creates it with that mode and refuses unsafe existing directories. Sockets use
mode `0600`. Locks and `daemon.log` stay in the cache directory.
`SEYK_DAEMON_DIR` overrides the normal directory for sockets, locks and logs.
An explicit override must fit the same socket length limit and does not fall back.

## Local-content mode

For automatic local-content collections on Unix, opt into a fixed narrower
policy with `seyk mcp --local-content /absolute/real/folder`. It requires explicit
directory roots and always runs in-process, bypassing the daemon. The MCP
initialize response advertises `serverInfo.name: seyk-local-content-v1`; clients relying on this
policy must require that name and JSON schema 6 or later, and refuse an older helper
instead of dropping the flag. Use `--no-watch` to disable background indexing.

Local-content cached queries retain their indexed-only scope; their background
indexer discovers new files. Model-loading and model-free fallback keep the
same local-content access policy.

The same policy filters discovery and every content read/open: hidden paths,
`.icloud` markers, Library/cache/caches/build/dist/target/node_modules/__pycache__
directory names (ASCII case-insensitive), and structural Rust outputs under
target(s)/[up to two namespace components]/debug or release followed by
incremental/.fingerprint/build/deps/examples are excluded. Ordinary `targets/data`
remains eligible. Include/exclude options cannot widen this fixed policy. Roots
and file/directory links are refused, including in-root aliases; supply real
POSIX paths, since `/tmp` can itself be a symlink on macOS. Roots are pinned by
directory descriptors and current device/inode checks. Other Seyk modes retain
their existing discovery and explicit-link behavior.

Observed macOS dataless files are refused before content open, with descriptor
type/identity/flags checked again afterward. Neither this mode nor the ordinary
filesystem guard guarantees atomic prevention of hydration: cloud state can
change between the metadata check and open, and directories can move while
traversed. Discovery is metadata-only and path-based, with the fixed policy and
confined metadata checks before descending. A returned seekable audio handle
is not rechecked at each later read. Model/cache/materializer reads and custom
resource providers are separate from this content boundary.

## Privacy

**Privacy.** Seyk itself sends nothing anywhere, but a cloud-hosted agent sends
the passages it reads to its provider. Point the server only at folders you are
happy for that agent to read.

## Indexing and inline search latency

A search covered by a running watcher uses the existing index once all discovered
supported resources are current. While the model loads, it uses an independent
in-memory keyword index and returns `degraded: {"mode": "lexical_only", "reason":
"model_loading"}`. New or changed resources awaiting embedding trigger the same
fallback with `reason: "not_yet_embedded"`. Missing text weights use `reason: "no_model"`.
If only vision encoder files are missing, semantic text search stays available.
Images and scanned PDF pages are skipped with diagnostics naming the missing files
and `seyk model download`. After loading, `index_status.model_diagnostics` lists
this condition; it is empty before model loading and when no optional model is
missing. This does not set a lexical-only degradation marker. The active profile
matches explicit `--text-only`, preserving reusable text vectors for identical
inputs under that profile.
The keyword index covers text and PDF text, respects request filters and root
confinement, and creates no vectors. Preparing it still takes discovery and text
parsing time, and there is no hard latency bound. Ready-state ranking is unchanged.
Local-content cached queries keep their indexed-only scope; the background
indexer discovers new files. Fallback keeps the local-content provider policy.

CLI JSON, MCP and library results share this optional marker (protocol 8). With no
active overlapping watcher, a search may sync inline, but it still shares the
engine-wide indexing lease with watchers on disjoint roots. It can wait for an
entire remaining pass, including resource commits and deletion/pruning. The
mutex provides no fairness or maximum-wait guarantee. Cached-only searches do
not acquire this lease.

A synthetic SQLite probe (debug build, five samples, one segment per file,
batches of 16, fake inference at 2 ms per document segment) measured median
inline latency for a one-file/one-segment request: **7.2 ms with no watcher
load**, **268.1 ms with 64 watcher files/segments**, and **841.5 ms with 256
watcher files/segments**. All data lived in fresh temporary directories. Each
watcher stopped after its first pass and was joined before the next sample. The
request arrived at the watcher's first embedding batch. These measure the MCP
handler and a real Hub watcher, excluding model loading, ONNX inference and
transport; they do not establish production bounds.

Reproduce with:

```sh
cargo test --bin seyk measure_disjoint_watcher_inline_sync -- --ignored --nocapture
```

Every sample prints its load beside latency and lease wait. Searches that sync
inline can wait behind indexing on another root. A maximum wait time would
require a separately evaluated finer GC/lease design.

## Per-hit match provenance

Every full MCP hit and CLI JSON search hit includes an additive `match` object:

```json
{"semantic_rank": 2, "keyword_rank": 1, "filename_rank": null, "matched_terms": ["cafe", "restore"]}
```

The library exposes this as `SearchHit::r#match` (`MatchReason`). Ranks are
one-based positions in the contributing candidate lists before fusion and final
truncation, so they can exceed the requested result limit. `keyword_rank` is the
BM25 leg. A `null` rank means that leg did not contribute (including a disabled
keyword weight); semantic-only searches have no keyword or filename ranks.
Semantic ranks combine all query phrasings using their best score. Filename ranks
follow the engine's filename candidate order, not relevance within the file.

`matched_terms` contains normalized query tokens from complete matching BM25
groups, deduplicated in query order across all phrasings. Case and accents follow
the index's Unicode tokenizer; standalone stopwords are excluded. An identifier
such as `INC-18372` contributes its tokens only when the entire group matches one
indexed field. Matches can come from name, path, heading or body. Terms are
confirmed against the existing keyword index, never recovered from source text,
and no new text is persisted in the cache. Without a contributing keyword leg,
the list is empty.

Existing `matched`, full-mode `matched_query`, scores and CLI/library `signals` keep their
meaning. Current JSON `schema_version` is **11**. Version 7 added optional result-kind
counts, per-root indexing progress and `index_status`; version 8 adds optional
lexical-only `degraded` metadata and its reason. Version 9 added per-profile derived image-code counts to cache statistics. Version 10 adds `model_diagnostics` to MCP index status.
Version 11 makes MCP search compact by default; full CLI extraction accepts versions 2–11.
Match provenance does not change existing fields or extraction requirements. Older
search JSON without `match` deserializes to empty provenance. No cache migration is
needed. Rust callers constructing `SearchHit` literals add `r#match:
Default::default()`; custom cache implementations implement
`EmbeddingCache::lexical_matched_terms`, returning one query-token list per key
in request order. The engine calls this batch method only for final returned
hits ranked by the keyword leg. SQLite checks each query group in one statement
over the selected rowids; candidate ranking does no term-membership probes.

For reproducible model-free character/byte accounting, see
[offline MCP payload accounting](benchmarks/mcp-payloads.md).
