# Standalone index progress protocol v1

`seyk index <root> --json --progress jsonl` writes live progress as one compact
JSON object per line on stderr. The final stdout JSON report keeps its existing
schema and formatting. Progress is opt-in so existing stderr consumers keep their
current behavior. The flag also works without `--json`.

Use one root per invocation. Multiple roots and `--watch` are rejected with this
flag because the engine reports counts for a whole pass. Without the flag,
multiple roots and watch mode keep their existing behavior.

```sh
seyk index ./notes --text-only --json --progress jsonl >report.json 2>progress.jsonl
```

## Event fields

```json
{"v":1,"type":"progress","root":"./notes","phase":"indexing","by_kind":{"text":{"indexed":12,"waiting":8,"skipped":0}},"done":12,"total":20,"eta_seconds":4}
```

| Field | Meaning |
| --- | --- |
| `v` | Progress protocol version, currently `1`. Independent of stdout `schema_version`. |
| `type` | Always `"progress"`. |
| `root` | Caller-supplied root spelling, without tilde expansion or canonicalization. `"."` when omitted. Non-UTF-8 bytes use replacement characters. |
| `phase` | `"discovering"`, `"indexing"`, `"complete"`, or `"failed"`. |
| `by_kind` | Existing `IndexProgress.by_kind` snapshot. Keys are `text`, `pdf`, `image`, and `audio`, present only when discovered. |
| `done` | Sum of `indexed + skipped` across kinds. Counts resources, not segments or pages. |
| `total` | Sum of `indexed + skipped + waiting`, or `null` before the first resource snapshot. |
| `eta_seconds` | Existing resource ETA, or `null` when unavailable or terminal. |

Each kind has nonnegative integer `indexed`, `waiting`, and `skipped` counts.
`indexed` includes cached reuse. A mixed PDF counts as one PDF resource. Skipped
resources count as done. ETA requires at least two newly indexed resources and
1 ms elapsed. Cached reuse and skips do not train the estimate.

## Ordering and completion

The stream starts with `discovering`, with empty `by_kind`, `done: 0`, and unknown
total and ETA. The first engine resource snapshot changes the phase to `indexing`.
Within `indexing`, events are at least 250 ms apart. Phase changes and the terminal
event bypass this limit. Counts never decrease, and total stays fixed once known.
There is no heartbeat while the engine has no new resource snapshot.

After indexing returns, a final `complete` event contains the latest snapshot,
including any updates suppressed by throttling. An empty root still produces an
indexing snapshot and a complete event with `done: 0` and `total: 0`.

A model-loading or indexing error produces a final `failed` event with the latest
available counts. Pending resources remain pending. Invalid arguments can fail
before the stream starts. Process termination can interrupt the stream without a
terminal event. Treat the process exit status and final stdout report as the
command result. `complete` can include skipped resources.

Progress records contain no file paths, resource IDs, titles, source text, or
error messages beyond the root supplied by the caller. Existing diagnostics,
including model/cache warnings and fatal errors, can also appear on stderr as
plain text. Consumers must distinguish progress JSON objects from diagnostics.
A closed stderr pipe does not stop indexing.
