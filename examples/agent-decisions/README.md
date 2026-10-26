# Example: let an agent cite decision records

`decisions/` holds six invented architecture decision records for a made-up
ticket tracker called Lantern. Use them to try Seyk on the kind of content an
agent should cite rather than guess: why a choice was made, and where that is
written down.

Run the commands from this folder. Every output below was captured from a real
run **without model weights**, so it shows keyword and file-name matches only.
With the model downloaded (`seyk model download`), the default hybrid search
also matches by meaning, so a question can find a passage that shares none of
its key terms.

## Find the passage

```console
$ seyk search "how do background jobs write to the database?" decisions -n 2 --no-cache
1.000  decisions/0007-storage-sqlite.md:20-25  § 0007. Store tickets in SQLite, not Postgres > Consequences
       Installation is one binary and one file. Only one process can write at a time, so background jobs queue their writes through the main service instead of opening their own connections. Full-text search uses SQLite FTS5 rather than an…

0.857  decisions/0007-storage-sqlite.md:13-18  § 0007. Store tickets in SQLite, not Postgres > Decision
       We store tickets in a single SQLite database file in write-ahead-log mode, next to the application. Backups copy that one file with the SQLite online backup API. We keep the schema portable so a move to Postgres stays possible if a…

Searched 8 segments: 6 files (0 unchanged, 6 new or changed), 0 segments embedded in 0.0s
  Keyword and filename matches only (model unavailable)
```

Each hit names the file, the line range and the heading path. Scores rank hits
within one search; they are not comparable across searches.

## Extract it as evidence

`extract` re-reads the file, checks it against the hash recorded at search
time, and prints the exact bytes of the hit:

```console
$ seyk search "how do background jobs write to the database?" decisions --json -n 1 --no-cache > hits.json
$ seyk extract --from-json hits.json
## Consequences

Installation is one binary and one file. Only one process can write at a
time, so background jobs queue their writes through the main service instead
of opening their own connections. Full-text search uses SQLite FTS5 rather
than an external search server.
```

If the file changes after the search, extraction refuses instead of returning
different text. With `--typed-errors`, the failure is one stable line and exit
code 4:

```console
$ echo "Note: revisit in 2027." >> decisions/0007-storage-sqlite.md
$ seyk extract --from-json hits.json --typed-errors
error: resource_changed
```

A matching hash shows the file is unchanged since the search. It does not show
that the decision is still right, or who wrote it.

## Give the records to an agent

Point the MCP server at the absolute path of the `decisions` folder:

```bash
claude mcp add seyk -- seyk mcp "$PWD/decisions"
codex mcp add seyk -- seyk mcp "$PWD/decisions"
```

The agent gets `search`, `extract` and `index_status` tools. For a question
such as "why did we pick SQLite over Postgres?", `search` returns passages of
`0007-storage-sqlite.md` with line ranges the agent can read and cite.
See [Set up an agent](../../docs/mcp.md#set-up-an-agent) for absolute binary paths and Codex timeouts.

`tests/agent_decisions_example.rs` runs the search and extraction above
without model weights, so this example keeps working.
