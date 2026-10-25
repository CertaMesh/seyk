# Caching and storage

## How caching works

The original files stay the source of truth. The cache holds file metadata,
segment locations, vectors and a keyword index (see the privacy note below).

There are four ways to search:

1. **Lazy cache (default).** For each file: if size and modification time match
   the cache, its vectors are reused without reading it. Otherwise the file is read
   and hashed. If the content hash is unchanged, only its metadata and version are updated.
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
the old vectors are never looked up. `cache stats` lists the profiles.

The `text-literals-v1` model variant treats Gemma special-token spellings in
text and titles as ordinary text. Earlier versions could skip media-token
literals or cache a different embedding for control-token literals such as
`<bos>`. This policy changes the cache profile, so the first index/search after
upgrading re-embeds the entire selected collection (including unchanged files).
The cache stores no reconstructable source text for an exact selective repair.
Ordinary-text token ids and the segmenter version are unchanged.

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

**Privacy.** Seyk sends no files, queries or embeddings over the network. Treat the cache
as sensitive. Embeddings can leak information about the text they came from, and
the keyword index (a SQLite FTS5 table) stores **which words occur in each passage
and how often, not their order**. Each field is tokenized and sorted before it
is indexed. These word frequencies are still sensitive, like embeddings. The
cache directory is created `0700` and the file `0600`. SQLite
`secure_delete=ON` zeroes deleted ordinary records in updated pages; zeroing of
the main database and removal of historical WAL images complete at the next
scrub's checked checkpoint. Removed ordinary records and keyword postings are
**erased after the next scrub**, not necessarily when a file is removed or
replaced. A scrub runs FTS5 `optimize` for each pending profile, then checks a
WAL TRUNCATE checkpoint. Forensic tests with the bundled SQLite verify removal
of readable prefix-compressed suffixes, not just whole token strings. A word
still present in another live record remains indexed. Scrub cost depends on the profile's index size.

A scrub happens at the end of a one-shot `seyk index` or `seyk search` command,
when cleanup is pending. Watcher, daemon and MCP use a background worker after
about **60 seconds with no further changes**; it never scrubs inline in an MCP
request or during an indexing pass. A running scrub holds the cache mutex, so
a concurrent search can be delayed by the scrub's duration. Further writes,
including fresh inserts while cleanup is already pending, postpone it. Continuous
changes can delay scrubbing indefinitely. The worker retries blocked cleanup at roughly 60-second
intervals. It sleeps until the quiet-period/retry deadline, or checks every 60
seconds when idle to notice other processes' writes, rather than polling every
second. Observing another process's write starts a fresh quiet period. If cleanup
is pending, a short-lived MCP process or `seyk index --watch` stopped within about
60 seconds of its last change leaves the scrub pending; there is no scrub on exit.
Engine drop wakes an idle worker immediately. An active scrub may finish its
current iteration; drop does not cancel or join it.

Run `seyk cache scrub` to scrub immediately without deleting live records.
`seyk cache clear` deletes the entire index, runs `VACUUM`, and truncates the WAL,
including cleanup-marker frames. Schema migrations also mark compaction pending;
VACUUM is performed by the next scrub. Fresh inserts, unchanged metadata touches,
identical vectors and zero-row deletions do not request a scrub or an explicit
checkpoint. Core/library index and search APIs defer scrubbing; integrations can
call `SearchEngine::scrub_cache`, `EmbeddingCache::scrub`, or start the idle worker.

Pending profile scrub, WAL cleanup and migration/clear compaction are durable
`meta` keys, with no cache schema or profile change. They survive interruption; pending work stays recorded until the scrub's
checked TRUNCATE succeeds. If a reader
blocks it, the marker stays and a warning is emitted; sync and search still
succeed. Old content may remain readable in the database's stale FTS blocks and
in the WAL until the scrub completes. Once other readers close, rerun
`seyk cache scrub` using the same `--cache-path` (if supplied), or rerun the
one-shot command. Clear is a fallback, not the required cleanup-only retry.
Human `cache stats` reports pending scrubbing. Cache stats JSON keeps its
existing shape and omits `scrub_pending`; use human `cache stats` or the library
status field to check for pending scrubbing.

After a completed scrub, removed records do not remain readable in the current
database or `-wal` file. Scrubbing can leave new WAL frames containing only live
cleanup metadata; clear also truncates these housekeeping frames. For remnants
left by older binaries, run `seyk cache clear` once. This is logical file cleanup,
not physical secure erasure: SSD wear levelling, filesystem snapshots, backups,
open reader snapshots and crash-interrupted cleanup may retain earlier copies.
Use OS disk encryption and manage snapshots/backups according to your privacy
needs.

Protection against a lost or stolen disk depends on OS disk encryption: check
`fdesetup status` on macOS (FileVault), `lsblk -f` / `cryptsetup status <mapping>`
on Linux (LUKS), or `manage-bde -status` on Windows (BitLocker). Seyk does not
encrypt SQLite independently. A copied cache exposes vectors, word frequencies,
resource paths, hashes and metadata to anyone who can read it; file permissions
and disk encryption do not protect a copy on another unencrypted volume.

Seyk applies backup hints only to its default platform cache directory, or a
cache directory it creates during this open. For a custom `SEYK_CACHE` /
`--cache-path` in an existing directory, it leaves the directory's tags,
permissions and exclusion flag untouched. To opt in, give a dedicated directory
that does not exist yet, such as `--cache-path /private/local/seyk/cache.sqlite3`.
Seyk writes the standard `CACHEDIR.TAG` for backup tools that honor it and, on
macOS, sets the directory's own backup-exclusion resource value using Apple's
Core Foundation API (`kCFURLIsExcludedFromBackupKey`) without root. Tagging is
best-effort: failures warn and continue, foreign tags are preserved, and
filesystems without atomic no-clobber hard-link publication skip the tag.
Temporary tag files older than one hour are swept best-effort on preparation.
These are backup hints, not encryption or a guarantee against every backup tool.
Linux cache directories have no automatic backup guarantee.
`~/Library/Caches` is excluded by Time Machine on the development Mac; we have
not verified a universal macOS iCloud exclusion guarantee. iCloud Drive sync is
separate from Time Machine.

A cache inside a detected iCloud Drive, Dropbox, OneDrive, Google Drive or
Syncthing folder produces one warning per process on stderr and continues.
Detection covers conventional locations, Dropbox configured roots, OneDrive
environment roots, macOS CloudStorage providers and Syncthing `.stfolder`
markers; arbitrary custom roots without these signals may be missed.
`seyk cache stats --json` reports `backup_excluded` (the directory's own macOS
resource flag, `null` when unknown) and `in_synced_folder` (advisory boolean).
Both values are captured once at cache open, without subprocesses, and reused
by stats and MCP daemon status's `cache_status` (`null` until the daemon opens
its cache, or if stats fail). Search responses do not include cache status.
These snapshots do not reflect subsequent directory changes until reopen and
are distinct from inherited Time Machine exclusions. Protocol version 5 adds
these fields; extraction accepts search documents from versions 2–8.

## Schema versions and downgrades

CLI/MCP output uses JSON schema 8. `seyk extract` accepts search envelopes with
schema 2 through 8. The cache is at schema 6; upgrades migrate older caches and
preserve existing vectors, profiles and keyword indexes. Current binaries
refuse a newer cache schema; v0.1.0 deletes a cache it does not recognize, so
keep a copy or use a separate `--cache-path` if downgrading.

Older Seyk binaries reset caches whose schema they do not recognize. Use separate
`--cache-path` values when running builds with different cache schemas.

## Move an index with cache import

```bash
seyk cache import --from old.sqlite --map mapping.jsonl --profile PROFILE_ID --cache-path new.sqlite --json
```

To move an indexed collection into a new directory, use `cache import` with an
explicit destination and profile ID from `seyk cache stats --json`. The destination
must be absent or a completely empty schema-v6 cache. Import never overwrites or
merges populated caches, initializes a model, or downloads weights.

The mapping file is JSONL v1 (one object per nonblank line):

```json
{"version":1,"old":"file:///old-epoch/note.md","new":"file:///new-epoch/note.md","content_hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}
```

Replace the example hash with the lowercase, 64-character BLAKE3 of the complete
file. Both identities must be absolute local file URIs; Unix non-UTF-8 paths use
percent encoding. Unknown fields, versions, duplicate old/new identities and
malformed hashes are errors. There is no prefix remapping or implicit traversal.

Seyk reads only each new file through its local-file access policy and hashes its
opened handle in bounded memory. The observed hash must equal the cached hash;
the supplied hash is an additional guard. Mismatches drop the entire resource
with a `hash_mismatch` count; other verified mappings can still commit. Missing,
disallowed or unstable files and changes to preparation kind or filename title
also drop the resource. An empty mapping creates an empty seed. Reports contain
counts of resources, segments, unique vectors, keyword rows, drop reasons and
unmatched mappings, with no source excerpts or vector bytes. `report.drops` always
lists every reason (`hash_mismatch`, `missing_hash`, `preparation_changed`,
`unavailable_or_unstable`, `unmapped`), including zeros. Check counts rather than
key presence to detect dropped resources.

The import preserves locators and exact vectors under the same profile, rebuilding
name/path keyword fields from new identities and retaining only order-free
heading/body fields. Unmapped evidence, old path labels, other profiles and orphan
vectors are excluded. This skips re-extraction of unchanged evidence (including
PDF parsing and audio transcription); ordinary indexing already reuses vectors
across paths when its model input is identical. PDF/audio rows retain their
composite preparation descriptor and rely on the normal runtime compatibility
check after import. Extraction still verifies content hashes; later changes fail.

Only v6 sources are accepted: open an older cache once with this Seyk first.
The source is read-only in one SQLite read transaction, including committed WAL
contents. Keep its `-wal`/`-shm` files together; SQLite read-only WAL access needs
existing sidecars or directory write access and otherwise fails. Destination
schema/data writes commit together; cache corruption or SQL errors roll everything
back. A failed new destination may remain an empty SQLite shell and can be retried.
Unix access rejects symlink components and checks handle/path identity around the
single streamed hash. Windows access is path-based and is not pinned. This is not
an atomic filesystem snapshot; use immutable projections for reliable staging.
