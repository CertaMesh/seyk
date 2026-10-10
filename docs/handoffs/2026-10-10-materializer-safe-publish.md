# Safe publication of ONNX cache files

The old materializer removed an invalid destination, attempted a hard link,
then copied directly onto the destination on **any** link error. Concurrent
callers could both observe a miss; one could install the source hard link before
the other encountered `AlreadyExists`. On macOS, copying onto that link empties
the shared source inode. This can corrupt provisioned Hugging Face weights,
including weights linked into another test's private directory.

The regression forces that interleaving with barriers and disposable bytes:
two publishers finish their cache checks, a source hard link is installed, and
both publishers continue. Substituting the original remove/link/copy sequence
made the regression fail with an empty source. With safe publication it passes,
checking both bytes and BLAKE3 hashes of the source and destination. A second
concurrent regression repairs a same-length corrupt destination linked to a
different source, preserving both source files. A third checks content reuse
and repair of partial files. The three fixed regressions also passed 100/100
suite repetitions (300 tests), with eight concurrent processes while the
golden-suite CPU stress workers and workspace verification were running.

## Publication contract (including review fixes)

- Reuse is metadata-only for regular destinations: shared device/inode on Unix,
  or volume serial/file index through `same-file` on Windows. Copies instead
  require a bounded, versioned verified-at-publish sidecar whose source and
  target identities, lengths and nanosecond mtimes match current metadata.
  Malformed, stale or symlink stamps cause safe republishing; stamp contents
  never choose a path or authorize a write through an existing entry.
- New hard links use the filesystem's no-clobber operation.
- `AlreadyExists` rechecks reuse, then repairs by linking to a unique temporary
  name and renaming over the destination. Copying is limited to cross-device
  or hard-link permission errors. A failed link never opens the destination for writing. Copies go to
  fresh, exclusively created temporary files in the destination directory.
- A staged copy must match the source's size and BLAKE3 hash before an atomic
  rename replaces the destination directory entry. Failure removes the staging
  file and returns an I/O/integrity error; it never publishes a partial copy.
  The copy hashes bytes as it writes them, comparing against one source pre-hash.
  Its writer closes before recording stamp metadata, because Windows finalizes
  last-write timestamps when writer handles close ([file-time contract](https://learn.microsoft.com/en-us/windows/win32/sysinfo/file-times)).
- Temporary names carry a recognizable prefix and creation timestamp. Cache
  misses sweep abandoned names older than a day without using inherited source
  mtimes. A guard also cleans up successful POSIX same-inode rename no-ops.
- Copy stamps publish before the data rename. A crash or competing copy between
  those renames can leave mismatched stats; the next load safely republishes.
- Callers still require immutable provisioned sources. This change cannot
  reconstruct weights already corrupted by an earlier process.

## Load experiment

The review round adds identity assertions to both concurrent regressions. Both
failed against the initial PR implementation because its loser/repair became a
copy; both now retain the source inode. The expanded eleven-test materializer
suite passed **100/100 repetitions (1,100 tests)** with eight concurrent processes
and twelve CPU stress workers, alongside workspace verification.

A 150,000-byte disposable fixture measures reuse via digest and copy-chunk
counters: initial hard-link publication calls neither, initial forced-EXDEV
publication hashes the source once and hashes again while copying, and **50
subsequent publications on each path make zero additional digest/copy calls**.
The counter observes every weight-digest helper call, including direct calls
from `matches()`. An isolated disposable probe with the original hashing matcher
failed this assertion at 101 calls versus one; the final matcher passes all
eleven tests. The final backend executable also passed another 100/100 loaded
suite repetitions (1,100 tests) after the writer-close/counter refinements.
Fault regressions cover source hashing, partial copying, verification, stamp
writing, stamp rename, and target rename. Each preserves source and old target
bytes and leaves no staging entry. Additional cases cover malformed/stale stamps,
symlink destinations/stamps, real rename failure, digest mismatch, unexpected
link errors, and stale-temp cleanup. Runner regressions with fake executables
pass on Python 3.9 and 3.14, including early aborts and hard-link fixture rejection.

Build the unmodified and fixed `audio_golden` executables separately:

```sh
cargo test -p seyk-embeddinggemma --features audio --test audio_golden --no-run --offline --locked
python3 scripts/test/audio_golden_load.py /path/to/audio_golden \
  --hf-home "$TMPDIR/disposable-hf-home" --runs 12 --parallel 3 \
  --stress-workers 12 --output /path/to/results
```

The runner invokes all three golden tests, including the two ignored tests,
in each process. Twelve independent SHA-256 workers supply CPU load. It records
exit status, duration, the specific ORT exception, and SHA-256 hashes before
and after the entire experiment. It waits for completion without an outer
timeout and changes no test timeout or retry policy.
Use Python 3.9 or newer. The runner rejects multiply linked fixture files and
homes outside the system temporary directory, and the configured (`HF_HOME`)
or default (`~/.cache/huggingface`) HF home, even when those live under temp. Early aborts, missing test headers
and executable launch errors are recorded as failures; the cohort continues to
the summary and final fixture hash check.

For this experiment, pinned local revision
`daa72c51243991dfcaf9f9137d2c573d8f7790c0` was copied read-only into a disposable
HF home, with fresh destination files and verified hashes. Its ONNX snapshot
paths point to disposable blobs inside that home. The after experiment uses a
second, independently copied home with a new cache key. No real HF inode is
shared with either fixture, and no weights were downloaded. The real cache was
never opened for writing.

On this macOS ARM host with Rust 1.99.0, the complete cohorts finished as follows:

| Measurement | Before | After |
| --- | ---: | ---: |
| Full-suite invocations | 12 | 12 |
| Invocations failing with the ORT exception | 1 (8.3%) | 1 (8.3%) |
| `missing_audio_files_leave_text_available` ORT failures | 1/12 | 1/12 |
| `reference_embedding_and_text_cache_identity` ORT failures | 0/12 | 1/12 |
| Fixture SHA-256 hashes unchanged | Yes | Yes |
| Longest suite invocation | 133.823 s | 95.661 s |

The baseline failure was at `audio_golden.rs:125`, with
`Encountered unknown exception in Initialize()` while loading the temporary
text graph. The fixed cohort's failing invocation reported that same error at
line 125 and at line 60 while loading the materialized text graph. Its other
eleven invocations passed all three tests. Counts are per suite invocation,
not per native throw: the fixed cohort's single failing invocation contained
two error reports.

These observations separate the verified source-corruption defect from the
still-unexplained native initialization flake. The safety fix preserves source
files and repairs stale destinations, but does **not** resolve the ORT flake.
The cohorts ran amid other host work and workspace verification; their duration
difference is not a performance comparison. A native throw backtrace remains
necessary to identify the ORT failure. This PR is part of Solo todo #4461.

The review follow-up ran another complete twelve-invocation cohort with three
parallel processes and twelve CPU stress workers, using a third independently
copied disposable pinned snapshot. It passed **12/12** with **zero ORT exceptions**
and unchanged fixture SHA-256 hashes. The longest invocation was 171.212 s.
This cohort exercised the revised shared-inode publication/reuse path. The final
copy-writer-close adjustment changes the copy-only path, which is exercised by
the forced-EXDEV regressions. The passing cohort does not explain the native
exception observed in the two earlier cohorts and does not close todo #4461.

The review-B addendum extends the suite to thirteen tests. An injected persist
operation rejects publication after a forced-EXDEV copy has been verified and
its stamp published. The original hard-linked target, source bytes and hashes
survive, with no staging entries. A Unix case rejects replacement of a symlink
and also asserts that its original link and referent survive unchanged. These
complement the existing partial-copy, verification and actual digest-mismatch
failures. A disposable probe bypassing the persist seam fails the new test;
the correctly wired helper passes all thirteen tests. The expanded backend
suite passed **100/100 loaded repetitions (1,300 tests)** with eight concurrent
processes and twelve CPU stress workers, alongside workspace validation.

The runner adds a synthetic test for configured and default HF homes located
inside a temporary directory. Both cases were accepted before this guard and
are now rejected before any executable launches or results directory is created.
No real cache files are used by these tests.

## Verification

Formatting, workspace/all-targets clippy with `-D warnings`, and workspace
tests passed for default, `audio`, and `audio-stt` configurations with Rust
1.99.0. Rust 1.89.0 passed the CI MSRV gates: workspace/all-targets checks for
default and `audio-stt`, and workspace builds for default and `audio`.
Verification used the already provisioned ORT static archive through
`ORT_LIB_PATH`, with downloads disabled. Before each CLI test configuration,
a successful `seyk --version` probe completed macOS executable readiness;
no test deadline was changed.

During the first review round on main `41d5449`, the legacy no-model MCP CLI test
also reproduced its independent empty-response failure (`responses[0]` with
length zero). It does not load model files or execute the materializer. The
that round's default/audio workspace gates therefore integrated the reviewed CLI test
harness from PR #46 (`9cdac59`) temporarily, without including that file in this
PR. That round's audio-stt workspace run and MSRV gates passed with the main-based CLI
test. This PR's product code and source-preservation checks remain independent
of that test-harness change; the combined gate represents PR #48 landing after
PR #46. No failed test was blindly retried.

The review-B addendum repeated default, `audio`, and `audio-stt` workspace
verification with PR #46's now-merged CLI harness temporarily integrated. All
three passed, including the thirteen materializer regressions. The borrowed
CLI file is restored before committing; this addendum changes only the
materializer, fixture runner/tests, and this report. All formatting, three
clippy configurations, dependency/license checks, Python 3.9/3.14 regressions,
and the Rust 1.89 check/build gates also passed for the addendum.
