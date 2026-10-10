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

## Publication contract

- Reuse requires a regular destination with matching size and either matching
  device/inode identity on Unix or matching BLAKE3 content hashes.
- New hard links use the filesystem's no-clobber operation.
- A failed link attempt never opens the destination for writing. Copies go to
  fresh, exclusively created temporary files in the destination directory.
- A staged copy must match the source's size and BLAKE3 hash before an atomic
  rename replaces the destination directory entry. Failure removes the staging
  file and returns an error; it never publishes a partial copy.
- Callers still require immutable provisioned sources. This change cannot
  reconstruct weights already corrupted by an earlier process.

## Load experiment

Build the unmodified and fixed `audio_golden` executables separately:

```sh
cargo test -p seyk-embeddinggemma --features audio --test audio_golden --no-run --offline --locked
python3 scripts/bench/audio_golden_load.py /path/to/audio_golden \
  --hf-home /path/to/disposable-hf-home --runs 12 --parallel 3 \
  --stress-workers 12 --output /path/to/results
```

The runner invokes all three golden tests, including the two ignored tests,
in each process. Twelve independent SHA-256 workers supply CPU load. It records
exit status, duration, the specific ORT exception, and SHA-256 hashes before
and after the entire experiment. It waits for completion without an outer
timeout and changes no test timeout or retry policy.

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

## Verification

Formatting, workspace/all-targets clippy with `-D warnings`, and workspace
tests passed for default, `audio`, and `audio-stt` configurations with Rust
1.99.0. Rust 1.89.0 passed the CI MSRV gates: workspace/all-targets checks for
default and `audio-stt`, and workspace builds for default and `audio`.
Verification used the already provisioned ORT static archive through
`ORT_LIB_PATH`, with downloads disabled. Before each CLI test configuration,
a successful `seyk --version` probe completed macOS executable readiness;
no test deadline was changed.
