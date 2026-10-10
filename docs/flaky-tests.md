# Flaky-test investigation (2026-10-10)

Base: `41d5449`; Rust 1.99.0; macOS ARM64; provisioned local HF weights only.

## Solo todo 4425: deadline included first executable assessment

The freshly linked audio-stt CLI took **10.28 s** for `--version`, with rounded
user/system CPU times both **0.00 s**. Its next standalone MCP initialize response
arrived in **16 ms**, followed by tools/list at **17 ms**. The MCP engine factory
already runs on a background indexing thread; the handshake does not load ORT or
STT weights synchronously.

Fresh ad-hoc code-signing identities reproduce the first-launch delay. macOS
`syspolicyd` logs show Gatekeeper `evaluateScanResult` events for these identities.
The old test's deadline could kill the executable during that assessment, before
application code produced any stdout. Running an already assessed executable
repeatedly alone misses this cause.

| Experiment | Before | After |
| --- | ---: | ---: |
| Cold CLI, 24 runs, 8 parallel, 12 CPU stress workers | 7/24 failures (29.2%) | 0/24 failures |
| Warm CLI, 40 runs, 4 parallel, 12 CPU stress workers | 0/40 failures | 0/40 failures |

Other cargo builds and the parallel audio investigation also ran on the host.
The final cold experiment after the change included a successful **33.172 s**
overall run. Assessment is awaited once via a successful `--version` readiness
probe, outside the operation deadline; concurrent tests share this gate. The
entire MCP exchange retains **one 20 s deadline**, with no retries. The test now
keeps stdin open, awaits initialize, sends initialized/tools/list, and reads both
responses before disconnecting. Previously, immediate EOF could race pending
response tasks in rmcp's service loop.

The readiness probe relies on the enclosing test job's watchdog for a hung OS
launch; it deliberately does not impose an arbitrary application deadline on
code that has not started executing.

## Reproduce under load

Build with the pinned toolchain bin first in PATH. Cargo's `--no-run` output gives
the test executable to pass as `CLI_TEST` or `AUDIO_TEST` below. Use a different
output directory for each experiment so the cold code-signing identities differ.
The runner writes per-run logs and JSON counts, and exits nonzero on a test
failure. It never downloads or modifies weights.

```sh
export PATH="$HOME/.rustup/toolchains/1.99.0-aarch64-apple-darwin/bin:$PATH"
cargo test --offline --locked --features audio-stt --test cli --no-run
python3 scripts/bench/flaky_tests.py "$CLI_TEST" \
  mcp_server_lists_search_tool_without_loading_the_model \
  --binary target/debug/seyk --cold-cli --runs 24 --parallel 8 \
  --output /tmp/seyk-cold-after

# Omit --cold-cli for the warm experiment, with --runs 40 --parallel 4.
cargo test --offline --locked -p seyk-embeddinggemma --features audio \
  --test audio_golden --no-run
python3 scripts/bench/flaky_tests.py "$AUDIO_TEST" \
  missing_audio_files_leave_text_available --runs 12 --parallel 4 \
  --output /tmp/seyk-audio-investigation
```

## Solo todo 4461: unresolved

`missing_audio_files_leave_text_available` allocates a unique `tempfile::TempDir`
per test invocation. Every input is a completed hard link of a canonicalized HF
file. There is no copy fallback, shared destination, or symlink in its `onnx`
directory. Therefore `onnx_files_without_symlinks` returns that directory directly;
its size-based reuse and non-atomic copy fallback do not execute directly in
the failing test. This does not exclude another materializer touching the shared
source inode while the test uses its private hard link. The test replaces its own audio graph link before editing bytes,
so it does not overwrite the provisioned HF inode.

The shared materializer has a concrete source-corruption hazard: two writers can
both see an absent target, then one can create a hard link before the other's
`hard_link` call. The latter gets `AlreadyExists` and falls back to copying onto
the existing hard link. A disposable Rust 1.99.0 fixture on this machine confirmed
that `std::fs::copy(source, existing_hard_link)` returned `Ok(0)` and truncated
the source to zero bytes. This was tested on synthetic bytes, never HF weights.
Thus private test directories do not rule out corruption through a shared HF
inode. Size-only reuse and partial-copy publication are additional hazards.

The exact ORT exception remains unreproduced. The demonstrated copy hazard is
recorded on todo 4461 for a deterministic materializer regression and source-safe,
atomic publication fix; it is not sufficient evidence to call the original ORT
failure resolved. This PR changes neither ORT initialization nor materialization.
The unchanged audio test completed **12/12 runs** without failure, with four
parallel processes and twelve CPU stress workers. The slowest complete run was
119.897 s; sampling showed repeated audio-weight hashing after successful text
session initialization. An earlier experimental 180 s outer harness cutoff
censored runs during hashing and is excluded from the failure-rate calculation.
The committed runner waits for completion and does not change test timeouts.
The full `audio_golden --include-ignored` suite also passed **3/3** with its
neighboring model tests running concurrently. There is no demonstrated fix or
after-fix rate for 4461. This PR resolves only
todo 4425; 4461 remains open, with findings recorded in its Solo todo comment.
