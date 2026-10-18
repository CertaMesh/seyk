#!/bin/sh
# Run inside a disposable privileged Linux container with the repo and Cargo
# registry mounted read-only, a writable CARGO_TARGET_DIR, and --network none.
# RUSTUP_TOOLCHAIN must select an already-installed toolchain (>= MSRV).
set -eu
cd "$(dirname "$0")/../.."
test "$(uname -s)" = Linux
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT HUP INT TERM
cc -Wall -Wextra -Werror scripts/test/local_content_autofs.c -o "$scratch/autofs"
cargo test --offline --locked -p seyk-core --no-default-features --lib \
    --no-run --message-format=json > "$scratch/artifacts.jsonl"
binary=$(python3 - "$scratch/artifacts.jsonl" <<'PY'
import json
import sys
for line in open(sys.argv[1]):
    artifact = json.loads(line)
    if (artifact.get("reason") == "compiler-artifact"
            and artifact["target"]["name"] == "seyk_core"
            and artifact.get("executable")):
        print(artifact["executable"])
PY
)
test -n "$binary"
result=0
"$scratch/autofs" "$binary" || result=1
"$scratch/autofs" "$binary" ancestor || result=1
exit "$result"
