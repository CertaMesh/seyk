#!/usr/bin/env bash
# Generate the same notice bundle locally and on every release runner.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cargo_about() {
  nice -n 10 env CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="$HOME/.cache/seyk-target" PATH="$HOME/.cargo/bin:$PATH" cargo about "$@"
}

if [[ "$(cargo_about --version)" != "cargo-about 0.8.2" ]]; then
  echo 'Install cargo-about 0.8.2 with --locked (see THIRD_PARTY.md).' >&2
  exit 1
fi

# Keep the last complete bundle if generation fails. Local crates and every
# feature/platform are included so optional release features cannot lose notices.
output=THIRD_PARTY_LICENSES.html
temporary=$(mktemp "${output}.tmp.XXXXXX")
trap 'rm -f "$temporary"' EXIT
cargo_about generate --locked --workspace --all-features \
  --fail --config about.toml --output-file "$temporary" about.hbs
python3 - "$temporary" <<'PY'
from pathlib import Path
import hashlib
import html
import sys

notices = Path("third_party/onnxruntime/ThirdPartyNotices.txt").read_bytes()
expected = Path("third_party/onnxruntime/SHA256SUMS").read_text().split()[0]
if hashlib.sha256(notices).hexdigest() != expected:
    raise SystemExit("ONNX Runtime third-party notice checksum mismatch")

output = Path(sys.argv[1])
content = output.read_text(encoding="utf-8")
marker = "<!-- ONNX_RUNTIME_THIRD_PARTY_NOTICES -->"
if content.count(marker) != 1:
    raise SystemExit("expected one ONNX Runtime notice placeholder in the bundle")
output.write_text(content.replace(marker, html.escape(notices.decode("utf-8"))), encoding="utf-8")
PY
chmod 644 "$temporary"
mv "$temporary" "$output"
echo "Generated $output"
