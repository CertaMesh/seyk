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
mv "$temporary" "$output"
echo "Generated $output"
