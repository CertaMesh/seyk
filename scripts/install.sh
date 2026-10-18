#!/bin/sh
# Install the latest Seyk release binary.
#
#   gh api repos/CertaMesh/seyk/contents/scripts/install.sh \
#     -H "Accept: application/vnd.github.raw" | sh
#
# Uses the GitHub CLI to download and verify the release archive; run
# `gh auth login` first. Installs to $SEYK_INSTALL_DIR, default ~/.local/bin.
set -eu

repo="CertaMesh/seyk"
dir="${SEYK_INSTALL_DIR:-$HOME/.local/bin}"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) target="aarch64-apple-darwin" ;;
  Linux-x86_64) target="x86_64-unknown-linux-gnu" ;;
  Linux-aarch64 | Linux-arm64) target="aarch64-unknown-linux-gnu" ;;
  Darwin-x86_64)
    echo "No Intel Mac build: ONNX Runtime ships no prebuilt library for it." >&2
    echo "Build from source instead: cargo install --git https://github.com/$repo" >&2
    exit 1 ;;
  *) echo "Unsupported platform: $(uname -s) $(uname -m)" >&2; exit 1 ;;
esac

command -v gh >/dev/null || { echo "Needs the GitHub CLI (gh)." >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
gh release download --repo "$repo" --pattern "seyk-*-$target.tar.gz*" --dir "$tmp"

archive=$(ls "$tmp"/seyk-*-"$target".tar.gz)
if command -v sha256sum >/dev/null; then check="sha256sum -c"; else check="shasum -a 256 -c"; fi
(cd "$tmp" && $check "$(basename "$archive").sha256" >/dev/null) || {
  echo "Checksum mismatch for $(basename "$archive")" >&2; exit 1; }

tar -xzf "$archive" -C "$tmp"
mkdir -p "$dir"
install -m 755 "$tmp"/seyk-*-"$target"/seyk "$dir/seyk"
echo "Installed $("$dir/seyk" --version) to $dir/seyk"
case ":$PATH:" in *":$dir:"*) ;; *) echo "Add $dir to your PATH." ;; esac
