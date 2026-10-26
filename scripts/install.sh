#!/bin/sh
# Install a public Seyk release without authentication or elevated privileges.
# SEYK_VERSION=vX.Y.Z pins a release; otherwise use the latest published release.
# SEYK_DOWNLOADER=curl|wget|gh overrides automatic downloader selection.
# SEYK_INSTALL_DIR defaults to ~/.local/bin.

main() {
  set -eu

  repo="CertaMesh/seyk"
  dir="${SEYK_INSTALL_DIR:-$HOME/.local/bin}"
  fail() { echo "Seyk install: $*" >&2; exit 1; }

  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) target="aarch64-apple-darwin" ;;
    Linux-x86_64) target="x86_64-unknown-linux-gnu" ;;
    Linux-aarch64 | Linux-arm64) target="aarch64-unknown-linux-gnu" ;;
    Darwin-x86_64) fail "No Intel Mac build; build from source instead." ;;
    *) fail "Unsupported platform: $(uname -s) $(uname -m)" ;;
  esac

  downloader="${SEYK_DOWNLOADER:-}"
  if [ -z "$downloader" ]; then
    for candidate in curl wget gh; do
      if command -v "$candidate" >/dev/null 2>&1; then downloader=$candidate; break; fi
    done
  fi
  case "$downloader" in curl|wget|gh) ;; *) fail "Install curl or wget (or select gh with SEYK_DOWNLOADER=gh)." ;; esac
  command -v "$downloader" >/dev/null 2>&1 || fail "Downloader $downloader is not installed."
  if command -v sha256sum >/dev/null 2>&1; then
    check=sha256sum
  elif command -v shasum >/dev/null 2>&1; then
    check=shasum
  else
    fail "SHA-256 verification requires sha256sum or shasum."
  fi

  tmp=$(mktemp -d) || fail "Cannot create temporary directory."
  staged=""
  cleanup() {
    rm -rf "$tmp"
    if [ -n "$staged" ]; then rm -f "$staged"; fi
  }
  trap cleanup 0
  trap 'exit 1' HUP INT TERM
  fetch() {
    case "$downloader" in
      curl) curl --proto '=https' --proto-redir '=https' --tlsv1.2 --fail --silent --show-error --location "$1" -o "$2" ;;
      wget) wget --https-only -q -O "$2" "$1" ;;
    esac || fail "Download failed: $1 (check network access and that the release exists). For API rate limits, set SEYK_VERSION=vX.Y.Z to skip release discovery."
  }

  version="${SEYK_VERSION:-}"
  if [ -z "$version" ]; then
    if [ "$downloader" = gh ]; then
      version=$(gh api "repos/$repo/releases/latest" --jq .tag_name) || fail "No latest release available; check network access or set SEYK_VERSION."
    else
      fetch "https://api.github.com/repos/$repo/releases/latest" "$tmp/release.json"
      version=$(sed -n 's/^[[:space:]]*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' "$tmp/release.json")
    fi
  fi
  case "$version" in v[0-9]*) ;; *) fail "No valid release tag found; set SEYK_VERSION=vX.Y.Z." ;; esac
  case "$version" in *[!a-zA-Z0-9._-]*) fail "Invalid release tag: $version" ;; esac
  name="seyk-${version#v}-$target"
  archive="$name.tar.gz"
  if [ "$downloader" = gh ]; then
    gh release download "$version" --repo "$repo" --pattern "$archive" --pattern "$archive.sha256" --dir "$tmp" || fail "Could not download release $version."
  else
    base="https://github.com/$repo/releases/download/$version"
    fetch "$base/$archive" "$tmp/$archive"
    fetch "$base/$archive.sha256" "$tmp/$archive.sha256"
  fi
  [ -s "$tmp/$archive" ] || fail "Missing or empty archive: $archive."
  [ -s "$tmp/$archive.sha256" ] || fail "Missing or empty checksum: $archive.sha256."
  # Check only the selected archive, never paths supplied by the checksum file.
  expected=$(awk -v name="$archive" '$2 == name || $2 == "*" name { print $1 }' "$tmp/$archive.sha256")
  [ "${#expected}" -eq 64 ] || fail "Invalid SHA-256 checksum for $archive."
  case "$expected" in *[!a-fA-F0-9]*) fail "Invalid SHA-256 checksum for $archive." ;; esac
  if [ "$check" = sha256sum ]; then
    actual=$(sha256sum "$tmp/$archive")
  else
    actual=$(shasum -a 256 "$tmp/$archive")
  fi
  actual=${actual%% *}
  [ "$actual" = "$expected" ] || fail "Checksum mismatch for $archive."
  # Reject links, directories, special files, and duplicate binary members before extraction.
  tar -tvzf "$tmp/$archive" "$name/seyk" > "$tmp/member" || fail "Cannot inspect $archive."
  awk 'NR != 1 || substr($0, 1, 1) != "-" { exit 1 } END { if (NR != 1) exit 1 }' "$tmp/member" || fail "Archive must contain one regular seyk binary."
  tar -xzf "$tmp/$archive" -C "$tmp" "$name/seyk" || fail "Cannot extract $archive."
  [ -f "$tmp/$name/seyk" ] && [ ! -L "$tmp/$name/seyk" ] || fail "Extracted seyk is not a regular file."
  mkdir -p "$dir" || fail "Cannot create $dir; choose a writable SEYK_INSTALL_DIR."
  [ ! -d "$dir/seyk" ] || fail "Install destination is a directory: $dir/seyk."
  candidate="$dir/.seyk.$$"
  [ ! -e "$candidate" ] && [ ! -L "$candidate" ] || fail "Temporary install path already exists: $candidate."
  staged="$candidate"
  install -m 755 "$tmp/$name/seyk" "$staged" || fail "Cannot install to $dir; choose a writable SEYK_INSTALL_DIR."
  installed_version=$("$staged" --version) || fail "Downloaded binary cannot run on this system; the previous installation is unchanged."
  mv -f "$staged" "$dir/seyk" || fail "Cannot replace $dir/seyk."
  staged=""
  echo "Installed $installed_version to $dir/seyk"
  case ":$PATH:" in *":$dir:"*) ;; *) echo "Add $dir to your PATH." ;; esac
}

main "$@"
