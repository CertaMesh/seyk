# Install Seyk

Seyk 0.2.0 is the current release. Install it in one of three ways.

## Prebuilt binary

Each [release](https://github.com/CertaMesh/seyk/releases/tag/v0.2.0) has
archives for macOS (Apple Silicon), Linux (x86_64 and ARM64) and Windows
(x86_64), each with a `.sha256` file and the third-party licence notices. Two
platform limits apply:

- **Linux needs glibc 2.38 or newer** (Ubuntu 24.04, Debian 13, Fedora 39 and
  later), because ONNX Runtime's prebuilt library does.
- **There is no Intel Mac build**, because ONNX Runtime publishes no prebuilt
  library for it.

On macOS or Linux, run this. Both the installer source and the binary use the
same tag:

```sh
SEYK_VERSION=v0.2.0
curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL \
  "https://raw.githubusercontent.com/CertaMesh/seyk/refs/tags/$SEYK_VERSION/scripts/install.sh" \
  | env SEYK_VERSION="$SEYK_VERSION" sh
```

`scripts/install.sh` downloads the public archive with curl or wget, verifies
SHA-256, and installs to `~/.local/bin` without sudo. Set `SEYK_INSTALL_DIR` to
another writable directory, `SEYK_VERSION` to pin a different release tag, or
`SEYK_DOWNLOADER=gh` to use the GitHub CLI explicitly. The installer reports an
error if no release or matching asset is available.

On Windows, download the `.zip` archive from the release page and put
`seyk.exe` on your `PATH`.

## With Cargo

The crates are on crates.io. With Rust 1.89 or newer:

```bash
cargo install --locked seyk
```

## From source

To build the latest `main` instead of a release:

```bash
cargo install --locked --git https://github.com/CertaMesh/seyk
```

Cargo builds produce a self-contained binary. ONNX Runtime is linked statically; its prebuilt
library is downloaded by the `ort` crate **at build time**. For a fully offline
build, set `ORT_SKIP_DOWNLOAD=1` and point `ORT_LIB_LOCATION` at a local ONNX
Runtime build.

## Verify a release download

Release archives and their checksum files also receive GitHub build provenance
attestations. SHA-256 catches download corruption; provenance verification ties
the downloaded bytes to this repository's release workflow and tag. Neither
check establishes that the source code is safe.

Optionally, use an authenticated, current GitHub CLI to download and verify both
files before extracting or installing. Set `target` to `aarch64-apple-darwin`,
`x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` or
`x86_64-pc-windows-msvc`. Windows uses `.zip` instead of `.tar.gz`. This extra
verification is optional; `scripts/install.sh` still works with curl or wget
alone.

```sh
SEYK_VERSION=v0.2.0
target=aarch64-apple-darwin
archive="seyk-${SEYK_VERSION#v}-$target.tar.gz"
gh release download "$SEYK_VERSION" --repo CertaMesh/seyk \
  --pattern "$archive" --pattern "$archive.sha256"
for file in "$archive" "$archive.sha256"; do
  gh attestation verify "$file" --repo CertaMesh/seyk \
    --signer-workflow CertaMesh/seyk/.github/workflows/release.yml \
    --source-ref "refs/tags/$SEYK_VERSION" --deny-self-hosted-runners || exit 1
done
```

See the [GitHub CLI verification options](https://cli.github.com/manual/gh_attestation_verify)
for the identity and source-ref checks. Attestations are generated on tag pushes;
manual workflow runs only validate the archives and checksum files.

Older Seyk binaries reset caches whose schema they do not recognize. Use separate
`--cache-path` values when running builds with different cache schemas.

For agent setup, use an absolute binary path. GUI-launched agents may not inherit
your terminal's `PATH`. Run `command -v seyk` to find the installed binary.
Cargo normally installs it to `~/.cargo/bin/seyk`; the release installer defaults
to `~/.local/bin/seyk`. Expand `~` to your full home path in agent configuration.
Use an absolute directory to index too. See [agent setup](mcp.md#set-up-an-agent)
for Claude Code and Codex commands, including Codex timeout settings.

Next, [download the model](model-setup.md).
