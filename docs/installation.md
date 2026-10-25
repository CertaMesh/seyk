# Install Seyk

Prebuilt binaries for macOS (Apple Silicon), Linux (x86_64 and ARM64) and
Windows (x86_64) are attached to each
[release](https://github.com/CertaMesh/seyk/releases). Download the archive for
your platform and its `.sha256` file from there, or use the install script with
the GitHub CLI (run `gh auth login` first):

```bash
gh api repos/CertaMesh/seyk/contents/scripts/install.sh \
  -H "Accept: application/vnd.github.raw" | sh
```

The script picks your platform, checks the SHA-256 checksum and installs to
`~/.local/bin` (set `SEYK_INSTALL_DIR` to change that).

- **Linux needs glibc 2.38 or newer** (Ubuntu 24.04, Debian 13, Fedora 39 and
  later), because ONNX Runtime's prebuilt library does.
- **There is no Intel Mac build**, because ONNX Runtime publishes no prebuilt
  library for it.

From source (Rust 1.89 or newer):

```bash
cargo install --locked --git https://github.com/CertaMesh/seyk
```

The binary is self-contained. ONNX Runtime is linked statically; its prebuilt
library is downloaded by the `ort` crate **at build time**. For a fully offline
build, set `ORT_SKIP_DOWNLOAD=1` and point `ORT_LIB_LOCATION` at a local ONNX
Runtime build.

Older Seyk binaries reset caches whose schema they do not recognize. Use separate
`--cache-path` values when running builds with different cache schemas.

Next, [download the model](model-setup.md).
