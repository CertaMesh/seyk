# Install Seyk

From source (Rust 1.89 or newer):

```bash
cargo install --locked --git https://github.com/CertaMesh/seyk
```

There is no published release right now. Prebuilt binaries for macOS (Apple
Silicon), Linux (x86_64 and ARM64) and Windows (x86_64) return with the next
[release](https://github.com/CertaMesh/seyk/releases), each with a `.sha256`
file and the third-party licence notices. Two platform limits apply to them:

- **Linux needs glibc 2.38 or newer** (Ubuntu 24.04, Debian 13, Fedora 39 and
  later), because ONNX Runtime's prebuilt library does.
- **There is no Intel Mac build**, because ONNX Runtime publishes no prebuilt
  library for it.

The binary is self-contained. ONNX Runtime is linked statically; its prebuilt
library is downloaded by the `ort` crate **at build time**. For a fully offline
build, set `ORT_SKIP_DOWNLOAD=1` and point `ORT_LIB_LOCATION` at a local ONNX
Runtime build.

Older Seyk binaries reset caches whose schema they do not recognize. Use separate
`--cache-path` values when running builds with different cache schemas.

Next, [download the model](model-setup.md).
