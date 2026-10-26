# ONNX Runtime third-party notices

`ThirdPartyNotices.txt` is the unmodified upstream notice file for the native
ONNX Runtime 1.28.0 library statically linked by `ort-sys 2.0.0-rc.13`.

- Upstream tag: [`v1.28.0`](https://github.com/microsoft/onnxruntime/tree/v1.28.0).
- Resolved commit: `da9b5e364c465de65c49d91e696cd6485270757f`.
- Source file: [ThirdPartyNotices.txt](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.28.0/ThirdPartyNotices.txt).
- SHA-256: `0e07b95f3a8d6230037707c5c4a2b554d12c4cb67369669ac255635528ffcee2`.

The tagged file was checked byte-for-byte against the file at the resolved
commit. `SHA256SUMS` records its digest. The local licence generator verifies
that digest before embedding the full notice text in `THIRD_PARTY_LICENSES.html`.
Every release archive also includes this directory and the original text file.

The notice covers native dependencies, including abseil, protobuf, ONNX, re2,
flatbuffers, cpuinfo, GSL, nlohmann/json and Eigen. Eigen is the native MPL-2.0
exception documented in [THIRD_PARTY.md](../../THIRD_PARTY.md). Its source is
available from the [Eigen repository](https://gitlab.com/libeigen/eigen).
The corresponding dependency revisions and build integration are available in
the [pinned ONNX Runtime source tree](https://github.com/microsoft/onnxruntime/tree/da9b5e364c465de65c49d91e696cd6485270757f).

When changing the native runtime, update the notice file, tag, commit and digest
together. Keep the runtime MIT notice in `about.hbs` aligned with that version.
