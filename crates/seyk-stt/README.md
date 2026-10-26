# seyk-stt

Offline, on-demand Whisper transcription.

Implements the `seyk-core` speech-to-text interface with pinned Whisper tiny
ONNX models. Transcription runs locally when extraction requests it.
Model weights must be provisioned separately and are not bundled.

ONNX Runtime is downloaded by the `ort` dependency during a normal build.

```toml
[dependencies]
seyk-stt = "0.1.0"
```

[API documentation](https://docs.rs/seyk-stt) ·
[Rust library guide](https://github.com/CertaMesh/seyk/blob/main/docs/library.md) ·
[Repository](https://github.com/CertaMesh/seyk)

## License

Apache-2.0 for source code and MIT for bundled Whisper mel filters.
See [LICENSE](https://github.com/CertaMesh/seyk/blob/main/crates/seyk-stt/LICENSE), [LICENSE-MIT](https://github.com/CertaMesh/seyk/blob/main/crates/seyk-stt/LICENSE-MIT) and [NOTICE](https://github.com/CertaMesh/seyk/blob/main/crates/seyk-stt/NOTICE).
