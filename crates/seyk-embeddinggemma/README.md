# seyk-embeddinggemma

Local EmbeddingGemma 2 inference.

Runs text embedding on CPU through ONNX Runtime. Enable `image` for image
embedding or `audio` for audio embedding. Model files must be provisioned
locally; inference sends no content, queries or vectors to a remote service.

ONNX Runtime is downloaded by the `ort` dependency during a normal build.
The separate model weights are not bundled in this crate.

```toml
[dependencies]
seyk-embeddinggemma = "0.1.0"
```

[API documentation](https://docs.rs/seyk-embeddinggemma) ·
[Rust library guide](https://github.com/CertaMesh/seyk/blob/main/docs/library.md) ·
[Repository](https://github.com/CertaMesh/seyk)

## License

Apache-2.0. See [LICENSE](https://github.com/CertaMesh/seyk/blob/main/crates/seyk-embeddinggemma/LICENSE) and [NOTICE](https://github.com/CertaMesh/seyk/blob/main/crates/seyk-embeddinggemma/NOTICE).
