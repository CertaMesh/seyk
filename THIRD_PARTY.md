# Third-party components

Seyk is Apache-2.0. Dependency licence choices are checked by `cargo deny check
licenses` against `deny.toml` and the locked graph. GPL, AGPL, LGPL and SSPL
choices are not allowed. A permissive alternative in a dual-licence expression
may be selected. MPL requires explicit review, including existing
dependencies. The approved exception is only `option-ext 0.2.0` (MPL-2.0), an existing transitive dependency of `dirs`; no other MPL
package or version is permitted. Its unmodified source is available at
https://crates.io/crates/option-ext/0.2.0 and its MPL text accompanies the crate.

PDF text extraction uses pdf-extract 0.12.1 (MIT) and lopdf 0.42.0 (MIT), plus
MIT-licensed Adobe CMap, Type 1 encoding and font parsers. See their packaged
licences and source notices in the Cargo registry. The adapter does not bundle
font programs, a renderer or OCR assets. In particular, no Hayro fonts or OCR
models are included in Phase 4a.

EmbeddingGemma 2 weights and tokenizer files are not bundled in the executable.
The operator provisions them locally or uses the explicit model download
command. The pinned export declares Apache-2.0 in its
[model card](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/blob/daa72c51243991dfcaf9f9137d2c573d8f7790c0/README.md).
Preserve that licence and upstream notices when distributing the model assets.

ONNX Runtime is the existing native inference dependency (MIT); SQLite is
bundled by rusqlite (SQLite is public domain). Their upstream distributions
include notices for native/transitive components. This PDF phase adds no native
runtime. Release builders must preserve applicable upstream notices when
redistributing native artifacts or separately provisioned model assets.
