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
licences and source notices in the Cargo registry. The `pdf-render` feature adds Hayro 0.5.0 (Apache-2.0 OR MIT), its CPU
rasterizer and bundled fallback fonts. No OCR runtime or model assets are included.

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

## PDF rendering fonts

Hayro 0.5.0 uses hayro-interpret 0.5.0 and its 14 bundled Foxit fallback
fonts (BSD-3-Clause). They were extracted from PDFium; original code copyright
2014 Foxit Software Inc. The pinned source archive is
[hayro-interpret 0.5.0](https://crates.io/api/v1/crates/hayro-interpret/0.5.0/download).
No system fonts are loaded. Retain the following notice when distributing
executables with `pdf-render`.

| File | Bytes | SHA-256 |
|---|---:|---|
| FoxitDingbats.pfb | 29513 | `845c752392b6c914fb989c75a08b7792b88f542d2499042ef2889f8c814a16ed` |
| FoxitFixed.pfb | 17597 | `b6c8fe53f134b8b6d4578cd2d544df4cee9624c4efa8d51a560fb40ea296101b` |
| FoxitFixedBold.pfb | 18055 | `f1b7159702973f54fc86254ea38bcf3712b2a736eb3a0995751e9f4bb45ad603` |
| FoxitFixedBoldItalic.pfb | 19151 | `8a000945843bd31add06aee63bf9fd41b7578b4f3242a4b7cd46349ad9d24d4d` |
| FoxitFixedItalic.pfb | 18746 | `5007faf8320fa1fcc08b8894b356ad976f60b992ba29ee5272578bbdebaf3876` |
| FoxitSans.pfb | 15025 | `a02339e95d20db14cbdfc1c6baf359cd36a091d5434925c3a4040f60e1a1af9e` |
| FoxitSansBold.pfb | 16344 | `7bdfa36de1d5757dcfaf8b8a4be95c6e41c40fa70dbed9d78bf979a7b9959054` |
| FoxitSansBoldItalic.pfb | 16418 | `555ce45924338e2e4b32821910221ce85da691fd59a9723c4c971a7c5627fc8b` |
| FoxitSansItalic.pfb | 16339 | `7751cbefeda351708dae8c4a88db043411bc39ab2ae2097685da845d4d16f307` |
| FoxitSerif.pfb | 19469 | `4f57d2b9d884af8f907bf22df6019b52d86cbf6214fdbd02b1ae05472a543f35` |
| FoxitSerifBold.pfb | 19395 | `0bdf4b04e964139818d51eda03d566ba999fc3ba2421b1c6c51f9dc969022e80` |
| FoxitSerifBoldItalic.pfb | 20733 | `a406cac82583bf98175cb62c87ed5e95c45fbb34eece63a10b0a13e793cb2e10` |
| FoxitSerifItalic.pfb | 21227 | `610ae0687198045c4db5d1a6650fd5e92536631706382eb8b72e38df578d0ae9` |
| FoxitSymbol.pfb | 16729 | `47967d055530e7357088a08403115425643ec2cdfd6201ba8af0fbd7116c1539` |

```text
// Copyright 2014 PDFium Authors. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//    * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//    * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

Hayro also embeds a CC0-1.0 compact CMYK colour profile and a generated LAB
profile. Their provenance and CGATS notice accompany the pinned
hayro-interpret archive (`assets/README.md`, `assets/CGATS_LICENSE.txt`).
