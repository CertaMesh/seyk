# Third-party components

Seyk is Apache-2.0. Dependency licence choices are checked by `cargo deny check
licenses` against `deny.toml` and the locked graph. GPL, AGPL, LGPL and SSPL
choices are not allowed. A permissive alternative in a dual-licence expression
may be selected. MPL requires explicit review, including existing
dependencies. An approved exception is `option-ext 0.2.0` (MPL-2.0), an existing
transitive dependency of `dirs`. Its unmodified source is available at
https://crates.io/crates/option-ext/0.2.0 and its MPL text accompanies the crate.
Additional scoped exceptions for opt-in audio are listed below.

PDF text extraction uses pdf-extract 0.12.1 (MIT) and lopdf 0.42.0 (MIT), plus
MIT-licensed Adobe CMap, Type 1 encoding and font parsers. See their packaged
licences and source notices in the Cargo registry. The `pdf-render` feature adds
Hayro 0.5.0 (Apache-2.0 OR MIT), its CPU rasterizer and bundled fallback fonts. No OCR runtime or model assets are included.

EmbeddingGemma 2 weights and tokenizer files are not bundled in the executable.
The operator provisions them locally or uses the explicit model download
command. The pinned export declares Apache-2.0 in its
[model card](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/blob/daa72c51243991dfcaf9f9137d2c573d8f7790c0/README.md).
Preserve that licence and upstream notices when distributing the model assets.

ONNX Runtime (MIT) is the native inference dependency; SQLite is bundled by
rusqlite (SQLite is public domain). Their upstream distributions include notices
for native/transitive components. PDF support adds no native runtime. Release builders must preserve applicable upstream notices when
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

## Optional audio decoding (Phase 5a)

Unmodified Symphonia crates, approved for the optional `audio` feature. Each is
MPL-2.0, with the license text included at `LICENSE` in its crates.io source
archive. These are package+version exceptions only; new versions require new
approval.

- [symphonia 0.5.5](https://crates.io/crates/symphonia/0.5.5) — MPL-2.0; [source and license text](https://crates.io/api/v1/crates/symphonia/0.5.5/download).
- [symphonia-bundle-flac 0.5.5](https://crates.io/crates/symphonia-bundle-flac/0.5.5) — MPL-2.0; [source and license text](https://crates.io/api/v1/crates/symphonia-bundle-flac/0.5.5/download).
- [symphonia-bundle-mp3 0.5.5](https://crates.io/crates/symphonia-bundle-mp3/0.5.5) — MPL-2.0; [source and license text](https://crates.io/api/v1/crates/symphonia-bundle-mp3/0.5.5/download).
- [symphonia-codec-aac 0.5.5](https://crates.io/crates/symphonia-codec-aac/0.5.5) — MPL-2.0; [source and license text](https://crates.io/api/v1/crates/symphonia-codec-aac/0.5.5/download).
- [symphonia-codec-alac 0.5.5](https://crates.io/crates/symphonia-codec-alac/0.5.5) — MPL-2.0; [source and license text](https://crates.io/api/v1/crates/symphonia-codec-alac/0.5.5/download).
- [symphonia-codec-pcm 0.5.5](https://crates.io/crates/symphonia-codec-pcm/0.5.5) — MPL-2.0; [source and license text](https://crates.io/api/v1/crates/symphonia-codec-pcm/0.5.5/download).
- [symphonia-core 0.5.5](https://crates.io/crates/symphonia-core/0.5.5) — MPL-2.0; [source and license text](https://crates.io/api/v1/crates/symphonia-core/0.5.5/download).
- [symphonia-format-isomp4 0.5.5](https://crates.io/crates/symphonia-format-isomp4/0.5.5) — MPL-2.0; [source and license text](https://crates.io/api/v1/crates/symphonia-format-isomp4/0.5.5/download).
- [symphonia-format-riff 0.5.5](https://crates.io/crates/symphonia-format-riff/0.5.5) — MPL-2.0; [source and license text](https://crates.io/api/v1/crates/symphonia-format-riff/0.5.5/download).
- [symphonia-metadata 0.5.5](https://crates.io/crates/symphonia-metadata/0.5.5) — MPL-2.0; [source and license text](https://crates.io/api/v1/crates/symphonia-metadata/0.5.5/download).
- [symphonia-utils-xiph 0.5.5](https://crates.io/crates/symphonia-utils-xiph/0.5.5) — MPL-2.0; [source and license text](https://crates.io/api/v1/crates/symphonia-utils-xiph/0.5.5/download).

## Optional speech-to-text (Phase 5b-2)

`audio-stt` uses the separate FP32 graphs from
[Xenova/whisper-tiny at 5332fcc35e32a33b86612b9a57a89be7906102b1](https://huggingface.co/Xenova/whisper-tiny/tree/5332fcc35e32a33b86612b9a57a89be7906102b1).
The [card at that exact revision](https://huggingface.co/Xenova/whisper-tiny/blob/5332fcc35e32a33b86612b9a57a89be7906102b1/README.md)
declares `license: apache-2.0` and identifies Transformers.js. Its
1,160 bytes have SHA-256 `cdd395427d195f122aee69c00e34183f2fadd8bc217aef35ca7c43395b96d29d`.
There is no separate LICENSE or NOTICE file in that revision's tree. Preserve
the [exact card](crates/seyk-stt/assets/XENOVA-CARD.md) and
[Apache-2.0 text](crates/seyk-stt/assets/APACHE-2.0.txt) with the attribution.
We do not rely on the unlicensed onnx-community export. Model weights and
runtime tokenizer/configuration assets are not bundled; users explicitly
provision them with `seyk model download --stt`. The compiled
[manifest](crates/seyk-stt/assets/manifest.json) records every length and hash.

Original Whisper weights, reference mel code and mel filters come from
[OpenAI Whisper](https://github.com/openai/whisper/tree/86098128c0b4f24f0e2aa2994de830614b474227),
licensed MIT. The small bundled float32 mel bank is derived from its pinned
`whisper/assets/mel_filters.npz`; its SHA-256 and reference provenance are in
the manifest and fixture generator. Retain the following notice (also saved as
[OPENAI-MIT.txt](crates/seyk-stt/assets/OPENAI-MIT.txt)):

```text
MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

RustFFT is MIT OR Apache-2.0; tokenizers is Apache-2.0. ONNX Runtime is the MIT
native runtime also used by the optional embedding backend. Cargo-deny checks the
locked Rust graph; release packaging must also preserve the model, mel and
native-runtime notices described here. The speech-to-text dependencies add no
copyleft exception.

### Local speech test generation

macOS System Voice (`say`) output is for local testing only and must never be
committed or redistributed. Speech gates require `SEYK_STT_SPEECH_WAV`, generated
under ignored `tmp/` by `scripts/audio/stt_fixture.py`. Only public text/token
goldens and mathematical chirp/mel fixtures are bundled.

## Photo Anchor tables

Photo Anchor for EmbeddingGemma 2, by Pastel-Cloud OÜ (Pastel Labs).
Source: https://huggingface.co/pastel-labs/embeddinggemma-2-photo-anchor/tree/be9e066984345c92424c86bfcce3b17f3e3e8484
License: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/ .

The manifest and 256/768-dimensional anchor and calibration files are bundled
byte-for-byte, unmodified. No images or model weights are included. These
assets remain CC BY 4.0; Seyk's source code is Apache-2.0. Retain this attribution
in packages and binary distributions containing the tables.

The tables derive from 18,232 COCO 2017 train Flickr photos (CC BY 2.0 or
no known copyright restrictions). Individual credits are in the upstream
`data/photo_credits.csv` at the revision above. COCO annotations are CC BY 4.0;
see Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
The source embeddings used google/embeddinggemma-2 at
914f7f89142e33e77833254d9c9b90c3cef7303b (float16, Sentence Transformers).

Files live in `crates/seyk-embed/assets/photo-anchor/`.

| Bundled file | Bytes | SHA-256 |
|---|---:|---|
| `manifest.json` | 6241 | `4dbde655fdd44a1feb4b010685fe065ffbe9c1f996331b4f4a40c9a5b2580c1e` |
| `photo_anchor_256.f32.bin` | 1024 | `4e30ed052129ee745bb810fc1673e360857557e4a97daffdb2734e2b8dc5dc28` |
| `quant4_256.f32.bin` | 2048 | `fe029c4cf510f6a628a24bcdc3db097ce8c387e948080d5065ef81ff904a2353` |
| `photo_anchor_768.f32.bin` | 3072 | `a3978221be716a22f0d738efdc93df0de90fb8c490339011422ab15c2e8a3bb6` |
| `quant4_768.f32.bin` | 6144 | `5088d57dbb8ae53da9f45f94ce20aa9177160fcbc499d41a8ecb755270ba5560` |

EmbeddingGemma 2 use is also subject to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
