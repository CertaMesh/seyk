# Phase 4b compatibility spike — do not merge

Throwaway experiment at main f9af387. No product crate changes. See Solo project 73 pad 11511 for authority and the final gate report.

Reproduce from this directory with pinned Rust 1.99.0:

```
cargo check --locked --features ocr
cargo +1.89.0 check --locked --features render,native-ocr
cargo build --release --locked --features render,native-ocr,embed
target/release/seyk-4b-spike render
target/release/seyk-4b-spike embed
cargo deny --locked --all-features check licenses --config ../../deny.toml
```

The OCR check is expected to fail: upstream 0.6.1 is API-incompatible with ORT rc.13. No dependency fork or alternate runtime is used. Renderer/geometry builds and measurements are separate evidence, not proof that OCR ships.

`corpus.json` is wholly synthetic, authored for this experiment under the repository's Apache-2.0 licence. The 20 scanned PDFs contain only raster image XObjects. Their source text is rendered with Hayro's bundled Foxit fonts, never system fonts. Retrieval consumes the re-rendered PNG bytes at long edge 2048, image SearchDocument embeddings, default dimension 256 and Q8 precision. It performs no OCR. Fourteen fixed queries include ten semantic questions, two identifiers and two exact titles. No filename boosts or lexical search. Predeclared Recall@5 >=80%, clean printed OCR CER <=5%; no threshold tuning.

Rendered synthetic PDF/PNG intermediates exist only in ignored `generated/` for this experiment. This is not a product cache design. Corpus generation includes no private data or third-party content.

Model weights are explicitly provisioned outside git. The embedding command uses Seyk's existing pinned Gemma cache and makes no download call. First-party PP-OCR ONNX assets load through ORT rc.13 offline; their exact conversion recipe/version is unproven. OCR inference/CER remains blocked because the approved adapter does not compile. `download_models.py` is an explicit provisioning command and verifies the committed SHA-256/length pins. `profile.py` samples RSS at 50 ms intervals on macOS; it does not send content off the host.

The draft PR exists only to obtain native four-target build/link evidence. It must be closed after the report, never merged.

Hayro's 14 bundled Foxit fallback fonts use BSD-3-Clause; their notice is preserved in `FONT-LICENSE.txt`, and `evidence/bundled-fonts.json` records hashes. Cargo-deny checks the actual selected Rust tree; font licences are inspected separately because bundled assets have no Cargo package metadata. Renderer defaults resolve only these embedded fonts; the sandbox check denies network and system font directories.
