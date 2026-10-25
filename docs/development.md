# Development

This page is for contributors: how the model path is checked, how to run the
tests, and how to evaluate retrieval on your own notes. See also
[CONTRIBUTING.md](../CONTRIBUTING.md) and [Architecture](architecture.md).

## Correctness checks

The Rust/ONNX path is compared against the official `sentence-transformers`
implementation in float32 (`tests/parity.rs`, `tests/multimodal.rs`):

| Check | Result |
|---|---|
| Text token ids (with prompts) | identical |
| Text vectors, fp32, at 768/512/256/128 | worst cosine 1.00000 |
| Text vectors, q8 | worst cosine 0.99994 |
| Image resize size, patch positions, token ids | identical |
| Image vectors, fp32 | worst cosine 0.99999 |
| Image vectors, q8 | worst cosine 0.9979 |

The remaining image difference comes from the resize filter (a different bicubic
implementation than torchvision's). It changes individual pixels by at most 0.06.

## Testing

To reproduce cold CLI startup failures under load, use
[`scripts/test/flaky_tests.py`](../scripts/test/flaky_tests.py) with `--cold-cli`.
Build commands and options are in the script's `--help`.

```bash
cargo test --workspace             # no model needed: segmentation, cache, CLI, ranking
cargo test --workspace --release -- --ignored  # model-backed tests; image parity needs reference files
```

Model-backed tests read `SEYK_MODEL` (local model dir) and
`SEYK_TEST_PRECISION` (`q8` or `fp32`). The image parity tests also need a
reference dump from `scripts/reference_image_embeddings.py` (see the test for the
environment variables).

## Evaluating on your own notes

`examples/eval_corpus.rs` runs a list of questions against a folder and prints the
semantic hits next to a naive keyword baseline. It also flags hits that share no
content words with the question, which is the material only semantic search can find:

```bash
cargo run --release --example eval_corpus -- \
  --queries examples/queries.example.txt ~/notes
```

Everything runs locally.
