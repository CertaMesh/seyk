# Prepare a Photo Anchor calibration experiment

This tool fits candidate M1, M2, and M3 tables on a separate CC0 photo set. It leaves the product codec, bundled tables, cache format, and scan loop unchanged. Candidate tables are experimental artifacts outside the repository.

## Estimate the download

Install the pinned Python dependencies in an isolated environment, or use `uv`:

```sh
uv run --with-requirements scripts/photo_calibration/requirements.txt python scripts/photo_calibration/run.py
```

The default command estimates only. It makes no network requests and loads no models. The image download has a hard upper bound of **4.0 GB** for 2,000 photos at 2,000,000 bytes each. The builder rejects larger images without substituting another photo. Metadata and vectors use additional disk space. The default download budget is 4.0 GB. A count above 2,000 requires an explicitly approved larger `--approved-download-gb` bound before discovery or downloads. A smaller supplied bound also applies, even below 4.0 GB. Both entry points report the effective bound and whether further approval is required.

## Provision the inputs

1. Provision q8 ONNX weights through the normal model setup.
2. Provision the existing evaluation fixture with `scripts/fetch_photo_fixture.py`. The experiment verifies it offline and never downloads evaluation images.
3. Build the exporter before starting the experiment:

```sh
CARGO_BUILD_JOBS=4 nice -n 10 cargo build --release --locked --example photo_calibration_export
```

Use the build and inference resource limits of your machine. Obtain download and inference authorization before the next command.

## Run fitting and measurement

```sh
uv run --with-requirements scripts/photo_calibration/requirements.txt python scripts/photo_calibration/run.py --execute --model "$SEYK_MODEL" --exporter target/release/examples/photo_calibration_export
```

On Windows, pass the exporter path with its `.exe` extension. `--directory` changes the calibration destination. `--fixture` changes the evaluation fixture destination. Neither models nor evaluation images have a download fallback.

The command discovers broad-topic Commons JPEGs, verifies each image's CC0 licence name and URL through [Imageinfo metadata](https://www.mediawiki.org/wiki/API:Imageinfo), and pins the source URL, original hash, licence evidence, and downloaded SHA-256. It rejects non-960px thumbnails, redirects, duplicate source files, duplicate downloaded bytes, and evaluation fixture titles, IDs, or hashes. The selected set is saved before downloads. Image downloads retry HTTP 429 and 503 responses at most three times, honouring `Retry-After` seconds or dates with increasing delays. Invalid status, content type, image size, or redirects stop the download without retries or substitution. An interrupted run resumes that exact selection and cached vectors. A changed image or ineligible selected file stops the experiment.

The seeded BLAKE3 order assigns exactly 1,500 photos to A and 500 to held-out B at the default size. A has a further deterministic 80/20 fit/validation split. Neither B nor evaluation photos enter fitting. The 200 generic query captions are generated independently of evaluation queries and checked for overlap.

M1 chooses symmetric percentile bounds from 0.1, 0.25, 0.5, 1, and 2 percent by reconstruction MSE on A's validation partition. M2 searches independent lower and upper bounds on the same grid with per-coordinate query-energy weighting. A positive coordinate weight cannot change that coordinate's minimizer, so M2's additional freedom comes from asymmetric bounds. M3 widens only A coordinates with clipping above 3 percent to include their 0.5 and 99.5 percentiles, preserving other v1 table bytes.

Vector cache keys include the complete verified model-file SHA-256 map, model and input identities, task, and input bytes. Changing any model file, including same-size external weights or preprocessing files, produces new keys and prevents old vectors from being reused. The previous cache-key format is also invalidated.

The command writes `manifest.json`, `selection.json`, `job.json`, resumable vectors, `export.json`, candidate binary tables, and `candidates/evidence.json` under the calibration directory. The evidence pins model file hashes, manifest and export hashes, model/input identities, seed, table hashes, and per-coordinate choices. It reports clipping and reconstruction MSE separately for B and the evaluation fixture, score MAE and Spearman correlation over fixture pairs, and shortlist recall for k = 1, 5, 10 with R = k, 2k, 50 over B plus the fixture.

Before comparing candidates, the Python encoder and scorer must reproduce production Rust v1 code bytes and q4/f32 score bits, including ranking tie order. Each candidate is checked against sequential f32 decode with absolute limits 5e-5 at width 256 and 1.5e-4 at width 768. This is an offline arithmetic check; a future product implementation still needs independent Rust tests against QueryLut and sequential decode, cache migration tests, and semantic/hybrid engine gates.

The decision prefers M3 when its mean recall at R = 2k is within 0.5 percentage points of the best method. It keeps v1 unless the candidate gains at least 1 percentage point of mean recall or reduces MAE by at least 10 percent without losing recall at any measured cell. It also checks exact rerank order and nDCG@10 drop at most 0.005. A candidate is never marked merge-eligible by this tool. Do not tune again against held-out results or copy experimental tables into product assets without the remaining gates and attribution updates.

To recompute evidence without loading a model or using the network:

```sh
uv run --with-requirements scripts/photo_calibration/requirements.txt python scripts/photo_calibration/run.py --score-only
```

## Run timing after quality approval

The existing Criterion benchmark remains the timing tool. Prebuild its binary outside the quiet slot:

```sh
CARGO_BUILD_JOBS=4 nice -n 10 cargo bench -p seyk-core --bench similarity --no-run --locked
```

During an approved quiet slot, run the prebuilt benchmark executable directly:

```sh
<similarity-binary> 'photo_anchor_scan/' --bench --sample-size 30 --warm-up-time 0.5 --measurement-time 2
```

A future codec change needs separately prebuilt baseline and candidate binaries. Alternate baseline, candidate, baseline, candidate, baseline, recording load and memory at the slot boundaries. Compare the medians from Criterion's `estimates.json`. Candidate q4 must be within noise of baseline q4 and at most 2,000 microseconds per 1,000 vectors slower than f32 at both widths. This preparation change ships no fitted codec, so it makes no timing or retrieval improvement claim.
