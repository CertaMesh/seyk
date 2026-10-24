# Changelog

## Unreleased

- Add experimental `--image-scan q4` and `SearchRequest::image_scan(ImageScan::PhotoAnchorQ4)` for cached standalone images. Exact f32 remains the default and supplies all returned semantic scores.
- Cache schema v7 adds derived image codes and retains existing embeddings. Every opened cache upgrades, including exact-only users. This is a one-way bump: older binaries fail closed on v7. Use a separate cache with older binaries. Import accepts v6/v7 sources and copies f32 only; q4 sync re-derives codes.
- Photo evaluation files default to `~/.cache/seyk-bench/photo-fixture`, with `SEYK_PHOTO_FIXTURE` as an override. The pinned fetcher paces requests and retries rate limits at most three times per file.
