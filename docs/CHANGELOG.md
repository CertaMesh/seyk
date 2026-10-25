# Changelog

## Unreleased

- Isolate shared daemons by build version and cache schema as well as JSON schema/model options. A mismatched client starts a matching daemon without interrupting older sessions. Status/stop address only the invoking build; daemon status/logs and MCP server metadata report the build version. JSON schema remains v9 and cache schema v7; incompatible cache versions still require separate cache paths.

- Identify non-release builds in `seyk --version` with `+g<short-sha>` and `.dirty` for modified trees. Clean release-tag builds keep the plain version when the tag is present locally; `cargo install --git --tag` may show the SHA suffix. Builds without Git or repository metadata keep the plain version. Untracked files outside the watched source paths update `.dirty` on the next rebuild. Human-readable `seyk cache stats` now reports the SQLite cache schema version.
- Add opt-in `seyk index <root> --progress jsonl` for live resource counts and ETA on stderr. Final stdout JSON and default human output stay unchanged. See [the v1 protocol](index-progress.md).
- Add experimental `--image-scan q4` and `SearchRequest::image_scan(ImageScan::PhotoAnchorQ4)` for cached standalone images. Exact f32 remains the default and supplies all returned semantic scores.
- Cache schema v7 adds derived image codes and retains existing embeddings. Every opened cache upgrades, including exact-only users. This is a one-way bump: older binaries fail closed on v7. Use a separate cache with older binaries. Import accepts v6/v7 sources and copies f32 only; q4 sync re-derives codes.
- Photo evaluation files default to `~/.cache/seyk-bench/photo-fixture`, with `SEYK_PHOTO_FIXTURE` as an override. The pinned fetcher paces requests and retries rate limits at most three times per file.
