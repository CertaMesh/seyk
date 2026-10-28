# MCP selection reference v1

`ref` is `seyk-ref-v1:` followed by canonical unpadded base64url. It encodes
selection metadata, never source text. Decoding performs no resource, model or
cache reads. When the resource URI is omitted, the selection is the pair
`(ref,path)`. References are caller assertions, not access capabilities.

## Binary layout

Fields appear in the following order. Tags are fixed byte values, independent
of Rust enum layout or JSON field ordering.

| Field | Encoding |
|---|---|
| Domain and version | `53 52 01` hex |
| Flags | Bit 0 omits URI; all other bits must be zero |
| Resource URI | Length-prefixed UTF-8, present only if bit 0 is clear |
| Locator | One-byte tag plus payload below |
| Modality | 0 Document, 1 Text, 2 Image, 3 Audio, 4 Video |
| size_bytes | Unsigned u64 |
| modified_ns | Optional signed i64 |
| media_type | 0 None, 1 Some(application/pdf), 2 literal string |
| content_hash | 32 raw BLAKE3 bytes |

All five original metadata fields survive decoding. Extraction uses modality
and media type for dispatch, verifies content_hash, and returns the original
metadata including size_bytes and modified_ns.
No field is replaced by current file metadata.

| Locator tag | Payload in order |
|---|---|
| 0 Text | u64 byte_start, u64 byte_end, optional-u64 line_start, optional-u64 line_end |
| 1 PageText | u32 page, u64 byte_start, u64 byte_end, string text_version, 32-byte page_text_hash |
| 2 Page | u32 page |
| 3 Time | u64 start_ms, u64 end_ms |
| 4 ImageRegion | u32 x, u32 y, u32 width, u32 height |
| 5 WholeResource | None |

Unsigned integers use minimal unsigned LEB128. u64 takes at most ten bytes;
u32 takes at most five, with overflow and overlong forms rejected. Options
use byte 0 for None or 1 followed by the value for Some. Signed i64 uses
ZigZag followed by unsigned LEB128, including the full i64 range.

Strings use a minimal ULEB128 byte length and exact UTF-8 bytes. There is no
Unicode normalization, case folding or NUL termination. None differs from
Some(""). A literal media tag for exactly `application/pdf` is noncanonical;
it must use tag 1. No other string is interned.

Hash fields accept exactly 64 lowercase hex characters when encoding, store
32 bytes and decode to lowercase hex. Unknown outer/inner versions, tags,
reserved flag bits, invalid UTF-8/base64, nonzero unused base64 bits, padding,
whitespace, truncation and trailing bytes are invalid. There is no compression.

## Identity and bounds

The encoder omits ResourceId only when an absolute UTF-8 `path` reconstructs
exactly the same canonical ResourceId using `ResourceId::from_path`, without
filesystem access. Otherwise it stores the complete canonical URI, including
percent-encoded non-UTF8 file paths. URI decoding must round-trip canonically.
An omitted-URI reference requires the original absolute path; there is no
working-directory, tilde or root fallback. With an explicit URI, display path
is ignored. The decoder also accepts an explicit URI when the encoder would
omit it for the supplied path; decoding remains lossless, but does not enforce
that choice of representation. All display ranges are ignored for selection.

| Bound | Maximum |
|---|---|
| Decoded reference | 65,536 bytes |
| Entire encoded value including prefix | 87,394 ASCII characters |
| URI or supplied path | 16,384 UTF-8 bytes |
| text_version | 4,096 UTF-8 bytes |
| Literal media_type | 1,024 UTF-8 bytes |

The encoded bound is `12 + floor((4*N+2)/3)` for N decoded bytes. The decoder
checks lengths before allocating variable fields. The encoder fails explicitly
on an unrepresentable descriptor; it never trims metadata or silently drops a
hit.

Codec representation and selection validity are separate. For example,
Some(0) round-trips canonically, but line_start/line_end zero cannot select
evidence because lines are 1-based. Page zero, reversed byte/time/line ranges
and mismatched full/ref descriptors are invalid selections. Existing extraction
checks still validate spans, PageText version/hash and audio limits. Source
hashes are never inferred from current bytes.

Literal synthetic vectors in `tests/fixtures/mcp-refs.json` contain descriptor,
binary hex and exact base64url values. Their independent stdlib generator is
`scripts/test/make_mcp_ref_vectors.py`; it also freezes rejected inputs in
`tests/fixtures/mcp-refs-invalid.json`. Codec and selection tests check them at
separate layers. The [MCP input contract](mcp.md#extract-selection) defines the
strict selection table and access checks.
