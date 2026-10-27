# Offline MCP payload accounting

This check measures response characters and UTF-8 bytes through the production
MCP projector and reference codec. It does not run agents, load models, download
assets, index a corpus or measure provider token usage. A smaller payload does
not establish lower agent token usage or better answers.

## Public synthetic check

The manifest command creates its output exclusively, with mode 0600. Use a new
output name for each frozen input set. Keep manifests outside the repository.

```bash
python3 scripts/bench/mcp_payload_manifest.py --synthetic /private/tmp/mcp-synthetic.json
# Copy the printed SHA-256 into EXPECTED_SHA256.
CARGO_BUILD_JOBS=4 nice -n 10 cargo run --locked --example mcp_payload_sizes -- /private/tmp/mcp-synthetic.json EXPECTED_SHA256
```

The invented fixtures cover ten-hit text, PDF, audio, image-region and
whole-resource responses, with both omitted and explicit resource URIs. The
utility uses `src/bin/seyk/mcp_wire.rs` directly. `ContentBlock::json` supplies
the actual MCP compact JSON text and the reported text-block wrapping.

## Recorded-input check

```bash
python3 scripts/bench/mcp_payload_manifest.py /path/to/scored/runs/main /private/tmp/mcp-recorded.json
CARGO_BUILD_JOBS=4 nice -n 10 cargo run --locked --example mcp_payload_sizes -- /private/tmp/mcp-recorded.json EXPECTED_SHA256
```

The scored directory must contain `results.jsonl`, `labels.json` and each raw
transcript. Input tasks must appear in both records and labels, with recognized
arm/condition values; duplicates fail. The tool selects completed search results
by task and call ID, deduplicates transcript events and records summary/raw call
counts. It never selects smoke or diagnostic directories implicitly. The local
manifest retains exact payload strings, source paths, source SHA-256 hashes,
call IDs and inclusion records. Review its reconciliation before interpreting
results. Failed tools, malformed payloads and unrepresentable descriptors are
reported separately instead of silently omitted.

The second command requires the expected manifest SHA-256 and checks source
file hashes again. Record its command, Rust version, dependency lock hash and
serializer commit with the aggregate output. Outputs omit original task names,
paths, previews and prompts; per-task rows contain only condition, search count
and projected character/byte deltas. Do not commit manifests, private transcripts
or derived private fixture content. Commit only the utility and synthetic data.

## Interpretation

The four paired forms are as-recorded protocol-10 full, normalized protocol-10
full, new protocol-11 full and protocol-11 slim. Normalized forms use the same
compact JSON escaping and UTF-8 serialization. New full output includes its new
reference/time members. Payload text and MCP text-block wrapping are separate.
Report totals, per-call paired ratios, median/p90 sizes, the ten-hit subset and
reference binary/base64/JSON-member cost by locator kind and identity mode.
Percentiles use nearest rank, `ceil(p*n)`.

Per-task projections sum actual paired character/byte deltas across all its
searches. They are not the product of two medians, and they cannot be converted
to a causal change in provider input/cache/output token usage. Base64 tokenizes
differently from prose. There are no token-saving claims or chars/4 conversion.

Historical payloads lack raw skipped-resource identities. When distinct warning
counts cannot be reconstructed, the output marks warning inputs unknown and
reports the known-input projection only. Synthetic warning fixtures exercise
nonzero warning overhead separately; they do not estimate its historical
prevalence. Tool definitions are fixed overhead outside these response counts;
a tools/list comparison needs separately captured old/new definitions. This
utility makes no total-context or tool-definition savings claim.

The optional paid agent evaluation remains deferred. It needs separate
permission, fixed corpus/prompts/models/setup/grading, and correctness plus
citation and tokens per correct-and-cited answer. Offline payload accounting
does not complete that evaluation criterion.

## Measured payload sizes

The production serializer was run offline with Rust 1.99.0 on 59 successful
recorded search results. The immutable local manifest contains 92 scored tasks
and 63 tool results; four tool errors are excluded from payload comparison.
There are zero malformed or unrepresentable successful results and no unknown
query-warning inputs in this sample. All recorded hits use text locators with
omitted URIs. Other locator costs below come from public synthetic fixtures.

| Sample | Calls | Normalized v10 characters | v11 slim characters | Total size ratio |
|---|---:|---:|---:|---:|
| All successful recorded calls | 59 | 463,019 | 232,713 | 1.990 |
| Recorded ten-hit calls | 30 | 289,747 | 147,529 | 1.964 |
| Synthetic ten-hit text, omitted URI | 1 | 6,929 | 3,406 | 2.034 |
| Synthetic ten-hit PageText, omitted URI | 1 | 8,119 | 3,976 | 2.042 |

The 34 successful Claude calls reproduce the historical midpoint median of
9,535 as-recorded characters. This utility uses nearest-rank percentiles, so
its corresponding median is 9,529. The new slim midpoint median for those calls
is 4,904.5 characters, or 4,896 using nearest rank. Ratios above use sums of
paired payload lengths, not ratios of medians.

The retained preview and self-contained reference limit the reduction. Removing
descriptor JSON does not remove the metadata required for extraction. Recorded
text references have median value length 94 characters, p90 98, range 87–98
across 474 hits. Their JSON member costs eight more characters before the
separating comma. Synthetic reference value lengths are:

| Locator | Omitted URI | Explicit URI |
|---|---:|---:|
| Text | 104 | 134 |
| PageText | 166 | 195 |
| Page | 82 | 111 |
| Time | 99 | 128 |
| ImageRegion | 103 | 132 |
| WholeResource | 95 | 124 |

These values include the prefix and base64url text, but exclude the JSON key,
quotes and comma. Each synthetic column uses ten hits; an additional text
warning fixture repeats the same reference shape. The nonzero warning object
adds 61 characters to the otherwise identical synthetic slim response.

The public command above regenerates synthetic inputs; source, parser,
manifest, dependency-lock and runtime fingerprints for the recorded run remain
with its private local audit artifacts. No transcript content, source paths or
private derived fixtures are committed. Tool definitions and total agent
context are outside this comparison. The paid harness remains deferred, and
these measurements make no token, speed, accuracy or agent-outcome claim.
