# STT fixtures

`scripts/audio/stt_fixture.py` produces `mel-input.f32le` and
`mel-reference.f32le.gz`. They store mathematical input as little-endian
float32 samples and its reference as a gzip-compressed `[80, 3000]` mel tensor,
respectively.
`provenance.json` records their OpenAI reference commit, environment, tolerance,
and SHA-256 hashes. Its checksum map covers only these two outputs.

The script also produces local speech in ignored `tmp/` using macOS `say`.
The text, voice, and rate in `provenance.json` describe that procedure. They do
not establish which speech or decoder produced `token-goldens.json`.
Never commit or redistribute System Voice audio. Record only its PCM hash
when documenting a deliberate future capture.

`token-goldens.json` is a frozen decoder capture with 10-second and 30-second
token arrays. `token-goldens.provenance.json` records its provenance separately.
The historical decoder manifest identity, speech PCM hash, capture procedure,
and capture revision are `unknown`. The retained records do not establish them.
There is no committed speech PCM input from which to recover that hash.

The sidecar's `present_day_verification` records facts verified at commit
`912559f20b6a0ba8f340951f226609b03e6bf3ca`: the token file hash and the current
verifier's decoder manifest path and hash. These facts do not identify the
historical generator. Fixture checksum keys are relative to this directory;
the manifest path is relative to the repository root. The two mel hashes in
`provenance.json` also match the files at that commit.

The current verifier in `crates/seyk-stt/src/tests.rs` reads local 16 kHz mono
signed 16-bit little-endian PCM, scales samples by `1/32768`, appends 8000 zero
samples, then cycles and truncates to 10 or 30 seconds. Its ignored
`model_backed_transcript_cache_parity_silence_and_installation` test checks the
token arrays and transcript constants. This describes verification today,
not the unknown historical capture procedure.

The metadata correction preserves every fixture payload. Regenerating mel
fixtures does not refresh the token capture or its sidecar. A deliberate token
refresh must record its decoder manifest identity, input PCM hash, capture
procedure, and revision independently. Do not infer missing history from
current code or a new local speech generation.
