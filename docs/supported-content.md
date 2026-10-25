# Supported content

This page lists the file types Seyk indexes, the files it always refuses, and
how each kind of content becomes searchable evidence.

## Supported resources

| Kind | Extensions | Status |
|---|---|---|
| Markdown | `.md` `.markdown` `.mdown` `.mkd` | ✅ heading-aware segments |
| Plain text | `.txt` `.text` `.rst` `.org` `.adoc` `.tex` | ✅ paragraph segments |
| Source code | `.rs` `.py` `.js` `.ts` `.go` `.java` `.c` `.cpp` `.rb` `.php` `.sh` `.sql` `.toml` `.yaml` and more | ✅ blank-line blocks |
| Images | `.png` `.jpg` `.jpeg` `.webp` `.gif` `.bmp` `.tiff` | ✅ one vector per image (`WholeResource`) |
| Audio | `.wav` `.flac` `.mp3` `.m4a` | Opt-in `audio` feature; 10 s windows, `Time` evidence; search never transcribes, `audio-stt` extracts a transcript on request |
| Video | — | ❌ not implemented |
| PDF | Embedded text and scanned pages | Text evidence available; scan text extraction needs OCR |

A file passed explicitly on the command line is searched as plain text even with
an unknown extension, except the secret paths listed below. Directories skip `.git`, `target`, `node_modules`, `.hg`,
`.svn`, `__pycache__`, `.venv` and hidden entries by default.

## Files Seyk never indexes

Seyk blocks these names before reading file contents. Matching uses Unicode
compatibility normalization and full case folding on every platform (including
long `ſ`, Kelvin `K`, and compatibility spellings):

- Keys and certificate containers: `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.jks`, `*.keystore`.
- SSH private keys: `id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519`, `id_xmss`,
  including copies with a `.`, `_`, or `-` suffix (for example `id_rsa.backup`
  or `id_ed25519_sk`). Names ending in `.pub` are exempt from this SSH rule.
- Password stores: `*.kdbx`, `*.kdb`, `*.1pif`, and all contents of
  `*.agilekeychain` and `*.opvault` directories.
- Environment files: `.env`, `.env.*`, `*.env`, including examples and templates.
- Keychain exports: `*.keychain`, `*.keychain-db`.
- Credentials: `.netrc`, `netrc`, `_netrc`, `.pgpass`, `pgpass`, `pgpass.conf`,
  `credentials` (including `.aws/credentials`), `credentials.json`, and
  `.docker/config.json`.

There is no override. `--include`, `--hidden`, disabling default directory
exclusions, and library/MCP filters cannot admit these paths. The same policy
applies to watching, cached-only search, previews, and extraction. Discovery
skips blocked paths; explicitly naming one returns a `refusing secret file or
directory` error. Every supplied component, intermediate symlink target, and final canonical path
is checked. Resolution stops with an error after 40 symlink hops. Logical
directory names survive every hop, so an alias through a `.docker` directory
symlink still blocks its `config.json`, including paths containing `..`. Password-store directory components
(`*.opvault`, `*.agilekeychain`) block every descendant; other rules apply to
files only. An ordinary source directory such as `src/credentials/schema.rs`
remains searchable.

Previously cached entries for blocked paths are excluded from search and cannot
be extracted. This does not erase old cache data; use `seyk cache clear` to
remove it. The policy is centralized in `seyk-core/src/discovery.rs`. It checks
names, not contents: secrets renamed to ordinary filenames, hard links with
ordinary names, or credentials embedded in other documents are not detected.
Names are checked before files are opened; concurrent filesystem changes can
race those checks (TOCTOU). This policy is not a filesystem sandbox. A dotfiles target such as
`~/dotfiles/docker/config.json` is blocked by the Docker rule only when the
requested traversal passes through a `.docker` name.

## How text is segmented

Embedding a whole file would average all its ideas into one vector, and a query
matching one of them would score poorly. So text is split along its own structure:

- Markdown headings start a new segment and stay with their content. The heading
  path (`Projects > Pricing`) is sent to the model as the document title, as the
  model card recommends. Lines inside fenced code blocks are never taken for
  headings.
- Every paragraph of 60 characters or more is its own segment. Shorter fragments
  (labels, one-line notes) merge with their neighbours, up to about 1,000
  characters (~250 tokens).
- Paragraphs longer than 1,800 characters (~450 tokens) are cut at sentence ends,
  and the next piece repeats the last sentence. There is no overlap across paragraph
  breaks you chose yourself. Fenced code blocks count as one paragraph.

Sizes are counted in bytes, which equals characters for English text. Accented
and non-Latin text therefore gets somewhat shorter segments.

This uses a small fraction of the model's 8K context on purpose: shorter, focused
segments retrieve better and give precise locations. Each segment keeps its exact
byte range and line numbers.

## Images

Images go through the EmbeddingGemma 2 vision encoder with the default 280-token
budget and come back as `WholeResource` hits, so a text query like
"diagram of boxes connected by lines" finds the right picture. The Rust
preprocessing is a port of the Gemma 4 image processor. See
[Correctness checks](development.md#correctness-checks).

## PDFs

Normal builds include `pdf-render` and image embedding. Minimal library
consumers can enable `pdf` for embedded text or `pdf-render` for page rendering.
PDF pages with embedded text participate in semantic and keyword search. Hits
use `page_text`: a 1-based physical page, half-open UTF-8 byte range in the
extracted page text, extraction version and page-text hash. These are not byte
offsets in the PDF file. Explicit extraction checks the full PDF hash before
decoding, then verifies the derived page hash before returning text.

Scanned PDFs are found by meaning. Pages without usable embedded text are
rendered as whole-page images, respecting crop, rotation and a white background,
then embedded at a maximum long edge of 2048 pixels. Hits use a 1-based `page`
locator and have no text preview. Their text cannot be extracted yet; `extract`
returns an explicit OCR-unavailable error. OCR is tracked as a follow-up; no OCR
model is loaded or downloaded. Mixed PDFs keep their embedded-text hits and
scanned-page hits. With `--text-only`, a text-only embedder or a build without
`pdf-render`, image pages produce separate page diagnostics while supported text
pages remain searchable. Scan indexing is much slower than text indexing.
Rendering measured about 34 ms/page in the compatibility spike, while image
embedding takes seconds per page on CPU. The synthetic end-to-end scan test
measured about 6.7 s/page with one CPU thread. Use `--text-only` to skip image
embedding when indexing or searching large PDF collections.

Corrupt image streams can be omitted by the renderer. A page with image XObjects
that renders nearly uniform gets “scanned page could not be rendered” instead of
a hit; very faint or nearly blank scans can also trigger this guard. Partial
image omissions on otherwise nonblank pages remain a recall limitation. Reading
order, nonstandard fonts and sparse text over scanned pages remain limitations.
Every page content reference must resolve, decompress completely and pass strict
content decoding before any page output is accepted. Content streams currently
support no filter or FlateDecode without DecodeParms; other filters/parameters
are rejected explicitly because upstream decompression can silently return a
prefix after errors. PDFs over 512 pages, 1 MiB extracted text per page or 32
MiB total extracted text are rejected explicitly; other resources continue
indexing. These output limits do not bound upstream decompression, CPU time or
memory use.

Library integrations that install a `log` logger must filter PDF parser logs.
See [PDF parser logs and panics](library.md#pdf-parser-logs-and-panics).

PDF parser failures are isolated per page: valid pages keep their original page
numbers, and failed pages appear in `sync.page_diagnostics` with typed `code`
and sanitized `reason`. If every page fails, the resource also appears in
`sync.skipped`; `sync.skipped_counts` groups resource counts by code and reason.
CLI summaries group skips by code (at most five groups). MCP reports inline
diagnostics and watcher diagnostics in a separate `index_diagnostics` field;
`index.state` keeps its existing values, including `ready`. Protocol version 6 adds
these diagnostic fields; the cache schema remains unchanged. MCP page details are
capped at 128 entries, with the total reported separately.

An unchanged PDF is not re-parsed or rewritten after a deterministic failure.
Page diagnostics are replayed from an engine-local map keyed by resource and
preparation descriptor. They are not persisted: a fresh process reports them
only for resources it actually parses, and cannot replay failures from an
already-fresh persistent cache. Changed bytes or preparation versions trigger
a new parse.

Raw lopdf text fallback is deferred: lopdf 0.42 prefers Encoding over ToUnicode
and silently substitutes standard encoding after some font errors. That can
produce incorrect evidence. Page isolation recovers unaffected pages without
guessing glyph mappings. Subprocess isolation is a planned follow-up.
