# What Seyk is for

This page states what Seyk does, who it is for, what it promises and what it
leaves to the caller. [NORTH_STAR.md](NORTH_STAR.md) is the short decision
guide that agents working on Seyk follow; this page is the longer version for
people deciding whether to use or contribute to Seyk.

## Purpose

Seyk is local search for AI agents and applications over private files. It
finds the passages that match a question or an identifier and returns exactly
where they are: a line range, a PDF page, an audio time window. It never writes
answers. Agents and applications read the evidence and decide what it means.

Seyk exists because agents increasingly need to look things up in content that
should not leave the machine, such as decision records, design notes, private
documents and recordings. An agent's built-in grep finds exact strings fast but
misses paraphrases, and it cannot read PDFs or audio. Hosted retrieval means
uploading the content. Seyk is meant to be used **alongside grep**, not instead
of it.

The long-term aim is an embeddable engine: one Rust crate that gives local
embedding, hybrid ranking and exact locators, so an application does not have
to assemble them from separate libraries. That aim is pre-1.0 today (see
[Status](#status)).

## Who it is for

In order of what someone can adopt today:

1. **Developers giving an AI agent access to local files over MCP.** The
   alternative they already have is the agent's own grep and file tools. Seyk
   adds search by meaning, PDF and audio content, and exact locations, with
   nothing sent anywhere by Seyk itself.
2. **Developers searching their own files from the command line.** Same
   install, same engine, with JSON output for scripts.
3. **Rust developers embedding retrieval in an application.** Supported today
   as a Git dependency with a pre-1.0 API. This becomes a first-class path once
   the crates are published and indexing memory is bounded (see the
   [roadmap](roadmap.md)).

This order follows how mature each interface is, not user research.

## Core contract

Given resources the caller is allowed to read and a query, Seyk returns ranked
hits. Each hit names a resource and a location inside it, says which signals
found it, and may carry a short preview. On request, Seyk returns readable
evidence for a chosen hit after checking that the resource still matches the
hash recorded when it was indexed.

### What extraction returns, by kind of content

| Content | Location in a hit | What `extract` returns |
|---|---|---|
| Text, Markdown, code | byte and line range | the exact UTF-8 slice, after a whole-file BLAKE3 check |
| PDF with embedded text | page and range within that page's extracted text | the derived page text, after a whole-PDF check plus a parser-version and page-text-hash check |
| Audio | time window | a machine transcript, labelled as such, of samples read from a hash-checked file; transcripts can contain errors |
| Scanned PDF page | page | nothing readable yet (no OCR) |
| Image | whole resource | nothing readable |

`ImageRegion` is a reserved locator; nothing produces it today.

A hash match shows that the bytes Seyk read are the bytes it indexed, at the
moment of that read. It does not show who wrote the file, that the hash supplied
with a hit is authentic, or that the content is true. A transcript's accuracy
is never hash-checked.

### Seyk promises, and what stays with the caller

| Area | Seyk does | The caller is responsible for |
|---|---|---|
| Authorization | The MCP server only searches and extracts inside the folders it was started with. The filesystem provider refuses secret-looking file names (keys, password stores, `.env`) by name, with no override. | Deciding who may query which folders. CLI and library extraction add no access layer of their own; a custom resource provider sets its own rules. |
| Freshness | Uses filesystem size and mtime as cheap change hints. An equal-size rewrite with a restored mtime can leave stale vectors and keyword rows for ranking, even after `index`. Extraction checks the content hash and refuses changed text. Search omits a preview it cannot verify. | To force a re-read, run `seyk cache clear` for the cache in use, then search or index again. An agent that reads `path:lines` directly, without `extract`, gets no hash check. |
| Provenance | Every hit carries the resource, location, whole-resource hash and per-signal ranks (`match`). Derived text carries its method. | Judging authorship and authenticity. |
| Relevance | Ranks by meaning, keywords and file names, measured on public datasets ([benchmarks](benchmarks.md)). | Deciding whether a hit actually answers the question. Relevance is not guaranteed. |
| Meaning | Nothing. Seyk does not interpret, summarise or judge content. | Everything an answer needs. |
| Privacy | Sends no files, queries or embeddings over the network; only explicit model setup commands download. | What the agent does with passages it reads. A cloud-hosted agent sends them to its provider. |

The cache does not store passage bodies in their original word order, but it
does keep readable paths and headings, word frequencies, metadata and vectors.
Treat it as sensitive. See [Caching and storage](caching-and-storage.md).

## Non-goals

- Answering questions, summarising or generating text.
- Judging whether evidence is true. A hash match shows the source is unchanged
  since indexing, not that it is authentic or correct.
- Being a chat-with-your-documents app, an agent framework, an LLM
  orchestrator, an agent memory system or a hosted service. Seyk is a retrieval
  component inside them.
- Replacing grep. For every exact occurrence of a string or symbol, grep is
  still the right tool.
- PII protection or redaction. That belongs to the integrator or a downstream
  tool.
- Remote connectors, enterprise access control, HTTP APIs or distributed
  search, until the local engine is mature.

## Architectural principles

| Principle | Where it stands today |
|---|---|
| Local by default | Holds. Inference runs on the CPU through ONNX Runtime; searching never downloads anything. |
| Retrieval without generation | Holds. Speech-to-text runs only when a caller explicitly extracts an audio hit. |
| Precise locations | Holds for text, PDF text, scanned pages and audio. Images are located as whole files. |
| Check before returning text | Holds for extraction of text, PDF text and audio samples, with the scope described above. |
| Search and extraction stay separate | Holds. A hit exists without extraction; expensive extraction runs on demand. |
| Explicit trait seams, concrete implementations for current needs | Holds. Providers, extractors, embedders, caches and the vector index are traits, each with implementations Seyk actually uses. |
| Predictable resource use | **Not yet.** Indexing dense files can use several GB of memory ([#95](https://github.com/CertaMesh/seyk/pull/95)). |
| Stable interfaces | Partly. CLI and MCP JSON carry a `schema_version`, and extraction accepts older versions. The Rust API can change before 1.0. |

## Status

Seyk is at 0.1.0. Install from source; prebuilt binaries return with the next
release. The crates are not on crates.io yet. Video and OCR are not
implemented. Vector search is an exact comparison against every cached f32
vector by default. An experimental, opt-in Photo Anchor scan for standalone
images shortlists candidates from compressed codes and reranks them with f32;
it needs a persistent SQLite cache and a d256 or d768 profile, and gives no
recall guarantee on arbitrary libraries
([details](multimodal.md#experimental-photo-anchor-scan)). See [Known limitations](limitations.md) and the
[roadmap](roadmap.md).

## How Seyk compares

Other local tools also combine keyword and vector search, run models locally
and report file and line locations. Seyk's combination is: hybrid search with
per-signal provenance, locations across text, PDF pages and audio, extraction
that refuses changed files, and one engine behind a CLI, an MCP server and a
Rust library. Retrieval quality on public datasets is competitive with other
local tools, not ahead of them; see the
[local competitor runs](benchmarks.md#local-competitor-runs). Whether Seyk
helps agents finish real tasks is not measured yet.
