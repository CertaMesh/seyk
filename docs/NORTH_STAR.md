# Project north star — Seyk

> Last reviewed: 2026-10-07 by krishan.
> Source-of-truth for autonomous-agent decisions. When two valid paths fork, pick the one that serves the mission for the target users without violating non-goals or constraints; principles break remaining ties.

## Mission

Find where relevant information is in any collection of unstructured content, and return verifiable evidence (resource, exact location, extracted text) without sending that content to anyone else.

## Target users / adopters

- Applications and AI agents that need retrieval over unstructured content: CLI, MCP or Rust library.
- Any deployment, from one laptop to a shared server, with any content (files today, other sources later).
- Developers using only Seyk's embedding layer, and downstream layers such as Gaze that consume its evidence.

## Non-goals

- Answering questions, summarising or generating text. Seyk returns evidence; reasoning is the caller's job.
- PII protection or redaction. That belongs to Gaze or the integrator; Seyk never depends on Gaze.
- Remote connectors, enterprise IAM, HTTP APIs or distributed/vector-database search until the local engine is mature (Vision phases 9-10).
- Extracting or transcribing a whole corpus just to make it searchable. Extraction runs on demand, for chosen hits.
- Building every modality at once. OCR, PDF, audio and video come one proven phase at a time.

## Hard constraints

- **Tech:** Rust; MSRV 1.89, toolchain pinned to 1.99.0; CPU inference via ONNX Runtime; prebuilt binaries for macOS ARM, Linux x86_64/ARM64 and Windows x86_64.
- **Legal / privacy:** Content, queries and vectors are never sent to a third-party service. Network use is limited to an explicit model download and, in later phases, sources the operator configures. The cache never stores source text in a readable order. Private content used for evaluation never appears in the repo, PRs, docs or shared pads.
- **Correctness:** A vector is reused only for byte-identical model input under the same profile. Returned text is verified against the content hash; a mismatch is an error, never a guess.
- **Compatibility:** `seyk search "…" <paths>` keeps working. The EmbeddingGemma 2 parity tests stay green. Cache schema changes migrate and keep embeddings where they are still valid.

## Decision principles (priority-ordered)

1. **Correct evidence over more results.** Why: a wrong passage or stale byte range is worse than a missing hit; callers act on what Seyk returns. Beats: recall, speed.
2. **Privacy over convenience.** Why: Seyk indexes private writing and data; anything stored or sent is a liability. Beats: features, cache speed.
3. **Every phase ships working and measured.** Why: build the handle, blade and hinge first; an unmeasured ranking change is a guess. Beats: breadth, roadmap speed.
4. **Search and extract stay separate.** Why: extraction (OCR, speech-to-text) is expensive and optional; a hit must exist without it. Beats: convenience APIs that extract by default.
5. **Generic seams, one real implementation.** Why: resources, providers, extractors and embedders must allow future sources and models without a redesign, but nothing speculative ships. Beats: building future backends now.
6. **Lean on the platform before building it.** Why: OS disk encryption, SQLite and ONNX Runtime are maintained by others; Seyk adds the thin glue. Beats: in-house crypto, custom indexes.

## Success + anti-signals

**Going right:**
- One query style finds both "where did I write about someone becoming like their father?" and `father.md`, and the evaluation on a real folder holds or improves each phase.
- `engine.extract(&hit)` returns the exact matching text, hash-verified, for every supported resource type.
- A developer can embed text with only the embedding crates, without pulling in search, SQLite or the CLI.
- Gaze uses Seyk over a process boundary with no Seyk-side Gaze code.

**Going wrong:**
- A ranking or segmentation change merged without an evaluation run.
- The cache or logs hold readable source text, or content is sent to a third-party service.
- Seyk starts producing prose (answers, summaries) or imports a privacy/LLM stack.
- A phase leaves main un-releasable, or an abstraction ships with no real implementation behind it.
