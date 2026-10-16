# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub: open the repository's
**Security** tab and choose **Report a vulnerability**. Do not open a public
issue.

Include the Seyk version or commit, your platform, steps to reproduce, and the
impact you expect. We aim to acknowledge reports within seven days and will
keep you informed while a fix is prepared.

## Supported versions

Security fixes are made on the latest release and on `main`.

## In scope

Problems that break Seyk's guarantees, for example:

- content, queries or vectors leaving the machine without an explicit command;
- the cache or logs exposing readable source text;
- reads outside the configured roots, or symlink and path confinement bypasses;
- extraction returning text that does not match the verified content hash;
- model or asset integrity checks that can be bypassed.

## Out of scope

- Attacks that already require the same user account's access to the cache or
  model directories (see
  [Where things are stored](README.md#where-things-are-stored)).
- PII detection or redaction, which Seyk does not provide.
