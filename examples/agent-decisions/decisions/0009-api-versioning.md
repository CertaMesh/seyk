# 0009. Version the public API in the URL path

Date: 2026-04-22
Status: Accepted

## Context

Integrations built by customers call the Lantern REST API from scripts that
nobody maintains. A breaking change to a response shape silently broke an
invoicing script at one customer for two weeks.

## Decision

Every public endpoint lives under a version prefix such as `/api/v2/`. A
breaking change creates a new version. The previous version keeps working for
at least twelve months after its replacement ships, and responses from a
deprecated version carry a `Sunset` header with the shutdown date.

## Consequences

We maintain two versions side by side for a year at a time. Client libraries
pin a version and never follow redirects between versions.
