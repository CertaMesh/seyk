# 0001. Record architecture decisions

Date: 2026-01-12
Status: Accepted

## Context

Design choices for the Lantern ticket tracker were made in chat threads and
calls. Six months later nobody could say why the sync service polls instead of
using webhooks, and two people argued for a week about a choice that had
already been settled.

## Decision

We write a short decision record for every choice that is hard to reverse:
storage, wire formats, public APIs, hosting and security boundaries. Each
record has a number, a status, the context, the decision and its consequences.
Records are never deleted. A record that no longer holds is marked
Superseded and links to its replacement.

## Consequences

New team members read the decision log instead of asking. Writing a record
takes about half an hour, which is cheap next to re-arguing a settled choice.
