# 0007. Store tickets in SQLite, not Postgres

Date: 2026-03-18
Status: Accepted

## Context

Every Lantern customer runs their own instance, usually on a single small
virtual machine. Most instances hold fewer than 200,000 tickets. Running a
separate Postgres server for each customer doubled the support load: upgrades,
backups, connection limits and disk alerts all became our problem.

## Decision

We store tickets in a single SQLite database file in write-ahead-log mode,
next to the application. Backups copy that one file with the SQLite online
backup API. We keep the schema portable so a move to Postgres stays possible
if a customer outgrows one machine.

## Consequences

Installation is one binary and one file. Only one process can write at a
time, so background jobs queue their writes through the main service instead
of opening their own connections. Full-text search uses SQLite FTS5 rather
than an external search server.
