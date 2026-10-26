# 0015. Use server-side sessions instead of JWTs for the web app

Date: 2026-08-27
Status: Accepted. Supersedes the token section of 0003.

## Context

The web app stored a signed JWT in local storage. Logging a user out, or
locking an account after a password reset, could not take effect until the
token expired an hour later.

## Decision

The web app uses an opaque session ID in an HttpOnly, SameSite=Lax cookie. The
session record lives in the database, so revoking it takes effect on the next
request. API tokens for integrations stay as they are; they are long-lived
secrets that customers rotate themselves.

## Consequences

Every web request now reads one session row. With SQLite in WAL mode this
read is cheap and needs no extra service.
