# 0004. Poll the mail server instead of using push notifications

Date: 2026-02-03
Status: Accepted

## Context

Lantern turns support emails into tickets. Our two mail providers offer push
notifications, but each uses a different protocol, and one drops events when
our endpoint is down for more than ten minutes. Missed events mean missed
tickets, and nobody notices until a customer complains.

## Decision

The sync service polls each mailbox every 60 seconds and remembers the last
message ID it processed. Polling is the same for both providers, and a restart
simply resumes from the stored ID.

## Consequences

A new email can take up to a minute to become a ticket. That is acceptable for
support mail, which has a four-hour response target. Polling costs roughly
1,440 requests per mailbox per day, well inside both providers' limits.
