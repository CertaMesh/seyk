# 0012. Keep attachments out of the database

Date: 2026-06-09
Status: Accepted

## Context

Customers attach screenshots, log files and PDFs to tickets. When attachments
lived inside the SQLite file chosen in decision 0007, one customer's database
grew to 40 GB and nightly backups took over three hours.

## Decision

Attachments go to a directory on local disk, or to S3-compatible object
storage when the customer configures it. The database stores only the file
name, size, content type and a SHA-256 checksum of each attachment. The
checksum is checked whenever an attachment is downloaded.

## Consequences

The database stays small and backups are fast again. Backups now have two
parts, the database file and the attachment store, and the restore guide
covers both.
