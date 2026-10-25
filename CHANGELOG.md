# Changelog

## Unreleased

### Added

- `seyk extract --typed-errors` opts into stable runtime failure exit codes 1
  and 3 through 11, with one privacy-safe `error: <stable_code>` stderr line.
  Clap retains usage exit 2 and help/version exit 0. Default failure behavior,
  success output, and JSON schema version 9 are unchanged. See the
  [exit-code table](docs/cli.md#typed-extraction-errors).
