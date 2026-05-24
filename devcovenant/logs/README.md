# Run Logs
**Last Updated:** 2026-02-27
**Project Version:** 1.0.1.dev1

## Overview
This directory is the canonical runtime log root for DevCovenant command runs.
Per-run folders are local execution artifacts used for debugging and audit
triage. They are repository runtime state, not source-of-truth docs.

## Layout
Planned per-run folders contain stable artifacts such as:
- `run.json`
- `summary.txt`
- `summary.json`
- `stdout.log`
- `stderr.log`
- `tail.txt`

The latest-run pointer now lives in
`devcovenant/registry/runtime/latest.json`, not inside `devcovenant/logs/`.

## Workflow
Use this triage order:
1. `summary.txt`
2. `tail.txt` (if present)
3. `stderr.log` / `stdout.log`

Treat generated log contents as local runtime state. Commit only tracked docs
in this directory. The shared `run_logging` runtime allocates and updates
these artifacts for command runs.
