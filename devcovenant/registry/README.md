# Registry State
**Last Updated:** 2026-04-27
**Project Version:** 1.0.1b5

## Table of Contents
- [Overview](#overview)
- [Tracked Registry](#tracked-registry)
- [Runtime Registry](#runtime-registry)
- [Lifecycle and Ownership](#lifecycle-and-ownership)
- [Troubleshooting Notes](#troubleshooting-notes)
- [Workflow](#workflow)

## Overview
`devcovenant/registry/` is the single registry root for a managed repository.
It separates deterministic tracked governance metadata from disposable runtime
state.

Do not edit generated payloads by hand. Refresh and gate commands own registry
materialization.

## Tracked Registry
`devcovenant/registry/registry.yaml` is the only tracked registry artifact.
It stores deterministic repository governance metadata such as:
- resolved policy entries and metadata traces
- resolved profile inventory
- tracked inventory data for package-owned/generated surfaces

This file belongs in git so a cloned repository carries its current
DevCovenant governance state.

## Runtime Registry
`devcovenant/registry/runtime/` stores untracked runtime-local state:
- `gate_status.json` for concise gate lifecycle state
- `session_snapshot.json` for heavy gate baseline/snapshot/run-event payloads
- `latest.json` for the latest run-pointer metadata

Runtime registry files are disposable. They are not package payload, not git
truth, and not a trust anchor.

## Lifecycle and Ownership
Registry regeneration occurs during full-refresh paths:
- `devcovenant refresh`
- `devcovenant deploy`
- `devcovenant upgrade`
- gate pre-commit stages (`devcovenant gate --open`, required non-lifecycle
  `devcovenant gate --verify`, and `devcovenant gate --close`) through gate-owned
  check orchestration

Ownership model:
- tracked registry file: deterministic generated repo metadata
- runtime registry files: local runtime/session state
- logs: local run artifacts under `devcovenant/logs/`

## Troubleshooting Notes
If integrity checks report registry drift:
1. Run `devcovenant refresh`.
2. Re-run `devcovenant run`.
3. Re-run `devcovenant gate --close`.

If drift persists, compare AGENTS policy block content against
`devcovenant/registry/registry.yaml` and verify descriptor/profile edits were
refreshed.

## Workflow
1. Run a refresh-producing command.
2. Confirm `devcovenant/registry/registry.yaml` is regenerated when inputs
   changed.
3. Use `devcovenant/registry/runtime/` only for live runtime inspection.
4. Run `devcovenant gate --verify` before `devcovenant run` in active sessions.
5. Validate with tests and `gate --close`.
