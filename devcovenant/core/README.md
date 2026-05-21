# Core Runtime
**Last Updated:** 2026-04-27
**Project Version:** 1.0.1b5

## Overview
`devcovenant/core/` holds the runtime implementation for command behavior,
policy execution, metadata resolution, refresh orchestration, selector logic,
and translator routing.

The core is intentionally flat.
There are no subpackages under `devcovenant/core/`.
Instead, related logic stays in cohesive modules that own one clear runtime
surface.

## Module Families
The flat core is organized by responsibility families.
The families are conceptual; the code stays in `devcovenant/core/*.py`.

- foundations:
  `policy_contract.py`, `runtime_errors.py`, `repository_paths.py`,
  `selectors.py`, `document_exemptions.py`, `tracked_registry.py`,
  `profile_registry.py`, `runtime_profile.py`, `project_governance.py`,
  `policy_metadata.py`, `policy_registry.py`, `policy_commands.py`,
  `policy_runtime_actions.py`
- command and execution:
  `cli_support.py`, `execution.py`, `run_events.py`, `run_logs.py`,
  `cleanup.py`
- workflow and repo state:
  `workflow_support.py`, `gate_runtime.py`, `refresh_runtime.py`,
  `repository_validation.py`
- policy runtime:
  `policy_runtime.py`, `policy_autofix.py`, `translator.py`
- managed content:
  `agents_blocks.py`, `managed_docs.py`, `asset_materialization.py`

## Why Flat
The old split across `flow/`, `runtime/`, `services/`, `lib/`, and
`contracts/` made the core harder to scan because one feature often lived in
several separate folders at once.
A fully collapsed design would be just as hard to maintain when unrelated
responsibilities land in the same file.
The flat core keeps the runtime in one visible surface while still preserving
modular ownership.

## Workflow
1. Update the target core module.
2. Update the mirrored tests under `tests/devcovenant/core/`.
3. Update docs affected by behavior changes.
4. Run the gate sequence:
   - `devcovenant gate --open`
   - `devcovenant gate --verify`
   - `devcovenant run`
   - `devcovenant gate --close`
5. Keep `SPEC.md`, `PLAN.md`, and the reference maps aligned when contracts
   change.
