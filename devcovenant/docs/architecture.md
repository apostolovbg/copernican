# DevCovenant Architecture
**Last Updated:** 2026-04-27
**Project Version:** 1.0.1b5

## Overview
This document explains how DevCovenant is put together.
Use it when you need the internal ownership map rather than the day-to-day
operator flow.
If you only need to use the product, start with `README.md`,
`installation.md`, `config.md`, and `workflow.md` first.

## Flat Core
DevCovenant keeps its implementation in a flat `devcovenant/core/*.py`
surface.
There are no subpackages under `devcovenant/core/`.
The goal is to keep the runtime easy to scan without scattering one feature
across retired layer folders or mixing unrelated responsibilities into one
file.

The root CLI remains intentionally thin.
It resolves the top-level command first and then loads the heavier runtime
machinery needed for that command path.
That keeps help and other lightweight entrypoints fast.

## Module Families
The flat core still has clear responsibility families.
The grouping is logical, not directory-driven.

### Foundations
These modules define the shared contracts and repo-state helpers used by the
rest of the runtime.

- `devcovenant/core/policy_contract.py`
  typed policy contracts such as `CheckContext`, `Violation`, and fixer
  interfaces
- `devcovenant/core/runtime_errors.py`
  explicit error contracts and stable error codes
- `devcovenant/core/repository_paths.py`
  repo-relative path rendering plus cached text, bytes, YAML, and AST access
- `devcovenant/core/selectors.py`
  selector sets and watchlist derivation
- `devcovenant/core/document_exemptions.py`
  managed-marker and doc-header exemption fingerprints
- `devcovenant/core/tracked_registry.py`
  tracked registry path and persistence helpers
- `devcovenant/core/profile_registry.py`
  profile discovery, registry materialization, and profile-derived overlays
- `devcovenant/core/runtime_profile.py`
  workflow runtime-profile payload and rendering helpers
- `devcovenant/core/project_governance.py`
  governance-state parsing, release-heading rules, and placeholder rendering
- `devcovenant/core/policy_metadata.py`
  policy metadata parsing, typed decoding, and metadata bundle resolution
- `devcovenant/core/policy_registry.py`
  policy descriptor loading and tracked policy registry operations
- `devcovenant/core/policy_commands.py`
  policy command declarations, lookup, payload parsing, and validation
- `devcovenant/core/policy_runtime_actions.py`
  policy-owned runtime-action loading and execution helpers

### Customization Support
These modules own the repo-to-custom shadow-copy flow and the shipped test
blueprint descriptors that keep builtin tests package-friendly.

- `devcovenant/core/customization.py`
  builtin-to-custom path resolution, copy/remove helpers, and custom test
  mirror materialization
- `devcovenant/core/test_blueprints.py`
  loading, serializing, materializing, and comparing `test_blueprints.yaml`
  descriptors

### Command And Execution
These modules own command parsing, console behavior, subprocess handling,
and run evidence.

- `devcovenant/core/cli_support.py`
  CLI flag parsing, output-mode policy, and runtime error normalization
- `devcovenant/core/execution.py`
  command dispatch, managed re-exec, subprocess streaming, and workflow-run
  execution
- `devcovenant/core/run_events.py`
  workflow run-event adapters and event collection
- `devcovenant/core/run_logs.py`
  run log allocation, summary artifacts, and workflow profiling payloads
- `devcovenant/core/cleanup.py`
  cleanup selection and the `clean` command implementation
- `devcovenant/custom.py`
  top-level `custom` command dispatcher for builtin shadow copies and test
  mirror materialization

### Workflow And Repo State
These modules own the gate lifecycle, refresh orchestration, workflow
contracts, and repo-shape validation.

- `devcovenant/core/workflow_support.py`
  runtime registry paths, workflow contract resolution, and workflow
  validation
- `devcovenant/core/gate_runtime.py`
  `gate --open`, `--verify`, `--close`, status rendering, and gate-session
  snapshots
- `devcovenant/core/refresh_runtime.py`
  full refresh orchestration, policy registry refresh, dependency artifact
  refresh, and managed-file regeneration
- `devcovenant/core/repository_validation.py`
  manifest inventory, integrity validation, and structure validation

### Policy Runtime
These modules assemble check context, execute policies, and support
translator-aware analysis.

- `devcovenant/core/policy_runtime.py`
  check-context construction, policy check execution, file scoping, and
  policy reporting
- `devcovenant/core/policy_autofix.py`
  shared autofix contracts and autofix execution helpers
- `devcovenant/core/translator.py`
  translator declarations, runtime resolution, and `LanguageUnit` support
- `devcovenant/custom/policies/builtin_test_blueprint_sync/\
builtin_test_blueprint_sync.py`
  custom policy that keeps builtin test blueprints and custom mirrors in sync

### Managed Content
These modules own generated AGENTS content, managed docs, and asset
materialization.

- `devcovenant/core/agents_blocks.py`
  generated AGENTS policy-block rendering
- `devcovenant/core/managed_docs.py`
  managed header/block rendering, adoption rules, and doc synchronization
- `devcovenant/core/asset_materialization.py`
  profile asset rendering and desktop asset materialization

## Built-In Engine Checks
DevCovenant always runs three engine-level checks.
They are part of the runtime itself rather than descriptor-backed inventory
policies.

- workflow validation in `devcovenant/core/workflow_support.py`
- integrity validation in `devcovenant/core/repository_validation.py`
- structure validation in `devcovenant/core/repository_validation.py`

Repository-tweakable settings for these checks still live in ordinary config
sections such as `paths`, `workflow`, and `integrity`.

## Evidence Flow
DevCovenant is built around recorded evidence, not only around pass/fail
checks.
The normal flow is:

1. collect and interpret repository files
2. resolve workflow and policy definitions
3. run engine checks and configured policies
4. record evidence about what happened

The main evidence locations are:

- per-run log folders under `devcovenant/logs/`
- tracked registry state in `devcovenant/registry/registry.yaml`
- runtime session state under `devcovenant/registry/runtime/`
- managed governance output in `AGENTS.md`

## Workflow Ownership
Workflow shape is saved separately from policy state.
Core owns the reserved anchors:

- `open`
- `verify`
- `close`

Profiles own the declared workflow runs between `verify` and `close`.
Those runs define whether a run is enabled or required, how it executes,
how freshness is checked, and how evidence is recorded.

Workflow definition and validation live in
`devcovenant/core/workflow_support.py`.
Gate lifecycle and session-state handling live in
`devcovenant/core/gate_runtime.py`.
Execution of workflow child commands lives in
`devcovenant/core/execution.py`.

## Managed Docs And Generation
Managed documents are built outputs.
The managed-doc runtime owns descriptor parsing and validation, managed
header rendering, managed block rendering, adoption rules, and preservation
rules for authored content.
AGENTS-specific policy-block rendering is intentionally isolated in
`devcovenant/core/agents_blocks.py`.

`LICENSE` stays a deliberate exception to the usual metadata-heavy header
shape.
For license docs, managed-doc rendering syncs only the top title line and
preserves the legal body after the file is first seeded.

## Registry Ownership
The tracked registry and runtime registry are different.

- `devcovenant/registry/registry.yaml`
  durable repo state, resolved metadata, and tracked policy/runtime facts
- `devcovenant/registry/runtime/`
  transient gate, workflow, and latest-run evidence

In code, tracked-registry helpers live in
`devcovenant/core/tracked_registry.py`.
Runtime evidence paths and workflow registry helpers live in
`devcovenant/core/workflow_support.py`.
Repo-relative path rendering and cached file access live in
`devcovenant/core/repository_paths.py`.
Manifest and structure checks live in
`devcovenant/core/repository_validation.py`.

Tracked repo artifacts must stay repo-relative and content-derived.
Absolute local checkout paths belong only to transient runtime diagnostics,
not to tracked repository state.

## Package Boundary
The published package ships the docs, builtin policies, builtin profiles,
assets, translators, and runtime modules that DevCovenant needs to operate.
That includes the shipped `devcovenant/runtime-requirements.lock` bootstrap
file and DevCovenant's packaged license files under `devcovenant/licenses/`.
It does not ship live repository state such as:

- `devcovenant/config.yaml`
- tracked registry outputs
- runtime registry data
- timestamped log folders
- build debris

That package boundary is owned by `pyproject.toml`, `MANIFEST.in`, and the
packaging tests.
