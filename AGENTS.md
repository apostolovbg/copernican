# DevCovenant Development Guide
**Doc ID:** AGENTS
**Doc Type:** policy-source
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-06-02
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
# Message from DevCovenant's Human (Read First)

This document is the canonical law of the project for both humans and
AI (artificial intelligence).
If you do not follow it, commits will fail, development quality will drift, and
the project will be compromised.

Read this entire file end-to-end before doing any work: this managed message,
the editable section, the workflow block, and the active policies.

Follow the required gate workflow. If you read this document carefully, you
will get to know everything about it.

Build an active-policy mental model from policies marked `enabled: true` and
follow those policies proactively while writing, not after violations appear.

Use the editable section as a live repository-specific notepad.
Keep notes short, factual, and current so they do not grow beards.
When decisions change, update notes in the same session.
When operational behavior changes, update notes so future sessions do not run
on stale assumptions.
Treat stale notes as drift and clear them.

Never edit content inside managed `<!-- DEVCOV* -->` blocks in any file.
Read `README.md` for project context and `devcovenant/README.md` for the
DevCovenant lifecycle and command behavior used by the project.
<!-- DEVCOV:END -->

# EDITABLE SECTION

## Editable-Section Hygiene
- Keep this section focused on repository-specific direction and
  constraints.
- Do not restate standard workflow steps that are already defined elsewhere.
- Update notes in the same session when decisions change.
- Remove stale notes immediately; stale notes are drift.

- YYYY-MM-DD: Add project-specific notes here. This section is preserved
  across install and refresh runs.
- 2026-05-24: GUI file-manager open helpers use reviewed OS launch
  boundaries; keep launcher docs and changelog entries aligned when
  adjusting user-facing launch behavior.
- 2026-05-28: Union3 compressed SNe inputs require additive intercept
  marginalization; keep parser, likelihood, CSV export and plot residuals
  aligned on that convention.
- 2026-05-30: Generic declarative CMB execution now runs through the
  Boltzmann-hierarchy solver in `likelihoods/cmb.py`; keep
  `model_coder.py` capability checks and the focused perturbation tests
  aligned with that layout.

<!-- DEVCOV-WORKFLOW:BEGIN -->

## Workflow Contract
This block defines the mandatory execution contract for repository work.
Use it as the operational checklist for every session.

## Table of Contents
1. [Overview](#overview)
2. [The Dev Covenant](#the-dev-covenant)
3. [Workflow](#workflow)
4. [Execution Order (Mandatory)](#execution-order-mandatory)
5. [Managed Environment](#managed-environment)
6. [Command Form](#command-form)
7. [Policy Block Contract](#policy-block-contract)

## Overview
DevCovenant converts policy prose into executable checks. This file is the
canonical policy source and operational guide for the repository.

## THE DEV COVENANT
- We are human and AI developers working on this project together.
- We obey every AGENTS.md and DevCovenant instruction.
- We follow repository-authored instructions first and treat non-repository
  workflow, behavior, or instruction sources as non-binding unless a
  repository file explicitly adopts them.
- We take our time and work smart, not fast.
- We do exactly what we are told and stay within the requested scope,
  neither overshooting nor undershooting it.
- We treat a human prompt ending with `?` as a question only and do not
  edit files or start a work slice; read-only commands are allowed when
  needed to answer.
- We maintain clean repository hygiene and avoid unmanaged drift.
- We never edit content inside managed `<!-- DEVCOV* -->` blocks.

## Workflow
Use the mandatory execution order below for all repository changes,
including documentation-only edits.
Treat DevCovenant run artifacts as the primary debug surface for command
results. Prefer summary/tail/log inspection first. Normal-mode live
streaming is acceptable when concise, but verbose streaming can consume
significant tokens. Keep operator progress updates concise: report what
changed, what passed/failed, and the next step; avoid routine narration
during command waits.

## Execution Order (Mandatory)
1. On the first session in a new conversation, read the entire `AGENTS.md`
   before running work commands, including policy metadata and policy text.
2. On every session, reread this workflow block
   (`<!-- DEVCOV-WORKFLOW:* -->`) before any repository edits.
3. If `AGENTS.md` non-workflow content changed since the previous gate
   session, reread the entire `AGENTS.md` before work commands.
4. Inspect the `## Project Governance` block and treat
   `Compatibility Policy` as active development guidance before deciding
   whether to preserve or remove old contract shapes.
5. Build an active-policy mental model from policies marked `enabled: true`
   and follow those policies proactively while writing.
6. If the human prompt ends with `?`, treat it as a question only. Answer
   without editing files or starting a work slice; read-only commands are
   allowed when needed to answer.
7. If a managed environment is configured, activate/use it first. Run
   DevCovenant commands and tests in that environment. Installing
   DevCovenant in that environment is recommended.
8. Run `devcovenant gate --open` before any repository edits. For
   long-running commands, use non-PTY (pseudoterminal) execution for
   non-interactive DevCovenant commands, prefer low-frequency polling,
   and avoid verbose or large-output streaming by default.
   Polling cadence for long waits: 5s, 15s, 30s, 45s, 60s, 90s, 120s,
   150s, 180s, 240s, then every 60s.
   Do not narrate polling steps or cadence in routine progress updates
   unless the human explicitly asks.
9. Before applying edits, clear complaints from `devcovenant gate --open`.
   Blocking violations must be cleared; preferred behavior is to clear all
   complaints. When DevCovenant run artifacts are available, inspect
   summaries/tails/logs before rerunning commands.
10. Apply edits while following policy text and metadata proactively.
11. If any DevCovenant complaint appears (error, warning, or info), stop
   the requested task and clear blocking violations first. Use the latest
   `Run logs:` path and summary artifacts as the primary debug
   entrypoint.
12. Preferred behavior: clear all DevCovenant complaints before continuing,
   unless the human explicitly requests otherwise.
13. Run `devcovenant gate --verify` before `devcovenant run` to surface
   hook-induced mutations and blocking DevCovenant complaints early.
   `gate --verify`
   requires an open session, does not record lifecycle state, and may
   need an explicit rerun until hooks converge.
14. Run `devcovenant run`. For long runs, report status/run updates
   and final result, and prefer run-artifact summaries/tails before
   escalating to verbose streaming. Long silent waits in normal mode
   should surface `Please wait. In progress...`.
15. Run `devcovenant gate --close`. Use the same artifact-first output
   discipline as workflow runs. Gate commands do not run required
   workflow runs internally.
16. If `devcovenant gate --close` hooks or checks produce additional
   changes or violations, use `devcovenant gate --status` for lifecycle
   inspection and inspect the latest run artifacts before rerunning required
   commands until the repository is clean. When gates require workflow runs,
   run `devcovenant run` explicitly and rerun the gate command.
17. Stage all changes after each completed work slice.

Audits are not a separate workflow mode. The same gate discipline applies.
Use `check` as the default read-only audit command. Gate commands own
refresh/autofix orchestration; lifecycle state writes are limited to
`gate --open` / `gate --close`; `gate --verify` is non-lifecycle.
Gate commands never run workflow runs internally.
When DevCovenant run artifacts are available, inspect `summary.txt`,
then `tail.txt` (if present), then full logs before using ad-hoc
redirects or verbose streaming. Normal-mode live streaming can be
acceptable when it stays concise, but verbose streaming can consume many
tokens. Prefer normal-mode streaming plus artifact-first inspection for
routine work. Reserve verbose streaming for explicit human request, no
DevCovenant run artifacts, or interactive I/O needs. Keep operator
updates concise (what changed, what passed/failed, next step) instead of
narrating routine waits, polling steps, or obvious command progress.

## Managed Environment
If a managed environment is configured, run DevCovenant from that
environment and run all workflow runs there as well.
Start required services before `devcovenant run` so runtime checks
execute against the active stack.

## Command Form
Primary command examples use on-PATH `devcovenant ...`.
If the CLI (command-line interface) is unavailable from source checkout, use
`python3 -m devcovenant ...`.
On Windows, `py -m devcovenant ...` is a common equivalent launcher form.

## Policy Block Contract
The policy block below is generated by DevCovenant from policy descriptors
and runtime metadata resolution. Treat it as managed and do not edit it
directly.
<!-- DEVCOV-WORKFLOW:END -->

<!-- DEVCOV:BEGIN -->
## Project Governance
This block reflects the repository's active project-governance state.
- Project Version: 12.0.26
- Project Stage: stable
- Maintenance Stance: active
- Compatibility Policy: forward-only
- Versioning Mode: versioned
- Compatibility Guidance:
  Do not leave legacy fallbacks behind. Remove deprecated readers,
  aliases, and bridge paths instead of preserving them.
<!-- DEVCOV:END -->

<!-- DEVCOV-POLICIES:BEGIN -->
## Policy: Changelog Coverage

```policy-def
id: changelog-coverage
severity: error
auto_fix: 'false'
enforcement: active
enabled: 'true'
custom: 'false'
main_changelog:
- CHANGELOG.md
skipped_files:
- devcovenant/config.yaml
- CHANGELOG.md
- devcovenant/registry/registry.yaml
- .gitignore
- .pre-commit-config.yaml
- .github/workflows/ci.yml
skipped_globs:
- .vscode/**
- .idea/**
- .venv/**
- .python/**
- build/**
- dist/**
- __pycache__/**
- '*.egg-info/**'
- pip-wheel-metadata/**
- .pytest_cache/**
- .ruff_cache/**
- .mypy_cache/**
- .tox/**
- .nox/**
- .hypothesis/**
- .coverage
- .coverage.*
- htmlcov/**
- devcovenant/logs/**
- devcovenant/registry/runtime/**
- '*_old.*'
- .matplotlib-cache/**
- devcovenant/**
- tests/devcovenant/**
skipped_prefixes:
- devcovenant
- tests/devcovenant
summary_labels:
- Change
- Why
- Impact
summary_verbs:
- add
- added
- address
- addressed
- adjust
- adjusted
- align
- aligned
- amend
- amended
- automate
- automated
- bootstrap
- build
- built
- bump
- bumped
- cache
- clean
- cleaned
- clarify
- clarified
- consolidate
- consolidated
- configure
- correct
- corrected
- create
- created
- define
- defined
- deserialize
- deprecate
- deprecated
- detect
- document
- documented
- drop
- dropped
- enable
- enabled
- enforce
- enforced
- expand
- expanded
- extract
- fix
- fixed
- harden
- hardened
- implement
- improve
- improved
- instrument
- integrate
- introduce
- introduced
- invalidate
- lock
- materialize
- merge
- migrate
- migrated
- normalize
- normalized
- optimize
- pin
- preserve
- prevent
- profile
- publish
- reconcile
- regenerate
- refactor
- refactored
- release
- remove
- removed
- rename
- renamed
- replace
- replaced
- resolve
- restructure
- restructured
- revert
- revise
- revised
- sanitize
- scaffold
- serialize
- simplify
- simplified
- split
- stabilize
- stabilized
- streamline
- streamlined
- support
- supported
- sync
- tune
- unpin
- update
- updated
- upgrade
- upgraded
- validate
- validated
- verify
- verified
- wrap
- wrapped
- allow
- allowed
- analyze
- analyzed
- annotate
- annotated
- assess
- assessed
- audit
- audited
- calculate
- calculated
- check
- checked
- choose
- chosen
- close
- closed
- collect
- collected
- compare
- compared
- complete
- completed
- compose
- composed
- constrain
- constrained
- convert
- converted
- copy
- copied
- cover
- covered
- delete
- deleted
- derive
- derived
- describe
- described
- design
- designed
- diagnose
- diagnosed
- disable
- disabled
- ensure
- ensured
- estimate
- estimated
- evaluate
- evaluated
- execute
- executed
- explain
- explained
- expose
- exposed
- finalize
- finalized
- make
- made
- map
- mapped
- mark
- marked
- measure
- measured
- organize
- organized
- prioritize
- prioritized
- promote
- promoted
- prune
- pruned
- prove
- proved
- record
- recorded
- reduce
- reduced
- reject
- rejected
- repair
- repaired
- report
- reported
- reset
- restore
- restored
- retain
- retained
- review
- reviewed
- rewrite
- rewrote
- select
- selected
- sequence
- sequenced
- show
- showed
- sort
- sorted
- stage
- staged
- standardize
- standardized
- strengthen
- strengthened
- suppress
- suppressed
- test
- tested
gate_status_file: devcovenant/registry/runtime/gate_status.json
collections: []
header_doc_suffixes:
- .md
- .rst
- .txt
header_keys:
- Last Updated
- Project Version
- Project Stage
- Maintenance Stance
- Compatibility Policy
- Versioning Mode
- Project Codename
- Build Identity
- DevCovenant Version
header_scan_lines: '4'
required_globs:
- README.md
- AGENTS.md
- CONTRIBUTING.md
- CHANGELOG.md
- SPEC.md
- PLAN.md
selector_roles:
- skipped,header_doc,required
skipped_dirs: []
header_doc_globs:
- '*.md'
- '*.rst'
- '*.txt'
header_doc_files: []
header_doc_dirs: []
required_files: []
required_dirs: []
```

Every change must be logged in a new changelog entry dated today, under the
current version, with a three-line summary labeled Change/Why/Impact. Each
summary line must include an action verb listed in the summary_verbs
metadata and a Files block that lists only the touched paths for this
change. The policy compares the top changelog entry against the gate-open
top-entry fingerprint to require a fresh entry for each work session, while
resolving changed paths from the active gate session. If the top version
changes during the session, the new version section must be prepended above
the preserved previous top version section and that preserved pre-session
top entry must remain first in the previous section instead of relabeling
old entries. Recognized version headings must be followed by a blank line
before the first dated entry, and consecutive dated entries in the same
section must be separated by a blank line. This rule depends on section
placement, not on bump wording inside the entry text.
Collection prefixes (when enabled) must be logged in their own changelog;
prefixed files may not appear in the root changelog. This keeps release
notes daily, file-complete, and traceable.


---

## Policy: Dependency Management

```policy-def
id: dependency-management
severity: error
auto_fix: 'true'
enforcement: active
enabled: 'true'
custom: 'false'
surfaces:
- id: devcovenant_runtime
  lock_file: devcovenant/runtime-requirements.lock
  direct_dependency_files: []
  third_party_file: devcovenant/licenses/THIRD_PARTY_LICENSES.md
  licenses_dir: devcovenant/licenses
  report_heading: '## License Report'
  manage_licenses_readme: 'true'
  generate_hashes: 'true'
  audit_service: pypi
  required_paths:
  - devcovenant/runtime-requirements.lock
  hash_targets:
  - id: linux-py311
    marker: sys_platform == "linux" and python_version == "3.11"
    pip:
      platform: manylinux2014_x86_64
      implementation: cp
      python_version: '3.11'
      abi: cp311
  - id: linux-py312
    marker: sys_platform == "linux" and python_version == "3.12"
    pip:
      platform: manylinux2014_x86_64
      implementation: cp
      python_version: '3.12'
      abi: cp312
  - id: linux-py313
    marker: sys_platform == "linux" and python_version == "3.13"
    pip:
      platform: manylinux2014_x86_64
      implementation: cp
      python_version: '3.13'
      abi: cp313
  - id: linux-py314
    marker: sys_platform == "linux" and python_version == "3.14"
    pip:
      platform: manylinux2014_x86_64
      implementation: cp
      python_version: '3.14'
      abi: cp314
  - id: windows-py311
    marker: sys_platform == "win32" and python_version == "3.11"
    pip:
      platform: win_amd64
      implementation: cp
      python_version: '3.11'
      abi: cp311
  - id: windows-py312
    marker: sys_platform == "win32" and python_version == "3.12"
    pip:
      platform: win_amd64
      implementation: cp
      python_version: '3.12'
      abi: cp312
  - id: windows-py313
    marker: sys_platform == "win32" and python_version == "3.13"
    pip:
      platform: win_amd64
      implementation: cp
      python_version: '3.13'
      abi: cp313
  - id: windows-py314
    marker: sys_platform == "win32" and python_version == "3.14"
    pip:
      platform: win_amd64
      implementation: cp
      python_version: '3.14'
      abi: cp314
  - id: macos-py311
    marker: sys_platform == "darwin" and python_version == "3.11"
    pip:
      platform: macosx_11_0_x86_64
      implementation: cp
      python_version: '3.11'
      abi: cp311
  - id: macos-py312
    marker: sys_platform == "darwin" and python_version == "3.12"
    pip:
      platform: macosx_11_0_x86_64
      implementation: cp
      python_version: '3.12'
      abi: cp312
  - id: macos-py313
    marker: sys_platform == "darwin" and python_version == "3.13"
    pip:
      platform: macosx_11_0_x86_64
      implementation: cp
      python_version: '3.13'
      abi: cp313
  - id: macos-py314
    marker: sys_platform == "darwin" and python_version == "3.14"
    pip:
      platform: macosx_11_0_x86_64
      implementation: cp
      python_version: '3.14'
      abi: cp314
- id: root_workspace
  lock_file: requirements.lock
  direct_dependency_files:
  - requirements.in
  - copernican/runtime-requirements.lock
  dependency_files:
  - requirements.in
  - devcovenant/runtime-requirements.lock
  - copernican/runtime-requirements.lock
  third_party_file: licenses/THIRD_PARTY_LICENSES.md
  licenses_dir: licenses
  report_heading: '## License Report'
  manage_licenses_readme: 'true'
  generate_hashes: 'true'
  audit_service: pypi
  required_paths:
  - requirements.in
  - devcovenant/runtime-requirements.lock
  - copernican/runtime-requirements.lock
  hash_targets:
  - id: linux-py311
    marker: sys_platform == "linux" and python_version == "3.11"
    pip:
      platform: manylinux2014_x86_64
      implementation: cp
      python_version: '3.11'
      abi: cp311
  - id: windows-py311
    marker: sys_platform == "win32" and python_version == "3.11"
    pip:
      platform: win_amd64
      implementation: cp
      python_version: '3.11'
      abi: cp311
  - id: macos-py311
    marker: sys_platform == "darwin" and python_version == "3.11"
    pip:
      platform: macosx_13_0_x86_64
      implementation: cp
      python_version: '3.11'
      abi: cp311
  - id: macos-py311-arm64
    marker: sys_platform == "darwin" and python_version == "3.11"
    pip:
      platform: macosx_13_0_arm64
      implementation: cp
      python_version: '3.11'
      abi: cp311
- id: package_runtime
  lock_file: copernican/runtime-requirements.lock
  direct_dependency_files:
  - pyproject.toml
  dependency_files:
  - pyproject.toml
  third_party_file: copernican/lib/licenses/THIRD_PARTY_LICENSES.md
  licenses_dir: copernican/lib/licenses
  report_heading: '## License Report'
  manage_licenses_readme: 'true'
  generate_hashes: 'true'
  audit_service: pypi
  required_paths:
  - pyproject.toml
  - copernican
  hash_targets:
  - id: linux-py311
    marker: sys_platform == "linux" and python_version == "3.11"
    pip:
      platform: manylinux2014_x86_64
      implementation: cp
      python_version: '3.11'
      abi: cp311
  - id: windows-py311
    marker: sys_platform == "win32" and python_version == "3.11"
    pip:
      platform: win_amd64
      implementation: cp
      python_version: '3.11'
      abi: cp311
  - id: macos-py311
    marker: sys_platform == "darwin" and python_version == "3.11"
    pip:
      platform: macosx_13_0_x86_64
      implementation: cp
      python_version: '3.11'
      abi: cp311
  - id: macos-py311-arm64
    marker: sys_platform == "darwin" and python_version == "3.11"
    pip:
      platform: macosx_13_0_arm64
      implementation: cp
      python_version: '3.11'
      abi: cp311
license_source_overrides:
- id: click
  kind: archive_url
  url: https://files.pythonhosted.org/packages/source/c/click/click-{version}.tar.gz
  member_globs:
  - click-{version}/LICENSE.txt
  - click-{version}/docs/license.md
selector_roles: dependency
dependency_globs: []
dependency_files: []
dependency_dirs: []
```

Manage dependency-maintenance artifacts as one coherent policy surface.
When dependency inputs change, lockfiles and repository
compliance artifacts must stay synchronized together.
Artifact ownership is declared through structured `surfaces` metadata.
Each surface defines one lock/report/license bundle with its own lock file,
direct dependency inputs, dependency selectors, license report path,
license directory, and optional hash-lock target matrix.
Surface selectors support role-based taxonomy for mixed ecosystems:
`intent`, `resolved`, and `package_manifest`.
Each surface may declare nested selector keys:
`dependency_files`, `dependency_globs`, `dependency_dirs`,
`dependency_role_files`, `dependency_role_globs`,
`dependency_role_dirs`.
Surface selectors decide when checks and license/report refresh must react.
Direct dependency inputs decide when a lock refresh must recompile.
Surfaces may also declare `audit_service` and `audit_ignore_ids`.
When audit is enabled, policy checks audit the frozen lock contents directly
against published vulnerability data instead of trusting the current host's
marker selection, so Linux-only or Windows-only vulnerable pins do not hide
on another platform.
Policy checks remain read-only. Autofixers may invoke declared policy
runtime actions, and explicit policy-born CLI commands may invoke those same
runtime actions manually. Remediation messaging may differ when autofix is
enabled versus disabled. When one Python surface enables
`generate_hashes`, DevCovenant resolves the full configured target closure
from `hash_targets` and writes an all-target hash lock or fails explicitly.
Advisory lookups use UTC-stamped runtime-local cache entries, not tracked
repository files, so repeated gates reuse recent results without adding
hidden CI-only audit logic.
When a direct dependency does not bundle upstream license files in installed
metadata, repositories may declare `license_source_overrides` keyed by
normalized package name. Builtin overrides currently support `archive_url`
sources with templated `url` and `member_globs` fields.
Hash mode does not patch a host-local compile result or depend on
GitHub-specific dependency logic. Artifact refresh remains
deterministic/idempotent.


---

## Policy: Docstring And Comment Coverage

```policy-def
id: docstring-and-comment-coverage
severity: error
auto_fix: 'false'
enforcement: active
enabled: 'true'
custom: 'false'
include_suffixes:
- .py
exclude_prefixes:
- build
- dist
- node_modules
- devcovenant
- tests
include_prefixes: []
include_globs:
- '*.py'
exclude_suffixes: []
exclude_globs:
- build/**
- dist/**
- node_modules/**
- devcovenant/**
- tests/**
force_include_globs:
- devcovenant/custom/**/*.py
- tests/devcovenant/custom/**/*.py
selector_roles:
- include
- exclude
- force_include
include_files: []
include_dirs: []
exclude_files: []
exclude_dirs: []
force_include_files: []
force_include_dirs: []
```

Source files must include a docstring or nearby explanatory comment so
intent stays visible even as code evolves. Adapters decide how each
language satisfies the requirement.


---

## Policy: Documentation Growth Tracking

```policy-def
id: documentation-growth-tracking
severity: warning
auto_fix: 'false'
enforcement: active
enabled: 'true'
custom: 'false'
selector_roles:
- user_facing
- user_visible
- doc_quality
include_prefixes: []
exclude_prefixes:
- devcovenant
- tests/devcovenant
user_facing_prefixes: []
user_facing_exclude_prefixes:
- tests
- devcovenant
- tests/devcovenant
user_facing_suffixes:
- .py
- .js
- .ts
- .tsx
- .vue
- .go
- .rs
- .java
- .kt
- .swift
- .rb
- .php
- .cs
- .yml
- .yaml
- .json
- .toml
user_facing_files:
- .pre-commit-config.yaml
- pyproject.toml
user_facing_globs:
- .github/workflows/*.yml
- .github/workflows/*.yaml
- '*.py'
- '*.js'
- '*.ts'
- '*.tsx'
- '*.vue'
- '*.go'
- '*.rs'
- '*.java'
- '*.kt'
- '*.swift'
- '*.rb'
- '*.php'
- '*.cs'
- '*.yml'
- '*.yaml'
- '*.json'
- '*.toml'
user_facing_keywords:
- api
- endpoint
- endpoints
- route
- routes
- routing
- service
- services
- controller
- controllers
- handler
- handlers
- client
- clients
- webhook
- webhooks
- integration
- integrations
- sdk
- cli
- ui
- view
- views
- page
- pages
- screen
- screens
- form
- forms
- workflow
- workflows
user_visible_files:
- ./README.md
- ./ABOUT.md
- ./SECURITY.md
- ./SUPPORT.md
- ./CITATION.cff
- ./AGENTS.md
- ./CONTRIBUTING.md
- ./SPEC.md
- ./PLAN.md
doc_quality_files:
- ./README.md
- ./ABOUT.md
- ./SECURITY.md
- ./SUPPORT.md
- ./AGENTS.md
- ./CONTRIBUTING.md
- ./SPEC.md
- ./PLAN.md
required_headings:
- Overview
require_toc: 'false'
min_section_count: '3'
min_word_count: '120'
doc_routes:
- copernican/lib/** => README.md, ABOUT.md
- copernican/lib/gui/** => README.md, ABOUT.md, SUPPORT.md
- copernican/rng_minigames/** => README.md, ABOUT.md
- engines/** => README.md, ABOUT.md
- pyproject.toml => ABOUT.md, SECURITY.md, CITATION.cff
- copernican_settings.yml => README.md, ABOUT.md, SUPPORT.md
- copernican/workflow.py => README.md, ABOUT.md, SUPPORT.md
require_mentions: 'true'
mention_min_length: '3'
mention_stopwords:
- devcovenant
- tools
- common
- custom
- policy
- policies
- script
- scripts
- py
- js
- ts
- json
- yml
- yaml
- toml
- md
- readme
- plan
- spec
include_suffixes: []
include_globs: []
exclude_suffixes: []
exclude_globs:
- devcovenant/**
- tests/devcovenant/**
force_include_globs: []
user_facing_exclude_globs:
- .vscode/**
- .idea/**
- .venv/**
- .python/**
- build/**
- dist/**
- __pycache__/**
- '*.egg-info/**'
- pip-wheel-metadata/**
- .pytest_cache/**
- .ruff_cache/**
- .mypy_cache/**
- .tox/**
- .nox/**
- .hypothesis/**
- .coverage
- .coverage.*
- htmlcov/**
- devcovenant/logs/**
- devcovenant/registry/runtime/**
- models/**
- copernican/datasets/**
- tests/**
- devcovenant/**
- tests/devcovenant/**
user_facing_exclude_suffixes: []
user_facing_dirs: []
user_visible_globs: []
user_visible_dirs: []
doc_quality_globs: []
doc_quality_dirs: []
include_files: []
include_dirs: []
exclude_files: []
exclude_dirs: []
user_facing_exclude_files:
- devcovenant/config.yaml
user_facing_exclude_dirs:
- tests/**
force_include_files: []
force_include_dirs: []
```

When user-facing files change (as defined by the user-facing selectors and
keywords), the documentation set listed here must be updated. User-facing
includes API surfaces, integration touchpoints, and any behavior that affects
the user's experience or workflow. Updated docs should mention the relevant
components by name so readers can find changes quickly. The policy also
enforces documentation quality standards such as clear core headings,
minimum depth, and readable structure. When `doc_routes` is configured,
each user-facing change must match at least one route and touch all mapped
docs.


---

## Policy: Last Updated

```policy-def
id: last-updated
severity: error
auto_fix: 'true'
enforcement: active
enabled: 'true'
custom: 'false'
include_suffixes:
- .md
allowed_globs:
- devcovenant/README.md
- devcovenant/core/README.md
- devcovenant/custom/README.md
- devcovenant/registry/README.md
- devcovenant/logs/README.md
- devcovenant/builtin/policies/README.md
- devcovenant/builtin/profiles/README.md
- devcovenant/custom/policies/README.md
- devcovenant/custom/profiles/README.md
- devcovenant/docs/*.md
- devcovenant/docs/**/*.md
- README.md
- AGENTS.md
- CONTRIBUTING.md
- CHANGELOG.md
- SPEC.md
- PLAN.md
- ABOUT.md
- SECURITY.md
- SUPPORT.md
- copernican/README.md
- copernican/ABOUT.md
- copernican/SECURITY.md
- copernican/SUPPORT.md
- copernican/docs/*.md
- copernican/docs/**/*.md
allowed_files: []
allowed_suffixes: []
required_files: []
required_globs:
- devcovenant/README.md
- devcovenant/core/README.md
- devcovenant/custom/README.md
- devcovenant/registry/README.md
- devcovenant/logs/README.md
- devcovenant/builtin/policies/README.md
- devcovenant/builtin/profiles/README.md
- devcovenant/custom/policies/README.md
- devcovenant/custom/profiles/README.md
- devcovenant/docs/*.md
- devcovenant/docs/**/*.md
- README.md
- AGENTS.md
- CONTRIBUTING.md
- CHANGELOG.md
- SPEC.md
- PLAN.md
selector_roles:
- include
- allowed
- required
include_globs:
- '*.md'
include_files: []
include_dirs: []
allowed_dirs: []
required_dirs: []
```

Docs must include a `Last Updated` header in the generated header zone so
readers can trust recency. The auto-fix updates UTC dates for touched
allowlisted docs while respecting allowlist selectors.


---

## Policy: Line Length Limit

```policy-def
id: line-length-limit
severity: warning
auto_fix: 'false'
enforcement: active
enabled: 'true'
custom: 'false'
max_length: '79'
allow_long_url_lines: 'true'
url_prefixes:
- https://
- http://
- ftp://
- ftps://
- sftp://
- ssh://
- ws://
- wss://
- file://
- git://
- svn://
- 'mailto:'
- 'tel:'
- 'magnet:'
- 'torrent:'
- 'data:'
- 'urn:'
allow_long_lines: 'true'
long_lines_contain:
- =>
long_lines_between: []
include_suffixes:
- .py
- .md
- .rst
- .txt
- .yml
- .yaml
- .json
- .toml
- .cff
exclude_prefixes:
- build
- dist
- node_modules
- devcovenant
exclude_globs:
- .vscode/**
- .idea/**
- .venv/**
- .python/**
- build/**
- dist/**
- __pycache__/**
- '*.egg-info/**'
- pip-wheel-metadata/**
- .pytest_cache/**
- .ruff_cache/**
- .mypy_cache/**
- .tox/**
- .nox/**
- .hypothesis/**
- .coverage
- .coverage.*
- htmlcov/**
- licenses/*.txt
- devcovenant/licenses/*.txt
- devcovenant/logs/**
- devcovenant/registry/runtime/**
- node_modules/**
- '**/*.egg-info/**'
- .matplotlib-cache/**
- copernican/rng_minigames/registry.json
- validation/manifests/**
- licenses/THIRD_PARTY_LICENSES.md
- copernican/lib/licenses/THIRD_PARTY_LICENSES.md
- copernican/lib/licenses/*.txt
- copernican/datasets/**
- devcovenant/**
include_prefixes: []
include_globs:
- '*.py'
- '*.md'
- '*.rst'
- '*.txt'
- '*.yml'
- '*.yaml'
- '*.json'
- '*.toml'
- '*.cff'
exclude_suffixes: []
force_include_globs:
- devcovenant/custom/**
- tests/devcovenant/custom/**
- tests/**/*.py
- tests/**/*.md
- copernican/datasets/**/cosmo_parser_*.py
- copernican/datasets/**/metadata_*.yml
- copernican/datasets/**/metadata_*.yaml
selector_roles:
- include
- exclude
- force_include
url_globs:
- https:/**
- http:/**
- ftp:/**
- ftps:/**
- sftp:/**
- ssh:/**
- ws:/**
- wss:/**
- file:/**
- git:/**
- svn:/**
- mailto:/**
- tel:/**
- magnet:/**
- torrent:/**
- data:/**
- urn:/**
url_files: []
url_dirs: []
include_files: []
include_dirs: []
exclude_files: []
exclude_dirs: []
force_include_files: []
force_include_dirs: []
```

Keep lines within the configured maximum so documentation and code remain
readable. Reflow long sentences or wrap lists rather than ignoring the limit.
Optional metadata escape hatches can allow long lines for URL-heavy content
or explicit marker patterns when repositories need targeted flexibility.


---

## Policy: Managed Environment

```policy-def
id: managed-environment
severity: error
auto_fix: 'false'
enforcement: active
enabled: 'true'
custom: 'false'
expected_paths:
- .venv
expected_interpreters:
- .venv/bin/python
- .venv/Scripts/python.exe
command_search_paths: []
cleanup_protected_paths: []
required_commands:
- pre-commit
allow_current_interpreter_fallback: 'false'
manual_commands:
- '{current_python} -m venv .venv'
- '{managed_python} -m pip install -r requirements.lock'
managed_commands:
- bootstrap=>{current_python} -m venv .venv
- bootstrap=>{managed_python} -m pip install -r requirements.lock
- open=>{current_python} -m venv .venv
- open=>{managed_python} -m pip install -r requirements.lock
```

DevCovenant must run from one execution environment described by this
policy's metadata. `expected_paths` and `expected_interpreters` point to
that target environment, while optional `command_search_paths` add extra
PATH entries for resolving required commands and optional
`cleanup_protected_paths` define extra roots that cleanup must never delete.
`required_commands` declare the commands that must resolve once the target
interpreter is selected, using the managed PATH plus any declared command
search paths. `manual_commands` document how a human can create or repair
the environment, and stage-scoped `managed_commands` define how
DevCovenant may prepare it automatically. Command templates may reference
`{current_python}` and `{current_bin}` for the currently running
interpreter, and `{managed_python}`, `{managed_bin}`, `{managed_root}`,
`{repo_root}` for the selected target environment. User-facing guidance
renders those path tokens with display-safe paths so local absolute roots do
not leak into routine messages.
The target environment may live inside the repository or outside it, as long
as the metadata declares the interpreter path or environment root that
DevCovenant should use. The policy itself is environment-neutral: a
repository may declare a local `.venv`, a bench-managed environment, a
container-managed environment, a system interpreter, or another tool-owned
layout in its active profile stack. DevCovenant should not assume the
builtin defaults profile picks one of those layouts for it.
Active managed-environment policy reuses the current interpreter when it
already satisfies the contract, re-executes CLI commands in the selected
interpreter when needed, and only runs bootstrap commands when the target
environment is still missing or invalid. Stage-scoped `managed_commands`
accept `open`, `run`, `close`, `bootstrap`, `managed`, and `all` prefixes;
bootstrap-mode commands may still reuse `open` bootstrap commands once
when the target environment is not ready. When no automatic bootstrap
commands are declared, `bootstrap` stage operations may keep using the
current interpreter only when `allow_current_interpreter_fallback` is
explicitly true; otherwise `open`, `run`, `close`, and `managed` remain
strict about the target environment so incomplete profiles fail loudly
instead of falling through to wrapper adapters or alternate policy
sources. If the resolved interpreter is missing or not executable,
DevCovenant fails explicitly instead of falling through to wrapper
adapters or alternate policy sources. When enabled with empty metadata,
the policy emits a warning so teams fill the required context.


---

## Policy: Modules Need Tests

```policy-def
id: modules-need-tests
severity: error
auto_fix: 'false'
enforcement: active
enabled: 'true'
custom: 'false'
include_suffixes:
- .py
include_prefixes: []
exclude_prefixes:
- build
- dist
- node_modules
- tests
- devcovenant
exclude_globs:
- build/**
- dist/**
- node_modules/**
- tests/**
- devcovenant/**
watch_dirs:
- tests
tests_watch_dirs:
- tests
mirror_roots:
- devcovenant/custom=>tests/devcovenant/custom
- copernican/datasets/bao=>tests/copernican/datasets/bao
- copernican/datasets/cmb=>tests/copernican/datasets/cmb
- copernican/datasets/gw=>tests/copernican/datasets/gw
- copernican/datasets/sne=>tests/copernican/datasets/sne
mirror_test_name_templates:
- python=>test_{stem}.py
test_style_requirements:
- python=>python_unittest
include_globs:
- '*.py'
exclude_suffixes: []
force_include_globs:
- devcovenant/custom/**/*.py
watch_files: []
placeholder_test_methods:
- test_placeholder
placeholder_text_markers:
- placeholder-marker-alpha
- placeholder-marker-beta
- placeholder-marker-gamma
selector_roles:
- include,exclude,watch,tests_watch,force_include
include_files: []
include_dirs: []
exclude_files: []
exclude_dirs: []
watch_globs: []
tests_watch_globs: []
tests_watch_files: []
force_include_files: []
force_include_dirs: []
```

In-scope non-test modules must have corresponding tests under configured
test roots. The rule is metadata-driven and supports mirror enforcement for
selected source roots. The policy enforces structural source-to-test
alignment and rejects stale mirrored tests. Placeholder tests are not
allowed. Python test files must use unittest.TestCase-style definitions;
workflow execution runs the declared unittest command directly.


---

## Policy: Name Clarity

```policy-def
id: name-clarity
severity: warning
auto_fix: 'false'
enforcement: active
enabled: 'true'
custom: 'false'
exclude_prefixes:
- build
- dist
- node_modules
- devcovenant
include_suffixes:
- .py
include_prefixes: []
include_globs:
- '*.py'
exclude_suffixes: []
exclude_globs:
- build/**
- dist/**
- node_modules/**
- devcovenant/**
force_include_globs:
- devcovenant/custom/**/*.py
- tests/devcovenant/custom/**/*.py
- tests/**/*.py
selector_roles:
- exclude
- include
- force_include
exclude_files: []
exclude_dirs: []
include_files: []
include_dirs: []
force_include_files: []
force_include_dirs: []
```

Identifiers should be descriptive enough to communicate intent without
reading their implementation. Avoid cryptic or overly short names unless
explicitly justified.


---

## Policy: No Future Dates

```policy-def
id: no-future-dates
severity: error
auto_fix: 'true'
enforcement: active
enabled: 'true'
custom: 'false'
```

Dates in changelogs or documentation must not be in the future. Auto-fixers
should correct accidental placeholders to today’s date.


---

## Policy: No Print Outside Output Runtime

```policy-def
id: no-print-outside-output-runtime
severity: error
auto_fix: 'false'
enforcement: active
enabled: 'true'
custom: 'false'
selector_roles:
- include
- exclude
- force_include
allowed_file_files: []
allowed_file_dirs: []
include_suffixes: []
include_prefixes: []
include_globs: []
exclude_suffixes: []
exclude_prefixes: []
exclude_globs: []
force_include_globs: []
include_files: []
include_dirs: []
exclude_files: []
exclude_dirs: []
force_include_files: []
force_include_dirs: []
sink_call_targets:
- python=>print
- python=>builtins.print
sink_attr_targets: []
sink_macro_targets: []
allowed_symbol_targets: []
allowed_file_globs: []
allow_waiver_comment: []
```

Enforce metadata-driven direct-output sink boundaries across configured
languages. Language sink definitions come from profile overlays, while
repository profiles define in-scope selectors and boundary allowlists.


---

## Policy: No Raw Errors

```policy-def
id: no-raw-errors
severity: error
auto_fix: 'false'
enforcement: active
enabled: 'true'
custom: 'false'
selector_roles:
- include
- exclude
- force_include
include_suffixes:
- .py
include_prefixes: []
include_globs:
- '*.py'
exclude_suffixes: []
exclude_prefixes:
- build
- dist
- node_modules
- devcovenant
- tests
exclude_globs:
- build/**
- dist/**
- node_modules/**
- devcovenant/**
- tests/**
force_include_globs:
- devcovenant/custom/**/*.py
- tests/devcovenant/custom/**/*.py
include_files: []
include_dirs: []
exclude_files: []
exclude_dirs: []
force_include_files: []
force_include_dirs: []
forbid_bare_except: 'true'
forbid_raise_exception: 'true'
forbid_broad_exception_handlers: 'true'
forbid_silent_exception_pass: 'true'
broad_exception_waiver_markers:
- DEVCOV_ALLOW_BROAD_ONCE
broad_exception_waiver_between:
- DEVCOV_BROAD_BEGIN=>DEVCOV_BROAD_END
```

Enforce explicit error surfaces and block raw exception anti-patterns.
This policy flags bare `except`, broad `except Exception` handlers,
generic `raise Exception(...)`, and silent `except Exception: pass`
handlers in selected source files. Broad-handler waivers are explicit
through marker comments or marker regions.


---

## Policy: Package Artifact Mirror

```policy-def
id: package-artifact-mirror
severity: error
auto_fix: 'true'
enforcement: active
enabled: 'true'
custom: 'false'
file_mirrors: []
dir_mirrors: []
dir_skip_paths: []
```

Ensure package-shipped artifacts that are true mirrors stay in exact sync
with their canonical repository-root sources. `file_mirrors` and
`dir_mirrors` declare `source=>target` pairs. `dir_skip_paths` declares
repo-relative mirror exceptions in `source_dir=>relative/path` form.
Dependency lockfiles and third-party license inventories are not mirrors
unless the repository explicitly declares them here. Auto-fix rewrites the
configured exact mirrors from their source paths, preserves separately
mirrored files that live inside mirrored directories, preserves package-owned
skipped paths, and removes stale mirrored files.


---

## Policy: Package Doc Sync

```policy-def
id: package-doc-sync
severity: error
auto_fix: 'true'
enforcement: active
enabled: 'true'
custom: 'false'
sync_pairs:
- README.md=>copernican/README.md
- ABOUT.md=>copernican/ABOUT.md
- SECURITY.md=>copernican/SECURITY.md
- SUPPORT.md=>copernican/SUPPORT.md
- CITATION.cff=>copernican/CITATION.cff
- docs/api_overview.md=>copernican/docs/api_overview.md
- docs/architecture.md=>copernican/docs/architecture.md
- docs/bao_compound_dataset_format.md=>copernican/docs/bao_compound_dataset_format.md
- docs/cli_guide.md=>copernican/docs/cli_guide.md
- docs/data_overview.md=>copernican/docs/data_overview.md
- docs/dataset_licenses.md=>copernican/docs/dataset_licenses.md
- docs/dataset_metadata.md=>copernican/docs/dataset_metadata.md
- docs/design_overview.md=>copernican/docs/design_overview.md
- docs/gui_guide.md=>copernican/docs/gui_guide.md
- docs/gui_overview.md=>copernican/docs/gui_overview.md
- docs/latex_syntax.md=>copernican/docs/latex_syntax.md
- docs/minigames.md=>copernican/docs/minigames.md
- docs/orchestration_services.md=>copernican/docs/orchestration_services.md
- docs/packaging.md=>copernican/docs/packaging.md
- docs/run_manifest.md=>copernican/docs/run_manifest.md
- docs/security_changes.md=>copernican/docs/security_changes.md
omit_block_pairs:
- <!-- REPO-ONLY:BEGIN -->=><!-- REPO-ONLY:END -->
rewrite_repo_relative_links: 'false'
```

Ensure package-facing documentation files stay synchronized with their
canonical repository-source docs. Configured `sync_pairs` map
`source=>target` doc paths. Configured `omit_block_pairs` remove
repo-only sections between paired begin/end markers before comparison.
When `rewrite_repo_relative_links` is true, repo-relative Markdown links are
rewritten to `main`-branch repository URLs resolved from `pyproject.toml`,
and repo-relative images are rewritten to the repository's `main`-branch raw
URLs so PyPI renders them. Auto-fix rewrites the configured package docs from
their source docs after those transforms.


---

## Policy: Raw String Escapes

```policy-def
id: raw-string-escapes
severity: warning
auto_fix: 'false'
enforcement: active
enabled: 'false'
custom: 'false'
include_suffixes:
- .py
- .pyi
- .pyw
- .js
- .jsx
- .ts
- .tsx
- .go
- .rs
- .java
- .cs
- .kt
- .swift
- .php
- .rb
selector_roles:
- include
- exclude
- force_include
include_globs:
- '*.py'
- '*.pyi'
- '*.pyw'
- '*.js'
- '*.jsx'
- '*.ts'
- '*.tsx'
- '*.go'
- '*.rs'
- '*.java'
- '*.cs'
- '*.kt'
- '*.swift'
- '*.php'
- '*.rb'
include_files: []
include_dirs: []
exclude_globs: []
exclude_files: []
exclude_dirs: []
force_include_globs: []
force_include_files: []
force_include_dirs: []
language_globs: []
language_files: []
language_dirs: []
language_suffixes: []
literal_patterns: []
raw_literal_patterns: []
suspicious_escape_patterns: []
```

Warn when in-scope string literals contain suspicious bare backslashes.
Detection is language-aware: Python uses tokenizer spans, while other
languages use metadata-driven literal and escape patterns.


---

## Policy: Read Only Directories

```policy-def
id: read-only-directories
severity: error
auto_fix: 'false'
enforcement: active
enabled: 'true'
custom: 'false'
include_globs:
- copernican/datasets/**
include_suffixes: []
include_prefixes:
- copernican/datasets
exclude_suffixes: []
exclude_prefixes: []
exclude_globs:
- copernican/datasets/**/cosmo_parser_*.py
- copernican/datasets/**/metadata_*.yml
- copernican/datasets/**/metadata_*.yaml
force_include_globs: []
selector_roles:
- include
- exclude
- force_include
include_files: []
include_dirs: []
exclude_files: []
exclude_dirs: []
force_include_files: []
force_include_dirs: []
```

Protect declared read-only directories from modification. If a directory must
be editable, update this policy definition first.


---

## Policy: Security Scanner

```policy-def
id: security-scanner
severity: error
auto_fix: 'false'
enforcement: active
enabled: 'true'
custom: 'false'
exclude_globs:
- tests/**
- '**/tests/**'
- devcovenant/**
include_suffixes:
- .py
include_prefixes: []
include_globs:
- '*.py'
exclude_suffixes: []
exclude_prefixes:
- devcovenant
force_include_globs:
- devcovenant/custom/**/*.py
- tests/devcovenant/custom/**/*.py
- tests/**/*.py
selector_roles:
- exclude
- include
- force_include
exclude_files: []
exclude_dirs: []
include_files: []
include_dirs: []
force_include_files: []
force_include_dirs: []
scanners:
- id: bandit
  kind: bandit
  config_file: bandit.yaml
  include_suffixes:
  - .py
```

Scan source files for risky constructs like `eval`, `exec`, or
`shell=True`. Repositories may also attach reviewed external scanners
through structured `scanners` metadata, so local gates, local `check`,
and generated CI all use the same policy-owned security path instead of
separate workflow glue. Use only documented reviewed allow markers such
as translator `security-scanner: allow` comments or backend-native
suppressions like `# nosec` when a security review approves the
exception.


---

## Policy: Tests Coverage

```policy-def
id: tests-coverage
severity: error
auto_fix: 'false'
enforcement: active
enabled: 'true'
custom: 'false'
enforce_symbol_fidelity: 'true'
symbol_kinds:
- function
- class
symbol_name_min_length: '3'
symbol_assertion_window: '2'
fixture_marker_pattern: \bDEVCOV_FIXTURE_OK:\s*(?P<reason>\S.*)
assertion_signal_patterns:
- '*=>\bassert\b'
- python=>\bassert\b
- python=>\bself\.assert[A-Za-z_]*\s*\(
tautology_patterns:
- '*=>^\s*assert\s*\(\s*true\s*\)\s*;?\s*$'
- '*=>^\s*assert\s+true\s*;?\s*$'
- rust=>^\s*assert!\s*\(\s*true\s*\)\s*;?\s*$
- python=>^\s*assert\s+True\s*$
- python=>^\s*self\.assertTrue\s*\(\s*True\s*\)\s*$
include_suffixes:
- .py
include_prefixes: []
exclude_prefixes:
- build
- dist
- node_modules
- tests
- devcovenant
exclude_globs:
- build/**
- dist/**
- node_modules/**
- tests/**
- devcovenant/**
watch_dirs:
- tests
tests_watch_dirs:
- tests
include_globs:
- '*.py'
exclude_suffixes: []
force_include_globs:
- devcovenant/custom/**/*.py
watch_files: []
selector_roles:
- include,exclude,watch,tests_watch,force_include
include_files: []
include_dirs: []
exclude_files: []
exclude_dirs: []
watch_globs: []
tests_watch_globs: []
tests_watch_files: []
force_include_files: []
force_include_dirs: []
```

In-scope modules with related tests must include assertion signals in those
related test files. This policy enforces assertion-quality coverage for
structural source-to-test relationships, while modules-need-tests enforces
source-to-test structural alignment itself.
Tautological assertions (for example always-true checks) do not count as
assertion signal unless explicitly annotated as fixture-only using comment
marker `DEVCOV_FIXTURE_OK: <reason>` immediately above the assertion.


---

## Policy: Version Governance

```policy-def
id: version-governance
severity: error
auto_fix: 'false'
enforcement: active
enabled: 'false'
custom: 'false'
scheme: semver
enforce_bumping: 'true'
canonical_versions_required: 'true'
version_file: copernican/VERSION
changelog_file: CHANGELOG.md
changelog_header_prefix: '## Version'
ignored_prefixes:
- devcovenant
- models
semver_scope_tags_required: 'true'
pep440_allow_prereleases: 'true'
pep440_allow_dev_releases: 'true'
pep440_allow_post_releases: 'true'
calver_pattern: []
custom_regex_pattern: []
custom_adapter_path: []
selector_roles: ignored
ignored_globs:
- devcovenant/**
- models/**
ignored_files: []
ignored_dirs: []
```

When enabled, this policy governs repository version format and optional
version progression under the configured `scheme`. Repositories may enable
`enforce_bumping` to require forward version movement, while format-only
custom schemes may leave bump enforcement disabled. Scheme adapters may be
builtin or repo-defined and may add extra release rules to the latest
changelog entry.
Activation is controlled by `config.yaml -> policy_state`. Repositories
should choose `version-governance.scheme` explicitly in profile or config
metadata, then tune bump enforcement to match the selected scheme.


---

## Policy: Version Sync

```policy-def
id: version-sync
severity: error
auto_fix: 'true'
enforcement: active
enabled: 'false'
custom: 'false'
version_file: copernican/VERSION
target_roles:
- docs
- changelog
- package_manifest
role_extractors:
- docs=>project_version_line
- changelog=>changelog_header_version
- package_manifest=>manifest_project_version
role_legality_schemes:
- package_manifest=>pep440
target_role_files:
- docs=>README.md
- docs=>AGENTS.md
- docs=>CONTRIBUTING.md
- docs=>SPEC.md
- docs=>PLAN.md
- changelog=>CHANGELOG.md
- package_manifest=>pyproject.toml
- docs=>ABOUT.md
- docs=>SECURITY.md
- docs=>SUPPORT.md
target_role_globs: []
target_role_dirs: []
changelog_file: CHANGELOG.md
changelog_header_prefix: '## Version'
selector_roles:
- target
target_globs: []
target_files: []
target_dirs: []
```

All version-bearing targets must match the canonical version file (default
`VERSION` or a configured override).
Target selection is role-based via `target_roles` and role selectors
(`target_role_files`, `target_role_globs`, `target_role_dirs`) with
`role=>selector` entries. Version extraction is role-driven via
`role_extractors` and explicit extractor names
(`project_version_line`, `devcovenant_version_line`,
`changelog_header_version`, `manifest_project_version`). Manifest
extraction remains format-aware (TOML/JSON/YAML) while selector routing
stays role-based. Version-sync validates and compares every extracted
value through the active `version-governance` scheme so canonical docs,
managed-doc `DevCovenant Version` headers, changelog entries, manifests,
and any opted-in legal text stay synchronized even when repositories use
non-SemVer version formats. Optional `role_legality_schemes` entries add
stricter ecosystem legality for selected roles without collapsing
repo-level version governance back into one packaging-only scheme.
Autofix repairs mismatched declared targets by rewriting the target's
version token to the current canonical version.
<!-- DEVCOV-POLICIES:END -->
