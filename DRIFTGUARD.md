# Development Laws (DriftGuard Policy)

This repository uses **DriftGuard** as a live policy layer for human+AI
development.

- The **canonical machine-readable policy** is defined in
  `driftguard/repo_policy.yml`. This file describes:
  - which files belong to which surfaces (docs, interfaces, library code, etc.),
  - which rules apply to each surface,
  - which drift metrics are tracked, and with what thresholds.
- This document is a human-readable summary. In case of mismatch,
  `driftguard/repo_policy.yml`—as enforced by the DriftGuard code under
  `driftguard/`—is the source of truth. This page replaces the legacy embedded
  sections in `README.md` and `AGENTS.md`.

### Docs and metadata

- Only the following files must carry a `Last Updated:` header:
  - `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `CHANGELOG.md`
  - all `docs/**/*.md`
  - `CITATION.cff`, `LICENSE.md`, `THIRD_PARTY_LICENSES.md`
  - `copernican.py`
  - `start.sh`, `start.command`, `start.bat`
  - `copernican_lib/config_schemas/**/*.yml`
  - `models/**/*.yml`
- All other files (library code, tests, tools, workflows, lockfiles, etc.)
  must **not** have a `Last Updated` header.
- `Last Updated` dates must never be in the future. DriftGuard enforces this
  for the files that require headers.
- A single semantic version `X.Y.Z` must be consistent across:
  - `copernican_lib/VERSION`
  - `README.md` (where it appears)
  - `CITATION.cff`
  - packaging metadata.
- Any non-trivial change in behaviour, interface, or policy must add an entry
  to the latest section of `CHANGELOG.md`.
- Conflict-marker rules are not enforced by policy; Git/merge tools handle
  them.

### Code and tests

- New Python modules under `copernican_lib/` and `engines/` must be accompanied
  by new or updated tests under `tests/`.
  DriftGuard treats “new module with no test changes” as a hard violation.
- Library and engine code (`copernican_lib/**/*.py`, `engines/**/*.py`) must
  not use bare `print()`.
  Use the logging / console abstraction instead.
- Tests and tooling scripts may use `print()` where appropriate.
- Bugfixes:
  - Any bugfix noted in `CHANGELOG.md` should be backed by a test that would
    fail without the fix.
  - DriftGuard currently treats “bugfix with no test change” as a warning, not
    a hard error.

### Tools and workflows

- The codebase is formatted and linted with:
  - **Black** (formatting),
  - **isort** (import ordering),
  - **Ruff** (linting and simple quality checks),
  - **Flake8** (additional linting),
  - **pre-commit-hooks** (`end-of-file-fixer`, `trailing-whitespace`) for basic
    hygiene.
- **DriftGuard** does not replace these tools; it complements them by enforcing
  project-specific policy and drift limits.
- A local dev script (e.g. `python tools/dev_suite.py`) should be run before
  committing. It:
  - runs formatting and linting,
  - runs DriftGuard in fast mode on staged files (and may apply safe
    autofixes),
  - runs a relevant subset of tests.
- Run the **full program unit test suite in `/tests` before every commit** using
  both `python -m pytest -q` and `python -m unittest discover -v`. The CI
  pipeline also runs both commands to keep commits aligned with the policy and
  provide an audit trail for test coverage.

### CI and drift

- CI runs `driftguard check` in full mode on the whole repository:
  - Hard violations (broken policy) fail the build.
  - Drift metrics (e.g. TODO count, test–module coupling, document age) are
    computed and logged.
- Drift metrics have thresholds configured in `driftguard/repo_policy.yml`.
  Crossing a threshold emits a warning and may eventually be promoted to a hard
  error.
- The goal is **positive drift**:
  - adding tests when new modules appear,
  - keeping documentation and metadata fresh,
  - gradually reducing TODO/FIXME debt,
  - tightening policy over time rather than loosening it.
