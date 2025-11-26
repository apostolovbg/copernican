# Development Laws (DriftGuard Policy)
**Last Updated:** 2025-11-26

This repository uses **DriftGuard** as a live policy layer for human+AI
development. `DRIFTGUARD.md` is the human source of truth and must remain in
lockstep with `driftguard/repo_policy.yml` and the enforcement code under
`driftguard/`. Any policy change in one location must be mirrored in the other
two immediately and accompanied by tests.

- The **canonical machine-readable policy** is defined in
  `driftguard/repo_policy.yml`. This file describes:
  - which files belong to which surfaces (docs, interfaces, library code, etc.),
  - which rules apply to each surface,
  - which drift metrics are tracked, and with what thresholds.
- This document is a human-readable summary that must remain synchronised with
  `driftguard/repo_policy.yml` and the enforcement code under `driftguard/`.
  Any drift among these surfaces is a policy violation.

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
  - runs a quick DriftGuard sanity check on staged files (and may apply safe
    autofixes) before you run the full policy suite,
  - runs a relevant subset of tests.
- Before any commit, run `python -m pytest -q`, `python -m unittest discover -v`
  and `driftguard check --scope=staged --mode=full` on staged changes. Fix any
  failures from those commands before committing; commits without a fresh full
  DriftGuard pass are prohibited.
- Keep Python sources **Black-clean before committing**. DriftGuard runs the
  `formatter-clean` rule across the full policy surfaces—even when Git shows a
  clean working tree—and fails if Black would reformat tracked or staged
  Python files to prevent CI from rewriting files mid-run.
- Run the **full program unit test suite in `/tests` before every commit and
  every task** using both `python -m pytest -q` and
  `python -m unittest discover -v`. The CI pipeline runs both commands under
  the `Unit tests` and `Pytest` steps before executing DriftGuard so the policy
  engine evaluates a verified codebase.

### Operational discipline and returning laws

- Always run `python -m pytest -q`, `python -m unittest discover -v`, and the
  **full** `driftguard check --scope=staged --mode=full` before every commit in
  every session. Treat hard and soft violations as urgent blockers.
- `CHANGELOG.md` entries must follow the template and list every touched file
  or subsystem. Compare `git diff --name-only` against the newest changelog
  entry before committing so nothing slips past the DriftGuard hook.
- Comment code extensively and keep comments aligned with behaviour. Capture
  both the “what” and the “why” for simple and complex logic alike. Update
  nearby comments immediately whenever behaviour changes so code and comments
  stay synchronised.
- Document every module, function and class with clear “what” and “why”
  explanations. Use concise, descriptive identifiers, prefer raw strings or
  explicit escapes to avoid invalid sequences, and keep lines under
  79 characters.
- Update documentation—including AGENTS, README and `docs/`—whenever behaviour
  or structure changes. Treat documentation refresh as integral to every task
  and keep `Last Updated` headers chronological.
- Bump project version per SemVer (`MAJOR.MINOR.PATCH`) whenever features,
  fixes or breaking changes land; keep version markers in sync.
- Commit only after all tests and DriftGuard pass on every supported platform;
  add tests alongside new functionality or behavioural changes and refresh
  dependencies when packages change.
- Treat `start.command`, `start.bat` and `start.sh` equally. When one launcher
  is modified, review and update the others as needed, considering compliance
  and security requirements and how changes affect the start scripts. Always
  run through the managed virtual environment seeded by these launchers.
- Audit licenses for new dependencies and update `THIRD_PARTY_LICENSES.md` and
  `licenses/` when packages change. Refresh dependency locks whenever manifests
  move.
- Validate every timestamp before recording it. Confirm the real current date
  and keep changelog chronology consistent without gaps or future-dated
  entries.
- Preserve human-authored edits wherever possible. Respect existing structure,
  wording and metadata unless correcting objective errors called out by a
  human contributor.

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
