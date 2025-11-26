# Contributing to the Copernican Suite
**Last Updated:** 2025-11-26

**DriftGuard is the supreme development authority.** All contributors must obey
the rules codified in `driftguard/repo_policy.yml` and summarised in
[DRIFTGUARD.md](DRIFTGUARD.md). Failure to follow DriftGuard guidance will
compromise the Copernican Suite and will lead to rejected commits. **Before any
commit, run `python -m pytest -q`, `python -m unittest discover -v`, and
`driftguard check --scope=staged --mode=full` on staged changes every session;
committing without a fresh full DriftGuard pass is forbidden.** Resolve any
reported issues before committing and prefer `driftguard fix` when the
automated fix is safe. Keep Python sources Black-clean; DriftGuard checks
tracked policy surfaces even when Git is clean and fails if Black would
reformat any Python file to avoid CI churn.

Thank you for considering a contribution. DriftGuard defines the canonical
policy for this repository: follow the chain `DRIFTGUARD.md` (human summary) →
`driftguard/repo_policy.yml` (machine-readable spec) → the DriftGuard code
under `driftguard/`. This replaces the legacy guidance that was embedded in
`README.md` and `AGENTS.md`. Review those sources before opening a pull
request.

Quick checklist:

1. Run `python tools/dev_suite.py` before committing. It applies Black,
   isort, Ruff, Flake8 and executes DriftGuard's fast fix/check cycle alongside
   targeted tests.
2. Run `driftguard check --scope=repo --mode=full` when preparing a pull
   request to mirror CI and confirm policy compliance and drift metrics.
3. Document non-trivial changes in the latest `CHANGELOG.md` section with an
   ISO date, author and the files or subsystems touched; keep version markers
   in `copernican_lib/VERSION`, `README.md` and `CITATION.cff` aligned.
4. Add or refresh tests for new modules and bugfixes, and update relevant
   documentation. Keep `Last Updated` headers only on the doc and metadata
   surfaces listed in the DriftGuard policy and avoid future-dated entries.
5. Use the managed launchers (`start.sh`, `start.command`, `start.bat`) to work
   inside the project virtual environment.

Pull requests that do not meet these requirements may be rejected.
Contributions must comply with the Copernican Suite License, which forbids
redistributing the suite in full and prohibits patent claims.
