# Contributing to the Copernican Suite
**Last Updated:** 2025-11-26

Thank you for considering a contribution. DriftGuard defines the canonical
policy for this repository: `driftguard.yml` is authoritative, and the shared
"Development Laws (DriftGuard Policy)" section in `README.md` and `AGENTS.md`
summarises the human-readable rules. Review those sources before opening a
pull request.

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
