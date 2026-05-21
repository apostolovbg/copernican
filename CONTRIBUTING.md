# Contributing
**Doc ID:** CONTRIBUTING
**Doc Type:** contributing-guide
**Project Version:** 12.0.1
**Last Updated:** 2026-05-21
**DevCovenant Version:** 1.0.1b5

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
It defines the standard contributor workflow used by repositories that use
DevCovenant. Add repository-specific contributor notes below the managed
section.
<!-- DEVCOV:END -->

Thank you for considering a contribution. Before opening a pull request, please
read `AGENTS.md` for the full development specification. Log every change in
`CHANGELOG.md` with the date, author and the files or subsystems you touched,
and compare `git diff --name-only` with your entry before pushing so the
`copernican-policy` hook remains satisfied. The quick checklist is:

1. Run `pre-commit run --all-files` to apply Black, Isort, Ruff, Flake8 and the
   Copernican policy hook that validates version metadata, date fields and
   enforces print-free library modules.
2. Run the test suite with `python -m unittest discover` or via the launchers'
   *Run the unit test suite* option.
3. Document your changes in `CHANGELOG.md` using the `- YYYY-MM-DD: summary
   (author)` format.
4. Update documentation where needed, including `README.md` and `AGENTS.md`.
5. Ensure your code is well commented and follows the project's style.

Pull requests that do not meet these requirements may be rejected.
Contributions must comply with the Copernican Suite License, which forbids
redistributing the suite in full and prohibits patent claims.

## Compliance as Workflow
Treat the AGENTS laws and DevCovenant policies as the workflow itself: read
them before working, obey them while coding, and re-run the mandated commands
before every commit (`pre-commit run --all-files`, `python3
devcovenant_check.py check --mode=startup` and dependency lock updates when
requirements change). Log the action in `CHANGELOG.md` including the law
number/policy ID to prove compliance.
