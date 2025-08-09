# Contributing to the Copernican Suite

Thank you for considering a contribution. Before opening a pull request, please
read `AGENTS.md` for the full development specification. The quick checklist is:

1. Run `pre-commit run --files <changed files>` to apply Black, Isort, Ruff and Flake8 checks.
2. Run the test suite with `python -m unittest discover` or `copernican.py --run-tests`.
3. Document your changes in `CHANGELOG.md` using the `- YYYY-MM-DD: summary (author)` format.
4. Update documentation where needed, including `README.md` and `AGENTS.md`.
5. Ensure your code is well commented and follows the project's style.

Pull requests that do not meet these requirements may be rejected. Contributions must comply with the Copernican Suite License, which forbids redistributing the suite in full and prohibits patent claims.
