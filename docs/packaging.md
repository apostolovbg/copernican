# Packaging Guide
**Last Updated:** 2025-08-30

This document explains how to prepare the suite for development or packaging.

## Install Python 3.12+

The `start.*` launchers always download a private Python 3.12+ into
`.python`, ignoring any system interpreter. They refuse to run when another
virtual environment is active so the repository's `.venv` is always used.
If the download fails the scripts exit with guidance.

## Bootstrap the virtual environment

Run the launcher in the project root. It recreates or upgrades `.venv` on every
start and ignores globally installed packages:

- `start.bat` on Windows
- `start.command` on macOS
- `start.sh` on Linux

The script creates or reuses `.venv` from the bundled interpreter, upgrades
`pip`, installs packages from `requirements.lock` with hash verification and
installs the project itself with `pip install --no-deps .`. ArviZ is installed
separately to work around its stale NumPy requirement. Re-run the launcher
after pulling updates to refresh the environment.

`requirements.lock` pins exact versions and SHA256 hashes for all runtime
dependencies. Adding or updating a package requires editing this file and the
license summary in `THIRD_PARTY_LICENSES.md`.

## Build optional distributions

Inside `.venv` run:

```bash
pip build .
```

The command writes source archives and wheels to the `dist/` directory.

### Custom version strings

Set the ``COPERNICAN_VERSION`` environment variable before building to
override the version derived from Git. This is useful for CI jobs on
feature branches. For example, a build off ``work`` might use:

```bash
export COPERNICAN_VERSION="1.2.1-alpha+work.$(git rev-parse --short HEAD)"
pip build .
```

## Verify the build

After installation or building a distribution, run the test suite to
confirm everything operates correctly:

```bash
python -m unittest discover -v
```

The tests exercise the reference ΛCDM model and basic data parsers and
should complete within a few seconds.

## Troubleshooting

- **No module named pip**: run `python -m ensurepip --upgrade` and relaunch
  launcher.
- **Packages not updating**: run `pip install -U pip` followed by
  `pip install --require-hashes -r requirements.lock`.
- **Permission denied**: avoid `sudo pip`; use a writable directory or the
  provided `.venv`.
- **Virtual environment missing**: ensure a launcher was used or activate with
  `source .venv/bin/activate` (`.venv\\Scripts\\activate` on Windows).
