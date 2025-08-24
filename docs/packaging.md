# Packaging Guide

This document explains how to prepare the suite for development or packaging.

## Install Python 3.11+

The `start.*` launchers verify Python 3.11+ before bootstrapping the suite.
If the interpreter is missing or outdated they display one of the following
commands and exit:

- **Debian/Ubuntu**: `sudo apt install python3.11 python3.11-venv`
- **macOS**: `brew install python@3.11`
- **Windows**: `winget install -e --id Python.Python.3.11`

Run the command for your platform, then re-run the launcher.

## Bootstrap the virtual environment

Run the launcher in the project root:

- `start.bat` on Windows
- `start.command` on macOS
- `start.sh` on Linux

The script verifies Python 3.11+, then creates or reuses `.venv`, upgrades
`pip`, installs packages from `requirements.lock` with hash verification and
installs the project itself with `pip install --no-deps .`. Re-run it after
pulling updates to refresh the environment.

`requirements.lock` pins exact versions and SHA256 hashes for all runtime
dependencies. Adding or updating a package requires editing this file and the
license summary in `THIRD_PARTY_LICENSES.md`.

## Build optional distributions

Inside `.venv` run:

```bash
pip build .
```

The command writes source archives and wheels to the `dist/` directory.

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
