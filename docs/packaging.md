# Packaging Guide
**Last Updated:** 2025-10-22
This document explains how to prepare the suite for development or packaging.

## Install Python 3.12+

The `start.*` launchers always download a private Python 3.12+ into
`.python`, ignoring any system interpreter. They refuse to run when
another virtual environment is active so the repository's `.venv` is
always used. If the download fails the scripts exit with guidance.

On Windows the launcher now constructs the download URL without command
continuations, exports it to PowerShell via environment variables and
creates the `.python` directory before extraction. The download and
extraction steps run through dedicated subroutines outside the
`if not exist "%PYBIN%" (...)` block, so `cmd.exe` no longer mis-parses
PowerShell closures when the block finishes. Release metadata such as the
Python version, release identifier and architecture tag are now computed
before the conditional runs, guaranteeing `%DOWNLOAD_URL%` expands to the
expected GitHub asset without enabling delayed expansion. Those guard rails
prevent empty-URL failures when PowerShell parses the `Invoke-WebRequest`
call and ensure the bootstrapper refuses to continue if the download URL ever
collapses to an empty string while still restoring the interactive menu.

## Bootstrap the virtual environment

Run the launcher in the project root. It recreates or upgrades `.venv` on every
start and ignores globally installed packages:

- `start.bat` on Windows
- `start.command` on macOS
- `start.sh` on Linux

The script creates or reuses `.venv` from the bundled interpreter, upgrades
`pip`, installs packages from `requirements.lock` and
installs the project itself with `pip install --no-deps .`. ArviZ is installed
separately to work around its stale NumPy requirement. Re-run the launcher
after pulling updates to refresh the environment.

`requirements.lock` pins exact versions for all runtime
dependencies. Adding or updating a package requires editing this file and the
license summary in `THIRD_PARTY_LICENSES.md`.

Development helpers such as `pre-commit` are installed without the
`--no-deps` flag so their own dependencies are pulled in automatically.
This keeps the environment consistent across platforms without manually
tracking every transient requirement.

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
  `pip install -r requirements.lock`.
- **Permission denied**: avoid `sudo pip`; use a writable directory or the
  provided `.venv`.
- **Virtual environment missing**: ensure a launcher was used or activate with
  `source .venv/bin/activate` (`.venv\\Scripts\\activate` on Windows).
