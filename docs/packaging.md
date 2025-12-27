# Packaging Guide
This document explains how to prepare the suite for development or packaging.

CAMB only publishes wheels for Python 3.11 today, so the suite intentionally
sticks to that interpreter until upstream catches up. Blocking newer Python
versions avoids forcing contributors to compile CAMB locally during bootstrap.

## Install Python 3.11

The `start.*` launchers always download a private Python 3.11 interpreter into
`.python`, ignoring any system interpreter. They now delete legacy downloads
that fall outside the Python 3.11 series and rebuild `.venv` automatically
whenever it points at an unsupported interpreter. They refuse to run when
another virtual environment is active so the repository's `.venv` is always
used. This guard also prunes stray Python 3.12 downloads before the environment
is recreated, keeping the managed toolchain inside the supported window. If the
download fails the scripts exit with guidance.

On Windows the launcher now constructs the download URL without command
continuations, exports it to PowerShell via environment variables and creates
the `.python` directory before extraction. The download and extraction steps
run through dedicated subroutines outside the `if not exist "%PYBIN%" (...)`
block, so `cmd.exe` no longer mis-parses PowerShell closures when the block
finishes. Release metadata such as the Python version, release identifier and
architecture tag are now computed before the conditional runs, guaranteeing
`%DOWNLOAD_URL%` expands to the expected GitHub asset without enabling delayed
expansion. Those guard rails prevent empty-URL failures when PowerShell parses
the `Invoke-WebRequest` call and ensure the bootstrapper refuses to continue if
the download URL ever collapses to an empty string while still restoring the
interactive menu.

## Bootstrap the virtual environment

Run the launcher in the project root. It recreates or upgrades `.venv` on every
start and ignores globally installed packages:

- `start.bat` on Windows
- `start.command` on macOS
- `start.sh` on Linux

The script creates or reuses `.venv` from the bundled interpreter, upgrades
`pip` to the latest stable release, installs packages from `requirements.lock`
and installs the project itself with `pip install --no-deps .`. Because macOS
still ships the legacy `setuptools 79.0.1` wheel through `ensurepip`,
`pyproject.toml` pins package discovery to the `copernican_lib`, `engines`,
`models` and `models.*` namespaces. The explicit include list ensures
reinstalling the suite never hits the "Multiple top-level packages discovered"
guard while still packaging any nested plugin modules. ArviZ now ships as the
widely available `0.16.1` release alongside `numpy==1.26.4`, `scipy==1.12.0`,
`matplotlib==3.8.2` and `pandas==2.2.1`, ensuring every platform pulls
published wheels instead of attempting source builds. Re-run the launcher after
pulling updates to refresh the environment.

Option 6 in the start menu now toggles between installing and uninstalling the
`copernican-suite` wheel inside `.venv`. The helper runs `python -m pip show
copernican-suite` before showing the menu so the entry always matches the
package state, giving you one clear path to install the release wheel or remove
it before testing fresh code.

`requirements.lock` pins exact versions for all runtime dependencies. Adding or
updating a package requires editing this file and the license summary in
`THIRD_PARTY_LICENSES.md`. This release refreshed nearly every pin to match
published wheels, so remember to keep the license table in sync when adjusting
future dependencies.

Development helpers such as `pre-commit` are installed without the `--no-deps`
flag so their own dependencies are pulled in automatically. This keeps the
environment consistent across platforms without manually tracking every
transient requirement.

## Build optional distributions

Inside `.venv` run:

```bash
python -m build
```

The command writes source archives and wheels to the `dist/` directory.

### Keep the tracked version in sync

Update both the README heading and `copernican_lib/VERSION` before building a
release candidate. The runtime version helper reads the tracked file, so
packaged wheels display the intended identifier even before a Git tag is cut.
Keeping the two locations aligned prevents confusion between development
snapshots and tagged releases.

### Regenerating dependency locks

Run `make lock` whenever `requirements.in` changes. The helper installs `pip-
tools==7.4.1` on demand, strips the interpreter banner from the 3.11 runs
produce byte-for-byte identical lockfiles in CI.  Developers can either rely on
the pre-commit hook to provision the tool automatically or install the optional
`dev` extra (`pip install .[dev]`) when preparing packaging updates locally.

### Custom version strings

Set the ``COPERNICAN_VERSION`` environment variable before building to override
the version derived from Git. Export the same value stored in
`copernican_lib/VERSION` so the package metadata, runtime banner and
documentation all agree. This is useful for CI jobs on feature branches. For
example, a build off ``work`` might use:

```bash
export COPERNICAN_VERSION="1.2.1-alpha+work.$(git rev-parse --short HEAD)"
python -m build
```

## Verify the build

After installation or building a distribution, run both test suites to confirm
everything operates correctly. The `tests/test_engine_mcmc.py` module now
exercises the sampler's reseeding helper so automated builds catch any
regression that might reintroduce ``nan`` walkers. Run:

```bash
python -m pytest -q
python -m unittest discover -v
```

The tests exercise the reference ΛCDM model and basic data parsers and should
complete within a few seconds.

## Troubleshooting

- **No module named pip**: run `python -m ensurepip --upgrade` and relaunch
  launcher.
- **Packages not updating**: run `pip install -U pip` followed by `pip install
  -r requirements.lock`.
- **Permission denied**: avoid `sudo pip`; use a writable directory or the
  provided `.venv`.
- **Virtual environment missing**: ensure a launcher was used or activate with
  `source .venv/bin/activate` (`.venv\\Scripts\\activate` on Windows).
