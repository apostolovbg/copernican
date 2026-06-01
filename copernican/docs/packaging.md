# Packaging Guide
This document explains how to prepare Copernican for development or packaging.

CAMB only publishes wheels for Python 3.11 today, so Copernican intentionally
sticks to that interpreter until upstream catches up. Blocking newer Python
versions avoids forcing contributors to compile CAMB locally during bootstrap.

## Install Python 3.11

Run Copernican from the managed `.venv` with `python -m copernican` or the
`copernican` console script. The repository keeps Python 3.11 as the supported
runtime because the pinned dependency set is validated against that
interpreter across the supported platforms. If the environment drifts,
recreate `.venv` and reinstall the locked requirements before continuing.

## Bootstrap the virtual environment

Create or refresh `.venv` with `python -m venv .venv`, activate it, then run
`python -m pip install -r requirements.lock` and `python -m pip install .`
inside that environment. The package discovery list in `pyproject.toml`
includes the runtime package `copernican`, the helper package `copernican.lib`,
the engine package `engines`, and the model catalogue `models`, which keeps
reinstalls aligned with the source tree while still packaging the nested
modules. ArviZ now ships as the widely available `0.16.1` release alongside
`numpy==1.26.4`, `scipy==1.12.0`, `matplotlib==3.8.2` and `pandas==2.2.1`,
ensuring every platform pulls published wheels instead of attempting source
builds. Refresh the environment after pulling updates.

`requirements.lock` pins exact versions for all runtime dependencies. Adding
or updating a package requires editing this file and the license asset summary
in `licenses/THIRD_PARTY_LICENSES.md`. The `[tool.setuptools.data-files]`
section installs those license assets, while
`[tool.setuptools.package-data]` ships `copernican/VERSION` and the bundled
datasets inside the package. Keep the license table in sync when adjusting
future dependencies.

Development helpers such as `pre-commit` are installed without the
`--no-deps` flag so their own dependencies are pulled in automatically. This
keeps the environment consistent across platforms without manually tracking
every transient requirement.

## Build optional distributions

Inside `.venv` run:

```bash
python -m build
```

The command writes source archives and wheels to the `dist/` directory.

### Keep the tracked version in sync

Update both the README heading and `copernican/VERSION` before building a
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
