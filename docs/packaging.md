# Packaging Guide
This document explains how to prepare Copernican for development or packaging.

CAMB only publishes wheels for Python 3.11 today, so Copernican intentionally
sticks to that interpreter until upstream catches up. Blocking newer Python
versions avoids forcing contributors to compile CAMB locally during bootstrap.

## Table of Contents

- [Choose the Python launcher](#choose-the-python-launcher)
- [Bootstrap the virtual environment](#bootstrap-the-virtual-environment)
- [Launch Copernican](#launch-copernican)
- [Build optional distributions](#build-optional-distributions)
  - [Keep the tracked version in sync](#keep-the-tracked-version-in-sync)
  - [Regenerating dependency locks](#regenerating-dependency-locks)
- [Verify the build](#verify-the-build)
- [Troubleshooting](#troubleshooting)

## Choose the Python launcher

Copernican still targets Python 3.11. Use the launcher on your computer that
starts that interpreter. On many Unix-like systems that is `python` or
`python3`. On Windows it is usually `py -3`.

## Bootstrap the virtual environment

Create or refresh `.venv`. This makes a private environment for Copernican.
Use the Python 3 launcher that is already on your machine. If your
machine names that launcher differently, substitute the right name in
the commands below.

macOS and Linux:

Create the virtual environment. This makes a private `.venv` folder for
Copernican.

```
python3 -m venv .venv
```

Activate the environment. This tells your terminal to use the Python inside
`.venv`.

```
source .venv/bin/activate
```

Install the locked dependencies. This puts the exact package versions
Copernican expects into the environment.

```
python -m pip install -r requirements.lock
```

Windows PowerShell:

Create the virtual environment. This makes a private `.venv` folder for
Copernican.

```
py -3 -m venv .venv
```

Activate the environment. This tells PowerShell to use the Python inside
`.venv`.

```
.venv\Scripts\Activate.ps1
```

Install the locked dependencies. This puts the exact package versions
Copernican expects into the environment.

```
python -m pip install -r requirements.lock
```

Windows cmd:

Create the virtual environment. This makes a private `.venv` folder for
Copernican.

```
py -3 -m venv .venv
```

Activate the environment. This tells cmd to use the Python inside `.venv`.

```
.venv\Scripts\activate.bat
```

Install the locked dependencies. This puts the exact package versions
Copernican expects into the environment.

```
python -m pip install -r requirements.lock
```

## Launch Copernican

Use the same launch steps as the top README. The commands below are the same
on macOS, Linux and Windows.

Start the command-line interface. This runs Copernican in text mode.

```
python -m copernican --cli
```

Start the graphical interface. This opens the GUI window.

```
python -m copernican --gui
```

If Copernican is already installed in the same `.venv`, use these commands
instead.

```
copernican --cli
```

```
copernican --gui
```

Each run keeps its own run logs inside the generated
`~/copernican_output/copernican-run_*` folder.

`copernican/workflow.py` owns the launch flow for both CLI and GUI, and
`copernican/lib/global_settings/defaults.yml` supplies the GUI-facing
defaults that shape that flow through `copernican/lib/settings.py`.

The GitHub Actions governance job now boots the repo-local `.venv`
before it runs DevCovenant so CI matches the managed-environment
contract used in local work.

Union3 compressed SNe data require additive intercept marginalization in the
SNe likelihood, CSV export and plot residual paths so all residual views use
the same convention.

## Build optional distributions

Inside `.venv` run:

```
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

```
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
