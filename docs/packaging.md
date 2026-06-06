# Packaging Guide
This document explains how to prepare Copernican for development or
packaging.

CAMB only publishes wheels for Python 3.11 today, so Copernican keeps
that interpreter until upstream catches up. Use this guide from the
folder that contains the Copernican files. Every command below assumes
that folder is the current working directory. The bootstrap below
downloads Python 3.11 into `.python`, then builds `.venv` from that
local interpreter. The system Python stays untouched.

## Table of Contents

- [Bootstrap the private interpreter](#bootstrap-the-private-interpreter)
- [Create the venv](#create-the-venv)
- [Activate the environment](#activate-the-environment)
- [Install the locked dependencies](#install-the-locked-dependencies)
- [Launch Copernican](#launch-copernican)
- [Build optional distributions](#build-optional-distributions)
  - [Keep the tracked version in sync](#keep-the-tracked-version-in-sync)
  - [Regenerating dependency locks](#regenerating-dependency-locks)
- [Verify the build](#verify-the-build)
- [Troubleshooting](#troubleshooting)

## Bootstrap the private interpreter

Open a terminal anywhere. Then `cd` into the folder that contains the
Copernican files before you start. The commands below assume that
current directory.

Open a terminal in the folder that contains the Copernican files. The
commands below assume that current directory.

This step downloads Python 3.11 into `.python`. The commands are
copy/paste-safe on each platform.

macOS and Linux:

Download the Python 3.11 build.

```
mkdir -p .python
arch="$(uname -m)"
case "$(uname -s)" in
    Darwin)
        plat="apple-darwin"
        ;;
    Linux)
        plat="unknown-linux-gnu"
        ;;
    *)
        echo "Unsupported platform." >&2
        exit 1
        ;;
esac
base="https://github.com/astral-sh/python-build-standalone/releases"
file="download/20251028/cpython-3.11.14+20251028-${arch}-${plat}"
file="${file}-install_only.tar.gz"
url="$base/$file"
curl -fL "$url" | tar -xz -C .python --strip-components=1
```

Windows PowerShell:

Download the Python 3.11 build.

```
New-Item -ItemType Directory -Force .python | Out-Null
$base = "https://github.com/astral-sh/python-build-standalone/releases"
$file = "download/20251028/cpython-3.11.14+20251028-amd64-pc-windows-msvc"
$file = "${file}-install_only.tar.gz"
$url = "$base/$file"
Invoke-WebRequest -Uri $url -OutFile python.tar.gz
tar -xzf python.tar.gz -C .python --strip-components=1
Remove-Item python.tar.gz
```

Windows cmd:

Download the Python 3.11 build.

```
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
    "$base = 'https://github.com/astral-sh/python-build-standalone/releases'; ^
     $file = 'download/20251028/'; ^
     $file = $file + 'cpython-3.11.14+20251028-'; ^
     $file = $file + 'amd64-pc-windows-msvc'; ^
     $file = $file + '-install_only.tar.gz'; ^
     $url = $base + '/' + $file; ^
     New-Item -ItemType Directory -Force .python | Out-Null; ^
     Invoke-WebRequest -Uri $url -OutFile python.tar.gz; ^
     tar -xzf python.tar.gz -C .python --strip-components=1; ^
     Remove-Item python.tar.gz"
```

## Create the venv

macOS and Linux:

```
./.python/bin/python3 -m venv .venv
```

Windows PowerShell:

```
.\.python\python.exe -m venv .venv
```

Windows cmd:

```
.\.python\python.exe -m venv .venv
```

## Activate the environment

macOS and Linux:

```
source .venv/bin/activate
```

Windows PowerShell:

```
.venv\Scripts\Activate.ps1
```

Windows cmd:

```
.venv\Scripts\activate.bat
```

## Install the locked dependencies

This puts the exact package versions Copernican expects into the
environment.

```
python -m pip install -r requirements.lock
```

## Launch Copernican

Use the same launch steps as the top README. The commands below match
the same launch flow on every supported platform.

Start the command-line interface. This runs Copernican in text mode.

```
python -m copernican --cli
```

Start the graphical interface. This opens the GUI window.

```
python -m copernican --gui
```

The GUI opens directly in the active `.venv` on every supported
platform.

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
