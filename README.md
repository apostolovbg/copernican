# copernican
**Doc ID:** README
**Doc Type:** repo-readme
**Project Version:** 12.0.26
**Last Updated:** 2026-07-09
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->

<!-- DEVCOV:END -->

**Version:** 12.0.26

![Copernican banner](https://raw.githubusercontent.com/apostolovbg/copernican/main/copernican/docs/banner_github.png)

## Overview
Copernican is a Python toolkit for evaluating cosmological models against
SNe Ia, BAO, and CMB observations. It gives researchers one manifest-driven
workflow for selecting data, choosing a model, running the sampler, and
keeping the results tied to the exact inputs that produced them.

The same manifest can drive the command-line interface or the GUI. That keeps
interactive runs and scripted runs on the same configuration surface, with the
same seed handling, dataset selection, engine choice, and output layout.

Copernican is built for reproducibility. Every run writes a manifest, logs,
summary artifacts, plots, and chain outputs into a per-run directory under
`~/copernican_output/`, so a result can be replayed or audited later without
guessing which options were used.

The package includes the model library, trusted dataset parsers, sampler
engines, validation manifests, and supporting analysis tools needed for the
full workflow. It also supports both standard backend CMB contracts and the
native declared-graph route for custom theories, so the same application can
handle conventional and extended cosmology models.

The native declared-graph route now materializes scalar hierarchies,
exposes `PP / phiphi`, `TP / Tphi`, and `EP / Ephi`, and uses exact
curved-sky lensing remapping for lensed spectra.
The native solver in `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
keeps the exact remapper, collision terms, and source-grid refinement
visible in regression coverage instead of hiding them behind solver-side
shortcuts.
The canonical native CMB sign, gauge, source, projection, and spectrum
conventions now live in `copernican/docs/cmb_solver.md`, so later solver
slices change equations against one explicit physical contract rather than
redefining state meaning in code.
The exact remapper now consumes the declared `PP` spectrum directly,
without a solver-side lensing bridge or hidden remap scale.
Its scalar contract now keeps the hierarchy and source terms aligned with
the physical runtime graph instead of acceptance-only damping scaffolding.
It also keeps the physical collision-rate Thomson coupling, the CAMB-style
low-multipole polarization source moment, exact photon and polarization
hierarchy sources, and q-resolved massive-neutrino q-bin moments
aligned for native runs. The scalar compiler also seeds
Newtonian and synchronous metric roles from leading-order physical
initial-condition series instead of heuristic constants. The q-grid path
now materializes direct per-bin density, momentum, and shear moments.
The native collision substep now resolves exact Thomson relaxation from
compiled collision metadata instead of hard-coded state names, and
`lensed_BB` keeps declared primordial B-mode sources visible.
The public `CMBLike` likelihood now also accepts stacked spectrum blocks
when the data frame carries a `spectrum` column, so TT/TE/EE/BB/PP/TP/EP
blocks can be flattened into one covariance surface. Row-order indexing
keeps stacked mixed-spectrum tables aligned with the requested theory
blocks. The temperature projection also carries a physical acoustic
phase, so changing the primordial tilt reshapes the TT spectrum instead
of only rescaling it.
The regression suite also proves vector and tensor sector
classification, plus a gauge-invariant native comparison, so the
remaining sector and gauge claims are explicit rather than implicit.
The native transfer and spectrum accumulation stay in extended
precision until the public solver converts the final values to float64.

Copernican ships as a managed Python application. The repository keeps the
bootstrap interpreter, virtual environment, and locked dependencies in view so
source checkouts and installed copies follow the same launch path.

## Launch Copernican

Start in the repository root. The commands below bootstrap the managed
environment, install the locked dependencies, and launch the CLI or GUI.

### Bootstrap the private interpreter

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

### Create the managed virtual environment

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

### Activate the environment

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

### Install the locked dependencies

This installs the exact package versions Copernican expects into the active
environment.

```
python -m pip install -r requirements.lock
```

### Run Copernican

Start the command-line interface.

```
python -m copernican --cli
```

Start the graphical interface.

```
python -m copernican --gui
```

The GUI opens directly in the active `.venv` on every supported platform.

If Copernican is installed in the same `.venv`, use these commands instead.

```
copernican --cli
```

```
copernican --gui
```

See [docs/packaging.md](docs/packaging.md#launch-copernican) for the
packaging notes that sit alongside these commands.

Each run keeps its own run logs inside the generated
`~/copernican_output/copernican-run_*` folder.

## Repository Layout
- `copernican/lib/` contains shared runtime helpers, GUI scaffolding,
  analysis tools, plotting helpers, and the native CMB internals.
- `copernican/models/` houses the YAML model definitions and their metadata.
- `copernican/engines/` collects the sampler back ends.
- `copernican/datasets/` bundles the trusted datasets and parser metadata.
- `copernican/validation/` holds the validation runner and reference
  manifests.
- `docs/` contains the long-form manual set.
- `ABOUT.md`, `AGENTS.md`, `CHANGELOG.md`, `CITATION.cff`, `PLAN.md`,
  `SECURITY.md`, and `SUPPORT.md` describe the front-door package contract.

## Run Builder and GUI
The GUI keeps the same manifest model as the CLI. The Run Builder walks
through seed, model, dataset, engine, and plan panels; the Save Manifest page
stays locked until each step has a selection; and the Start Run action
renames the workspace to `copernican-run_<timestamp>` before launching the
worker. The Run Settings panel mirrors the CLI prompts for walkers, burn-in,
production steps, and pool size so GUI runs and CLI runs use the same run
metadata.

The Run Monitor streams stdout and stderr into a log box, tails the per-run
log file, and keeps the cancel controls disabled until a run exists. Metadata
dialogs open with the system default application and use the same launch
behavior as the rest of the GUI.

## Analysis Workspace
The Analysis tab provides Run Summary, Posteriors, and Comparisons tools.
Run Summary ingests a saved run folder and renders the manifest, parameter
summary, and log in a scrollable panel. Posteriors lists `posterior-*.nc`
snapshots and renders trace and histogram views in the shared plot viewer.
Comparisons loads two run folders and reports parameter shifts, dataset count
deltas, and χ² differences in a structured view.

## Validation
The Validation tab runs the reference manifest against the shipped datasets,
streams the CLI output into the GUI, and stores the resulting summary in
`~/VALIDATION.md` alongside the per-run output directory. The manifest keeps
the regression baseline deterministic so validation reports stay repeatable.

## Documentation and Policy
The package docs mirror the root docs so installed copies and repository
copies stay aligned. `docs/gui_guide.md` explains the GUI, `docs/cli_guide.md`
explains the CLI, `docs/run_manifest.md` covers manifest structure, and
`docs/packaging.md` covers setup and distribution tasks.

## Maintenance Helpers
Command-line users can work without the GUI:

- `python -m copernican --catalogue-summary`
- `python -m copernican --revalidate-dataset <dataset_id>`
- `python -m copernican --list-manifests`
- `python -m copernican --show-manifest <path>`
- `python -m copernican --run-validation`
- `python -m copernican --analysis-summary <run_dir>`
- `python -m copernican --analysis-compare <base_run> <alternative_run>`
- `python -m copernican --analysis-posterior <run_dir>`

## Repository Policy
Read `AGENTS.md` before making changes, keep the package docs mirrored, and
follow the gate workflow so edited files, manifests, and generated artifacts
stay in sync.
