# copernican
**Doc ID:** README
**Doc Type:** repo-readme
**Project Version:** 12.0.26
**Last Updated:** 2026-07-28
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
interactive runs and scripted runs on one configuration surface, with the same
seed handling, control-model and test-model selection, dataset selection,
engine choice, and output layout.

Copernican is built for reproducibility. Every run writes a manifest, logs,
summary artifacts, plots, and chain outputs into a per-run directory under
`~/copernican_output/`, so a result can be replayed or audited later without
guessing which options were used.

The package includes the model library, trusted dataset parsers, sampler
engines, validation manifests, and supporting analysis tools needed for the
full workflow. Every bundled CMB model declares the native graph contract;
models with available CMB output use the same native solver, while a model
without a defensible perturbation closure reports CMB output as unavailable.

The CMB subsystem's physical state convention, hierarchy equations, collision
operators, gauge routes, line-of-sight sources, spectrum units, lensing
inputs, numerical controls, and independent-reference boundaries are
documented in
[`copernican/docs/cmb_solver.md`](copernican/docs/cmb_solver.md).

All bundled CMB model manifests, including
[`model_lcdm.yml`](copernican/models/model_lcdm.yml), declare
`standard: false` and execute through the native graph route without using an
external Boltzmann backend when CMB output is available. The explicit
[`model_lcdm_ccmbs.yml`](copernican/models/model_lcdm_ccmbs.yml) artifact
documents the native LambdaCDM contract. Each model's species and source
closures remain theory-specific; CAMB and CLASS are independent scientific
reference tools used by tests, not production spectrum engines.

Native projection requests use bounded radial-kernel caches and ell-batched
work arrays, keeping memory use tied to the declared numerical envelope while
preserving the model's requested multipole range. Compatible Fourier-mode
grids share radial recurrence work before projection.

Generated scalar gauges use one compiled declared equation graph and shared
deterministic Runge-Kutta substeps, so gauge-equivalent contracts follow the
same numerical trajectory rather than diverging because of basis-specific
adaptive stepping.

Native CMB execution separates contract-static graph compilation,
cosmology-static background and collision tables, and request-specific
projection work. Every scalar mode uses the same compiled equation program;
bounded cache identities keep repeated cosmology proposals from rebuilding
contract structure. Projection kernels may batch only radial-kernel work,
never scalar physics equations.
Native numerical contracts may declare `evolution_phase_step` for the
declaration-driven evolution schedule. Tensor projections reserve their
spin-2 high-k quadrature tail rather than applying an empirical spectrum
correction.
The generated massless-neutrino hierarchy uses an explicit `F_2 / 2`
anisotropic-stress convention, including its metric source and initial data.

Native runtime acceptance also records phase timings and enforces the bounded
180-second full-spectrum and 60-second joint-MCMC budgets used by the managed
test surface.

Generated scalar contracts validate Einstein energy, momentum, and shear
residuals across the accepted evolution history and expose named anchor
diagnostics in the runtime envelope. State and residual units are checked
before projection.

Native accuracy tiers can opt into phase-aware transfer-node selection,
visibility-aware source-time refinement, and line-of-sight quadrature checks
through `adaptive_transfer`, `adaptive_source`, and `adaptive_projection`
accuracy controls. Each surface reports its measured refinement error and
fails clearly when the declared tolerance is not met within its node budget.

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
through seed, control model, test model, dataset, engine, and plan panels. The
control model defaults to `model_lcdm.yml`, while the test model is selected
independently. The Save Manifest page stays locked until each step has a
selection; and the Start Run action renames the workspace to
`copernican-run_<timestamp>` before launching the worker. The Run Settings
panel mirrors the CLI prompts for walkers, burn-in, production steps, and pool
size so GUI runs and CLI runs use the same run metadata.

The manifest stores one comparison request containing both model identities.
Compatibility checks cover declared observables, units, multipole grids, and
spectrum roles before execution. Summaries, CSV files, posterior artifacts,
plot footers, and residual labels use the resolved control/test pair rather
than assuming an LCDM control.
Posterior plotting reads that pair from the saved manifest, and direct
plotting calls provide the same comparison object.

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
