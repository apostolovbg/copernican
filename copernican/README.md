# copernican
**Doc ID:** README
**Doc Type:** repo-readme
**Project Version:** 12.0.26
**Last Updated:** 2026-06-21
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->

<!-- DEVCOV:END -->

**Version:** 12.0.26

![Copernican banner](https://raw.githubusercontent.com/apostolovbg/copernican/main/copernican/docs/banner_github.png)

Copernican is a Python toolkit that helps researchers test cosmological
models against SNe Ia, BAO and CMB observations with a single
manifest-driven workflow. `python -m copernican` orchestrates everything
from model loading through sampler execution while the private `.python`
bootstrap interpreter and managed `.venv` keep the pinned Python 3.11
environment portable across macOS, Linux and Windows. Developers must
consult [AGENTS.md](AGENTS.md) and the DevCovenant policies before
making any edits because the repository enforces its laws through
pre-commit checks.

The CMB surface now includes a declared-math graph engine for
`standard: false` contracts. It evolves one declared graph per `k`
mode, applies algebraic constraints and closures inside that graph,
and projects the declared observables into transfer functions and
spectra with bounded `k` sampling and cached Bessel tables. The
background path consumes the declared background graph, computes a
Peebles-style recombination history, integrates the declared
reionization ODE, and builds the visibility and optical-depth grids
before the perturbation and line-of-sight steps run. Declared
background outputs now feed native density, pressure,
equation-of-state, and curvature quantities directly; the perturbation
runtime can mix `tau`, `eta`, `a`, `z`, or other declared monotonic
background coordinates on equation left-hand sides; and end-anchored
boundary conditions can drive the native shooter when they replace the
missing start-state slots.

## Launch Copernican

Open a terminal anywhere. Then `cd` into the folder that contains the
Copernican files. Every block below assumes that folder is the current
working directory.

The first block downloads a private Python 3.11 interpreter into
`.python`. The second block builds `.venv` from that local interpreter.
After that, activate `.venv`, install the locked dependencies, then run
the CLI and GUI.

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

This puts the exact package versions Copernican expects into the
environment.

```
python -m pip install -r requirements.lock
```

### Run Copernican

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

If Copernican is already installed in the same `.venv`, use these
commands instead.

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

`copernican/workflow.py` owns the launch flow for both CLI and GUI, and
`copernican/lib/global_settings/defaults.yml` supplies the GUI-facing
defaults that shape that flow through `copernican/lib/settings.py`.

The GitHub Actions governance job now boots the repo-local `.venv`
before it runs DevCovenant so CI matches the managed-environment
contract used in local work.

Union3 compressed SNe data require additive intercept marginalization in the
SNe likelihood, CSV export and plot residual paths so all residual views use
the same convention.

## Table of Contents

- [Highlights](#highlights)
- [Overview](#overview)
- [Directory layout](#directory-layout)
- [Run Builder & GUI](#run-builder-gui)
- [Analysis workspace](#analysis-workspace)
- [Validation](#validation)
- [Documentation & policy](#documentation-policy)
- [Maintenance helpers](#maintenance-helpers)
- [Law & policy compliance reminder](#law-policy-compliance-reminder)

## Highlights
- **Manifest-driven orchestration:** `python -m copernican` consumes model,
  data and engine selections, writes every run into
  `~/copernican_output/copernican-run_*`, and reuses
  `copernican/lib/run_pipeline.py` helpers so CLI and GUI paths stay
  consistent.
- **Modular library layout:** `copernican/lib/` hosts shared helpers
  (plotting, analysis, diagnostics, GUI scaffolding and dataset registries)
  while `copernican/models/`, `copernican/engines/`,
  `copernican/validation/`, `copernican/datasets/` and `~/copernican_output/`
  remain the canonical asset roots.
- **CMB capability checks:** `copernican/lib/model_coder.py` keeps the CMB
  backend capability flags close to the perturbation compiler so
  `standard: false` models run through the declared-math CMB graph engine in
  `copernican/lib/likelihoods/cmb/`, where `cmb.py` owns the public entry
  point, `camb_solver.py` owns the standard backend route, and
  `copcmb_solver.py` owns the native declared solver. `engine_adapter.py`
  now hands the precompiled native runtime into that package directly instead
  of rebuilding a CAMB-style contract in the native hot path. Non-standard
  contracts now declare one immutable graph of variables, equations,
  constraints, closures, sources, initial conditions, observable mappings,
  and numerical requirements. Unsupported symbols, unsolved variables,
  missing initial conditions, missing observables, incompatible
  projection-role bindings, and unsupported projection kernels fail fast.
  Run manifests also record the compiled graph summary, declared background
  and recombination provenance, and the selected
  production CMB execution route so native runs prove their CAMB-free
  prediction path from route metadata rather than from `standard: false`
  alone.
- **Run Builder & GUI:** a navigation rail keeps the Run Builder, Run Monitor,
  Analysis workspace and validation tools at your fingertips while metadata
  dialogs, builder panels, and the package entrypoint preserve the same
  launch flow.
- **Analysis workspace:** Run Summary, Posteriors, Diagnostics and Comparisons
  tabs rely on `copernican.lib.analysis`, `posterior_explorer`, and the shared
  `PlotViewer` so summaries, comparisons and posterior plots stay in sync with
  CLI helpers.
- **Validation & documentation:** Fixed Planck 2018 manifests drive the
  validation suite, while `docs/` guides, `README.md`, `CHANGELOG.md` and the
  AGENTS/DevCovenant policies capture every procedural rule. Reference-backed
  recombination, visibility, and spectrum checks stay in the normal governed
  test suite, while `copernican/validation/` remains the separate
  publication-style LCDM reference runner that uses the same manifest
  infrastructure.
- **Maintenance helpers:** CLI commands such as run summaries, comparisons,
  validations and dataset revalidation keep operators productive without the
  GUI.
- **Law & policy reminder:** Always obey the DevCovenant laws in `AGENTS.md`
  and run `pre-commit run --all-files` before finishing work.

## Overview
 - `python -m copernican` acts as a manifest-first orchestrator: it consumes a
   manifest describing the selected models, datasets, sampler settings and
   environment hints, reuses `copernican/lib/run_pipeline.py` helpers and
   writes every run log inside
   `~/copernican_output/copernican-run_*` with a matching
   `run_manifest_<timestamp>.yml` for reproducibility. Command-line flags
   such as `--manifest`, `--output-dir`, `--gui`, `--cli` and `--no-gui`
   let CI and operators pick the desired entry point while the managed
   interpreter and dependencies come from `.venv`. The launch flow itself
   lives in `copernican/workflow.py`, and
   `copernican/lib/global_settings/defaults.yml` carries the GUI-side
   defaults that shape the same workflow through `copernican/lib/settings.py`.
 - `copernican/lib/` contains shared utilities (analysis helpers, likelihoods,
   diagnostics, GUI scaffolding, plotting helpers, dataset registries, etc.) so
   engines stay lightweight and consistent across backends.
- `copernican/engines/` collects sampler back ends. The default
  `copernican.engines.engine_mcmc`
  couples `emcee` with ArviZ when available; the nested sampler mirrors the
  same schema while exposing evidences. Both reuse the shared progress
  renderer and manifest helpers. The MCMC initializer uses a tolerance
  cutoff for tiny singular values to keep walker startup stable across
  platforms.
- `copernican/models/` houses YAML model definitions with priors,
  transforms and dataset compatibility metadata. Each definition is
  converted into a picklable engine adapter so manifest generation stays
  deterministic even under multiprocessing. CMB-valid models declare a
  backend contract under `cmb` with `backend`, `param_map`, `grids`,
  `values`, `calls` and a mandatory `perturbations` block; `standard:
  true` keeps the perturbation sections empty, while `standard: false`
  requires a supported declared background graph, declared perturbation
  graph, line-of-sight transfer functions and explicit backend mapping for
  the compiled declarative CMB engine. Non-standard models declare their
  own equations, closures and observable projections directly, while the
  available background and perturbation symbols stay limited to the
  documented CMB engine context so unsupported names fail loudly. Transfer
  components keep source-term roles separate from reviewed projection
  kernels, and `custom_line_of_sight` can project declared source sums
  through explicit kernels without hiding unsupported BB or lensing inputs.
  Saved manifests carry the graph summary, projection contracts, background
  and recombination provenance, and the selected execution route so audits can
  distinguish backend-standard CAMB prediction from native declared-graph
  prediction.
- `copernican/datasets/` bundles vetted observations and parsers. The
  loaders validate SHA256 digests, register citations, and tag each manifest
  with the hashes used for the run; the directory remains read-only except
  when a human explicitly edits the datasets.
- `copernican/validation/` holds the manifest runner and manifests used by
  the validation suite. The latest summary lives in `~/VALIDATION.md` so
  package installs do not write validation state into the package tree.
- `copernican/lib/gui/` provides a Tkinter scaffold with the navigation rail,
  Run Builder, Run Monitor, Analysis workspace, validation helpers and a Help
  page that renders Markdown assets inline; the package entrypoint launches
  the GUI inside the managed environment after logging the environment.

## Directory layout
- `copernican/models/`: YAML definitions for every supported cosmological
  model.
- `copernican/engines/`: Computational backends; the default MCMC engine
  records diagnostics and writes NetCDF chains.
- `copernican/datasets/`: Trusted datasets grouped by type (`sne`, `bao`,
  `cmb`) with parser metadata. Parsers compute SHA256 values and register
  their digests via `dataset_registry`.
- `~/copernican_output/`: Per-run folders such as
  `copernican-run_<timestamp>/` that store manifests, parameter summaries,
  logs and NetCDF chains.
- `copernican/validation/`: Reference manifests, runner helpers and summaries
  used by the validation suite.
- `docs/`: Guides covering architecture, GUI/CLI workflows, manifest
  structure, datasets and the DevCovenant policies.
- `ABOUT.md`, `AGENTS.md`, `CHANGELOG.md`, `CITATION.cff`,
  `SECURITY.md`, `SUPPORT.md`, `licenses/THIRD_PARTY_LICENSES.md`:
  Governance, release history, citation, support and security metadata.

## Run Builder & GUI
The navigation rail keeps quick actions and an always-visible logo square, so
launching the Run Builder or monitor never steals focus. Run Builder mirrors
the CLI stages: seed, model, data, engine and plan panels require one selection
per panel, the Save Manifest page remains locked until every stage reports a
selection, and saved manifests live under
`~/copernican_output/copernican_run_NEW_CONFIG/` so Confirm only becomes
available when a manifest exists. Start Run renames that workspace to
`copernican-run_<timestamp>` before spawning the CLI worker, while cancel and
clear remove temporary folders. The Run Settings panel mirrors the CLI
prompts (walkers, burn-in, production, pool size) so GUI runs and CLI runs
share the same configuration metadata. Quick actions keep the dataset catalog
health overview, import manifest flow and output directory helpers within
reach. The Models step also includes a `Load model...` action that opens a
file picker for any valid `.yml` or `.yaml` model path, matching the CLI's
exact-path loading rule.
Folder-open actions use the operating system's native handlers so the GUI can
open output locations without changing the launch behavior that operators
already expect.

The Run Monitor threads CLI stdout/stderr into a log box that tails the
per-run `copernican-run_<timestamp>.txt` file inside
`~/copernican_output/copernican-run_<timestamp>/`, mirrors the
counter-based progress updates from the sampler and keeps the Cancel/
Hard Stop buttons disabled until a run starts. A “Lock log to latest
entry” checkbox pins the view so operators can watch batches finish
without scrolling. Metadata dialogs size themselves to the longest line,
add an “Open file…” action that launches the OS editor and keep
horizontal resizing locked while allowing unlimited vertical growth.

## Analysis workspace
The Analysis tab now hosts Run Summary, Posteriors and Comparisons alongside a
placeholder Diagnostics panel. Run Summary ingests
`~/copernican_output/copernican-run_*` folders and loads the manifest,
parameter summary and log to render dataset counts, R-hat/ESS diagnostics,
per-model χ² components, BAO `r_s` values and timestamps inside a scrollable
panel. Its action buttons reload the summary, export structured
`analysis-summary_<timestamp>.yml`/`.json` files via
`copernican.lib.analysis.save_run_summary` and copy the JSON payload onto the
clipboard.

Posteriors uses `copernican.lib.posterior_explorer` to list `posterior-*.nc`
snapshots and renders a trace/hist overview inside the shared
`copernican.lib.gui.plot_viewer.PlotViewer`. The tab keeps controls for fitting
to screen, restoring the original limits, and toggling the drag-enabled pan so
you can inspect any region without re-creating the plot. The Comparisons tab
lets you point at two run directories, refresh Δχ²/parameter shifts and
dataset count deltas, and export or copy the structured comparison summary
that the new `copernican.lib.analysis.compare_runs` helper produces.

Every run now also writes ArviZ-powered corner plots and parameter
histograms into the `~/copernican_output/copernican-run_*` folders so the
Analysis workspace can render them inside the PlotViewer without re-running
the sampler. Use `python -m copernican --analysis-posterior \
~/copernican_output/copernican-run_*` to rerun
`copernican.lib.analysis.plot_posterior`, producing the overview, corner and
histogram assets from each `posterior-*.nc` snapshot on demand.

## Validation
The Validation tab runs `python -m copernican --run-validation`, streams CLI
output into a log box, and stores the summaries inside
`copernican/validation/output/<manifest_stem>/validation_run_<timestamp>/`
plus `~/VALIDATION.md`. The fixed Planck 2018 manifest evaluates a reference
model against Union3 UNITY SNe, BOSS DR12 BAO and Planck 2018 Lite CMB data
with constant priors, so regression checks stay deterministic. “Cancel
validation” terminates the worker, “Clear validation” removes the outputs
and summary, and GUI progress bars mirror the counter-based batches the
sampler emits. The Validation button stays disabled while a worker runs so
overlapping validation jobs cannot start.

## Documentation & policy
Release notes live in `CHANGELOG.md`, licensing details appear in
`licenses/THIRD_PARTY_LICENSES.md`, and the package-root doc set now also
includes `ABOUT.md`, `SECURITY.md`, `SUPPORT.md` and `CITATION.cff`. The GUI
Help panel renders `README.md` (banner included) plus the CLI/GUI guides from
`docs/gui_guide.md` and `docs/cli_guide.md`. The Analysis workspace and
package entrypoint wiring are covered by `docs/gui_overview.md` and
`docs/gui_guide.md`, while dataset and manifest hygiene appear across
`docs/data_overview.md`, `docs/run_manifest.md` and the DevCovenant policies.

The root package docs are the authored copies. `package-doc-sync` mirrors
them into `copernican/` so the GUI, package metadata and support surfaces can
open the package-root files directly without duplicating the content model.

Law 11 of `AGENTS.md` codifies the documentation expansion commitment: every
code change should grow the written record, and DevCovenant tooling
automatically verifies policy sync before accepting commits.

## Maintenance helpers
Command-line operators who skip the GUI can still run maintenance helpers:

- `python -m copernican --catalogue-summary` prints the dataset/model/engine
  inventory health summary.
- `python -m copernican --revalidate-dataset <dataset_id>` reruns the parser
  trust check for a specific dataset and reports the digest result.
- `python -m copernican --list-manifests` lists every timestamped run folder
  and `--show-manifest <path>` pretty-prints a saved manifest.
- `python -m copernican --run-validation` executes the lightweight validation
  suite, prints the reference summary, stores it in `~/VALIDATION.md`, and
  exits without opening the GUI.
- `python -m copernican --analysis-summary <run_dir>` loads the
  manifest/log/parameter summary from the specified run, prints the
  diagnostics table and (with `--analysis-summary-output <dir>`) writes
  structured `analysis-summary_<timestamp>.yml/.json` exports just like the
  GUI’s Run Summary tab.
- `python -m copernican --analysis-compare <base_run> <alternative_run>`
  aligns two run directories, prints the Δχ²/parameter summary and (with
  `--analysis-compare-output <dir>`) saves `analysis-comparison_<timestamp>`
  JSON/YAML files matching the Analysis Comparisons tab.
- `python -m copernican --analysis-posterior <run_dir> \
  --analysis-posterior-output <file.png>` builds a trace/hist overview using
  `copernican.lib.posterior_explorer` and writes the figure to the requested
  path so you can reproduce the PlotViewer output without the GUI.

## Law & policy compliance reminder
Before beginning work, read every law in `AGENTS.md`, obey the DevCovenant
policies in `devcovenant`, and finish every change with `pre-commit run --all-
files`.
