# copernican
**Doc ID:** README
**Doc Type:** repo-readme
**Project Version:** 12.0.26
**Last Updated:** 2026-08-16
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
sampler-engine choice, and output layout.

Copernican is built for reproducibility. Every run writes a manifest, one
canonical run log, summary artifacts, plots, and chain outputs into a per-run
directory under `~/copernican_output/`, so a result can be replayed or audited
later without guessing which options were used.

The package includes the model library, trusted dataset parsers, sampler
engines, validation manifests, and supporting analysis tools needed for the
full workflow. Every bundled CMB model declares the native graph contract;
models with available CMB output use the same native solver, while a model
without a defensible perturbation closure reports CMB output as unavailable.
USMF2 now executes through that same native route. Its theory-specific
shrink-field closure, sourced initial conditions, metric constraints,
projection sources, and public observables are evaluated directly from the
declared graph without a CAMB, CLASS, or LCDM substitution.

The CMB subsystem's physical state convention, hierarchy equations, collision
operators, gauge routes, line-of-sight sources, spectrum units, lensing
inputs, numerical controls, and independent-reference boundaries are
documented in
[`copernican/docs/cmb_solver.md`](copernican/docs/cmb_solver.md).

The compiled contract audit in `copernican.lib.cmb_contract` produces a
machine-testable model-by-capability matrix for `TT`, `TE`, `EE`, `BB`, `PP`,
`TP`, and `EP`. It derives species, sectors, hierarchy families, projection
roles, and early unsupported-combination diagnostics from declarations rather
than theory names or model filenames.

Native spectrum requests run that capability preflight before background
construction, so unsupported observables fail without beginning evolution.

Before native scalar evolution begins, Copernican audits the requested k grid
against the model's declared numerical limits and preflights every mode on the
coupled Einstein constraint surface. The runtime records the ordered mode set,
residual terms, normalization scales, and constraint provenance with the
spectrum data so scientific failures can be traced without silently skipping
high-k modes.

Evolved scalar Einstein residuals use the sum of absolute declared equation
terms as their dimensionless normalization. Copernican records the maximum
location and physical regime, tolerance and normalization provenance, and
coarse-to-intermediate-to-reference convergence evidence; a grid below its
declared reference eta resolution is reported as under-resolved rather than
accepted as a physical result.
The generated shear residual evaluates its declared correction directly so
nearly equal metric potentials do not create a floating-point-only breach.

All bundled CMB-capable models execute through the Copernican native
declared-graph CMB engine. The CLI and GUI do not expose a CMB solver or
backend selector; the selected control and test models provide the physical
contracts for the same engine. CAMB and CLASS are independent scientific
reference tools used by tests, not production spectrum engines.

A default package installation has no CAMB or CLASS dependency. The repository
workspace lock includes CAMB only for independent scientific-reference tests;
the packaged runtime lock and installed license inventory contain native
production dependencies only.

Requested spectra resolve only declared model dependencies. A model without
a defensible perturbation closure reports the affected CMB outputs as
unavailable rather than substituting another engine or a zero spectrum. See
[`copernican/docs/cmb_solver.md`](copernican/docs/cmb_solver.md) for the graph,
physical, numerical, lensing, caching, and reference conventions.

CMB likelihoods and result tools preserve exact spectrum names and row order,
including repeated or noncontiguous multipoles. Scalar, vector, tensor,
total, lensed, unlensed, lensing-potential, and diagnostic surfaces retain
separate metadata in plots, diagnostics, and long-form CSV output. Runtime
payloads distinguish computed, unrequested, physically zero, and unavailable
spectra, while cache identities bind the graph, parameters, numerical grids,
accuracy controls, requested spectra, and multipole sequence.

Native CMB validation records the resolved numerical envelope in each run
manifest. The named final tier requires bounded background, transfer,
source, hierarchy, momentum-grid, and lensing controls and rejects
under-resolved requests before expensive evolution. Cross-sector refinement
tests enforce the documented `TT`, `TE`, `EE`, `PP`, lensed `BB`, q-grid,
and hierarchy thresholds. Explicit graphs without a sector registry retain
their active sector identity from compiled observable and tensor-character
metadata, so runtime envelopes cannot silently omit executed sectors.

Native MCMC workers prepare immutable graph structure once per model and
reuse it across parameter proposals. Bounded caches distinguish structural,
parameter-dependent, and complete-result data, while request diagnostics
record cold, warm, and exact-hit states with phase timings and work units.
Primordial-only parameter rebounds reuse bounded transfer products and rerun
only primordial power integration; changed cosmological parameters retain
separate transfer identities, and adaptive refinement keeps its full path.
Exact split collision half-steps absorb collision stiffness, so their
magnitude does not create redundant Runge-Kutta microsteps after a declared
tight-coupling transition. Only valid parameter-domain exclusions become
rejected proposals; contract, convergence, non-finite, constraint,
capability, and performance failures stop execution with typed diagnostics.

Native CMB performance budgets are enforced separately at the workload
boundary:
180 seconds for a cold full spectrum, 5 seconds for a warm parameter rebound,
and 1 second for an exact cache hit. Reports retain deterministic median and
p95 samples for each state, so practical MCMC throughput remains visible
without weakening numerical accuracy or silently reusing stale results.

Ensemble fit results also retain an `ensemble_performance` record with total
and per-stage timings, requested and effective worker counts, the CPU-derived
worker limit, nominal proposal evaluations, and failed-request counts. Spawned
workers request one numerical thread, and the record marks any process or
numerical oversubscription. The reference validation manifest fixes seed 0,
five burn-in steps, ten production steps, 32 walkers, and a three-worker pool;
its copied run manifest and parameter summary preserve this provenance.

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
through seed, control model, test model, dataset, sampler engine, and plan
panels. The control model defaults to `model_lcdm.yml`, while the test model
is selected independently. The Save Manifest page stays locked until each
step has a selection; and the Start Run action renames the workspace to
`copernican-run_<timestamp>` before launching the worker. The Run Settings
panel mirrors the CLI prompts for walkers, burn-in, production steps, and pool
size so GUI runs and CLI runs use the same run metadata.

Sampler convergence diagnostics are evaluated over sampled coordinates.
Parameters locked by equal lower and upper bounds remain in posterior outputs
with deterministic diagnostics: R-hat is `1.0`, and bulk and tail effective
sample sizes equal the retained draw count. If ArviZ cannot produce finite
diagnostics for a sampled coordinate, the engine uses its conservative
internal finite estimator.

The manifest stores one comparison request containing both model identities.
Compatibility checks cover declared observables, units, multipole grids, and
spectrum roles before execution. Summaries, CSV files, posterior artifacts,
plot footers, and residual labels use the resolved control/test pair rather
than assuming an LCDM control.
Posterior plotting reads that pair from the saved manifest, and direct
plotting calls provide the same comparison object.

The run worker is the sole owner of the canonical log file. It sends
structured severity-preserving events to the Run Monitor's in-memory log box,
while progress snapshots use a separate channel. GUI forwarding never writes
back to the worker file, so each event and selected dataset-ingestion record
appears once. Metadata dialogs open with the system default application and
use the same launch behavior as the rest of the GUI.
Run and validation workers use the active interpreter from the directory that
contains the importable `copernican` package, so source checkouts do not
require an editable package installation.
Immediately before launch, the GUI writes the displayed confirmation snapshot
to the run workspace. The worker therefore receives the same models, datasets,
seed, and sampler settings shown in the Run Builder.

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
copies stay aligned. `README.md` is the canonical source for
`copernican/README.md`. `docs/gui_guide.md` explains the GUI,
`docs/cli_guide.md` explains the CLI, `docs/run_manifest.md` covers manifest
structure, and `docs/packaging.md` covers setup and distribution tasks.

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
