# copernican
**Doc ID:** README
**Doc Type:** repo-readme
**Project Version:** 12.0.26
**Last Updated:** 2026-09-01
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
sampler choice, CMB solver choice, and output layout.

Copernican is built for reproducibility. Every run writes a manifest, one
canonical run log, summary artifacts, plots, and chain outputs into a per-run
directory under `~/copernican_output/`, so a result can be replayed or audited
later without guessing which options were used.

The package includes the model library, trusted dataset parsers, sampler
samplers, validation manifests, and supporting analysis tools needed for the
full workflow. Every bundled CMB model declares the declared graph contract;
models with available CMB output use the same declared solver, while a model
without a defensible perturbation closure reports CMB output as unavailable.
USMF2 now executes through that same declared route. Its theory-specific
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

Declared spectrum requests run that capability preflight before background
construction, so unsupported observables fail without beginning evolution.

Before declared scalar evolution begins, Copernican audits the requested k grid
against the model's declared numerical limits and preflights every mode on the
coupled Einstein constraint surface. The runtime records the ordered mode set,
residual terms, normalization scales, and constraint provenance with the
spectrum data so scientific failures can be traced without silently skipping
high-k modes.
Generated scalar contracts keep their declared regular superhorizon metric
seed during this preflight. The nearly singular early-time energy constraint
is validated rather than solved for a replacement potential, preventing
low-k round-off from collapsing the initial metric and corrupting every
projected spectrum.

Final generated scalar spectra promote the declared Fourier ladder to at least
512 modes. This resolves the rapidly oscillating spherical-Bessel phase that
a 64-mode smoke-test ladder aliases into a jagged CMB curve. The hierarchy
also uses quarter-cycle Runge-Kutta stages through recombination, then returns
to the declared phase step for the late integrated Sachs-Wolfe tail.
The bundled final LambdaCDM contract additionally declares a bounded
phase-aware line-of-sight grid. Its runtime envelope records the effective
eta-node count and spacing, so raw projection resolution is inspectable rather
than hidden by plotting.

Evolved scalar Einstein residuals use the sum of absolute declared equation
terms as their dimensionless normalization. Copernican records the maximum
location and physical regime, tolerance and normalization provenance, and
coarse-to-intermediate-to-reference convergence evidence; a grid below its
declared reference eta resolution is reported as under-resolved rather than
accepted as a physical result.
The generated shear residual evaluates its declared correction directly so
nearly equal metric potentials do not create a floating-point-only breach.

All bundled CMB-capable models execute through the Copernican declared
declared-graph CMB solver. The manifest selects the solver independently from
the sampler; `ccmbs_numpy` is the default CCMBS reference backend, and its
identity and capabilities are persisted in provenance. CAMB and CLASS are
independent scientific reference tools used by tests, not production spectrum
solvers.

A default package installation has no CAMB or CLASS dependency. The repository
workspace lock includes CAMB only for independent scientific-reference tests;
the packaged runtime lock and installed license inventory contain declared
production dependencies only.

Requested spectra resolve only declared model dependencies. A model without
a defensible perturbation closure reports the affected CMB outputs as
unavailable rather than substituting another solver or a zero spectrum. See
[`copernican/docs/cmb_solver.md`](copernican/docs/cmb_solver.md) for the graph,
physical, numerical, lensing, caching, and reference conventions.

CMB likelihoods and result tools preserve exact spectrum names and row order,
including repeated or noncontiguous multipoles. Scalar, vector, tensor,
total, lensed, unlensed, lensing-potential, and diagnostic surfaces retain
separate metadata in plots, diagnostics, and long-form CSV output. Runtime
payloads distinguish computed, unrequested, physically zero, and unavailable
spectra, while cache identities bind the graph, parameters, numerical grids,
accuracy controls, requested spectra, and multipole sequence.

Declared CMB validation records the resolved numerical envelope in each run
manifest. The named final tier requires bounded background, transfer,
source, hierarchy, momentum-grid, and lensing controls and rejects
under-resolved requests before expensive evolution. Cross-sector refinement
tests enforce the documented `TT`, `TE`, `EE`, `PP`, lensed `BB`, q-grid,
and hierarchy thresholds. Explicit graphs without a sector registry retain
their active sector identity from compiled observable and tensor-character
metadata, so runtime envelopes cannot silently omit executed sectors.

The fixed-parameter CCMBS diagnostics API is independent of samplers and
plots. `discover_bundled_cmb_plugins()` enumerates all bundled CMB models;
`run_cmb_model_diagnostic()` captures raw transfer components, raw and public
TT/TE/EE spectra, runtime-envelope metadata, and doubled-k-grid refinement
errors before plotting. `run_bundled_cmb_diagnostics()` applies the same
report shape to the full corpus, preserving explicit non-convergence instead
of hiding it. `compare_cmb_spectra_to_reference()` adds backend-neutral
fractional auto-spectrum and normalized cross-spectrum checks for independent
fixed-point fixtures. Generated scalar contracts validate explicit metric
derivatives and compiler-backed source and closure expressions before runtime;
`Psi_tau` and `Phi_history_tau` use explicit runtime history-gradient
bindings, so missing derivatives cannot become implicit zero histories. BAO
has a fixed-background regression that deliberately breaks the CMB entrypoint
and verifies that BAO evaluation remains usable.
Full observable parity is available through
`compare_full_cmb_observable_parity()` and `build_cmb_parity_report()` in
`copernican.lib.likelihoods.cmb.diagnostics`. These helpers compare raw
`C_ell` or `D_ell` arrays for TT, TE, EE, BB, PP, TP, EP, lensed surfaces,
and explicit scalar/vector/tensor/total sectors. They retain finite and
physical-shape checks, cross-spectrum sign checks, C-to-D convention checks,
independent refinement evidence, fixture digests, and optional parameter
response points; missing or mismatched rows are rejected rather than
interpolated or silently substituted.
Fixed-point reports also carry compact raw source-history samples and
independently recomputed metric, visibility, polarization, and ISW residuals;
an unavailable or failed audit remains explicit rather than being treated as
parity evidence. Raw projection certification records those residuals for the
generated-hierarchy slice, while final certification remains the acceptance
boundary.
Generated histories also retain fine and coarse eta arrays in
`source_history_refinement`. The `source_history_bundle_digest` binds those
arrays, raw source terms, hierarchy and initial-state residuals, derivative
provenance, audit controls, and the independent closure decision to one
deterministic SHA-256 value. Explicit model graphs report a typed
`not_applicable` digest instead of fabricating generated-history evidence.
Production contracts may additionally declare a doubled scalar
`k_sample_count` convergence rule. CCMBS evaluates the declared TT, TE, and
EE surfaces at both resolutions, records per-spectrum errors in the runtime
envelope, and raises a typed convergence failure when the declared tolerance
is not met. `audit_bundled_cmb_contracts()` separately inventories every
bundled declaration for contract version, gauge, sectors, spectra, hierarchy
families, numerical bounds, and runtime-envelope consistency; this structural
audit does not substitute for hierarchy or CAMB scientific validation.
`audit_bundled_cmb_declarations()` adds the model-facing classification used
by the declaration slice: `ready` means the model-owned contract and source
graph are complete, `rejected` means an enabled declaration is structurally
incomplete, and `unavailable` is reserved for a model that explicitly disables
CMB output. The report also records whether execution uses the shared
generated scalar hierarchy or an explicit model-authored scalar graph, along
with any theory-specific source equations. QRSF and TORG retain their named
baryon-locked source closures; USMF2 is classified as an explicit scalar
graph, never as a generated-hierarchy or LCDM fallback.
The final certification helpers build a deterministic matrix from those raw
reports, reject missing residual or reference evidence, retain every raw
array, and write a hash-addressed JSON record. The independent test-owned
CAMB fixture freezes the LCDM parameter point, multipole ordering, `D_ell`
normalization, TT/TE/EE tolerances, and CAMB provenance. BAO isolation tests
deliberately make the CMB entrypoint unavailable while requiring identical
fixed-background values and covariance handling.
`build_final_cmb_certification_report()` extends that boundary with solver
and dataset identities, fixture hashes, per-model raw evidence digests, and
explicit integrity decisions. `assess_bao_cmb_isolation()` compares captured
BAO values, covariance metadata, and typed failures without invoking CCMBS.
Slice One records the pre-repair corpus baseline before any shared physics is
changed. `CMB_CORPUS_BASELINE_REQUEST` freezes the initial-guess parameters,
ordered TT/TE/EE multipoles, k/eta node counts, source-anchor rule, and
doubled-k refinement. `run_bundled_cmb_corpus_baseline()` evaluates each
frozen `BUNDLED_CMB_MODEL_FILENAMES` row once, attaches contract and
source-graph audits, and preserves raw source histories, transfer components,
public and raw spectra, residual vectors, numerical metadata, cache
identities, and typed decisions before a plot is made.

USMF2 follows named finite node-count tiers rather than a wall-clock limit.
An incomplete progression is recorded as `unclassified` with its remaining
work and never relabelled `unavailable`. `build_cmb_corpus_baseline_report()`
and `write_cmb_corpus_baseline_report()` serialize the ten exact rows in a
canonical order with per-row and report SHA-256 digests. Rejected evidence is
therefore preserved as a scientific baseline rather than hidden as a pass.

The later strict bundled-model matrix uses `CMB_CERTIFICATION_TIER` for final
scientific acceptance. `run_bundled_cmb_matrix()` evaluates each filename
through CCMBS, while `build_bundled_cmb_matrix_report()` and
`write_bundled_cmb_matrix_report()` require explicit contract identity,
source residuals, doubled-grid evidence, and scalar/batch cache-isolation
evidence before counting a model as accepted. The exact ordered batch audit
is exposed by `assess_scalar_batch_cache_evidence()`; an unmeasured check
remains an explicit non-passing decision. A complete matrix exposes
`decision_complete: true` even when a model is explicitly rejected, while
`success` remains false until every row is scientifically accepted. The
opt-in matrix execution can also evaluate two bounded parameter points through
both public scalar and ordered batch paths, recording per-point equality and
cache identities; a model with no independently variable parameter remains
explicitly unavailable for that comparison.
Slice Eleven adds the full corpus boundary. `declared_cmb_spectrum_names()`
reads every angular-power observable from a compiled model, so a matrix
request cannot silently collapse to TT/TE/EE. `run_bundled_cmb_full_matrix()`
executes those per-model requests, including BB, PP, TP, EP, and any lensed
names a future contract declares. `build_bundled_cmb_full_matrix_report()`
requires an exact declared/requested match and records one classification per
model: accepted, rejected with typed evidence, or unavailable only when the
declaration explicitly disables CMB output. Raw spectra, transfer histories,
source residuals, doubled-grid evidence, scalar/batch comparisons, and cache
identities remain nested in each row; missing or unclassified models are never
promoted to a passing corpus result. `write_bundled_cmb_full_matrix_report()`
serializes the same hashable matrix without dropping any raw fields.
Tier-controlled parity uses the same declared numerical overrides and workload
for both paths. The cached scalar wrapper retains its typed result, raw
unscaled spectra, request metadata, solver identity, and phase provenance so
the audit compares solver products rather than plot output.
The final LambdaCDM declaration uses 2048 k, eta, and evolution nodes. CCMBS
keeps the generated hierarchy history at the declared LOS phase resolution so
acoustic sources are not aliased by sparse-history interpolation. Joint MCMC
evaluation may defer the doubled-grid comparison, but it cannot bypass the
declared final phase-grid floor; a low-node smoke path is restricted to the
explicit diagnostic matrix.
Source histories are cached independently from complete spectra only when
their structural contract, parameter identity, solver, source grid, and
requested source roles match exactly. Runtime envelopes report source-cache
hits and misses, while the phase-aware k status records the physical node
requirement and whether the bounded grid resolves it. A contract may require
that phase check explicitly; an under-resolved grid is then rejected with its
metrics instead of being presented as converged.
Irregular phase-aware k ladders use positive composite-trapezoid weights in
log-k; Simpson weights are reserved for uniform ladders so sparse Bessel
phases cannot create negative lobes or alternating aliases. Fixed-point
diagnostics also record ordered TT peaks and troughs, damping, TE sign
changes, and EE peaks directly from raw arrays before plotting.
The generated scalar hierarchy uses the standard `Pi = Theta_gamma,2 +
E_gamma,0 + E_gamma,2` collision moment, includes it once in the visibility
monopole, and applies the `3/4` E-source coefficient. The deprecated split
temperature quadrupole terms are explicit zeroes.
Coarsened projection batches preserve zero-width optional vector and tensor
sectors without indexing absent kernels, so scalar refinement remains warning
free under NumPy's strict empty-axis rules.

Generated scalar manifests publish a source-closure summary with derivative
kind, conformal-time coordinate, binding, dependencies, source-role ownership,
closure names, and Einstein residual names. Runtime envelopes repeat that
provenance and require finite, eta-aligned raw source-history fields before
independent residual recomputation. Generated constraints use strict term
lookup, so an omitted term raises a typed failure instead of being normalized
against its own residual. Explicit model graphs are recorded as
`not_applicable` for this generated summary and are never silently rewritten.

Declared MCMC workers prepare immutable graph structure once per model and
reuse it across parameter proposals. Bounded caches distinguish structural,
parameter-dependent, and complete-result data, while request diagnostics
record cold, warm, and exact-hit states with phase timings and work units.
The public `compute_cmb_spectrum_batch` contract evaluates an ordered
sequence of declared contracts and returns one serializable result per input,
including typed failures and cache provenance. It starts as an exact
scalar-to-batch adapter, so parameter-dependent state remains isolated while
shared-structure and vectorized kernels are validated independently.
Each successful result also records the requested multipole and spectrum
order, solver identity, and optional raw-spectrum payload. The diagnostics
parity audit compares public arrays exactly and compares raw arrays and
runtime metadata whenever those fields are present; missing cache identities
or required request metadata cannot pass as evidence.
The MCMC sampler exposes `cmb_batch_size` as an explicit opt-in setting;
the default `0` keeps the exact scalar sampler and fallback path unchanged.
Sampler progress also reports worker-pool launch and walker initialization
as explicit phases, including elapsed time, measured rate, remaining work, and
ETA in both CLI output and GUI progress snapshots. Burn-in and production
count iterations separately from cumulative walker evaluations. Worker logs
record runtime preparation duration so expensive startup is distinguishable
from posterior evaluation.
Primordial-only parameter rebounds reuse bounded transfer products and rerun
only primordial power integration; changed cosmological parameters retain
separate transfer identities, and adaptive refinement keeps its full path.
Exact split collision half-steps absorb collision stiffness, so their
magnitude does not create redundant Runge-Kutta microsteps after a declared
tight-coupling transition. Only valid parameter-domain exclusions become
rejected proposals; contract, convergence, non-finite, constraint, and
capability failures stop execution with typed diagnostics. Runtime telemetry
records phase timings, cache states, and work units without imposing a
wall-clock limit on valid solver evaluations. Large fixed-point requests use
deterministic ordered mode and projection chunks, and their configured and
effective numerical controls remain visible in the runtime envelope.

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
  analysis tools, plotting helpers, and the declared CMB internals.
- `copernican/models/` houses the YAML model definitions and their metadata.
- `copernican/samplers/` collects the sampler back ends.
- `copernican/datasets/` bundles the trusted datasets and parser metadata.
- `copernican/validation/` holds the validation runner and reference
  manifests.
- `docs/` contains the long-form manual set.
- `ABOUT.md`, `AGENTS.md`, `CHANGELOG.md`, `CITATION.cff`, `PLAN.md`,
  `SECURITY.md`, and `SUPPORT.md` describe the front-door package contract.

## Run Builder and GUI
The GUI keeps the same manifest model as the CLI. The Run Builder walks
through seed, control model, test model, dataset, sampler, and plan
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
diagnostics for a sampled coordinate, the sampler uses its conservative
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
- `python -m copernican --analysis-compare <base_run> <alternate_run>`
- `python -m copernican --analysis-posterior <run_dir>`

## Repository Policy
Read `AGENTS.md` before making changes, keep the package docs mirrored, and
follow the gate workflow so edited files, manifests, and generated artifacts
stay in sync.
