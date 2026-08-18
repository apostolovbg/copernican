# Design Overview
**Project Version:** 12.0.26

This document expands on the high-level summary in the README by tracing
how the Copernican organises its architecture. The command-line launcher
(`python -m copernican`) steers each run, the `copernican/lib/` package
gathers shared infrastructure, and the `copernican/samplers/`,
`copernican/models/` and `copernican/datasets/` directories plug into that
foundation to deliver repeatable analyses.
The `copernican/lib/cli/` namespace houses the dependency scanner and menu
renderers invoked by the launcher. Keeping those prompts in a dedicated package
trims the startup import surface so users reach the Stage 1 seed dialog faster
while retaining the existing logging, validation and manifest pipelines
described throughout this document.
## Table of Contents
- [Architectural map](#architectural-map)
- [Stage-by-stage flow](#stage-by-stage-flow)
 - [Dependency priming and logging](#dependency-priming-and-logging)
 - [Stage 1 orchestration](#stage-1-orchestration)
 - [Stage 2 sampling and progress](#stage-2-sampling-and-progress)
 - [Stage 3–4 post-processing](#stage-3–4-post-processing)
 - [Stage 5 visualisation](#stage-5-visualisation)
 - [Stage 6 outputs and manifests](#stage-6-outputs-and-manifests)
- [Dataset integrity and parsers](#dataset-integrity-and-parsers)
- [Plugins & Posterior](#plugin-interface-and-posterior-construction)
- [Console and error handling](#console-and-error-handling)
- [Deps & Packaging](#dependency-management-and-packaging)
- [Package entrypoint and CI](#package-entrypoint-and-ci)
- [Future probes and extensibility](#future-probes-and-extensibility)
## Architectural map
* `python -m copernican` assembles run manifests, dispatches dataset loaders
 and prepares sampler inputs so Stage 2 sampling always starts from a
 consistent configuration. The launcher keeps Stage 1 focused on
 reproducibility by leading with the seed dialog, surfaces every validation
 error encountered during model parsing or sampler import and leaves a
 deliberate spacer after logging initialisation so the console flow stays
 tidy without redundant banners.
* `copernican/lib/` contributes the reusable building blocks—data ingestion,
 posterior construction, validation checks, plotting helpers and diagnostics.
 Samplers and parsers import from this package instead of reimplementing
 numerical plumbing. Shared progress helpers live in
 `copernican.lib.progress`, which exposes `BatchProgressBar`. The helper
 writes simple counter lines such as “Burn-in stage batch 1: 3/200 steps
 completed (1%)”, preserves the listener contract and exposes a no-op
 suspension context so diagnostics can print between updates without a
 carriage-return renderer.
* `copernican/samplers/` contains sampler backends such as the default
 ``sampler_mcmc.py``. Samplers consume `ModelPlugin` definitions,
 evaluate joint likelihoods spanning SNe Ia, BAO and CMB data and surface
 ArviZ-powered convergence diagnostics for downstream tooling. When ArviZ is
 unavailable the code falls back to a conservative Gelman–Rubin summary while
 logging the downgrade. CMB contracts use the declared-math graph CCMBS
 solver through the registry in
 `copernican/lib/likelihoods/cmb/`, where `cmb.py` owns the public likelihood
 surface, `orchestrators/ccmbs.py` owns declared execution coordination,
 `solvers/` owns backend selection, and `runtime/` owns numerical helpers.
 Solver selection remains independent from sampler selection.
 `runtime/background.py`, `runtime/evolution.py`,
 `runtime/projection.py`, and `runtime/cache.py` split the declared path
 into background tables, declared evolution, line-of-sight projection,
 and bounded cache ownership. Nested sampling and ensemble MCMC both rely
 on the shared Stage 2 helper so the counter lines and listener events
 stay consistent regardless of sampler backend.
* `copernican/models/` holds YAML descriptions that declare bounds, priors,
 transforms and dataset compatibility. Each file is compiled into a picklable
 :class:`copernican.lib.model_adapter.ModelPlugin` so multiprocessing pools
 can reconstruct Stage 2 state deterministically. Adapter validation allows
 only vetted attributes and functions and preserves constants, transforms,
 priors and structured declared CMB contracts exactly as written in the model
 file.
* `copernican/datasets/` curates vetted catalogues with parser code and
 metadata that record citations, licensing information and SHA256
 digests. Loaders validate the digests before the observations flow
 into the likelihood pipeline. Parsers must register under the
 `dataset_id` stated in their metadata so discovery remains
 deterministic, and loaders reject symbolic links or paths that would
 escape the repository tree. Folders named `placeholder` are ignored so
 unfinished datasets never pollute the menus.
* `docs/` stores focused guidance on data formats, manifest contents,
 packaging, LaTeX conventions and the scripting API. README sections link into
 these files rather than repeating their full contents.
* The GUI scaffold mirrors the CLI flow. Diagnostics logging begins when the
 GUI loads, capturing environment checks and exposing severity filters plus
 downloads from Settings → Diagnostics. The Run Builder produces a manifest
 snapshot at the "Start Run" confirmation stage, starts the run log with that
 manifest context before outputs exist and streams the lines into the Run
 Monitor with severity filters, copy/export actions and alert anchors that can
 jump directly to the relevant log snippet. Pause, cancellation and hard-stop
 controls mark the manifest as paused, cancelled or aborted and capture
 whether outputs were kept, deleted or archived for audit trails.
* Dataset, model and sampler panes expose compatibility badges, parser digests,
 citations and licenses alongside actions that open the containing folders,
 view metadata files or revalidate trusted parser hashes. Manifest files can
 be pulled back into the Run Builder through "Duplicate & Edit" so the GUI
 pre-fills model, dataset, and sampler selections for iterative
 experiments.
## Declared CCMBS Solver
CMB contracts use the declared-math graph CCMBS solver in
`copernican/lib/likelihoods/cmb/` through the selectable solver registry.
The stable default execution identity is `ccmbs_numpy`. User interfaces keep
control/test model, sampler, and CMB solver selections independent.
* One immutable graph carries variables, derived quantities,
 differential equations, algebraic constraints, closures, source terms,
 initial conditions, observable mappings, validity notes, and numerical
 requirements. Variable metadata such as rank, spin, parity, tensor
 character, gauge role, source role, and projection role stay attached to
 graph nodes instead of selecting a solver family.
* Source terms are declared as named graph expressions. Observable mappings
 choose the projection kernel and bind the named source terms that feed
 each transfer component or spectrum.
* Background quantities come from the declared model expressions when they
 exist, otherwise from the physical defaults resolved by the helper. The
 sampler evaluates `H(a)`, conformal time, `chi(z)`, angular-diameter
 distance, baryon and photon densities, relativistic neutrino density, the
 baryon-photon sound speed, and the sound horizon.
* Recombination uses a Peebles-style hydrogen ODE with detailed-balance
 photoionization, equilibrium helium ionization, and a
 photoionization-balance reionization ODE calibrated to the requested
 optical depth. The background tables expose `x_e(z)`, `n_e(z)`,
 optical depth `tau(eta)`, `tau_dot`, and the visibility function
 `g(eta) = -tau_dot * exp(-tau)`.
* The package split keeps the declared boundary explicit:
 `runtime/background.py` owns background-state construction,
 `runtime/evolution.py` owns compiled graph stepping,
 `runtime/projection.py` owns transfer projection and spectrum assembly,
 `runtime/cache.py` owns bounded cache storage, and the remaining runtime
 modules own adaptive grids, convergence, performance, and lensing helpers.
* Perturbations evolve whichever declared variables expose differential
 equations. Constraints and closures resolve algebraic targets inside the
 same graph before the declared observables are projected.
* Declared equations, constraints, closures, sources, and conditions may
 reference the background symbols `a`, `z`, `eta`, `H`, `Hconf`, `tau`,
 `tau_dot`, `visibility`, `k`, `seed`, `sound_horizon`, `sound_speed`,
 `sound_speed_sq`, `baryon_sound_speed_sq`, `collision_rate`,
 `free_streaming`,
 `tight_coupling_drag`, and `tight_coupling_ratio`, plus any solved or
 derived graph quantity. The default `seed` is a unit-normalized transfer
 seed unless the contract overrides it, while declared relation targets
 such as `Phi` and `Psi` only exist after the graph solves them. The safe
 math functions exposed by the expression evaluator remain available too.
 Unsupported symbols and incompatible gauges fail fast during
 validation or step evaluation instead of falling back to a decorative
 metadata path.
* Line-of-sight transfer functions integrate the visibility-weighted sources
 against spherical Bessel kernels. The source set includes Sachs-Wolfe,
 Doppler, early and late ISW, the visibility-weighted monopole and velocity,
 and the polarization quadrupole projection used for E modes.
 `spin2_b_mode` requires a declared `polarization_b` source ancestry, and
 `line_of_sight_lensing_potential` requires a declared `potential` source.
* The spectra are built from the primordial power law
 `P_R(k) = A_s * (k / k_pivot) ** (n_s - 1)` and integrated into `TT`,
 `TE`, `EE`, and any declared `BB`, lensing-potential, or model-defined
 transfer target. The declared path keeps the numerical settings explicit and
 rejects unsupported sectors, missing declared equations, and theory-specific
 solver keys before any spectrum is produced.
## Stage-by-stage flow
### Dependency priming and logging
The launcher opens by checking Python dependencies and offering to install any
missing ones. It caches the AST scan so repeat invocations can skip re-sourcing
modules whose paths and modification times have not changed. Relative imports
inside the bundled likelihood package are ignored during the scan so the
console never reports those internal modules as missing; unexpected warnings
usually mean the managed `.venv` was skipped. A short NumPy/SciPy calculation
verifies that compiled binaries match the available CPU features before heavy
work begins.
The manifest executor initialises logging before it emits run events. Console
messages flow through :mod:`copernican.lib.console_output`, while direct
`print` and `input` calls are captured without wrapping process streams. The
worker owns the only run file and emits structured records to the GUI's
memory-only monitor. Repository paths can be rendered relatively in the file;
external output paths remain absolute. System details and timestamps use UTC.
### Stage 1 orchestration
Stage 1 focuses on reproducibility and validation:
* The seed selector honours ``COPERNICAN_SEED`` when set, otherwise offers to
 accept the default value, enter a custom integer or generate a random seed.
 The choice is logged and written to the manifest before any sampling.
* Model parsing normalises YAML files via
 :mod:`copernican.lib.model_spec_validator` and compiles the expressions into
 NumPy-ready callables through :mod:`copernican.lib.model_coder`. Sampler
 adapters built with :func:`copernican.lib.model_adapter.build_plugin`
 collect bounds, priors, transforms, and optional structured declared CMB
 contracts.
 Validation errors are aggregated and displayed as bullet points before the
 user is asked whether to restart Stage 1 or exit entirely.
* Sampler selection is dynamic: any file matching
 `copernican/samplers/sampler_*.py` appears in the menu. Prompts reflect
 the selected backend so ensemble MCMC users configure burn-in, walkers and
 worker pools while nested sampling users pick live-point budgets and
 evidence tolerances. A confirmation summary makes the intended plan explicit
 and provides options to restart the questionnaire or cancel the run cleanly.
### Stage 2 sampling and progress
Once both models and datasets are prepared, Stage 2 draws from the joint
posterior built by :func:`copernican.lib.posterior.make_logposterior`. The
helper injects Jacobian corrections for transformed parameters, applies bounds
and assembles the combined SNe, BAO and CMB likelihoods via
:class:`copernican.lib.likelihoods.JointLike`. Samplers receive a picklable
:class:`copernican.lib.posterior.PosteriorEvaluator` so multiprocessing pools
can reuse the same callable safely.
The shared helper in :mod:`copernican.lib.progress` keeps interactive output
stable across samplers. It writes counter lines such as “Burn-in stage
batch 1:
3/200 steps completed (1%)”, emits the same ``batch_start``,
``progress_update`` and ``batch_finish`` events that feed the GUI progress
panels, and offers a suspension context so diagnostics can print between
updates without disrupting the counter output. When a batch ends it logs a
completion line before the next batch begins, giving both terminals and log
files a clear record of progress. If ArviZ is installed the sampler records
R-hat and effective sample sizes for every parameter on each batch; otherwise
it logs a conservative Gelman–Rubin fallback.
Before those sampling batches, MCMC emits separate worker-pool and walker-
initialization phases. Their structured records include elapsed seconds,
completed evaluations, throughput, and estimated remaining time, so the GUI
monitor does not remain at a generic “Initialising” state while expensive
initial posterior points are evaluated. Worker runtime preparation duration
is logged independently from walker evaluation.
### Stage 3–4 post-processing
BAO and CMB analyses reuse the sampler output. BAO observables are computed
from maximum-posterior parameters and logged alongside residual norms so
operators can monitor fit quality as plots render. CMB spectra pull additional
adapter-provided constants from `cmb.param_map` when present, validate the
declared `cmb.perturbations` contract, compile the typed perturbation IR and
stream TT/TE/EE residual statistics to the console. Both stages respect
dataset independence statements stored in
:mod:`copernican.lib.dataset_registry` so assumptions remain explicit in
manifests and plots.
### Stage 5 visualisation
Stage 5 produces publication-ready figures. `copernican.lib.plotter` responds
to the number of parameters by adjusting canvas size, font scale and corner-
plot grid dimensions. Footer guard bands keep three lines of metadata clear of
the axes: the model comparison, dataset description and citation. Footer
spacing maintains both a fixed gap above the axes and a clearance above the
canvas edge so long labels or future gravitational-wave annotations do not
collide with data. The corner-input preparation path thins samples when
necessary and labels every parameter using the names stored on the adapter
before ArviZ or the Matplotlib renderer receives the data.
### Stage 6 outputs and manifests
All artefacts land in a run-specific `output/copernican-run_YYYYMMDD_HHMMSS`
directory. The manifest recorded by :mod:`copernican.lib.run_manifest` captures
the suite version, model filenames, sampler choice, sampler settings, dataset
hashes, CMB metadata, seed and Git state. CSV summaries, NetCDF chains and
Matplotlib figures share the same naming scheme so downstream notebooks and
manuscripts can reference them consistently.
## Dataset integrity and parsers
Dataset loaders live in :mod:`copernican.lib.dataset_registry` and expose
decorators that register parser functions for each `dataset_id`. Parser
dictionaries follow explicit ``*_PARSER_REGISTRY`` names and are collected
via the ``get_parser_registries`` helper so discovery code cannot be confused
with individual loader functions. SHA256 digests for every non-parser file in a
dataset directory are computed and stored on the returned DataFrame `.attrs`
mapping. Parsers only load when their hashes match the vetted list stored in
the corresponding metadata file. Each loader logs whether a covariance matrix
was used or diagonal errors were applied and records the dataset version in the
manifest. Independence statements are centralised so manifests and console
summaries always describe which probes are assumed uncorrelated.
## Plugin interface and posterior construction
Adapters produced by :func:`copernican.lib.model_adapter.build_plugin`
expose dataset compatibility flags (`valid_for_distance_metrics`,
`valid_for_bao`, `valid_for_cmb`) and structured `cmb` background and
perturbation contracts for samplers that compute spectra. The interface
includes required attributes and functions listed in
:mod:`copernican.lib.model_adapter`; validation errors identify missing hooks
and incompatible contracts, preventing samplers from receiving incomplete
models. The perturbation compiler produces a typed IR that records the
declared derivative equations, derived symbols, dependencies, and declared
execution summary before any scientific execution begins. Posterior
evaluation routes through
:func:`copernican.lib.posterior.make_logposterior`, which merges priors,
transforms and likelihood callables into a picklable evaluator suitable for
spawn-based worker pools on macOS and Linux.
## Console and error handling
All console I/O flows through :mod:`copernican.lib.console_output` so the
logger records it once. Unicode encoding errors use ASCII replacements on the
direct pre-logging path. The launcher enables `faulthandler` and registers
handlers for SIGILL, SIGSEGV and SIGFPE so a crash produces a stack trace on
the console and canonical log before exiting. Python warnings are forwarded
to the central logger, and ``COPERNICAN_STRICT_WARNINGS`` can promote them to
errors during CI runs.
## Dependency management and packaging
The dependency-management surface regenerates the pinned `requirements.lock`
file and matching license inventory so dependency refreshes stay
deterministic. The packaging guide in `docs/packaging.md` details how to
build wheels and source distributions while keeping runtime metadata aligned
with the tracked version file.
## Package entrypoint and CI
The package entrypoint keeps the managed virtual environment active and
receives environment variables through `python -m copernican`. The CI suite
mirrors that entrypoint by invoking the same dependency checks, sampler smoke
tests and metadata validators across Linux, macOS and Windows runners.
## Future probes and extensibility
The current architecture keeps samplers, parsers and models pluggable. New
datasets can register via the loader decorators, and new samplers need only
honour the progress, logging and adapter interfaces to fit seamlessly into the
menu system. Placeholder directories allow in-progress probes—such as
gravitational-wave standard sirens—to coexist without appearing in user menus
until their metadata and digests are finalised.
