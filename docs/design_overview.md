**Last Updated:** 2025-11-25

This document expands on the high-level summary in the README by tracing how
the Copernican Suite organises its architecture. The command-line launcher
(`copernican.py`) steers each run, the `copernican_lib/` package gathers shared
infrastructure, and the `engines/`, `models/` and `data/` directories plug into
that foundation to deliver repeatable analyses.

The `copernican_lib/cli/` namespace now houses the dependency scanner and menu
renderers invoked by the launcher. Keeping those prompts in a dedicated
package trims the startup import surface so users reach the Stage 1 seed dialog
faster while retaining the existing logging, validation and manifest pipelines
described throughout this document.

## Architectural map

* `copernican.py` assembles run manifests, dispatches dataset loaders and
  prepares engine inputs so Stage 2 sampling always starts from a consistent
  configuration. The launcher keeps Stage 1 focused on reproducibility by
  leading with the seed dialog, surfaces every validation error encountered
  during model parsing or engine import and leaves a deliberate spacer after
  logging initialisation so the console flow stays tidy without redundant
  banners.
* `copernican_lib/` contributes the reusable building blocks—data ingestion,
  posterior construction, validation checks, plotting helpers and diagnostics.
  Engines and parsers import from this package instead of reimplementing
  numerical plumbing. Shared progress helpers live in
  `copernican_lib.progress`, along with the spinner pump, notifier bridge and
  suspension context that keeps console output coherent even when diagnostics
  print between walker updates. The helpers record the first render emitted by
  each batch and always wipe the console line with a spacer on teardown so
  captured transcripts never trap stale 0% bars.
* `engines/` contains back ends such as the default ``cosmo_engine_mcmc.py``.
  Engines consume `EnginePlugin` definitions, evaluate joint likelihoods
  spanning SNe Ia, BAO and CMB data and surface ArviZ-powered convergence
  diagnostics for downstream tooling. When ArviZ is unavailable the code falls
  back to a conservative Gelman–Rubin summary while logging the downgrade.
  CI runners that cannot call CAMB can opt into the ``COPERNICAN_FAKE_CMB``
  shortcut while production evaluations still query the physics engine. Nested
  sampling and ensemble MCMC both rely on the shared Stage 2 renderer so the
  carriage-return bar, spinner and walker metrics stay consistent regardless of
  backend.
* `models/` holds YAML descriptions that declare bounds, priors, transforms and
  dataset compatibility. Each file is compiled into a picklable
  :class:`copernican_lib.plugins.EnginePlugin` so multiprocessing pools can
  reconstruct Stage 2 state deterministically. Plugin validation allows only
  vetted attributes and functions and preserves constants, transforms and
  priors exactly as written in the model file.
* `data/` curates vetted catalogues with parser code and metadata that record
  citations, licensing information and SHA256 digests. Loaders validate the
  digests before the observations flow into the likelihood pipeline. Parsers
  must register under the `dataset_id` stated in their metadata so discovery
  remains deterministic, and loaders reject symbolic links or paths that would
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

Logging is initialised immediately after the cache check. Console messages and
prompts flow through :mod:`copernican_lib.console_output` so patched `print`
and `input` hooks in :mod:`copernican_lib.logger` can mirror them into the log
file without duplication. The logger strips repository paths from messages,
records system details and timestamps in UTC and keeps a deliberate blank
spacer after initialisation so Stage 1 banners align with prior releases while
avoiding redundant status text.

### Stage 1 orchestration

Stage 1 focuses on reproducibility and validation:

* The seed selector honours ``COPERNICAN_SEED`` when set, otherwise offers to
  accept the default value, enter a custom integer or generate a random seed.
  The choice is logged and written to the manifest before any sampling.
* Model parsing normalises YAML files via
  :mod:`copernican_lib.model_spec_validator` and compiles the expressions into
  NumPy-ready callables through :mod:`copernican_lib.model_coder`. Engine
  plugins built with
  :func:`copernican_lib.engine_plugin_validation.build_plugin` collect bounds,
  priors, transforms and optional CMB parameter mappings.
  Validation errors are aggregated and displayed as bullet points before the
  user is asked whether to restart Stage 1 or exit entirely.
* Engine selection is dynamic: any file matching `engines/cosmo_engine_*.py`
  appears in the menu. Prompts reflect the selected backend so ensemble MCMC
  users configure burn-in, walkers and worker pools while nested sampling users
  pick live-point budgets and evidence tolerances. A confirmation summary makes
  the intended plan explicit and provides options to restart the questionnaire
  or cancel the run cleanly.

### Stage 2 sampling and progress

Once both models and datasets are prepared, Stage 2 draws from the joint
posterior built by :func:`copernican_lib.posterior.make_logposterior`. The
helper injects Jacobian corrections for transformed parameters, applies bounds
and assembles the combined SNe, BAO and CMB likelihoods via
:class:`copernican_lib.likelihoods.JointLike`. Engines receive a picklable
:class:`copernican_lib.posterior.PosteriorEvaluator` so multiprocessing pools
can reuse the same callable safely.

The shared renderer in :mod:`copernican_lib.progress` keeps interactive output
stable across engines. It paints a fifty-character carriage-return bar with
Unicode sub-blocks, a walker-progress meter and an animated spinner. Updates
flush on every write to keep terminals responsive, while a suspension context
allows diagnostic messages to print without corrupting the bar. When a batch
ends the renderer clears the line and inserts a spacer so transcripts never
contain half-drawn bars. If ArviZ is installed the engine records R-hat and
effective sample sizes for every parameter on each batch; otherwise it logs a
conservative Gelman–Rubin fallback.

### Stage 3–4 post-processing

BAO and CMB analyses reuse the sampler output. BAO observables are computed
from maximum-posterior parameters and logged alongside residual norms so
operators can monitor fit quality as plots render. CMB spectra pull additional
plugin-provided constants from `cmb.param_map` when present and stream TT/TE/EE
residual statistics to the console. Both stages respect dataset independence
statements stored in :mod:`copernican_lib.dataset_registry` so assumptions
remain explicit in manifests and plots.

### Stage 5 visualisation

Stage 5 produces publication-ready figures. `copernican_lib.plotter` responds
to the number of parameters by adjusting canvas size, font scale and
corner-plot grid dimensions. Footer guard bands keep three lines of metadata
clear of the axes: the model comparison, dataset description and citation.
Footer spacing maintains both a fixed gap above the axes and a clearance above
the canvas edge so long labels or future gravitational-wave annotations do not
collide with data. The corner-plot validator thins samples when necessary,
labels every parameter using the names stored on the plugin and exposes a
legacy wrapper so older tooling can still import `_validate_corner_inputs`
without linter noise.

### Stage 6 outputs and manifests

All artefacts land in a run-specific `output/copernican-run_YYYYMMDD_HHMMSS`
directory. The manifest recorded by :mod:`copernican_lib.run_manifest` captures
the suite version, model filenames, engine choice, sampler settings, dataset
hashes, CMB metadata, seed and Git state. CSV summaries, NetCDF chains and
Matplotlib figures share the same naming scheme so downstream notebooks and
manuscripts can reference them consistently.

## Dataset integrity and parsers

Dataset loaders live in :mod:`copernican_lib.dataset_registry` and expose
decorators that register parser functions for each `dataset_id`. Parser
dictionaries now follow explicit ``*_PARSER_REGISTRY`` names and are collected
via the ``get_parser_registries`` helper so discovery code cannot be confused
with individual loader functions. SHA256 digests for every non-parser file in a
dataset directory are computed and stored on the returned DataFrame `.attrs`
mapping. Parsers only load when their hashes match the vetted list stored in
the corresponding metadata file. Each loader logs whether a covariance matrix
was used or diagonal errors were applied and records the dataset version in the
manifest. Independence statements are centralised so manifests and console
summaries always describe which probes are assumed uncorrelated.

## Plugin interface and posterior construction

Plugins produced by
:func:`copernican_lib.engine_plugin_validation.build_plugin` expose dataset
compatibility flags (`valid_for_distance_metrics`, `valid_for_bao`,
`valid_for_cmb`) and optional `cmb.param_map` entries for engines that compute
spectra. The interface includes required attributes and functions listed in
:mod:`copernican_lib.plugins`; validation errors identify missing hooks and
incompatible parameter maps, preventing engines from receiving incomplete
models. Posterior evaluation routes through
:func:`copernican_lib.posterior.make_logposterior`, which merges priors,
transforms and likelihood callables into a picklable evaluator suitable for
spawn-based worker pools on macOS and Linux.

## Console and error handling

All console I/O flows through :mod:`copernican_lib.console_output` so the
logger can mirror it faithfully. Unicode encoding errors are caught and
replaced with ASCII fallbacks to keep runs alive on limited terminals. The
launcher enables `faulthandler` and registers handlers for SIGILL, SIGSEGV and
SIGFPE so any crash produces a stack trace on both the console and log before
exiting. All Python warnings are forwarded to the central logger, and the
``COPERNICAN_STRICT_WARNINGS`` environment variable can promote them to errors
during CI runs.

## Dependency management and packaging

`tools/update_lock.py` and the accompanying `make lock` target regenerate the
pinned `requirements.lock` file using `pip-tools`, avoiding implicit Python
version headers so CI runs remain deterministic. A small helper ensures
`pip-tools==7.4.1` is available before the lock step executes, even in clean CI
environments. The packaging guide in `docs/packaging.md` details how to build
wheels and source distributions while keeping runtime metadata aligned with the
tracked version file.

## Launcher parity and CI

The cross-platform launchers (`start.sh`, `start.command`, `start.bat`) all set
the managed virtual environment and pipe environment variables into
`copernican.py`. Each launcher keeps multi-line PowerShell calls inside helper
routines to avoid parser confusion on Windows. The CI suite mirrors these
launchers by invoking the same dependency checks, sampler smoke tests and
metadata validators across Linux, macOS and Windows runners.

## Future probes and extensibility

The current architecture keeps engines, parsers and models pluggable. New
datasets can register via the loader decorators, and new engines need only
honour the progress, logging and plugin interfaces to fit seamlessly into the
menu system. Placeholder directories allow in-progress probes—such as
gravitational-wave standard sirens—to coexist without appearing in user menus
until their metadata and digests are finalised.
