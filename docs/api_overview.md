# Copernican Suite API Overview

**Last Updated:** 2025-11-23

The suite exposes a lightweight API intended for advanced scripting.
Most functionality lives in the ``copernican_lib`` package which can be
imported directly without using the command-line interface.  The core
modules are:

- `model_parser.parse_model(path, cache_dir)` – validate and clean a
  `cosmo_model_*.yml` file.
- `model_coder.generate_callables(clean_path)` – compile sanitized model YAML
  into Python callables.
- `engine_interface.build_plugin(parsed_data, funcs)` – construct an
  :class:`copernican_lib.plugins.EnginePlugin` instance with dataset toggles,
  priors, bounds and distance functions ready for engine consumption.
- `copernican_lib.plugins` – home of the picklable plugin dataclass and
  validation helpers. Import `REQUIRED_ATTRIBUTES` and `REQUIRED_FUNCTIONS`
  from here when building custom tooling that needs to confirm interface
  compliance.
- `copernican_lib.progress` – shared progress bars, walker notifiers and
  sampler integration helpers. Engines import `BatchProgressBar`,
  `StepProgressEmitter` and `configure_sampler_progress_reporting` so live
  Stage 2 updates stay consistent even outside the default MCMC engine. The
  helpers record the first frame emitted for each batch, stream walker-level
  updates with Unicode sub-blocks and always clear the console on teardown so
  captured logs never contain stale bars, even when a sampler aborts early.
  Nested sampling and ensemble MCMC reuse the same renderer, keeping labels and
  spinners aligned regardless of backend choice.
- `copernican_lib.plotter.plot_corner(samples, plugin, data_attrs,
  plot_dir)` – render the Stage 2 posterior as an automatically thinned
  corner plot whose panel size and typography respond to the number of
  parameters. Figures clamp to a twelve-inch canvas, fonts scale with the
  derived panel width and the footer still details how samples were filtered or
  thinned.
  Footer guard bands preserve both the gap beneath the axes and the distance to
  the canvas edge, keeping metadata clear of the grid even with elongated axis
  labels or future gravitational-wave annotations. Contour thresholds remain
  strictly increasing, preserving Matplotlib compatibility while eliding
  redundant dataset text and retaining the citation line.
  Stage 5 calls this helper after the probe-specific figures so every run
  records the sampler geometry alongside Hubble, BAO and CMB outputs. The
  underlying `_prepare_corner_inputs` validator flattens samples, derives
  thinning statistics and remains reachable through the legacy
  `_validate_corner_inputs` wrapper so older tools import the familiar name
  without modification while lint hooks stay satisfied.
- `copernican_lib.posterior` – exposes
  :func:`copernican_lib.posterior.make_logposterior`, which now returns a
  picklable :class:`PosteriorEvaluator` combining priors, transforms and
  likelihood callables. Engines should always route posterior evaluations
  through this helper to keep multiprocessing safe.
- `copernican_lib.statistics` – shared chi-squared and BAO/CMB helper
  functions used by every engine.  Importing from this module keeps the
  numerical implementations in a single place so engines remain thin
  orchestration layers. The helpers expose SNe chi-squared evaluations that
  always return finite values for physically meaningful proposals so MCMC
  reseeding can fall back to them reliably.
  CI runners that lack CAMB can opt into ``COPERNICAN_FAKE_CMB=1`` so the CMB
  helpers return deterministic synthetic spectra instead of performing heavy
  physics evaluations, leaving production calculations untouched.
  - `data_loaders.load_sne_data(dataset_id)`,
    `load_bao_data(dataset_id)`,
    `load_cmb_data(dataset_id)` – load datasets by their identifiers. The
    interactive prompt lists the human readable `dataset_name` and description,
    but calls expect the `dataset_id`. Each loader logs a short summary
    describing the dataset and whether its covariance matrix was used or
    diagonal errors were applied.
- `console_output.write(msg)` – unified console printing function that is
  logged
  verbatim via `logger`.
- `console_output.ask(prompt)` – input helper that records prompts and
  responses in the run log.
- `logger.setup_logging(log_dir)` – initialise logging and patch
  `print`/`input` so all interactions are captured.
- `utils.get_timestamp(now=None)` – return a `YYYYMMDD_HHMMSS` string in
  Coordinated Universal Time for consistent filenames and manifests. The
  helper underpins logging, result writers and manifest builders so outputs
  from CI and local runs align chronologically.
- `chain_io.save_posterior(chain, param_names, path, metadata)` – store
  posterior samples in NetCDF format using ArviZ, or xarray when the
  dependency is unavailable during lightweight tests.
- `csv_writer.save_sne_results_detailed_csv`,
  `save_bao_results_csv` and `save_cmb_results_csv` – persist fitting
  results with filenames that encode the dataset, model and timestamp.

- `engines.cosmo_engine_mcmc.fit_sne_parameters` – returns a dictionary with
  posterior samples, joint chi-squared diagnostics for the SNe/BAO/CMB
  components, dataset-level point counts, burn-in length, acceptance fractions
  and a sanitised log-probability trace. BAO and CMB data frames can be passed
  via the `bao_data_df` and `cmb_data_df` keyword arguments to enable joint
  sampling in a single call. ``burn_in_steps`` overrides the default
  ``max(100, n_steps // 5)`` warm-up, keeping scripted workflows nimble, and
  the ``pool_size`` keyword enforces user-selected multiprocessing pools
  while automatically expanding the walker ensemble to keep every worker
  busy. The private `_reseed_invalid_walkers` utility reseeds walkers that
  emit `nan` coordinates after burn-in so downstream API consumers never need
  to handle undefined sampler states. When the CLI selects this backend, Stage 2
  prompts for production steps, burn-in length, walker counts and worker pools,
  mirroring the available function arguments for scripted workflows. The
  diagnostic bundle still includes R-hat and effective sample sizes when ArviZ
  is missing by falling back to an internal Gelman–Rubin estimator so headless
  tests remain deterministic.
- `engines.cosmo_engine_nested.fit_sne_parameters` – wraps a lightweight
  nested-sampling routine that evaluates the same plugin-provided posterior
  while reporting log-evidence estimates, live-point counts, enlargement
  factors and iteration diagnostics. The CLI surfaces backend-specific prompts
  for live points, evidence tolerances and enlargement fractions so
  interactive runs align with scripted calls that specify the same keyword
  arguments.
- `result_writer.save_summary(results, output_dir)` – serialize fitted
  parameters, 1σ errors, covariance matrices and the recorded sampling
  configuration—including nested-sampling metadata such as live-point counts
  and evidence tolerances—to JSON and YAML for later analysis.
  - `engines.cosmo_engine_mcmc` – lightweight `emcee` sampler for SNe
    posteriors. Walkers are initialised uniformly within declared
    parameter bounds, a burn-in run precedes production sampling and the
    returned dictionary includes log-probability traces, acceptance
    fractions, estimated autocorrelation times when the production chain is
    long enough and both MAP and posterior
    mean parameter summaries. Invalid proposals still return ``-np.inf``
    so callers see explicit rejections instead of opaque large negative
    sentinels, and verbose progress updates report percentage completion
    for burn-in and production stages. Future engines can adopt the same
    public API to remain plug compatible with the suite.
  - `engines.cosmo_engine_nested` – nested-sampling backend that draws live
    points within declared bounds, replaces the lowest-likelihood point with
    constrained proposals and tracks log-evidence accumulation alongside the
    familiar χ² component breakdown. The result dictionary mirrors the
    structure produced by the MCMC engine while adding nested-specific
    diagnostics so downstream tooling remains backend agnostic.

Plugins are validated through ``engine_interface.validate_plugin``—a thin
wrapper around :func:`copernican_lib.plugins.validate_plugin`—before use.
Chi-squared helpers assume this step has already succeeded, so validation
should occur once before any iterative evaluation begins. Engines expect the
attributes listed in ``copernican_lib.plugins.REQUIRED_ATTRIBUTES``. The
resulting :class:`EnginePlugin` exposes distance functions, CMB helpers and
initial parameter guesses derived from the model YAML while remaining fully
picklable for multiprocessing workloads.

## Standardised Dataset Format

All data parsers return a ``pandas.DataFrame`` with common columns and
metadata so that engines remain agnostic to the origin of the data.
`copernican_lib/data_loaders.py` reads ``metadata_*.yml`` files located next
to
the dataset tables and attaches the fields via the ``DataFrame.attrs``
dictionary after the parser returns. For supernovae datasets the table
contains
at minimum ``Name``, ``zcmb``, ``mu_obs`` and ``e_mu_obs``. Attributes such as
``covariance_matrix_inv`` and ``diag_errors_for_plot`` are also attached. BAO
and
CMB loaders follow the same pattern. New datasets can therefore be added
simply
by placing them under ``data/<type>/<source>/`` and providing a compatible
YAML
parser.

## Extending the API

Third-party tools may import these modules directly. A typical scripting
session looks like this:

```python
from copernican_lib import (
    model_parser, model_coder, engine_interface, data_loaders
)
import engines.cosmo_engine_mcmc as engine

cache = model_parser.parse_model(
    'models/cosmo_model_lcdm.yml', 'models/cache'
)
funcs, parsed = model_coder.generate_callables(cache)
plugin = engine_interface.build_plugin(parsed, funcs)
sne = data_loaders.load_sne_data('jla_2014')
result = engine.fit_sne_parameters(sne, plugin, burn_in_steps=20)
```

Because the API is intentionally thin, advanced users can orchestrate custom
pipelines or integrate the suite into larger optimisation frameworks without
relying on the command-line wrapper.

## Parameter Summary Format

The :mod:`result_writer` helper stores parameter estimates after optimisation
or sampling.  Files named ``parameter-summary_<timestamp>.json`` and ``.yml``
are created in the current run directory.  Each model entry contains
``parameters``, ``errors_1sigma`` and ``covariance_matrix`` with
``param_names`` and a numeric matrix.  When results originate from the MCMC
engine the
summary also records the burn-in length, production steps, posterior means,
log-probability arrays and the chi-squared value associated with the maximum
posterior sample.  The data is fully serialisable so external analysis tools
can parse it without importing NumPy or pandas.

Example::

    from copernican_lib import result_writer
    summary = {"LCDM": engine_results}
    result_writer.save_summary(summary, "output/run")
