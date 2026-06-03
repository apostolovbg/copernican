# Copernican API Overview

Copernican exposes a lightweight API intended for advanced scripting. Most
functionality lives in the ``copernican.lib`` package which can be imported
directly without using the command-line interface.  The core modules are:

- `model_spec_validator.validate_and_cache_model(path, cache_dir)` – validate
  and clean a `cosmo_model_*.yml` file.
- `model_coder.generate_callables(clean_path)` – compile sanitized model YAML
  into Python callables.
- `engine_adapter.build_plugin(parsed_data, funcs)` – construct an
  :class:`copernican.lib.engine_adapter.EnginePlugin` instance with dataset
  toggles, priors, bounds, distance functions and structured CAMB background
  and perturbation contracts ready for engine consumption.
- `copernican.lib.engine_adapter` – home of the picklable adapter dataclass,
  `EnginePlugin.CMB_CONTRACT`, `EnginePlugin.CMB_PERTURBATION_CONTRACT`,
  `EnginePlugin.CMB_PERTURBATION_STANDARD`, `EnginePlugin.CMB_PERTURBATION_IR`,
  `REQUIRED_ATTRIBUTES` and `REQUIRED_FUNCTIONS`. Import it when building
  custom tooling that needs to confirm interface compliance.
- `copernican.lib.progress` – shared progress reporting helpers. Engines import
  `BatchProgressBar` so CLI runs log simple counters such as “Burn-in stage
  batch 1: 3/200 steps completed (1%)” while still emitting the structured
  ``batch_start``, ``progress_update`` and ``batch_finish`` records that power
  the GUI progress monitors. The helper keeps stage metadata and the listener
  contract unchanged so every backend can report progress without depending on
  carriage-return renderers or spinner pumps.
- `copernican.lib.plotter.plot_corner(samples, plugin, data_attrs, plot_dir)` –
  render the Stage 2 posterior as an automatically thinned corner plot whose
  KDE/contour grid and marginals are now produced by ArviZ while the suite
  retains the responsive panel sizing, footers, and layout safeguards that keep
  metadata away from the axes. The additional footer line documents that the
  ArviZ backend generated the densities so automated workflows know when the
  helper relied on the shared plotting stack. The underlying
  `_prepare_corner_inputs` validator still flattens samples, derives thinning
  statistics and feeds `build_footer_lines`, keeping the `legacy` shim in place
  for earlier automation.
- `copernican.lib.plotter.plot_parameter_histograms(samples, plugin,
  data_attrs, plot_dir)` – generate a grid of per-parameter histograms rendered
  by ArviZ, complete with neutral info boxes, dataset-aware footers and
  quantile annotations so the GUI viewer can reuse the same assets. The helper
  uses `_prepare_corner_inputs` to thin the samples, lists the effective
  parameter names, and records how many finite draws survived before drawing
  the histograms so the exported files remain audit-friendly.
- `copernican.lib.posterior` – exposes
  :func:`copernican.lib.posterior.make_logposterior`, which now returns a
  picklable :class:`PosteriorEvaluator` combining priors, transforms and
  likelihood callables. Engines should always route posterior evaluations
  through this helper to keep multiprocessing safe.
- `copernican.lib.statistics` – shared chi-squared and BAO/CMB helper functions
  used by every engine.  Importing from this module keeps the numerical
  implementations in a single place so engines remain thin orchestration
  layers. The helpers expose SNe chi-squared evaluations that always return
  finite values for physically meaningful proposals so MCMC reseeding can fall
  back to them reliably. CI runners that lack CAMB can opt into
  ``COPERNICAN_FAKE_CMB=1`` so the CMB helpers return deterministic synthetic
  spectra instead of performing heavy physics evaluations, leaving production
  calculations untouched. Structured CMB contracts are required in the
  production path; the explicit legacy helper is reserved for tests.
  - `dataset_registry.load_sne_data(dataset_id)`, `load_bao_data(dataset_id)`,
    `load_cmb_data(dataset_id)` – load datasets by their identifiers. The
    interactive prompt lists the human readable `dataset_name` and description,
    but calls expect the `dataset_id`. Each loader logs a short summary
    describing the dataset and whether its covariance matrix was used or
    diagonal errors were applied.
- `console_output.write(msg)` – unified console printing function that is
  logged verbatim via `logger`.
- `console_output.ask(prompt)` – input helper that records prompts and
  responses in the run log.
- `logger.setup_logging(log_dir)` – initialise logging and patch
  `print`/`input` so all interactions are captured.
- `utils.get_timestamp(now=None)` – return a `YYYYMMDD_HHMMSS` string in
  Coordinated Universal Time for consistent filenames and manifests. The helper
  underpins logging, result writers and manifest builders so outputs from CI
  and local runs align chronologically.
- `chain_io.save_posterior(chain, param_names, path, metadata)` – store
  posterior samples in NetCDF format using ArviZ, or xarray when the dependency
  is unavailable during lightweight tests. Metadata is stamped on both the
  InferenceData root and the posterior group so callers opening just the
  posterior block still see the model, dataset and other provenance details.
- `csv_writer.save_sne_results_detailed_csv`, `save_bao_results_csv` and
  `save_cmb_results_csv` – persist fitting results with filenames that encode
  the dataset, model and timestamp.

- `copernican.engines.cosmo_engine_mcmc.fit_cosmology_parameters` –
   returns a dictionary with posterior samples, joint chi-squared
   diagnostics for the SNe/BAO/CMB components, dataset-level point counts,
   burn-in length, acceptance fractions and a sanitised log-probability
   trace. BAO and CMB data frames can be passed via the `bao_data_df` and
   `cmb_data_df` keyword arguments to enable joint sampling in a single
   call. ``burn_in_steps`` overrides the default ``max(100, n_steps // 5)``
   warm-up, keeping scripted workflows nimble, and the ``pool_size``
   keyword enforces user-selected multiprocessing pools while
   automatically expanding the walker ensemble to keep every worker busy.
   The private `_reseed_invalid_walkers` utility reseeds walkers that emit
   `nan` coordinates after burn-in so downstream API consumers never need
   to handle undefined sampler states. When the CLI selects this backend,
   Stage 2 prompts for production steps, burn-in length, walker counts and
   worker pools, mirroring the available function arguments for scripted
   workflows. A legacy ``fit_sne_parameters`` alias remains for backward
   compatibility but now logs a deprecation warning.
- `copernican.engines.cosmo_engine_nested.fit_cosmology_parameters` –
   wraps a lightweight nested-sampling routine that evaluates the same
   adapter-provided posterior while reporting log-evidence estimates,
   live-point counts, enlargement factors and iteration diagnostics. The
   CLI surfaces backend-specific prompts for live points, evidence
   tolerances and enlargement fractions so interactive runs align with
   scripted calls that specify the same keyword arguments. The legacy
   ``fit_sne_parameters`` name still resolves to this function but is
   deprecated.
- `result_writer.save_summary(results, output_dir)` – serialize fitted
  parameters, 1σ errors, covariance matrices and the recorded sampling
  configuration—including nested-sampling metadata such as live-point counts
  and evidence tolerances—to JSON and YAML for later analysis.
  - `copernican.engines.cosmo_engine_mcmc` – lightweight `emcee`
    sampler for SNe posteriors. Walkers are initialised uniformly within
    declared parameter bounds, a burn-in run precedes production
    sampling and the returned dictionary includes log-probability
    traces, acceptance fractions, estimated autocorrelation times when
    the production chain is long enough and both MAP and posterior mean
    parameter summaries. Invalid proposals still return ``-np.inf`` so
    callers see explicit rejections instead of opaque large negative
    sentinels, and verbose progress updates report percentage completion
    for burn-in and production stages. Future engines can adopt the
    same public API to remain plug compatible with the suite.
  - `copernican.engines.cosmo_engine_nested` – nested-sampling backend
    that draws live points within declared bounds, replaces the lowest-
    likelihood point with constrained proposals and tracks log-evidence
    accumulation alongside the familiar χ² component breakdown. The
    result dictionary mirrors the structure produced by the MCMC engine
    while adding nested-specific diagnostics so downstream tooling
    remains backend agnostic.

Engine adapters are validated through ``engine_adapter.validate_plugin``—a
thin wrapper around
:func:`copernican.lib.engine_adapter.validate_plugin`—before use. Chi-squared
helpers assume this step has already succeeded, so validation should occur
once before any iterative evaluation begins. Engines expect the attributes
listed in ``copernican.lib.engine_adapter.REQUIRED_ATTRIBUTES``. The resulting
:class:`EnginePlugin` exposes distance functions, CMB helpers, initial
parameter guesses, the structured CAMB contract derived from the model YAML
and the compiled perturbation IR while remaining fully picklable for
multiprocessing workloads.

## Table of Contents

- [Standardised Dataset Format](#standardised-dataset-format)
- [Extending the API](#extending-the-api)
- [Parameter Summary Format](#parameter-summary-format)
- [Run Analysis Helpers](#run-analysis-helpers)
- [Posterior Explorer](#posterior-explorer)
- [Run Comparison Helpers](#run-comparison-helpers)

## Standardised Dataset Format

All data parsers return a ``pandas.DataFrame`` with common columns and metadata
so that engines remain agnostic to the origin of the data.
`copernican/lib/dataset_registry.py` reads ``metadata_*.yml`` files located
next to the dataset tables and attaches the fields via the ``DataFrame.attrs``
dictionary after the parser returns. For supernovae datasets the table
contains at minimum ``Name``, ``zcmb``, ``mu_obs`` and ``e_mu_obs``.
Attributes such as ``covariance_matrix_inv`` and ``diag_errors_for_plot`` are
also attached. BAO and CMB loaders follow the same pattern. New datasets can
therefore be added simply by placing them under
``copernican/datasets/<type>/<source>/`` and providing a compatible YAML
parser.

## Extending the API

Third-party tools may import these modules directly. A typical scripting
session looks like this:

```python
from copernican.lib import dataset_registry, engine_adapter, model_coder
from copernican.lib import model_spec_validator
import copernican.engines.cosmo_engine_mcmc as engine

cache = model_spec_validator.validate_and_cache_model(
    "copernican/models/cosmo_model_ref_planck2018.yml",
    "copernican/models/cache",
)
funcs, parsed = model_coder.generate_callables(cache)
plugin = engine_adapter.build_plugin(parsed, funcs)
sne = dataset_registry.load_sne_data('jla_2014')
result = engine.fit_cosmology_parameters(sne, plugin, burn_in_steps=20)
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
engine the summary also records the burn-in length, production steps, posterior
means, log-probability arrays and the chi-squared value associated with the
maximum posterior sample.  The data is fully serialisable so external analysis
tools can parse it without importing NumPy or pandas.

Example::

    from copernican.lib import result_writer

    summary = {"ReferenceModel": engine_results}
    result_writer.save_summary(summary, "output/run")

## Run Analysis Helpers

The new :mod:`copernican.lib.analysis` module inspects an existing run
directory, reads the latest manifest/parameter summary, scans the generated
log, and assembles a structured
:class:`copernican.lib.analysis.RunAnalysisResult`.  The summary includes run
timing, diagnostics such as R-hat and ESS, dataset counts, and per-model chi-
squared plus BAO/CMB residual metadata so tools can report consistent tables or
JSON blobs without re-parsing log files manually.

Calling :func:`copernican.lib.analysis.analyze_run` on ``output/copernican-
run_...`` returns the dataclass, while :meth:`RunAnalysisResult.to_dict`
produces a serialisable representation for downstream APIs:

```python
from copernican.lib import analysis

result = analysis.analyze_run("output/copernican-run_20251207_200254")
print(result.model_summaries["ReferenceModel"].chi2["chi2_total"])
print(result.duration_seconds)
```

For workflows that need files rather than dataclasses, use
``analysis.save_run_summary`` to persist the structured result:

```python
from pathlib import Path
from copernican.lib import analysis

summary_paths = analysis.save_run_summary(
    Path("output/copernican-run_20251207_200254"),
    Path("reports/analysis"),
    formats=("yml", "json"),
)
print(summary_paths["yml"])  # analysis-summary_20251207_200254.yml
```

Each file contains the mapped fields from :class:`RunAnalysisResult`, including
datasets, diagnostics, ``model_summaries`` and links back to the manifest/log.

To regenerate the visual diagnostics, call ``analysis.plot_posterior(run_dir,
output_dir, kinds=("overview", "corner", "histograms"))``. The helper reads the
archived ``posterior-*.nc`` snapshots, builds an ArviZ-powered corner grid,
per-parameter histograms, and the compact trace/ histogram overview used inside
the GUI, and returns the written file paths so scripts can log or publish the
assets without needing to replicate the GUI plumbing.

```python
from copernican.lib import analysis

saved_paths = analysis.plot_posterior(
    Path("output/copernican-run_20251207_200254"),
    Path("reports/plots"),
    kinds=("overview", "corner", "histograms"),
)
print(saved_paths["corner"])
```

## Posterior Explorer

Posterior summaries rely on :mod:`copernican.lib.posterior_explorer` to locate
the ``posterior-*.nc`` snapshots inside a run directory and build a compact
trace/hist figure that reuses the shared analysis metadata. Use
``posterior_explorer.find_posterior_files(run_dir)`` to enumerate the NetCDF
files and ``posterior_explorer.create_posterior_overview_figure(result,
posterior_path)`` to draw the plot that appears inside the GUI’s Analysis
workspace. The GUI also exposes the plot through
:class:`copernican.lib.gui.plot_viewer.PlotViewer`, but you can use the same
helpers directly in scripts or notebooks:

```python
from pathlib import Path

from copernican.lib import analysis, posterior_explorer

run_dir = Path("output/copernican-run_20251207_200254")
result = analysis.analyze_run(run_dir)
posterior_files = posterior_explorer.find_posterior_files(run_dir)
if posterior_files:
    figure = posterior_explorer.create_posterior_overview_figure(
        result, posterior_files[0]
    )
    figure.savefig("reports/posterior_overview.png")
```

## Run Comparison Helpers

Comparisons between two runs reuse the same run analysis helpers to produce
consistent summary files and delta tables. Use
``analysis.compare_runs(base_result, alt_result)`` to compute a JSON-friendly
summary that includes descriptors for both runs, `duration_seconds` deltas,
dataset row-count differences, and per-model χ²/parameter comparisons.
``analysis.compare_run_dirs(base_dir, alt_dir)`` shortcuts the analysis step by
loading each output directory before calling ``compare_runs``. When you need a
file rather than an in-memory dictionary, call
``analysis.save_comparison_summary`` to persist the comparison as ``analysis-
comparison_<timestamp>.yml/json`` using the same timestamping helper that run
summaries rely on.
