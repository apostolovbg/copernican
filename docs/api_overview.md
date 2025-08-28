# Copernican Suite API Overview

**Last Updated:** 2025-08-28

The suite exposes a lightweight API intended for advanced scripting.
Most functionality lives in the ``copernican_lib`` package which can be
imported directly without using the command-line interface.  The core
modules are:

- `model_parser.parse_model(path, cache_dir)` – validate and clean a
  `cosmo_model_*.yml` file.
- `model_coder.generate_callables(clean_path)` – compile sanitized model YAML
  into Python callables.
- `engine_interface.build_plugin(parsed_data, funcs)` – construct a plugin
  object
  with attributes `MODEL_NAME`, `MODEL_DESCRIPTION`, `MODEL_ABSTRACT` and the
  distance and CMB functions required by engines.
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
- `chain_io.save_posterior(chain, param_names, path, metadata)` – store
  posterior samples in NetCDF format using ArviZ.
- `csv_writer.save_sne_results_detailed_csv`,
  `save_bao_results_csv` and `save_cmb_results_csv` – persist fitting
  results with filenames that encode the dataset, model and timestamp.
- `result_writer.save_summary(results, output_dir)` – serialize fitted
  parameters, 1σ errors and covariance matrices to JSON and YAML for later
  analysis.
  - `engines.cosmo_engine_comb` – reference engine providing high level
    optimisation routines such as ``fit_sne_parameters``,
    ``fit_combined_parameters``, ``calculate_bao_observables`` and generic
    ``chi_squared_*`` helpers. ``chi_squared_bao`` accepts arrays
    ``(z, observable_type, value, error)`` and an optional inverse
    covariance so repeated evaluations can reuse cached data. The engine
    evaluates SNe, BAO and CMB chi-squared terms concurrently when several
    datasets are provided. ``fit_combined_parameters`` accepts optional
    ``maxiter``, ``maxfun`` and ``prefit_maxiter`` arguments to bound
    optimisation time. Engines are regular Python modules that operate
    purely on data frames and plugin callables so alternative backends can
    be developed without modifying the rest of the codebase.
  - `engines.cosmo_engine_mcmc` – lightweight `emcee` sampler for SNe
    posteriors. Chi-squared helpers are re-exported so downstream code can
    reuse the same calculations as the combined engine.

Plugins are validated through ``engine_interface.validate_plugin`` before
use. Chi-squared helpers assume this step has already succeeded, so
validation should occur once before any iterative evaluation begins.
Engines expect the attributes listed in
``engine_interface.REQUIRED_ATTRIBUTES``.  The resulting object exposes
distance functions, CMB helpers and initial parameter guesses derived
from the model YAML.

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
import engines.cosmo_engine_comb as engine

cache = model_parser.parse_model(
    'models/cosmo_model_lcdm.yml', 'models/cache'
)
funcs, parsed = model_coder.generate_callables(cache)
plugin = engine_interface.build_plugin(parsed, funcs)
sne = data_loaders.load_sne_data('jla_2014')
result = engine.fit_sne_parameters(sne, plugin)
```

Because the API is intentionally thin, advanced users can orchestrate custom
pipelines or integrate the suite into larger optimisation frameworks without
relying on the command-line wrapper.

## Parameter Summary Format

The :mod:`result_writer` helper stores parameter estimates after optimisation
or sampling.  Files named ``parameter-summary_<timestamp>.json`` and ``.yml``
are created in the current run directory.  Each model entry contains
``parameters``, ``errors_1sigma`` and ``covariance_matrix`` with ``param_names``
and a numeric matrix.  The data is fully serialisable so external analysis
tools can parse it without importing NumPy or pandas.

Example::

    from copernican_lib import result_writer
    summary = {"LCDM": engine_results}
    result_writer.save_summary(summary, "output/run")
