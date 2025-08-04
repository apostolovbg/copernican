# Copernican Suite API Overview

The suite exposes a lightweight API intended for advanced scripting.
Most functionality lives in the ``copernican_lib`` package which can be
imported directly without using the command-line interface.  The core
modules are:

- `model_parser.parse_model(path, cache_dir)` – validate and clean a
  `cosmo_model_*.yml` file.
- `model_coder.generate_callables(clean_path)` – compile sanitized model YAML
  into Python callables.
- `engine_interface.build_plugin(parsed_data, funcs)` – construct a plugin object
  with attributes `MODEL_NAME`, `MODEL_DESCRIPTION`, `MODEL_ABSTRACT` and the
  distance and CMB functions required by engines.
  - `data_loaders.load_sne_data(name)`, `load_bao_data(name)`,
    `load_cmb_data(name)` – load datasets by their registered names. Each loader
    logs a short summary describing the dataset and whether its covariance matrix
    was used or diagonal errors were applied.
- `console_output.write(msg)` – unified console printing function that is logged
  verbatim via `logger`.
- `engines.cosmo_engine_comb` – reference engine providing high level
  optimisation routines such as ``fit_sne_parameters``,
  ``fit_combined_parameters``, ``calculate_bao_observables`` and generic
  ``chi_squared_*`` helpers.  Engines are regular Python modules that
  operate purely on data frames and plugin callables so alternative
  backends can be developed without modifying the rest of the codebase.

Plugins are validated through ``engine_interface.validate_plugin`` before
use. Engines expect the attributes listed in
``engine_interface.REQUIRED_ATTRIBUTES``.  The resulting object exposes
distance functions, CMB helpers and initial parameter guesses derived
from the model YAML.

## Standardised Dataset Format

All data parsers return a ``pandas.DataFrame`` with common columns and
metadata so that engines remain agnostic to the origin of the data.
`copernican_lib/data_loaders.py` reads ``metadata_*.yml`` files located next to
the dataset tables and attaches the fields via the ``DataFrame.attrs``
dictionary after the parser returns. For supernovae datasets the table contains
at minimum ``Name``, ``zcmb``,
``mu_obs`` and ``e_mu_obs``. Attributes such as ``covariance_matrix_inv``
and ``diag_errors_for_plot`` are also attached. BAO and CMB loaders
follow the same pattern. New datasets can therefore be added simply by
placing them under ``data/<type>/<source>/`` and providing a compatible
YAML parser.
