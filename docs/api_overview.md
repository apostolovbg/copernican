# Copernican Suite API Overview

The suite exposes a small API intended for advanced scripting. The
`copernican_lib` package contains the following key modules:

- `model_parser.parse_model_json(path, cache_dir)` – validate and clean a
  `cosmo_model_*.json` file.
- `model_coder.generate_callables(clean_path)` – compile sanitized model JSON
  into Python callables.
- `engine_interface.build_plugin(parsed_json, funcs)` – construct a plugin object
  with attributes `MODEL_NAME`, `MODEL_DESCRIPTION`, `MODEL_ABSTRACT` and the
  distance and CMB functions required by engines.
- `data_loaders.load_sne_data(name)`, `load_bao_data(name)`,
  `load_cmb_data(name)` – load datasets by their registered names.
- `engines.cosmo_engine_comb` – reference engine providing
  `fit_sne_parameters`, `fit_combined_parameters`, `calculate_bao_observables`
  and `chi_squared_*` helpers.

Plugins are validated through `engine_interface.validate_plugin` before use.
Engines expect the attributes listed in `engine_interface.REQUIRED_ATTRIBUTES`.

## Standardised Dataset Format

All data parsers return a `pandas.DataFrame` with common columns and
metadata so that the engines remain agnostic to the origin of the data.
For supernovae datasets the table contains at minimum `Name`, `zcmb`,
`mu_obs` and `e_mu_obs`.  Attributes such as `covariance_matrix_inv`,
`diag_errors_for_plot` and `dataset_name_attr` are attached via the
``DataFrame.attrs`` dictionary.  BAO and CMB loaders follow the same
pattern.  New datasets can therefore be added simply by placing them
under ``data/<type>/<source>/`` and providing a compatible parser.
