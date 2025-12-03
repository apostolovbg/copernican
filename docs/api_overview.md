# Copernican Suite API Overview

This document complements [`docs/architecture.md`](docs/architecture.md) by
describing the runnable units most users import when scripting runs outside the
CLI or GUI. The API is intentionally thin: it strings together the same
validation → manifest → executor flow that the interactive shell follows and
exposes enough utilities so tooling authors can reuse dataset loaders,
sampling helpers, and result writers without duplicating orchestration logic.

## Core Pipeline Helpers

- `copernican_lib.model_spec_validator.validate_and_cache_model(path, cache_dir)`
  – validates a `cosmo_model_*.yml`, sanitizes unknown keys, enforces that every
  parameter declares a `type`, injects `type: fixed` when the bounds collapse,
  and writes a timestamped cache file under `models/cache/`.
- `copernican_lib.model_coder.generate_callables(cache_path)` – compiles the
  sanitized YAML into NumPy-ready callables for distances, Hubble rate,
  nuisances, and any model-specific helper functions. The code generator ensures
  every expression is enumerated in LaTeX, never raw Python, mirroring the
  constraints enforced by `model_spec_validator`.
- `copernican_lib.engine_plugin_validation.build_plugin(parsed_yaml, funcs)`
  – produces the picklable `EnginePlugin` dataclass that describes priors,
  transforms, bounds, and dataset toggles (`valid_for_distance_metrics`,
  `valid_for_bao`, `valid_for_cmb`). Callers that need to inspect the plugin for
  diagnostics (e.g., GUIs or manifest checks) should use the same helper to stay
  aligned with `copernican_lib.plugins.REQUIRED_ATTRIBUTES`.
- `copernican_lib.plugins.validate_plugin(plugin)` – re-validates a plugin
  instance before any sampling begins. Engines use this to double-check
  configuration when rerunning manifests, while GUI helpers rely on the same
  call to keep the front-end responsive to invalid selections.

Every pipeline builder returns a picklable object so multiprocessing workers can
initialise their state without re-reading YAML files. The helper flow is
captured in `docs/architecture.md`, which expands on how CLI, GUI, and detached
runners share `copernican_lib/run_executor.execute_run_from_manifest`.

## Dataset and Registry Access

- `copernican_lib.dataset_registry.load_sne_data(dataset_id)` (similarly for
  BAO/CMB) – returns a `pandas.DataFrame` plus `.attrs` metadata that includes
  `dataset_name`, `citation`, `dataset_version`, `file_hashes`, and any pre-
  computed inverse covariance matrices. The loaders compute SHA256 digests for
  every non-parser file in the dataset directory so manifests can pinpoint the
  exact inputs that generated a result.
- `copernican_lib.dataset_registry.register_parser(path, dataset_id)` – used by
  each `cosmo_parser_*.py` to register itself with the central registry. Parsers
  must be hashed (`copernican_lib.dataset_registry.TRUSTED_PARSER_DIGESTS`) and
  they only load when the hash matches the trusted list, preventing tampering.
- `copernican_lib.dataset_registry.parse_metadata(path)` – loads
  `metadata_*.yml` files to attach `description`, `license`, BibTeX fields,
  `independence_assumptions`, and the canonical `dataset_id`.

Custom datasets simply need a new folder under `data/<type>/<source>/`, a
compatible parser, and metadata plus digests; see
[`docs/data_overview.md`](docs/data_overview.md) for a write-up of every bundled
dataset and the required metadata fields.

## Progress, Logging, and Console Utilities

- `copernican_lib.progress` – houses `BatchProgressBar`, `StepProgressEmitter`,
  and `configure_sampler_progress_reporting`, ensuring every engine reports
  burn-in/production percentages, walker-level motion, and spinner glyphs that
  the GUI reuses for the Run Monitor. The renderer uses carriage-returns so
  Linux/macOS/Windows terminals repaint the last active line, matching the
  behaviour introduced in version 7.6.14.
- `copernican_lib.console_output.write`, `console_output.ask`, and
  `logger.setup_logging` – unify console printing, prompt logging, and handler
  wiring so every prompt/print goes through the central logger. `faulthandler`
  plus SIGILL/SEGV/FPE handlers dump traces to both console and log file before
  exiting.
- `copernican_lib.utils.get_timestamp(now=None)` – returns a UTC
  `YYYYMMDD_HHMMSS` string used by run directories, logs, manifests, and plot
  footers so multiple machines and CI runners stay aligned.

## Engines and Result Writers

- `engines.cosmo_engine_mcmc.fit_cosmology_parameters(...)` – the default MCMC
  engine. Callers can pass `bao_data_df` and `cmb_data_df` to reuse pre-loaded
  datasets, enabling joint likelihood evaluations with shared wrappers such as
  `copernican_lib.statistics`. Invalid proposals return `-np.inf`, the walker
  pool grows automatically to fill `pool_size`, and `_reseed_invalid_walkers`
  reseeds `nan` coordinates after burn-in using jitter around the ensemble mean.
- `engines.cosmo_engine_nested.fit_cosmology_parameters(...)` – wraps nested
  sampling with live point controls, enlargement factors, and log-evidence
  diagnostics. The API mirrors the MCMC result dictionary while adding nested
  specific fields so downstream tooling can remain agnostic to the backend.
- `result_writer.save_summary(results, output_dir)` – serializes sampler outputs
  (JSON/YAML) with parameter summaries, covariance matrices, run settings,
  chi-squared breakdowns, and metadata such as walker counts and evidence
  tolerances.
- `chain_io.save_posterior(chain, param_names, path, metadata)` – writes NetCDF
  files with metadata stored in both the inference-data root and posterior group
  so any tool opening only the posterior block still recovers model/dataset
  identifiers.
- `copernican_lib.csv_writer` exports the SNe/BAO/CMB final tables with
  descriptive filenames that embed the dataset name, model, and timestamp.

## Scripting a Run

```python
from copernican_lib import (
    model_spec_validator, model_coder, engine_plugin_validation, dataset_registry,
    result_writer
)
import engines.cosmo_engine_mcmc as engine

cache = model_spec_validator.validate_and_cache_model(
    "models/cosmo_model_lcdm.yml", "models/cache"
)
funcs, parsed = model_coder.generate_callables(cache)
plugin = engine_plugin_validation.build_plugin(parsed, funcs)
sne = dataset_registry.load_sne_data("pantheon")

results = engine.fit_cosmology_parameters(
    sne,
    plugin,
    burn_in_steps=40,
    production_steps=200,
    pool_size=8,
    bao_data_df=dataset_registry.load_bao_data("bossdr12"),
    cmb_data_df=dataset_registry.load_cmb_data("planck2018lite"),
)

result_writer.save_summary({"LCDM": results}, "output/copernican-run_custom")
```

The example mirrors the CLI manifest steps: validate → encode → plugin →
datasets → engine → results. Scripting mode is ideal for notebooks and testing
frameworks that require fine-grained control over each stage while still
relying on the suite’s shared helpers.

## Parameter Summary Format

`result_writer` also saves `parameter-summary_<timestamp>.json/.yml`. Each entry
contains `parameters`, `errors_1sigma`, `covariance_matrix`, `burn_in`, and
`production_steps` for MCMC runs; nested samplers append their `live_point`
details and `evidence` traces. The files are JSON-friendly, so even tools that
avoid NumPy/pandas can parse them directly for report generation.
