# Orchestration Services

GUI clients, launcher scripts, and any future frontends should reuse the shared
orchestration helpers rather than re-implementing the CLI menu logic. The core
services live in `copernican_lib/orchestration.py` and cover validation,
manifest generation, run control, and logging so every entrypoint can call the
same helpers in the documented order.

## Service Map

1. **Configuration validation** (`copernican_lib.model_spec_validator`) –
   validates `cosmo_model_*.yml`, strips unknown keys, caches the sanitized YAML
   copy, and ensures every parameter has an explicit `type`. GUI builders call
   this before populating the manifest, and CLI runs always validate the chosen
   model before sampling.
2. **Manifest generation** (`copernican_lib.run_manifest`) – assembles the
   selected seed, model, dataset IDs, dataset file digests, engine plugin data,
   run settings, plan notes, Git state, and environment hints into
   `run_manifest_*.yml`. The builder writes the manifest to
   `output/copernican_run_NEW_CONFIG/`, saves `run_manifest_NEW_CONFIG.yml`,
   and once the run starts renames the workspace to `copernican-run_<timestamp>`.
   The manifest also records whether the GUI triggered the run, the parsed engine
   name, and every dataset digest so replays reproduce the same inputs.
3. **Run control** (`copernican_lib.run_executor.execute_run_from_manifest`) –
   loads the manifest, rebuilds dataset loaders, reconstructs the plugin, sets up
   logging (via `copernican_lib.logger`), and invokes the actual engine (MCMC,
   nested, or a future backend). It reuses the shared `copernican_lib.run_pipeline`
   so Stage 2 progress bars, diagnostics, and result writers are consistent for
   both CLI and GUI runs.

Any GUI or launcher that needs to start a run should instantiate
`orchestration.InProcessRunController`, supply the `RunRequest`, and listen for
`RunHandle` updates. Those dataclasses specify the minimum payloads required for
status tracking (progress, logs, widget enablement) and allow the GUI to cancel,
pause, or hard stop the worker.

## Manifest Structure

The manifest bundles:

- **Selections**: `seed`, `model_filename`, `datasets` (with `dataset_id`, path,
  metadata, digests), `engine`, and `plan_notes`.
- **Run Settings**: walkers, burn-in, production steps, pool size, diagnostics
  filters, and environment hints such as `COPERNICAN_STRICT_WARNINGS`.
- **Provenance**: git branch, commit hash, `COPERNICAN_VERSION` overrides,
  `python_version`, `operating_system`, `cpu_info`, `copernican_lib` version,
  and the dataset parser hashes.
- **Outputs**: `output_dir`, `manifest_name`, `log_path`, and boolean flags
  describing whether the GUI or CLI initiated the run.

The manifest is consumed by `copernican_lib.run_executor`, `copernican_lib/plotter`,
and `copernican_lib/gui/run_worker`. If `copernican_lib.version.get_version`
spins up, the manifest remains the authoritative record that ties a run to the
package release, dataset digests, and environment hints.

## Logging and Diagnostics

`copernican_lib.logger.setup_logging` configures file and console loggers, patches
`print`/`input`, and records `faulthandler` output and SIGILL/SEGV/FPE stack
traces before exiting when signals occur. It logs the Python version, OS,
CPU model, installed critical package versions, and the manifest summary.
`copernican_lib.console_output` wraps all terminal I/O so the GUI diagnostics
panel and CLI logs can stay synchronized.

Progress messages flush to stdout so long-running computations display
activity on Linux terminals. Warnings are forwarded to the central logger
unless `COPERNICAN_STRICT_WARNINGS=1` is set, in which case warnings become
errors and the run terminates fast during CI.

## Reuse Guidelines

- GUIs should call `copernican_lib.orchestration.InProcessRunController` rather
  than reimplement entire menus; the controller exposes `start`, `pause`, `resume`,
  `cancel`, and `hard_stop` hooks that delegate to `run_executor.execute_run_from_manifest`.
- The GUI worker sets `COPERNICAN_ALLOW_DIRECT=1` before invoking
  `copernican.main --manifest` so any helper importing `copernican` behaves
  identically to CLI manifests.
- Start scripts should point to `copernican.py --gui` or `--cli` and rely on
  `copernican_lib/orchestration` for manifest building, not local copies of the
  CLI menus. When the staged menu is needed for CI coverage, pass
  `COPERNICAN_ENABLE_STAGED_MENU=1` or `--enable-legacy-stage-menu`.

Following this service map preserves the single orchestrator approach and keeps
future frontends from diverging from the canonical data/model/engine pipeline.
