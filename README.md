**Version:** 11.0.1

# Copernican Suite

The Copernican Suite is a manifest-driven cosmology workbench that evaluates
models against SNe Ia, BAO and CMB data, renders diagnostics, and preserves the
entire run provenance in timestamped workspaces. Every launch passes through
`copernican.py`, validates datasets and parsers, generates a `run_manifest_*`
entry, and stores results under `output/copernican-run_<UTC timestamp>`. The
navigation choices (GUI vs CLI) reuse the same `copernican_lib` orchestration
services so GUI workers, CLI commands and future frontends share the same
plugins, loaders and execution flow.

## Table of Contents

1. [Purpose and Evolution](#purpose-and-evolution)
2. [Quick Start and Launchers](#quick-start-and-launchers)
3. [Architecture at a Glance](#architecture-at-a-glance)
4. [Command-Line Workflows](#command-line-workflows)
5. [Data, Models, and Datasets](#data-models-and-datasets)
6. [Plugins, Engines, and the API](#plugins-engines-and-the-api)
7. [GUI, Help, and Orchestration](#gui-help-and-orchestration)
8. [Developer Workflow and Policies](#developer-workflow-and-policies)
9. [Further Reading](#further-reading)

## Purpose and Evolution

The repository unifies CLI, GUI and backend engines within a single,
policy-enforced workspace. Since version 10.6.0 the focus has been on removing
legacy stage menus, refactoring manifests, and making GUI and CLI runs
identical. Highlights since then include:

- Manifest-first orchestration: `copernican.py` now builds per-run manifests
  (seed, model, dataset, engine, plan metadata) that every worker reads so GUI
  builders and CLI scripts stay in sync when a manifest is replayed.
- Dataset provenance: loaders now compute SHA256 digests for every non-parser
  file, store the hashes on the returned `DataFrame`, and copy them verbatim
  into the manifests. Parsers must register under their `dataset_id` so the
  loaders find the correct helper without discovery.
- GUI modernization: the navigation rail was rebuilt with a 240 px reserved
  column for the Copernican icon, single‑selection model/dataset lists, and a
  Live Run Monitor that mirrors CLI progress while respecting the headless
  renderer fallback.
- Policy automation: DevCovenant scripts now guard documentation, changelog
  coverage, line lengths, and test requirements for every new module so the
  suite remains consistent across releases.
- Metadata synchronization: version metadata now appears in `README.md`,
  `pyproject.toml`, `CITATION.cff`, and `copernican_lib/VERSION`; releasing a
  new version requires updating all four files simultaneously.

## Quick Start and Launchers

1. **Acquire the managed Python** – run `start.sh`/`start.command`/`start.bat`. The
   launchers download a vetted Python 3.11 interpreter into `.python`, build
   `.venv`, install pinned dependencies from `requirements.lock`, and configure
   the environment so `pip install --no-deps .` uses that interpreter. If the
   download fails the launcher prints guidance rather than continuing.
2. **Activate `.venv`** – each session must run inside the launcher-created
   virtual environment. The start scripts also rebuild `.venv` when their Python
   drifts outside 3.11 or the activation script disappears.
3. **Choose your interface** – `start.*` prompts for CLI/GUI. CLI runs
   `copernican.py --cli` (with options such as `--manifest` and `--output-dir`)
   and GUI runs `copernican.py --gui` using the shared orchestration service map
   (`copernican_lib/orchestration`).
4. **Stay manifest-first** – every execution writes `output/copernican_run_NEW_CONFIG/run_manifest_NEW_CONFIG.yml` and later renames the workspace to `copernican-run_<UTC timestamp>` once the run launches. `Run Builder` and CLI prompts both edit the manifest before invoking `copernican_lib.run_executor.execute_run_from_manifest`.

CLI-specific options include `--manifest` to point to a saved configuration,
`--output-dir` to override the default workspace, `--no-gui` to force CLI-only
runs and `--enable-legacy-stage-menu` or `COPERNICAN_ENABLE_STAGED_MENU=1` for
CI coverage. GUI launcher scripts detach the process (`pythonw` on Windows,
`nohup` on Unix) so closing the terminal does not kill the window.

## Architecture at a Glance

See [`docs/architecture.md`](docs/architecture.md) for a deep dive, but the
short story is:

- `copernican.py` orchestrates startup, argument parsing, dependency scanning,
  dataset/plugin validation, logging (`faulthandler`, SIG handlers, CPU/package
  snapshots), and runtime configuration before delegating to the shared
  `copernican_lib.run_executor`.
- `copernican_lib` houses dataset loaders, plugin builders, progress helpers,
  result writers, manifest utilities, GUI support, and console output wrappers.
- `models/` contains YAML-based definitions validated by `copernican_lib.model_spec_validator`, cached in `models/cache/`, and converted into picklable engine plugins via `copernican_lib.model_coder`.
- `engines/` bundles pure compute backends (`cosmo_engine_mcmc`, `cosmo_engine_nested`) that import the shared helpers from `copernican_lib` to stay lightweight.
- `data/` holds vetted SNe, BAO and CMB datasets with matching parsers, metadata, and hash digests; new datasets must register via `copernican_lib.dataset_registry`.
- `docs/` contains living references (API, data, GUI, orchestration, packaging, licensing), and `CHANGELOG.md` records every file touched since 10.6.0 to comply with the changelog-coverage policy.

### Manifest and Runtime Flow

The CLI, GUI builder, and any future frontend share:

1. Validate the selected model using `model_spec_validator`.
2. Build picklable callables via `model_coder.generate_callables`.
3. Build an engine plugin through `engine_plugin_validation.build_plugin`.
4. Revalidate datasets and compute digests using `dataset_registry`.
5. Assemble `run_manifest_*` with the selected engine, dataset hashes, Git state, run settings, and logger metadata.
6. Launch the real sampler via `copernican_lib.run_executor.execute_run_from_manifest`.

## Command-Line Workflows

`copernican.py --cli` is the default path. The interactive Stage 2 sampler menu now
asks for production steps, burn-in, walker count, and pool size after loading
the CMB dataset, suggesting safe minima from the models. Walkers are seeded
with `COPERNICAN_SEED` when provided; otherwise Stage 1 prompts for the seed
but allows accepting `0` or generating a random integer. Inputs flow into the
manifest and the log records the final choice for reproducibility.

Running `copernican.py --manifest <file>` rebuilds the same workspace even if the
starter menu is skipped. `copernican_lib/run_manifest.py` resizes metadata
panels, logs the manifest contents, and ensures the CLI, GUI and any detached
runner read the same configuration. `copernican_lib/plotter` reuses the manifest
for footer citations so Stage 5 visuals always match the chosen datasets, and
`copernican_lib.posterior` stores deterministic metadata (model name, dataset
ID, parameter names) inside both the NetCDF root and its posterior group.

Environment helpers flush warnings through the central logger (`console_output`)
and elevate warnings to errors when `COPERNICAN_STRICT_WARNINGS=1`. Before
heavy computation the suite performs a tiny NumPy/SciPy self-check to confirm
CPU instruction compatibility and suggests reinstalling the matched wheels if
the check fails.

## Data, Models, and Datasets

Datasets sit under `data/<type>/<source>/`, each with a parser module such as
`cosmo_parser_jla2014.py` and lifecycle metadata (`metadata_*.yml`). Parsers
register with `copernican_lib.dataset_registry` so the loaders can compute frame
attributes (`dataset_name`, `dataset_id`, `description`, `citation`, `license`,
`file_hashes`, `dataset_version`, `data_path`, `independence_assumptions`) and
attach the inverse covariance matrices the engines need. Parsers are hashed,
and the launcher rejects any parser whose SHA256 digest does not match
`TRUSTED_PARSER_DIGESTS` until the hash entry is updated and recorded in the
changelog.

Refer to [`docs/data_overview.md`](docs/data_overview.md) for a breakdown of every
included dataset, the parser expectations, and the hash verification process.
`docs/dataset_metadata.md` lists the metadata schema and explains how the
registry merges metadata into the returned `DataFrame`. The loaders now compute
table digests on every run and store them directly in the manifest so the CLI
and GUI can reproduce the exact inputs when re-running an analysis.

Models are written purely in YAML (`models/cosmo_model_*.yml`). Each file
declares `model_name`, `version`, `parameters`, `equations`, `abstract`, and a
`description` that doubles as the publication text. Additional fields like
`unit`, `latex_name`, `rs_expression`, compatibility flags (`valid_for_distance_metrics`, `valid_for_bao`, `valid_for_cmb`), and `cmb.param_map` appear as needed while remaining backward-compatible because the schema ignores unknown keys. The `README` and `docs/gui_overview.md` highlight how the GUI exposes metadata viewers and revalidation controls so operators can inspect models without leaving the interface.

## Plugins, Engines, and the API

`copernican_lib.plugins.build_engine_plugin` now constructs picklable dataclasses
describing bounds, priors, transforms, and dataset compatibility, and the
resulting `EnginePlugin` is also validated through `engine_plugin_validation`.
`copernican_lib.posterior.make_logposterior` returns a single `PosteriorEvaluator`
that merges priors, transforms, and likelihood functions so every engine shares
the same prior logic and multiprocessing safety.

The `engines` folder contains compute-only backends. The default
`cosmo_engine_mcmc` exposes a `fit_cosmology_parameters` entry point that
returns posterior samples, χ² breakdowns (SNe, BAO, CMB), dataset point counts,
burn-in lengths, acceptance fractions, and log-probability traces. The nested
sampler mirrors the same API while adding evidence diagnostics. Results seralize
through `result_writer.save_summary` (JSON/YAML) and `chain_io.save_posterior`
(NetCDF with metadata in both root and posterior groups).

Advanced users can script pipelines directly against the API. See
[`docs/api_overview.md`](docs/api_overview.md) for example sessions that mimic
the CLI’s manifest steps, describe shared helpers (`console_output`, `logger`,
`progress`, `statistics`), and document how to reuse dataset loaders or posterity
writers without the command-line wrapper.

## GUI, Help, and Orchestration

The GUI shares the same orchestration services described in
[`docs/orchestration_services.md`](docs/orchestration_services.md) and embeds
`README.md` inside the Help tab so users always have the same reference
material. The `docs/gui_overview.md` file explains the navigation rail, Run
Builder panels, Run Monitor progress bars, metadata dialogs (15/25 line rules,
Open file action), selectors, Run Settings, and the diagnostics console that
mirrors CLI logs while streaming Stdout/Stderr from the child worker. The GUI
locks the builder’s Save Manifest until all selections are made, writes temporary
workspaces (`copernican_run_NEW_CONFIG`), and renames them to
`copernican-run_<timestamp>` once the run is handed over to the worker.

`docs/launcher_gui.md` details how the start scripts detach the GUI, print the
hand-off message, and keep the staged menu disabled unless explicitly requested.
`copernican_lib/gui/run_worker.py` simply loads the JSON configuration, sets
`COPERNICAN_ALLOW_DIRECT=1`, and invokes `copernican.main` with `--manifest` so
GUI workers run the identical CLI workflow. The Run Monitor streams the same
dual progress bars used by the CLI and shows walker-level stats plus spinner
glyphs that match the carriage-return renderer introduced in version 7.6.14.

## Developer Workflow and Policies

Follow the DevCovenant and law directives in `AGENTS.md` before editing. The
README’s law section specifically requires updating both `README.md` and the
corresponding `docs/` page whenever a behaviour change or documentation broaden
is committed. Document changes go into `CHANGELOG.md` under the current version
by listing each touched file. Run `pre-commit run --all-files` at the start of
every session (DevCovenant runs automatically), use `.venv/bin/python devcovenant_check.py check --mode=startup`,
and rerun `devcovenant_check.py update-hashes` after updating policy scripts so
hashes stay in sync. Unit tests (preferred via `.venv/bin/python -m pytest`)
should run inside the managed `.venv`.

The `docs/documentation_policy.md` file lists formatting expectations for new
pages. Keep Markdown lines readable, images stored under `docs/`, and `Last
Updated` markers on allowlisted files only. The data directory remains
read-only except for parser or metadata updates, and any dataset hash change
must be documented in the changelog plus `docs/data_overview.md`.

## Further Reading

- [`docs/architecture.md`](docs/architecture.md) – Full component and data flow
  descriptions with diagrams for manifests, logging, and run directories.
- [`docs/api_overview.md`](docs/api_overview.md) – Expanded API reference for
  plugins, samplers, writers, and utilities.
- [`docs/data_overview.md`](docs/data_overview.md) – Detailed dataset listings,
  metadata schema, parser guidelines, and hash verification steps.
- [`docs/gui_overview.md`](docs/gui_overview.md) – GUI controls, Run Monitor,
  metadata dialogs, and launch behavior.
- [`docs/orchestration_services.md`](docs/orchestration_services.md) – Service map
  for orchestrators and the manifest runner used by every frontend.
- [`docs/launcher_gui.md`](docs/launcher_gui.md) – Launcher help text and detachment
  behavior.
- [`docs/run_manifest.md`](docs/run_manifest.md) – Manifest structure, gating, and
  workspace lifecycle.
- [`CHANGELOG.md`](CHANGELOG.md) – Release history with per-file reporting.

Explore `docs/validation/README.md` for model validation guidance and
`docs/packaging.md` if you plan to build wheels for deployment. The licensing
summary lives in `THIRD_PARTY_LICENSES.md`.
