# Copernican Architecture
**Project Version:** 12.0.26

The Copernican splits functionality across orchestration, data, model
adapters, engines, and presentation layers. This document captures how these
layers work together, describes the manifest lifecycle, and highlights key
guardrails such as policy enforcement and dataset validation.
## Table of Contents
- [Component Layers](#component-layers)
 - [Orchestration Layer](#orchestration-layer)
 - [Library Layer (`copernican.lib`)](#library-layer-copernicanlib)
 - [Engines Layer](#engines-layer)
 - [Presentation Layer](#presentation-layer)
- [Run Manifest Lifecycle](#run-manifest-lifecycle)
- [Data Provenance](#data-provenance)
- [Logging, Diagnostics, and Signals](#logging-diagnostics-and-signals)
- [Policies & Documentation](#policies-and-documentation-guardrails)
## Component Layers
### Orchestration Layer
- `python -m copernican` – entry point that handles argument parsing (`--gui`,
 `--cli`, `--manifest`, `--output-dir`), dependency scanning, logging
 (including `faulthandler` and SIG handlers), and then delegates to
 `copernican.lib.run_executor.execute_run_from_manifest`.
- `copernican/lib/run_executor.py` – reads manifests, revalidates datasets and
 model adapters, sets up logging via `copernican.lib.logger`, and invokes
 `copernican.lib.run_pipeline`.
- `copernican/lib/run_pipeline.py` – shared pipeline that drives Stage 1–5,
 updates manifest diagnostics, writes results, and orchestrates plotting plus
 NetCDF saving.
- `copernican/lib/orchestration.py` – exposes `InProcessRunController`,
 `RunRequest`, `RunHandle`, and `RunStatus` so GUI clients can
 start/pause/cancel runs without copying CLI logic.
### Library Layer (`copernican.lib`)
- `dataset_registry` – loads SNe/BAO/CMB datasets, verifies parser digests, and
 attaches metadata to `DataFrame.attrs`.
- `model_spec_validator`, `model_coder`, and `engine_adapter` – validate YAML
 models, cache sanitized copies, convert equations into callables, and
 assemble picklable engine adapters compliant with the expected interface.
- `engine_adapter` – ensures adapters declare required functions, structured
 native CMB contracts, and dataset compatibility before any engine consumes
 them.
- `posterior`, `statistics`, `chain_io`, `csv_writer`, `result_writer` –
 provide shared likelihoods, chi-squared helpers, NetCDF/CSV writers, and
 summary serialization that every engine reuses.
- `progress`, `console_output`, `logger`, `utils` – unify progress reporting,
 console I/O, logging, timestamp generation, and other utilities so CLI and
 GUI flows mirror each other via the counter-based `BatchProgressBar`.
- `gui` – contains Tkinter scaffolds, Run Builder controls, diagnostics panels,
 and the `run_worker` that spawns the CLI workflow with
 `COPERNICAN_ALLOW_DIRECT=1`.
### Engines Layer
- `copernican/engines/engine_mcmc.py` – ensemble MCMC sampler with
 `emcee`, walker reseeding for `nan` positions, `-np.inf` when proposals are
 invalid, and counter-based progress updates emitted via
 `copernican.lib.progress`.
- `copernican/engines/engine_nested.py` – nested sampling backend
 providing live point counts, enlargement factors, and log-evidence tracking
 while matching the MCMC result schema.
- Future engines must keep shared dependencies pure compute-only and rely on
 `copernican.lib.optim_utils` for shared helpers rather than importing CLI
 helpers themselves.
### Presentation Layer
- `README.md`, `docs/`, `AGENTS.md`, `CHANGELOG.md` – document the suite,
 policies, laws, and release history. Documentation updates must touch both
 `README.md` and the relevant `docs/` page, and every touched file must be
 mentioned in `CHANGELOG.md`.
- GUI and CLI UIs – the GUI draws from `docs/gui_overview.md` and
 `docs/gui_guide.md`, while the CLI prints the same
 `docs/orchestration_services.md` map via `python -m copernican --gui`.
- `output/` – each run writes to `copernican-run_<UTC>` directories containing
 logs, plots, NetCDF chains, and `run_manifest_*` files for reproducibility.
## Run Manifest Lifecycle
1. **Stage 1 seed selection** – accepts `COPERNICAN_SEED`, accepts 0, or
 generates a random value. The chosen seed is logged and written to the
 manifest.
2. **Stage 2 prompts** – after loading the CMB dataset, the CLI asks for burn-
 in, production steps, walkers, and pool size with recommended defaults
 derived from the selected models. The GUI Run Settings panel mirrors the
 same questions.
3. **Manifest composition** – `copernican.lib.run_manifest.build_manifest`
 aggregates seed, model metadata, dataset digests, engine settings, run plan
 notes, Git hash, environment hints, and adapter metadata. The manifest is
 saved in the temporary workspace until `copernican.lib.gui.run_worker` or
 the CLI worker renames the folder to `copernican-run_<timestamp>`.
4. **Execution** – `copernican.lib.run_executor.execute_run_from_manifest`
 rebuilds the dataset loaders, the adapter, and the run configuration,
 launches the selected engine, and streams diagnostics into the GUI Run
 Monitor or the console.
5. **Results** – `result_writer.save_summary` outputs JSON/YAML summaries,
 `chain_io.save_posterior` writes NetCDF files with metadata embedded on both
 inference-data root and posterior groups, and `csv_writer` exports dataset-
 specific tables. `copernican.lib.plotter` renders Stage 5 corner plots with
 enforced footer guard bands, keeping metadata clear of the axes.
## Data Provenance
- Parsers compute SHA256 digests for the metadata files, the parser source and
 the dataset files listed in the metadata's `data_files` sequence (falling
 back to recognised table extensions when omitted) and store those digests on
 `df.attrs['file_hashes']`. Documentation such as `README`s and `LICENSE`s is
 skipped so the manifest records only the inputs that affect the likelihood,
 keeping every replayed run consistent.
- `dataset_registry.TRUSTED_PARSER_DIGESTS` keeps line endings normalised to
 `\n`; the GUI revalidation button calls the same digest check the CLI uses so
 the dataset list reflects parser trust status.
- Metadata files determine citation text, authorship, license, and dataset IDs;
 they remain the reference for CLI metadata viewers and the GUI **View
 metadata** action.
## Logging, Diagnostics, and Signals
- Logging is initialised early, records Python/OS/package versions, and flushes
 warnings through `copernican.lib.logger`. `console_output` ensures prints and
 inputs route through the logger.
- `faulthandler` plus SIGILL/SIGSEGV/SIGFPE handlers capture stack traces on
 fatal signals and write them to both console and log paths before exiting so
 the per-run monitor log remains complete.
- Progress updates flush per line to ensure stage updates appear in long
 computations; Stage 2 emits walker updates and the same counter records so
 CLI and GUI log mirrors look identical.
## Policies and Documentation Guardrails
- `AGENTS.md` and repository policy enforce changelog coverage,
  documentation updates, line length, module tests, parser hashes, policy
  syncing, and other laws (commenting, documentation ties, etc.).
- Every policy change that sets `updated: true` requires scripting and tests
 before progressing; this is enforced via the gate workflow and
 dependency-management refresh path.
- Documentation updates must update both `README.md` and the related `docs/`
 file while noting touched paths in `CHANGELOG.md`.
- Datasets, parsers, and metadata remain read-only except when explicitly
 changed in `docs/data_overview.md` and the changelog.
This architecture keeps CLI, GUI, dataset loaders, engines, and documentation
aligned so new frontends can build upon the same manifest and adapter services
without diverging from the canonical execution flow.
