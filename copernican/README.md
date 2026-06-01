# copernican
**Doc ID:** README
**Doc Type:** repo-readme
**Project Version:** 12.0.26
**Last Updated:** 2026-06-01
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->

<!-- DEVCOV:END -->

**Version:** 12.0.26

![Copernican banner](docs/banner_github.png)

Copernican is a Python toolkit that helps researchers test cosmological models
against SNe Ia, BAO and CMB observations with a single manifest-driven
workflow. `python -m copernican` orchestrates everything from model loading
through sampler execution while the managed `.venv` keeps the pinned Python
3.11 environment portable across macOS, Linux and Windows. Developers must
consult [AGENTS.md](AGENTS.md) and
the DevCovenant policies before making any edits because the repository
enforces its laws through pre-commit checks.

Union3 compressed SNe data require additive intercept marginalization in the
SNe likelihood, CSV export and plot residual paths so all residual views use
the same convention.

## Table of Contents

- [Highlights](#highlights)
- [Overview](#overview)
- [Directory layout](#directory-layout)
- [Run Builder & GUI](#run-builder-gui)
- [Analysis workspace](#analysis-workspace)
- [Validation](#validation)
- [Documentation & policy](#documentation-policy)
- [Maintenance helpers](#maintenance-helpers)
- [Law & policy compliance reminder](#law-policy-compliance-reminder)

## Highlights
- **Manifest-driven orchestration:** `python -m copernican` consumes model,
  data and engine selections, writes every run into
  `output/copernican-run_*`, and reuses `copernican/lib/run_pipeline.py`
  helpers so CLI and GUI paths stay consistent.
- **Modular library layout:** `copernican/lib/` hosts shared helpers
  (plotting, analysis, diagnostics, GUI scaffolding and dataset registries)
  while `models/`, `engines/`, `copernican/datasets/` and `output/`
  remain the canonical asset roots.
- **CMB capability checks:** `copernican/lib/model_coder.py` keeps the CMB
  backend capability flags close to the perturbation compiler so
  `standard: false` models run through the generic declarative
  Boltzmann-hierarchy solver or fail clearly when a required capability
  is missing.
- **Run Builder & GUI:** a navigation rail keeps the Run Builder, Run Monitor,
  Analysis workspace and validation tools at your fingertips while metadata
  dialogs, builder panels, and the package entrypoint preserve the historic
  flow.
- **Analysis workspace:** Run Summary, Posteriors, Diagnostics and Comparisons
  tabs rely on `copernican.lib.analysis`, `posterior_explorer`, and the shared
  `PlotViewer` so summaries, comparisons and posterior plots stay in sync with
  CLI helpers.
- **Validation & documentation:** Fixed Planck 2018 manifests drive the
  validation suite, while `docs/` guides, `README.md`, `CHANGELOG.md` and the
  AGENTS/DevCovenant policies capture every procedural rule.
- **Maintenance helpers:** CLI commands such as run summaries, comparisons,
  validations and dataset revalidation keep operators productive without the
  GUI.
- **Law & policy reminder:** Always obey the DevCovenant laws in `AGENTS.md`
  and run `pre-commit run --all-files` before finishing work.

## Overview
 - `python -m copernican` acts as a manifest-first orchestrator: it consumes a
   manifest describing the selected models, datasets, sampler settings and
   environment hints, reuses `copernican/lib/run_pipeline.py` helpers and
  writes every run log inside `output/copernican-run_*` with a matching
  `run_manifest_<timestamp>.yml` for reproducibility.  Command-line flags such
  as `--manifest`, `--output-dir`, `--gui`, `--cli` and `--no-gui` let CI and
  operators pick the desired entry point while the managed interpreter and
  dependencies come from `.venv`.
 - `copernican/lib/` contains shared utilities (analysis helpers, likelihoods,
   diagnostics, GUI scaffolding, plotting helpers, dataset registries, etc.) so
   engines stay lightweight and consistent across backends.
 - `engines/` collects sampler back ends. The default `cosmo_engine_mcmc`
   couples `emcee` with ArviZ when available; the nested sampler mirrors the
   same schema while exposing evidences. Both reuse the shared progress
   renderer and manifest helpers.
 - `models/` houses YAML model definitions with priors, transforms and dataset
  compatibility metadata. Each definition is converted into a picklable
  engine adapter so manifest generation stays deterministic even under
  multiprocessing. CMB-valid models declare a backend contract under `cmb`
  with `backend`, `param_map`, `grids`, `values`, `calls` and a mandatory
  `perturbations` block; `standard: true` keeps the perturbation sections
  empty, while `standard: false` requires typed derivative equations and
  explicit backend mapping for the compiled declarative perturbation
  solver.
- `copernican/datasets/` bundles vetted observations and parsers. The
  loaders validate SHA256 digests, register citations, and tag each manifest
  with the hashes used for the run; the directory remains read-only except
  when a human explicitly edits the datasets.
- `copernican/lib/gui/` provides a Tkinter scaffold with the navigation rail,
  Run Builder, Run Monitor, Analysis workspace, validation helpers and a Help
  page that renders Markdown assets inline; the package entrypoint launches
  the GUI inside the managed environment after logging the environment.

## Directory layout
 - `models/`: YAML definitions for every supported cosmological model.
 - `engines/`: Computational backends; the default MCMC engine records
   diagnostics and writes NetCDF chains.
- `copernican/datasets/`: Trusted datasets grouped by type (`sne`, `bao`,
  `cmb`) with parser metadata. Parsers compute SHA256 values and register
  their digests via `dataset_registry`.
 - `output/`: Per-run folders such as `copernican-run_<timestamp>/` that store
   manifests, parameter summaries, logs and NetCDF chains.
 - `validation/`: Reference manifests, runner helpers and summaries used by the
   validation suite.
 - `docs/`: Guides covering architecture, GUI/CLI workflows, manifest
   structure, datasets and the DevCovenant policies.
- `AGENTS.md`, `CHANGELOG.md`, `licenses/THIRD_PARTY_LICENSES.md`,
  `CITATION.cff`:
  Governance, release history and citation/licensing metadata.

## Run Builder & GUI
The navigation rail keeps quick actions and an always-visible logo square, so
launching the Run Builder or monitor never steals focus. Run Builder mirrors
the CLI stages: seed, model, data, engine and plan panels require one selection
per panel, the Save Manifest page remains locked until every stage reports a
selection, and saved manifests live under `output/copernican_run_NEW_CONFIG/`
so Confirm only becomes available when a manifest exists. Start Run renames
that workspace to `copernican-run_<timestamp>` before spawning the CLI worker,
while cancel and clear remove temporary folders. The Run Settings panel mirrors
the CLI prompts (walkers, burn-in, production, pool size) so GUI runs and CLI
runs share the same configuration metadata. Quick actions keep the dataset
catalog health overview, import manifest flow and output directory helpers
within reach.
Folder-open actions use the operating system's native handlers so the GUI can
open output locations without changing the launch behavior that operators
already expect.

The Run Monitor threads CLI stdout/stderr into a log box that tails
`logs/runs/*.txt`, mirrors the counter-based progress updates from the sampler
and keeps the Cancel/Hard Stop buttons disabled until a run starts. A “Lock log
to latest entry” checkbox pins the view so operators can watch batches finish
without scrolling. Metadata dialogs size themselves to the longest line, add an
“Open file…” action that launches the OS editor and keep horizontal resizing
locked while allowing unlimited vertical growth.

## Analysis workspace
The Analysis tab now hosts Run Summary, Posteriors and Comparisons alongside a
placeholder Diagnostics panel. Run Summary ingests `output/copernican-run_*`
folders and loads the manifest, parameter summary and log to render dataset
counts, R-hat/ESS diagnostics, per-model χ² components, BAO `r_s` values and
timestamps inside a scrollable panel. Its action buttons reload the summary,
export structured `analysis-summary_<timestamp>.yml`/`.json` files via
`copernican.lib.analysis.save_run_summary` and copy the JSON payload onto the
clipboard.

Posteriors uses `copernican.lib.posterior_explorer` to list `posterior-*.nc`
snapshots and renders a trace/hist overview inside the shared
`copernican.lib.gui.plot_viewer.PlotViewer`. The tab keeps controls for fitting
to screen, restoring the original limits, and toggling the drag-enabled pan so
you can inspect any region without re-creating the plot. The Comparisons tab
lets you point at two run directories, refresh Δχ²/parameter shifts and dataset
count deltas, and export or copy the structured comparison summary that the new
`copernican.lib.analysis.compare_runs` helper produces.

Every run now also writes ArviZ-powered corner plots and parameter histograms
into the `output/copernican-run_*` folders so the Analysis workspace can render
them inside the PlotViewer without re-running the sampler.  Use `python -m
copernican --analysis-posterior output/copernican-run_*` to rerun
`copernican.lib.analysis.plot_posterior`, producing the overview, corner and
histogram assets from each `posterior-*.nc` snapshot on demand.

## Validation
The Validation tab runs `python -m copernican --run-validation`, streams CLI
output into a log box, and stores the summaries inside
`validation/output/<manifest_stem>/validation_run_<timestamp>/` plus the
gitignored `VALIDATION.md`. The fixed Planck 2018 manifest evaluates a
reference model against Union3 UNITY SNe, BOSS DR12 BAO and Planck 2018 Lite
CMB data with constant priors, so regression checks stay deterministic.
“Cancel validation” terminates the worker, “Clear validation” removes the
outputs and summary, and GUI progress bars mirror the counter-based batches
the sampler emits. The Validation button stays disabled while a worker runs so
overlapping validation jobs cannot start.

## Documentation & policy
Release notes live in `CHANGELOG.md`, licensing details appear in
`licenses/THIRD_PARTY_LICENSES.md`, and the GUI Help panel renders
`README.md` (banner included) plus the CLI/GUI guides from
`docs/gui_guide.md` and
`docs/cli_guide.md`. The Analysis workspace and package entrypoint wiring are
covered by `docs/gui_overview.md` and `docs/gui_guide.md`, while dataset and
manifest hygiene appear across `docs/data_overview.md`, `docs/run_manifest.md`
and the DevCovenant policies.

Law 11 of `AGENTS.md` codifies the documentation expansion commitment: every
code change should grow the written record, and DevCovenant tooling
automatically verifies policy sync before accepting commits.

## Maintenance helpers
Command-line operators who skip the GUI can still run maintenance helpers:

- `python -m copernican --catalogue-summary` prints the dataset/model/engine
  inventory health summary.
- `python -m copernican --revalidate-dataset <dataset_id>` reruns the parser
  trust check for a specific dataset and reports the digest result.
- `python -m copernican --list-manifests` lists every timestamped run folder
  and `--show-manifest <path>` pretty-prints a saved manifest.
- `python -m copernican --run-validation` executes the lightweight validation
  suite, prints the reference summary, stores it in `VALIDATION.md`, and exits
  without opening the GUI.
- `python -m copernican --analysis-summary <run_dir>` loads the manifest/log/
  parameter summary from the specified run, prints the diagnostics table and
  (with `--analysis-summary-output <dir>`) writes structured `analysis-
  summary_<timestamp>.yml/.json` exports just like the GUI’s Run Summary tab.
- `python -m copernican --analysis-compare <base_run> <alternative_run>` aligns
  two run directories, prints the Δχ²/parameter summary and (with `--analysis-
  compare-output <dir>`) saves `analysis-comparison_<timestamp>` JSON/YAML
  files matching the Analysis Comparisons tab.
- `python -m copernican --analysis-posterior <run_dir> --analysis-posterior-
  output <file.png>` builds a trace/hist overview using
  `copernican.lib.posterior_explorer` and writes the figure to the requested
  path so you can reproduce the PlotViewer output without the GUI.

## Law & policy compliance reminder
Before beginning work, read every law in `AGENTS.md`, obey the DevCovenant
policies in `devcovenant`, and finish every change with `pre-commit run --all-
files`.
