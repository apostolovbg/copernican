**Version:** 11.0.2

![Copernican Suite banner](docs/banner_github.png)

The Copernican Suite is a Python toolkit that helps researchers test
cosmological models against SNe Ia, BAO and CMB observations with a single
manifest-driven workflow. `copernican.py` orchestrates everything from model
loading through sampler execution while the launcher scripts keep a managed
Python 3.11 environment inside `.venv` so the tooling stays portable across
macOS, Linux and Windows. Developers must consult
[AGENTS.md](AGENTS.md) and the DevCovenant policies before making any edits
because the repository enforces its laws through pre-commit checks.

## Highlights
- **Manifest-driven orchestration:** `copernican.py` consumes theory, data and
  engine selections, writes every run into `output/copernican-run_*`, and
  re-uses `copernican_lib/run_pipeline.py` helpers so CLI and GUI paths stay
  consistent.
- **Modular library layout:** `copernican_lib/` hosts shared helpers (plotting,
  analysis, diagnostics, GUI scaffolding and dataset registries) while
  `models/`, `engines/`, `data/` and `output/` remain the canonical asset roots.
- **Run Builder & GUI:** a navigation rail keeps the Run Builder, Run Monitor,
  Analysis workspace and validation tools at your fingertips while metadata
  dialogs, builder panels, and the detached launcher preserve the historic flow.
- **Analysis workspace:** Run Summary, Posteriors, Diagnostics and Comparisons
  tabs rely on `copernican_lib.analysis`, `posterior_explorer`, and the shared
  `PlotViewer` so summaries, comparisons and posterior plots stay in sync with
  CLI helpers.
- **Validation & documentation:** Fixed Planck 2018 manifests drive the
  validation suite, while `docs/` guides, `README.md`, `CHANGELOG.md` and the
  AGENTS/DevCovenant policies capture every procedural rule.
- **Maintenance helpers:** CLI commands such as run summaries, comparisons,
  validations and dataset revalidation keep operators productive without the GUI.
- **Law & policy reminder:** Always obey the DevCovenant laws in `AGENTS.md`
  and run `pre-commit run --all-files` before finishing work.

## Overview
 - `copernican.py` acts as a manifest-first orchestrator: it consumes a manifest
   describing the selected models, datasets, sampler settings and environment
   hints, reuses `copernican_lib/run_pipeline.py` helpers and writes every run
   log inside `output/copernican-run_*` with a matching
   `run_manifest_<timestamp>.yml` for reproducibility.  Command-line flags such
   as `--manifest`, `--output-dir`, `--gui`, `--cli` and `--no-gui` let CI and
   operators pick the desired entry point while the launcher scripts guarantee
   the managed interpreter and dependencies.
 - `copernican_lib/` contains shared utilities (analysis helpers, likelihoods,
   diagnostics, GUI scaffolding, plotting helpers, dataset registries, etc.) so
   engines stay lightweight and consistent across backends.
 - `engines/` collects sampler back ends. The default `cosmo_engine_mcmc`
   couples `emcee` with ArviZ when available; the nested sampler mirrors the
   same schema while exposing evidences. Both reuse the shared progress renderer
   and manifest helpers.
 - `models/` houses YAML theories with priors, transforms and dataset
   compatibility metadata. Each definition is converted into a picklable
   plugin so manifest generation stays deterministic even under multiprocessing.
 - `data/` bundles vetted observations and parsers. The loaders validate SHA256
   digests, register citations, and tag each manifest with the hashes used for
   the run; the directory remains read-only except when a human explicitly
   edits the datasets.
 - `copernican_lib/gui/` provides a Tkinter scaffold with the navigation rail,
   Run Builder, Run Monitor, Analysis workspace, validation helpers and a Help
   page that renders Markdown assets inline; the start scripts hand off the
   detached GUI (using `pythonw` on macOS/Windows) after logging the environment.

## Directory layout
 - `models/`: YAML definitions for every supported cosmological model.
 - `engines/`: Computational backends; the default MCMC engine records diagnostics
   and writes NetCDF chains.
 - `data/`: Trusted datasets grouped by type (`sne`, `bao`, `cmb`) with parser
   metadata. Parsers compute SHA256 values and register their digests via
   `dataset_registry`.
 - `output/`: Per-run folders such as `copernican-run_<timestamp>/` that store
   manifests, parameter summaries, logs and NetCDF chains.
 - `validation/`: Reference manifests, runner helpers and summaries used by the
   validation suite.
 - `docs/`: Guides covering architecture, GUI/CLI workflows, manifest structure,
   launchers, datasets and the documentation policy itself.
 - `AGENTS.md`, `CHANGELOG.md`, `THIRD_PARTY_LICENSES.md`, `CITATION.cff`:
   Governance, release history and citation/licensing metadata.

## Run Builder & GUI
The navigation rail keeps quick actions and an always-visible logo square, so
launching the Run Builder or monitor never steals focus. Run Builder mirrors the
CLI stages: seed, model, data, engine and plan panels require one selection per
panel, the Save Manifest page remains locked until every stage reports a
selection, and saved manifests live under `output/copernican_run_NEW_CONFIG/` so
Confirm only becomes available when a manifest exists. Start Run renames that
workspace to `copernican-run_<timestamp>` before spawning the CLI worker, while
cancel and clear remove temporary folders. The Run Settings panel mirrors the
CLI prompts (walkers, burn-in, production, pool size) so GUI runs and CLI runs
share the same configuration metadata. Quick actions keep the dataset catalog
health overview, import manifest flow and output directory helpers within reach.

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
`copernican_lib.analysis.save_run_summary` and copy the JSON payload onto the
clipboard.

Posteriors uses `copernican_lib.posterior_explorer` to list `posterior-*.nc`
snapshots and renders a trace/hist overview inside the shared
`copernican_lib.gui.plot_viewer.PlotViewer`. The tab keeps controls for fitting
to screen, restoring the original limits, and toggling the drag-enabled pan so
you can inspect any region without re-creating the plot. The Comparisons tab
lets you point at two run directories, refresh Δχ²/parameter shifts and dataset
count deltas, and export or copy the structured comparison summary that the new
`copernican_lib.analysis.compare_runs` helper produces.

Every run now also writes ArviZ-powered corner plots and parameter histograms
into the `output/copernican-run_*` folders so the Analysis workspace can render
them inside the PlotViewer without re-running the sampler.  Use
`python copernican.py --analysis-posterior output/copernican-run_*` to rerun
`copernican_lib.analysis.plot_posterior`, producing the overview, corner and
histogram assets from each `posterior-*.nc` snapshot on demand.

## Validation
The Validation tab runs `python copernican.py --run-validation`, streams CLI
output into a log box, and stores the summaries inside
`validation/output/<manifest_stem>/validation_run_<timestamp>/` plus the
gitignored `VALIDATION.md`. The fixed Planck 2018 manifest evaluates a reference
ΛCDM model against Union3 UNITY SNe, BOSS DR12 BAO and Planck 2018 Lite CMB data
with constant priors, so regression checks stay deterministic. “Cancel
validation” terminates the worker, “Clear validation” removes the outputs and
summary, and GUI progress bars mirror the counter-based batches the sampler
emits. The Validation button stays disabled while a worker runs so overlapping
validation jobs cannot start.

## Documentation & policy
Release notes live in `CHANGELOG.md`, licensing details appear in
`THIRD_PARTY_LICENSES.md`, and the GUI Help panel renders `README.md` (banner
included) plus the CLI/GUI guides from `docs/gui_guide.md` and
`docs/cli_guide.md`. The brand-new Analysis workspace and launcher wiring are
covered by `docs/gui_overview.md` and `docs/launcher_gui.md`, while dataset
and manifest hygiene appear across `docs/data_overview.md`,
`docs/run_manifest.md` and the documentation policy itself (`docs/documentation_policy.md`).

Law 11 of `AGENTS.md` codifies the documentation expansion commitment: every
code change should grow the written record, and DevCovenant scripts automatically
verify policy sync before accepting commits.

## Maintenance helpers
Command-line operators who skip the GUI can still run maintenance helpers:

- `python copernican.py --catalogue-summary` prints the dataset/model/engine
  inventory health summary.
- `python copernican.py --revalidate-dataset <dataset_id>` reruns the parser trust
  check for a specific dataset and reports the digest result.
- `python copernican.py --list-manifests` lists every timestamped run folder and
  `--show-manifest <path>` pretty-prints a saved manifest.
- `python copernican.py --run-validation` executes the lightweight validation
  suite, prints the reference summary, stores it in `VALIDATION.md`, and exits
  without opening the GUI.
- `python copernican.py --analysis-summary <run_dir>` loads the manifest/log/
  parameter summary from the specified run, prints the diagnostics table and
  (with `--analysis-summary-output <dir>`) writes structured
  `analysis-summary_<timestamp>.yml/.json` exports just like the GUI’s Run
  Summary tab.
- `python copernican.py --analysis-compare <base_run> <alternative_run>` aligns
  two run directories, prints the Δχ²/parameter summary and (with
  `--analysis-compare-output <dir>`) saves `analysis-comparison_<timestamp>`
  JSON/YAML files matching the Analysis Comparisons tab.
- `python copernican.py --analysis-posterior <run_dir> --analysis-posterior-output
  <file.png>` builds a trace/hist overview using `copernican_lib.posterior_explorer`
  and writes the figure to the requested path so you can reproduce the PlotViewer
  output without the GUI.

## Law & policy compliance reminder
Before beginning work, read every law in `AGENTS.md`, obey the DevCovenant
policies in `devcovenant`, and finish every change with `pre-commit run --all-files`.
