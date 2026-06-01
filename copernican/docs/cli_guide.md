# Copernican CLI Guide
The command-line interface (CLI) drives the same manifest-based workflow as the
GUI. This guide explains how to launch the CLI, describes every interactive
prompt, shows how to feed saved manifests into the executor, and lists the key
environment variables that change behaviour. Keep `docs/gui_guide.md` at hand
when you want to mirror GUI behaviour or verify what the builder is doing on
each step.

## Entry Points
Always work from the managed `.venv` created for the package:

- `python -m copernican` (preferred)
- `copernican` when the console script is on `PATH`

The entrypoint reuses the pinned environment in `.venv` and installs
dependencies from `requirements.lock`.

After the environment check you can select:

1. **Start Copernican (GUI)** – Runs `python -m copernican --gui`
2. **Start Copernican (CLI)** – Runs the manifest-driven CLI workflow
3. **Run the unit test suite** – Delegates to `python -m unittest discover`
4. **Enable strict warning mode** – Sets `COPERNICAN_STRICT_WARNINGS=1` for the
   session so Python warnings raise errors
5. **Environment and dependency management** – Rebuilds `.venv` and reports the
   detected interpreter
6. **Install Copernican** – Runs `pip install .` inside the managed
   interpreter

You can call `python -m copernican --cli` directly after activating `.venv`
if you do not need the curated prompts. Additional switches include `--gui`,
`--no-gui`, `--manifest <path>` to execute a saved manifest, and
`--output-dir` to override where run directories are created.

## Interactive CLI Workflow
The CLI mirrors the Run Builder pages:

1. **Seed selection** – Accept the default seed (`0`), supply your own value or
   request a random seed. Setting `COPERNICAN_SEED` bypasses the prompt.
2. **Model selection** – Choose any `cosmo_model_*.yml` discovered under
   `models/`. The CLI validates YAML using the cached schema before generating
   engine adapters.
3. **Dataset selection** – Pick one dataset per category (SNe Ia, BAO, CMB).
   Parsers are verified by SHA256 digest before their modules are imported.
4. **Engine selection** – Choose a sampler backend from `engines/`. The default
   is `engines/cosmo_engine_mcmc.py` unless you override it. Engine metadata
   (walkers, burn-in, production steps, pool size) is gathered immediately
   after the engine choice. When a selected engine detects that every parameter
   is fixed (for example, when the validation manifest runs `Planck 2018
   Reference LambdaCDM`), the sampler now mirrors the reference values,
   fabricates identical chains, and still reports the configured worker pool
   count. This keeps diagnostics, plots and manifest metadata consistent even
   though no sampling steps are actually executed.
5. **Run plan / Manifest** – Provide notes for the run plan. The CLI then
   writes a manifest under `output/copernican_run_NEW_CONFIG/` using the same
   naming convention as the GUI. The manifest records dataset hashes, model
   metadata, engine knobs and Git information. The CLI run log for each
   manifest resides under the resulting `output/copernican-run_<timestamp>/`
   folder as `copernican-run_<timestamp>.txt`. GUI-launched runs also write a
   lighter monitoring log to `logs/runs/*.txt` so the Run Monitor can tail
   progress without editing the reproducibility artifacts.
6. **Confirm and Launch** – The CLI displays a summary, asks for confirmation
   and starts the worker. Logs stream to stdout and to `logs/copernican-
   run_<timestamp>.txt` in parallel.

Every stage logs progress and flushes stdout so long optimisations remain
responsive even on remote terminals. Menu prompts use numbered options to keep
keyboard-only navigation consistent on macOS, Linux and Windows shells.

## Utility Commands
Not every CLI task requires launching the manifest workflow. The following
flags execute their action and exit immediately:

- `--catalogue-summary` – Prints dataset counts by type, highlights untrusted
  parsers and reports how many models/engines were discovered.
- `--revalidate-dataset DATASET_ID` – Re-runs the parser hash check for a
  specific dataset id and warns when the digest diverges from the trusted
  value.
- `--list-manifests` – Lists timestamped run folders under the selected output
  directory and shows the most recent manifest file in each folder.
- `--show-manifest PATH` – Pretty-prints a saved manifest file so you can
  inspect it without opening a GUI metadata viewer.
- `--run-validation` – Executes the golden manifests under
  `validation/manifests/` (currently `reference_planck2018.yml`), runs the
  fixed `models/cosmo_model_ref_planck2018.yml`, writes the NEW_CONFIG/results
  into `validation/output/<manifest_stem>/validation_run_<timestamp>/` and
  saves the textual summary to `VALIDATION.md` before exiting. The manifest
  evaluates this fixed reference model against Union Through UNITY 2000 SNe,
  BOSS DR12 BAO and Planck 2018 Lite, and every parameter uses a `fixed` prior
  so the sampler still emits its reference trace and the plots keep drawing the
  comparison lines even though the values never wander from the Planck 2018
  anchor. The executor now persists a `run_manifest_<timestamp>.yml` copy
  inside each validation run directory so the manifest that drove the analysis
  stays alongside the outputs.

### Analysis helpers

- `--analysis-summary RUN_DIR` reads the manifest, parameter summary and log
  for the chosen run directory, prints the dataset counts, R-hat/ESS
  diagnostics and per-model χ² breakdown, and lets you export structured
  `analysis-summary_<timestamp>.yml/.json` files by also passing `--analysis-
  summary-output <dir>` and optional `--analysis-summary-formats yml,json`.
- `--analysis-compare BASE_DIR ALT_DIR` runs the structured comparator
  described in `copernican.lib.analysis`, prints the resulting JSON/YAML
  fragment with duration, dataset count and parameter deltas, and writes the
  same filer when combined with `--analysis-compare-output <dir>`.
- `--analysis-posterior RUN_DIR` reruns
  `copernican.lib.analysis.plot_posterior`, producing the ArviZ-powered
  overview, corner and histogram figures from the latest `posterior-*.nc`
  snapshot.  `--analysis-posterior-output` accepts either a directory
  (defaulting to the run folder) or a `.png` path. When given a directory all
  generated assets go inside it; supplying a `.png` stores the overview figure
  at that path while the corner/histogram files still accompany it in the same
  directory.

## Executing Saved Manifests
Both the CLI and GUI rely on
`copernican.lib.run_executor.execute_run_from_manifest`. To reuse a manifest:

1. Save or copy the manifest (e.g., `output/copernican-run_20251203_154118/
   run_manifest_20251203_154118.yml`).
2. Run `python -m copernican --manifest /path/to/manifest.yml`.
3. (Optional) Set `--output-dir` to store outputs in a deterministic folder for
   CI environments.

`copernican.lib.run_executor.execute_run_from_manifest` also saves a
timestamped `run_manifest_<timestamp>.yml` inside the provided output directory
before sampling begins, so CLI and validation runs archive the manifest even
when they only receive a reference to an existing YAML file.

The executor rebuilds the declared models via
`copernican.lib.engine_adapter.build_plugin`, reloads datasets using the
recorded hashes, and hands sampling to the selected engine. Progress updates
and log output match the GUI’s Run Monitor display.

## Environment Variables

- `COPERNICAN_STRICT_WARNINGS=1` – Elevates Python warnings to errors, useful
  in CI pipelines.
- `COPERNICAN_SEED=<value>` – Pre-fills the seed question.
- `COPERNICAN_DEP_CACHE_DIR=<path>` – Overrides the default `.cache/` location
  used by the dependency scanner.
- `COPERNICAN_DETACH_GUI=1` – Forces the GUI to detach if you need to keep the
  CLI prompt free while the window runs.

The staged CLI menu has been retired; there is no longer a
`COPERNICAN_ENABLE_STAGED_MENU` flag or equivalent toggle.

Review `AGENTS.md` for the rest of the configuration knobs, especially the
launcher policies and DevCovenant rules enforced in CI.

## Troubleshooting and Logs
The CLI prints an environment summary at startup (Python version, OS, CPU,
NumPy/SciPy versions) and enables `faulthandler` so fatal signals dump stack
traces to both stdout and the log file. Each run receives its own
`logs/copernican-run_<timestamp>.txt` with the per-walker progress indicators,
engine messages and warnings. Use `tail -f` or your preferred log viewer to
monitor long runs, and cross-reference the GUI Run Monitor if you transition
from CLI to GUI mid-analysis—the manifest files remain compatible across both
frontends.
