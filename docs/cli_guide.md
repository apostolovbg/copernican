# Copernican Suite CLI Guide
The command-line interface (CLI) drives the same manifest-based workflow as the
GUI. This guide explains how to launch the CLI, describes every interactive
prompt, shows how to feed saved manifests into the executor, and lists the key
environment variables that change behaviour. Keep `docs/gui_guide.md` at hand
when you want to mirror GUI behaviour or verify what the builder is doing on
each step.

## Launchers and Entry Points
Always work from the managed `.venv` created by the launcher scripts:

- `./start.command` (macOS), `./start.sh` (Linux) or `start.bat` (Windows)
- The launchers download the pinned Python 3.11, recreate `.venv` when needed,
  and install dependencies from `requirements.lock`

After the environment check you can select:

1. **Start Copernican Suite (GUI)** – Runs `python copernican.py --gui`
2. **Start Copernican Suite (CLI)** – Runs the manifest-driven CLI workflow
3. **Run the unit test suite** – Delegates to `python -m unittest discover`
4. **Enable strict warning mode** – Sets `COPERNICAN_STRICT_WARNINGS=1` for the
   session so Python warnings raise errors
5. **Environment and dependency management** – Rebuilds `.venv` and reports the
   detected interpreter
6. **Install Copernican Suite** – Runs `pip install .` inside the managed
   interpreter

You can call `python copernican.py --cli` directly after activating `.venv` if
you do not need the curated prompts. Additional switches include `--gui`,
`--no-gui`, `--manifest <path>` to execute a saved manifest, and `--output-dir`
to override where run directories are created.

## Interactive CLI Workflow
The CLI mirrors the Run Builder pages:

1. **Seed selection** – Accept the default seed (`0`), supply your own value or
   request a random seed. Setting `COPERNICAN_SEED` bypasses the prompt.
2. **Model selection** – Choose any `cosmo_model_*.yml` discovered under
   `models/`. The CLI validates YAML using the cached schema before generating
   plugins.
3. **Dataset selection** – Pick one dataset per category (SNe Ia, BAO, CMB).
   Parsers are verified by SHA256 digest before their modules are imported.
4. **Engine selection** – Choose a sampler backend from `engines/`. The default
   is `engines/cosmo_engine_mcmc.py` unless you override it. Engine metadata
   (walkers, burn-in, production steps, pool size) is gathered immediately after
   the engine choice.
5. **Run plan / Manifest** – Provide notes for the run plan. The CLI then writes
   a manifest under `output/copernican_run_NEW_CONFIG/` using the same naming
   convention as the GUI. The manifest records dataset hashes, model metadata,
   engine knobs and Git information.
6. **Confirm and Launch** – The CLI displays a summary, asks for confirmation
   and starts the worker. Logs stream to stdout and to
   `logs/copernican-run_<timestamp>.txt` in parallel.

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
- `--list-manifests` – Lists timestamped run folders under the selected
  output directory and shows the most recent manifest file in each folder.
- `--show-manifest PATH` – Pretty-prints a saved manifest file so you can
  inspect it without opening a GUI metadata viewer.
- `--run-validation` – Executes the golden manifests under
  `validation/manifests/` (currently `reference_planck2018.yml`), runs the fixed
  `models/cosmo_model_ref_planck2018.yml`, writes the NEW_CONFIG/results into
  `validation/output/<manifest_stem>/validation_run_<timestamp>/` and saves the
  textual summary to `VALIDATION.md` before exiting.
  The manifest evaluates this fixed reference model against Union Through UNITY
  2000 SNe, BOSS DR12 BAO and Planck 2018 Lite, and every parameter uses a
  `fixed` prior so the sampler still emits its reference trace and the plots keep
  drawing the comparison lines even though the values never wander from the
  Planck 2018 anchor.

## Executing Saved Manifests
Both the CLI and GUI rely on
`copernican_lib.run_executor.execute_run_from_manifest`. To reuse a manifest:

1. Save or copy the manifest (e.g., `output/copernican-run_20251203_154118/
   run_manifest_20251203_154118.yml`).
2. Run `python copernican.py --manifest /path/to/manifest.yml`.
3. (Optional) Set `--output-dir` to store outputs in a deterministic folder for
   CI environments.

The executor rebuilds the declared models via
`copernican_lib.plugins.build_engine_plugin`, reloads datasets using the
recorded hashes, and hands sampling to the selected engine. Progress updates and
log output match the GUI’s Run Monitor display.

## Environment Variables

- `COPERNICAN_STRICT_WARNINGS=1` – Elevates Python warnings to errors, useful in
  CI pipelines.
- `COPERNICAN_SEED=<value>` – Pre-fills the seed question.
- `COPERNICAN_DEP_CACHE_DIR=<path>` – Overrides the default `.cache/` location
  used by the dependency scanner.
- `COPERNICAN_DETACH_GUI=1` – Forces the GUI to detach if you need to keep the
  CLI prompt free while the window runs.

The staged CLI menu has been retired; there is no longer a `COPERNICAN_ENABLE_STAGED_MENU`
flag or equivalent toggle.

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
