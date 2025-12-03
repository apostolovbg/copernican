# Orchestration Services

This note maps the orchestration flow that GUI clients should reuse instead of
re-implementing CLI logic. The `copernican_lib/orchestration.py` module
summarises three GUI-safe services:

1. **Configuration validation**: `copernican_lib.model_spec_validator` exposes
   `validate_and_cache_model` for turning YAML models into cached callables
   without importing menu helpers.
2. **Manifest generation**: `copernican_lib.run_manifest.build_manifest`
   assembles dataset digests, plugin metadata and Git state for every run and
   remains identical across CLI and GUI launches.
3. **Run control**: `copernican_lib.result_writer.save_summary` serialises
   sampler outputs while `copernican_lib.cli.dependencies.get_runtime_options`
   keeps the runtime flags and logging posture aligned with the CLI. Manifest
   runners should now call `copernican_lib.run_executor.execute_run_from_manifest`
   so the shared pipeline in `copernican_lib/run_pipeline.py`, the dataset
   rebuild helpers in `copernican_lib/run_config.py`, and the YAML-backed model
   plugins all execute uniformly for both GUI and headless runs.

`copernican.main_workflow`, the console script entrypoint, now relays manifests
directly to `copernican_lib.run_executor.execute_run_from_manifest` so every
manifest-driven launch—CLI or GUI—shares the same runner.

`copernican.py --gui` prints this service map without entering the interactive
menus. GUI launchers should construct an
`orchestration.InProcessRunController` with run, pause, resume and cancel hooks
that call into the shared helpers above. The `RunRequest`, `RunHandle` and
`RunStatus` dataclasses document the minimum payloads required to drive the
existing pipeline while letting the GUI stream logs or status updates.

The GUI worker (`copernican_lib/gui/run_worker.py`) simply loads the JSON
configuration produced by the Run Builder, sets `COPERNICAN_ALLOW_DIRECT=1`,
and invokes `copernican.main` with `--manifest`. Any test or helper that
imports `copernican` directly should mirror that guard so the manifest CLI
remains usable without re-enabling the legacy menu workflow.

Forward-only remains the default: the staged menu is disabled unless a caller
sets `COPERNICAN_ENABLE_STAGED_MENU=1` or passes `--enable-legacy-stage-menu`.
CI can toggle that flag to exercise historical prompts without reintroducing
backward-compatible branches for regular users.
