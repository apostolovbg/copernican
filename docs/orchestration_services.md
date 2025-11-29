# Orchestration Services
**Last Updated:** 2025-11-24

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
   keeps the runtime flags and logging posture aligned with the CLI.

`copernican.py --gui` prints this service map without entering the interactive
menus. GUI launchers should construct an
`orchestration.InProcessRunController` with run, pause, resume and cancel hooks
that call into the shared helpers above. The `RunRequest`, `RunHandle` and
`RunStatus` dataclasses document the minimum payloads required to drive the
existing pipeline while letting the GUI stream logs or status updates.

Forward-only remains the default: the staged menu is disabled unless a caller
sets `COPERNICAN_ENABLE_STAGED_MENU=1` or passes `--enable-legacy-stage-menu`.
CI can toggle that flag to exercise historical prompts without reintroducing
backward-compatible branches for regular users.
