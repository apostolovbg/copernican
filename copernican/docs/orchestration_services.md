# Orchestration Services
This page maps the shared orchestration services that GUI clients should reuse
instead of re-implementing CLI logic.
## Table of Contents
- [Overview](#overview)
- [Service Map](#service-map)
- [GUI Workflow](#gui-workflow)
- [Managed Environment](#managed-environment)
- [Practical Rule](#practical-rule)
## Overview
The orchestration layer is the shared bridge between manifests, run control,
and user-facing launchers. It keeps the GUI and CLI on the same runtime path
so the same manifest, validation, and logging rules apply regardless of how a
run starts.
The `copernican/lib/orchestration.py` module exists to collect the GUI-safe
parts of that contract in one place. It keeps orchestration concerns out of the
individual front ends and avoids duplicating the same launch logic in multiple
call sites.
## Service Map
The shared services are:
1. **Configuration validation**.
 `copernican.lib.model_spec_validator.validate_and_cache_model` turns YAML
 models into cached callables without importing menu helpers or UI code.
2. **Manifest generation**.
 `copernican.lib.run_manifest.build_manifest` assembles dataset digests,
 engine-adapter metadata, the ordered control/test comparison, native CMB
 engine identity, and Git state for every run. The result is intentionally
 identical for CLI and GUI launches.
3. **Run control**.
 `copernican.lib.run_executor.execute_run_from_manifest` owns the manifest-
 driven execution path. It is the single runner both interfaces should call
 so the shared pipeline, dataset rebuild helpers, and YAML-backed adapters
 all execute uniformly.
4. **Run result writing**.
 `copernican.lib.result_writer.save_summary` serialises distinct control and
 test sampler outputs while the logging and dependency helpers keep runtime
 flags aligned with the CLI.
`copernican.main_workflow` relays manifests directly to the shared executor so
every manifest-driven launch shares the same runner.
## GUI Workflow
GUI launchers should construct an
`orchestration.InProcessRunController` with run, pause, resume, and cancel
hooks that call into the shared helpers above. The `RunRequest`, `RunHandle`,
and `RunStatus` dataclasses describe the minimum payloads required to drive
the pipeline while letting the GUI stream logs or status updates.
The GUI worker (`copernican/lib/gui/run_worker.py`) loads the JSON
configuration produced by the Run Builder, sets `COPERNICAN_ALLOW_DIRECT=1`,
and invokes `copernican.main` with `--manifest`. Any test or helper that
imports `copernican` directly should mirror that guard so the manifest CLI
remains usable through the shared executor.
The Settings surface persists choices through `copernican/lib/settings.py`
and the user settings file in the platform config directory. The Datasets,
GUI, and Tools tabs reuse the shared services listed above so deterministic
launches keep the same dataset hashes and GUI flags whether they start from
the command line or the Tkinter shell.
`python -m copernican --gui` prints this service map without entering the
interactive menus.
## Managed Environment
The orchestration contract assumes the active managed environment is
selected before launch. That keeps GUI and CLI execution aligned with the
same interpreter, the same dependency surface, and the same manifest
semantics.
CLI and GUI launches both exercise the unified manifest pipeline. They select
control and test models plus a sampler engine; CMB-capable models execute on
the fixed native declared-graph CMB engine.
## Practical Rule
If a GUI action can be expressed as a manifest-driven run, it should call the
shared orchestration services instead of inventing a second code path.
Manifest, run control, and logging behavior should stay identical between the
GUI and the CLI.
