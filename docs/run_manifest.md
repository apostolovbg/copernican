# Run Manifest

The suite writes a YAML manifest for every evaluation under the run's output
folder. The file is named `run_manifest_<timestamp>.yml`, where the timestamp
matches the start-of-run label used by the output directory and per-run log,
and records:

This manifest is the starting point for every run. GUI Drafts remain pending
until the operator confirms Start Run, at which point the manifest is finalised
(timestamped to the run directory) and fed to
`copernican/lib/run_executor.execute_run_from_manifest`. That helper rebuilds
the datasets via `copernican.lib.run_config`, instantiates the chosen engine,
and hands everything to `copernican/lib/run_pipeline.execute_run_pipeline` so
the CLI and GUI share an identical sampling, diagnostics, plotting and export
sequence. The executor reconstructs the selected model plugins directly from
their YAML caches before sampling, ensuring every run reuses the exact
symbolic definitions recorded in the manifest regardless of how the
configuration was authored.

When editing via the GUI the Save Manifest page first writes the working
configuration into
`output/copernican_run_NEW_CONFIG/run_manifest_NEW_CONFIG.yml` so you can
continue adjusting the builder without affecting real runs. Save buttons stay
disabled until seed, model, data and engine selections exist, and Start Run
renames the temporary workspace to the timestamped `copernican-run_<ts>` folder
and file before invoking the CLI worker so downstream tooling always sees the
canonical manifest. CLI `python -m copernican` invocations now pass the
manifest directly to `copernican.lib.run_executor.execute_run_from_manifest`
so the same manifest runner handles both interfaces, and
`copernican.main_workflow` just relays the manifest to that helper.

Headless runs can pin the manifest location with the `--manifest` flag to
`python -m copernican` so CI pipelines always collect the same path even
when output directories change.

- Copernican version under `copernican.version`.
- Selected model and engine names with their versions plus a `selection` block
  so GUI import/export can re-seed new runs without retyping choices.
- Parameter priors and the random seed captured at start confirmation.
- Dataset identifiers, names and release versions with SHA256 hashes of input
  files.
- Independence statements confirming that SNe, BAO and CMB likelihoods were
  treated as statistically separate when building the joint posterior.
- CMB adapter metadata summarising the backend contract plus the perturbation
  contract summary, including the contract version, gauge, standard flag,
  declared symbol names, source and observable names, equation,
  constraint, closure, and source counts, transfer-path summary, and
  backend mapping details for each CMB-capable model.
- Sampler configuration stored under ``configuration.run_settings`` so walkers,
  burn-in, production steps, pool/core hints and nested-sampling parameters
  stay tied to the manifest that produced a run.
- The Git commit hash and whether the tree was dirty.
- Lifecycle and retention metadata under ``status`` indicating whether outputs
  were prepared, paused, cancelled, aborted or completed and whether artefacts
  were kept, deleted or archived after a stop decision.
- The Run Builder snapshot under ``configuration`` plus the operator notes
  captured during the start confirmation stored in ``confirmation``.

Saving this manifest alongside plots and tables allows others to reproduce a
run exactly.  To rerun an analysis:

1. Checkout the commit listed under `git.commit` and ensure the dirty flag
   matches the worktree state.
2. Verify that each data file still produces the recorded SHA256 digest.
3. Configure the suite with the same model, priors, engine and seed.

When no ``COPERNICAN_SEED`` environment variable is present the program prompts
for a seed early in the run.  Users may accept the default ``0``, enter a
manual value or generate a random seed.  The chosen value is saved in the
manifest and main log so runs can be reproduced exactly.

The GUI mirrors the CLI behaviour by generating the manifest at the "Start Run"
confirmation stage rather than during draft editing. Pending manifests mark
``status.state`` as ``pending`` and set ``status.outputs`` to ``unprepared`` so
operators can review the configuration before directories or logs exist. The
confirmation panel exposes an **Insert manifest** button that records the
staged snapshot without launching a run, letting operators inspect metadata or
export the file before committing to Start Run. Starting the run flips the
status to ``running`` and the ``selection`` and ``configuration`` blocks
capture the chosen models, engine and dataset identifiers for reuse. Hard stops
or cancellations update ``status.state`` to ``aborted`` or ``cancelled`` and
embed a retention decision such as ``archived`` or ``deleted`` for downstream
provenance checks. The Home quick actions now also offer **Import manifest...**
so a manifest saved on one machine can be cloned, retimestamped for the current
run directory and inserted directly into the builder.
`copernican.lib.run_executor.execute_run_from_manifest` now ensures a
timestamped `run_manifest_<timestamp>.yml` file is saved beneath `output_dir`
before sampling begins. That copy surfaces inside CLI, GUI, and validation runs
so every run directory permanently archives the manifest that drove the
execution regardless of where the source manifest file originated.

The Stage 2 sampler now constructs its NumPy random number generator from the
shared :func:`copernican.lib.utils.get_random_seed` value.  That helper is
populated via :func:`copernican.lib.utils.set_random_seed`, which the CLI calls
after reading ``COPERNICAN_SEED`` or the interactive prompt.  When no explicit
seed is supplied the suite falls back to the deterministic default ``0`` so the
manifest's ``seed`` field always reflects the exact value fed into the engine.
Replaying a manifest therefore yields byte-for-byte identical chains, log-
probabilities and summary statistics as long as the same commit and dataset
hashes are used.

The manifest is intentionally human readable so it can be archived in lab
notebooks or cited in publications. Recording the suite version makes it clear
which behaviour and documentation set applied to the run, especially when a
development branch has diverged from the last tagged release.

When multiple selections point to the same YAML file the manifest will list
matching `MODEL_FILENAME` entries. That shared marker indicates the Stage 2
workflow reused a single posterior, keeping BAO and CMB chi-squared totals in
lock step for same-model regression checks.
