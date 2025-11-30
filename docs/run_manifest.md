# Run Manifest

The suite writes a YAML manifest for every evaluation under the run's output
folder. The file is named `run_manifest_<timestamp>.yml` and records:

Headless runs can pin the manifest location with the `--manifest` flag to
`copernican.py` so CI pipelines always collect the same path even when output
directories change.

- Copernican Suite version under `copernican.version`.
- Selected model and engine names with their versions plus a `selection` block
  so GUI import/export can re-seed new runs without retyping choices.
- Parameter priors and the random seed captured at start confirmation.
- Dataset identifiers, names and release versions with SHA256 hashes of input
  files.
- Independence statements confirming that SNe, BAO and CMB likelihoods were
  treated as statistically separate when building the joint posterior.
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

When no ``COPERNICAN_SEED`` environment variable is present the program
prompts for a seed early in the run.  Users may accept the default ``0``,
enter a manual value or generate a random seed.  The chosen value is saved
in the manifest and main log so runs can be reproduced exactly.

The GUI mirrors the CLI behaviour by generating the manifest at the "Start
Run" confirmation stage rather than during draft editing. Pending manifests
mark ``status.state`` as ``pending`` and set ``status.outputs`` to
``unprepared`` so operators can review the configuration before directories or
logs exist. Starting the run flips the status to ``running`` and the
``selection`` and ``configuration`` blocks capture the chosen models, engine
and dataset identifiers for reuse. Hard stops or cancellations update
``status.state`` to ``aborted`` or ``cancelled`` and embed a retention decision
such as ``archived`` or ``deleted`` for downstream provenance checks.

The Stage 2 sampler now constructs its NumPy random number generator from the
shared :func:`copernican_lib.utils.get_random_seed` value.  That helper is
populated via :func:`copernican_lib.utils.set_random_seed`, which the CLI calls
after reading ``COPERNICAN_SEED`` or the interactive prompt.  When no explicit
seed is supplied the suite falls back to the deterministic default ``0`` so the
manifest's ``seed`` field always reflects the exact value fed into the engine.
Replaying a manifest therefore yields byte-for-byte identical chains,
log-probabilities and summary statistics as long as the same commit and dataset
hashes are used.

The manifest is intentionally human readable so it can be archived in lab
notebooks or cited in publications. Recording the suite version makes it clear
which behaviour and documentation set applied to the run, especially when a
development branch has diverged from the last tagged release.

When both models point to the same YAML file the manifest will list matching
`MODEL_FILENAME` entries. That shared marker indicates the Stage 2 workflow
reused a single SNe posterior, keeping BAO and CMB chi-squared totals in lock
step for LCDM self-consistency checks.
