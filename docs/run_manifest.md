# Run Manifest
**Last Updated:** 2025-10-30

The suite writes a YAML manifest for every evaluation under the run's output
folder.  The file is named `run_manifest_<timestamp>.yml` and records:

- Copernican Suite version under `copernican.version`
- Selected model and engine names with their versions
- Parameter priors and the random seed
- Dataset identifiers with SHA256 hashes of input files
- The Git commit hash and whether the tree was dirty
- Per-engine extras such as MCMC burn-in length, production steps and
  acceptance fractions when the SNe sampler is used

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

The manifest is intentionally human readable so it can be archived in lab
notebooks or cited in publications. Recording the suite version makes it clear
which behaviour and documentation set applied to the run, especially when a
development branch has diverged from the last tagged release.

When both models point to the same YAML file the manifest will list matching
`MODEL_FILENAME` entries. That shared marker indicates the Stage 2 workflow
reused a single SNe posterior, keeping BAO and CMB chi-squared totals in lock
step for LCDM self-consistency checks.
