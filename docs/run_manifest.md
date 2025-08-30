# Run Manifest
**Last Updated:** 2025-08-30

The suite writes a YAML manifest for every evaluation under the run's output
folder.  The file is named `run_manifest_<timestamp>.yml` and records:

- Selected model and engine names with their versions
- Parameter priors and the random seed
- Dataset identifiers with SHA256 hashes of input files
- The Git commit hash and whether the tree was dirty

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
notebooks or cited in publications.
