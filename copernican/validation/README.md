# Validation manifests

## Overview

The validation suite runs *real* manifests through the same pipeline that
the GUI and CLI use for ordinary analyses. The manifest files live under
`copernican/validation/manifests/`, and the canonical manifest
`reference_planck2018.yml` compares the canonical CCMBS
`copernican/models/model_lcdm.yml` control with the fixed-parameter
`copernican/models/model_torg.yml` test model against the bundled Union
Through UNITY 2000 SNe, compound BAO, and Planck 2018 Lite datasets. Its exact
reference workload uses seed 0, five burn-in steps, ten production steps, 32
walkers, and a three-worker pool. The manifest records both model roles and
current hashes for every dataset asset consumed by the validation run.

## Reference model
`copernican/models/model_torg.yml` supplies the declared Temporal Opposing
Relational Geometry comparison contract. The LCDM control and TORG test run
through the same CCMBS solver, so the reference fixture exercises both
the scalar spectrum path and the model-comparison pipeline rather than a
fixed-parameter reference fixtures.

## Running validation
1. Activate the managed environment and run `python -m copernican` from the
   project root so the pinned Python 3.11 interpreter and dependencies are
   available.
2. Execute `python -m copernican --run-validation`. The command runs every
   manifest in `copernican/validation/manifests/`, writes its logs/plots into
   `copernican/validation/output/<manifest_stem>/validation_run_<timestamp>/`,
   and saves a textual summary of the pass/fail status plus the output
   directory to `~/VALIDATION.md`.
3. Alternatively, open the GUI and select the **Validation** navigation button.
   The same manifests execute in the background, the text pane shows the Run
   Monitor–style summary and a “Lock summary to latest entry” checkbox keeps
   the control scrolled to the bottom while the run is active.

Each validation run writes the standard artifacts (plots, CSVs, corner plots,
NEW_CONFIG) to its own
`copernican/validation/output/<manifest>/validation_run_YYYYMMDD_HHMMSS`
directory, just like a regular run. You can reuse the generated manifest or
configurations by opening the `run_manifest_<timestamp>.yml` file inside that
folder and copying it into the GUI builder.

See `ABOUT.md` for the package surface and `SUPPORT.md` for troubleshooting
and reporting guidance.

## Adding new manifests
To add another reference, drop a new YAML manifest into
`copernican/validation/manifests/` and describe the target model,
sampler and datasets exactly as the GUI would. The loader discovers all
`*.yml` files in that folder, so the CLI and GUI validation panels grow
automatically.
