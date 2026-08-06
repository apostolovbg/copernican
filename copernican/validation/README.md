# Validation manifests

## Overview

The validation suite runs *real* manifests through the same pipeline that
the GUI and CLI use for ordinary analyses. The manifest files live under
`copernican/validation/manifests/`, and the canonical manifest
`reference_planck2018.yml` compares the canonical native
`copernican/models/model_lcdm.yml` control with the fixed-parameter
`copernican/models/model_ref_planck2018.yml` test model against the publicly
released Union Through UNITY 2000 SNe, BOSS DR12 BAO, and Planck 2018 Lite
datasets. The manifest records both model roles and current hashes for every
dataset asset consumed by the validation run.

## Reference model
`copernican/models/model_ref_planck2018.yml` fixes the Hubble constant,
matter/baryon densities, photon density, effective neutrino number and
recombination redshift to the values reported in Planck Collaboration VI
(2018, Table 2). Each prior is declared as a `fixed` value so the run also
records the reference trace, the plotter draws the canonical comparison lines,
and the corner output highlights the sameness even though the parameters do
not wander. Because the parameters remain numerically locked, the manifest run
is deterministic, reproducible across environments and provides
the golden dataset used for validation and regression tracking.

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
engine and datasets exactly as the GUI would. The loader discovers all
`*.yml` files in that folder, so the CLI and GUI validation panels grow
automatically.
