# Validation About
**Doc ID:** VALIDATION-ABOUT
**Doc Type:** package-about
**Project Version:** 12.0.26
**DevCovenant Version:** 1.0.1b6

## Overview
The validation package owns the manifest-driven verification flow used by
`python -m copernican --run-validation` and the GUI Validation page. It keeps
the golden manifest, the runner, and the generated outputs together under
`copernican/validation/` so package installs and source checkouts share the
same layout for local validation work.

## Package Surface
- `copernican/validation/README.md` explains how to run the suite.
- `copernican/validation/manifests/` stores the manifest inputs.
- `copernican/validation/runner.py` executes the manifests through the
  standard pipeline.
- Validation writes its local summary marker to `~/VALIDATION.md`.

## Where To Start
Read the README first, then inspect the manifest and runner if you need to
extend or troubleshoot validation behavior.
