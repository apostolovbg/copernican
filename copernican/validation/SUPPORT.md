# Validation Support
**Doc ID:** VALIDATION-SUPPORT
**Doc Type:** package-support
**Project Version:** 12.0.26
**DevCovenant Version:** 1.0.1b6

## Overview
If validation does not behave as expected, start with the package-local
README, the manifest list, the runner, and the summary marker at
`~/VALIDATION.md`. That keeps the validation workflow self-contained inside
`copernican/validation/` and avoids confusion about root-level launcher state.

## Self-Service Help
- `copernican/validation/README.md` describes the manifest flow.
- `copernican/validation/ABOUT.md` explains the package surface.
- `copernican/validation/manifests/` holds the manifest inputs.
- `copernican/validation/runner.py` executes the manifests through the
  standard pipeline.

## Filing A Good Report
Include the manifest name, the command or GUI action, the output directory,
and any failure text. If the summary marker looks stale, mention whether
`~/VALIDATION.md` still contains the previous run.

## What Maintainers Need
The most useful reports include the Python version, platform, manifest path,
and the smallest log snippet that reproduces the failure.
