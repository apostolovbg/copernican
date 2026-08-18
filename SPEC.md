# Project Specification
**Doc ID:** SPEC
**Doc Type:** specification
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-08-18
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `SPEC.md` only for durable project rules below this block.
<!-- DEVCOV:END -->

## Overview
Copernican is a manifest-driven Python toolkit for cosmology workflows.
It evaluates models against SNe Ia, BAO, and CMB observations.

## Core Behavior
- Keep model evaluation declarative through YAML manifests.
- Preserve declared background expressions such as `Hz_expression`.
- Keep backend adapters narrow and behavior-preserving.
- Keep CMB capability checks beside `model_coder.py` so the declarative
  perturbation path can execute through the generic
  Boltzmann-hierarchy solver or fail clearly without a separate
  registry file.
- Validation runs and mirrored tests should stay behavior-focused, not
  cosmetic drift checks.
- Keep GUI folder-open actions on native OS handlers while preserving the
  existing launcher flow.
- Keep Union3 compressed SNe handling aligned on additive intercept
  marginalization across parser, likelihood and exported diagnostics.

## Repository Constraints
- Do not edit managed DevCovenant blocks directly.
- Keep generated artifacts in sync with their source config.
- Mirror source modules under `tests/` with matching package structure.
