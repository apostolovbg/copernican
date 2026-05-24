# Project Specification
**Doc ID:** SPEC
**Doc Type:** project-spec
**Project Version:** 12.0.11
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-05-24
**DevCovenant Version:** 1.0.1b6

## Overview
Copernican is a manifest-driven Python toolkit for cosmology workflows.
It evaluates models against SNe Ia, BAO, and CMB observations.

## Core Behavior
- Keep model evaluation declarative through YAML manifests.
- Preserve native background expressions such as `Hz_expression`.
- Keep backend adapters narrow and behavior-preserving.
- Validation runs and mirrored tests should stay behavior-focused, not
  cosmetic drift checks.
- Keep GUI folder-open actions on native OS handlers while preserving the
  existing launcher flow.

## Repository Constraints
- Do not edit managed DevCovenant blocks directly.
- Keep generated artifacts in sync with their source config.
- Mirror source modules under `tests/` with matching package structure.
