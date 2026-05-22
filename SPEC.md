# Project Specification
**Doc ID:** SPEC
**Doc Type:** project-spec
**Project Version:** 12.0.1
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-05-22
**DevCovenant Version:** 1.0.1b5

## Overview
Copernican is a manifest-driven Python toolkit for cosmology workflows.
It evaluates models against SNe Ia, BAO, and CMB observations.

## Core Behavior
- Keep model evaluation declarative through YAML manifests.
- Preserve native background expressions such as `Hz_expression`.
- Keep backend adapters narrow and behavior-preserving.

## Repository Constraints
- Do not edit managed DevCovenant blocks directly.
- Keep generated artifacts in sync with their source config.
- Mirror source modules under `tests/` with matching package structure.
