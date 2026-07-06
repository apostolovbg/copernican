# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-07-06
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

Use this plan to close the current CMB review in three slices. The
branch already has the native declared-graph baseline, exact curved-sky
lensing remapping, q-resolved massive-neutrino support, and proof tests
for the existing sector plumbing, but the review still requires the
final scalar normalization, generic collision handling, sector and
gauge completeness, and native validation closure.

The target condition is narrow and final:

* `standard: true` stays on the standard backend path.
* `standard: false` stays native, CAMB-free, and CLASS-free in
  production.
* The native path keeps exact curved-sky lensing, conventioned scalar
  equations, and physically governed sector execution instead of
  acceptance-only scaffolding.
* Review parity is earned only when the source, tests, docs, and gate
  artifacts all agree that the solver is physically complete.

This is a three-slice plan. Do not add extra slices. Keep the roadmap
detailed enough that it can be executed without improvising scope.

## Table of Contents

* [Problem Preamble](#problem-preamble)
* [Current Baseline](#current-baseline)
* [Overview](#overview)
* [Execution Rules](#execution-rules)
* [Execution Slices](#execution-slices)
* [Completion Standard](#completion-standard)

## Problem Preamble

Copernican already has a real native declared-graph CMB engine and exact
native curved-sky lensing remapping, but it does not yet have a
review-complete universal Boltzmann-hierarchy solver.

The remaining gap is the physics, sector, and proof work that lets the
native path:

* express final hierarchy physics instead of acceptance-only
  scaffolding;
* reproduce the full review target through native execution;
* support scalar normalization, polarization, lensing, gauge, neutrino,
  vector, and tensor completeness; and
* stay governed enough that review closure is truthful and repeatable.

This roadmap exists to close that gap in exactly three slices. Each
slice must end on a clean checkout that passes the appropriate governed
tests. A broader design that leaves the repository in a broken or
misleading state is not an acceptable outcome.

## Current Baseline

The previous CMB campaign is complete in its scope and becomes the
baseline for this roadmap.

Current facts:

* Copernican already has a working native declared-graph CMB path for
  `standard: false`.
* The exact curved-sky native lensing remapper exists.
* The native scalar route is already q-resolved and collision-aware, but
  the review still asks for stronger physical normalization, proof, and
  multi-sector coverage.
* The public surface still needs final truth checks around parity,
  gauge, initial conditions, stacked spectrum reconstruction, and
  convergence.
* The plan now focuses on three sequential slices, not a longer roadmap
  with hidden cleanup later.

## Overview

This plan has one job: close the review in source, then prove the
closure with governed verification. Slice One closes the scalar physics
and collision truth gaps. Slice Two closes the remaining sector and
gauge claims. Slice Three proves the native parity and convergence
claims and closes the plan only when the branch is clean.

## Execution Rules

* Slice One does implementation, tests, docs, and changelog work needed
  to close the remaining scalar truth gaps.
* Slice Two is proof-first and only repairs source truth gaps that its
  sector or gauge tests expose.
* Slice Three is validation-first and only repairs source truth gaps
  that its native parity or convergence tests expose.
* No slice may leave behind an approximate native lensing path, an
  acceptance-only scalar hierarchy, hardcoded collision handling, or
  alias-only gauge claims.
* Stage completed slice changes before moving on.
* Do not commit or push unless explicitly instructed.

Task markers mean:

* [open] active work for this roadmap;
* [closed] completed and validated for this roadmap.

## Execution Slices

### [open] Slice One - Scalar physics closure

Purpose:

Replace the remaining scalar review-era scaffolding with convention-
complete native physics and record the closure in tests, docs, and
changelog entries.

Depends on:

* Current native declared-graph baseline.
* Existing exact curved-sky native lensing remapper.
* Existing q-grid and collision metadata support.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `README.md`
* `CHANGELOG.md`

Scope:

* Replace the remaining scalar acceptance equations with explicit
  Einstein-Boltzmann content for photons, baryons, CDM, massless
  neutrinos, metric closure, and tight coupling.
* Make the scalar metric constraints use conventioned background
  weights so the equations are physically readable rather than proxy
  driven.
* Make massive-neutrino evolution q-resolved and keep any aggregate
  moments algebraically consistent with the q-bin hierarchy.
* Remove hardcoded collision-step assumptions and compile the Thomson
  relaxation from collision metadata.
* Keep exact curved-sky remapping as the only native lensed-spectrum
  path and preserve declared primordial `BB` through `lensed_BB`.
* Fix stacked multi-spectrum reconstruction so the public CMB surface
  can index requested spectra correctly.
* Update the user-facing docs, changelog, and tests so they describe and
  guard the completed scalar physics truth.

Tasks:

* Rewrite the scalar source and constraint expressions with explicit
  background weighting and documented variable conventions.
* Make the q-bin density, momentum, shear, and higher moments integrate
  with the correct physical weights.
* Move collision-step selection and coefficients to compiled operator
  metadata.
* Add tests for normalization, q weights, collision terms, and scalar-
  response changes.
* Update the touched docs and changelog entry for the slice.

Done when:

* The scalar hierarchy can be read as a physical Boltzmann system
  rather than a proxy-driven graph.
* q-resolved states and aggregate moments are consistent by construction.
* Collision terms are metadata-driven and no longer depend on hardcoded
  variable names.
* Tests prove the scalar physics changes observable outputs.

### [open] Slice Two - Sector and gauge closure

Purpose:

Implement real vector and tensor Boltzmann sectors and prove gauge
equivalence.

Depends on:

* Slice One.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`

Scope:

* Generate physical vector and tensor hierarchies, not just tagged
  synthetic sources.
* Add tensor and vector initial conditions, metric content, and
  sector-specific observables.
* Make Newtonian, synchronous, and gauge-invariant routes prove the
  same observables for the same physical model.
* Keep the sector and gauge proof tests explicit so labels do not
  masquerade as independent physics.

Tasks:

* Add vector and tensor hierarchy compilation from declared sector
  metadata.
* Add physical vector/tensor observable tests and sector mismatch
  guards.
* Add gauge-equivalence tests that compare the observables across routes
  for the same model.
* Preserve the existing q-resolution, source-refinement, and
  initial-condition proof tests.
* Update the touched docs and changelog entry for the slice.

Done when:

* Vector and tensor observables come from real sector content rather
  than one-off synthetic fixtures.
* Gauge-invariant and synchronous outputs match the Newtonian route for
  the same physical model.
* The proof tests for q resolution, source refinement, and initial
  conditions remain green.

### [open] Slice Three - Native validation and convergence closure

Purpose:

Validate the native solver against external reference data and
convergence thresholds.

Depends on:

* Slices One and Two.

Probable affected files:

* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `copernican/lib/likelihoods/cmb/native_lensing.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`

Scope:

* Add absolute native TT/TE/EE/PP/lensed parity checks against CAMB or
  CLASS using the native route, not the standard adapter.
* Add convergence tests for source refinement, k-grid resolution, and
  q-grid resolution so numerical settings prove accuracy rather than
  mere activation.
* Keep exact curved-sky lensing, the full spectrum family, and the
  multi-spectrum likelihood surface intact.
* Record the remaining runtime and validation claims clearly in docs
  and changelog.

Tasks:

* Add native-vs-reference spectrum parity tests across the full output
  family.
* Add convergence tests that show refinement moves spectra toward stable
  results.
* Add regression tests that keep stacked-spectrum reconstruction and
  declared BB-preserving lensing correct.
* Update the touched docs and changelog entry for the slice.

Done when:

* Native outputs match the reference data at meaningful tolerances.
* Refinement settings demonstrate convergence on the actual spectra, not
  only grid size changes.
* The branch can truthfully claim full review closure without hidden
  validation gaps.

## Completion Standard

This roadmap is complete only when all three slices are closed and the
repository can truthfully claim all of the following:

* Copernican ships a native Boltzmann-hierarchy CMB infrastructure whose
  `standard: false` route compiles complete declared theories upstream
  and executes them natively with no CAMB or CLASS production fallback.
* Standard-model native acceptance covers scalar normalization,
  polarization, lensing, gauge, q-resolved massive-neutrino, and
  required non-scalar sector behavior with documented reference
  tolerances.
* Nonstandard declared theories can define sectors, species,
  interactions, gauges, initial conditions, and projections through
  structured contracts rather than hidden solver branches.
* Runtime is governed by bounded caches, explicit convergence controls,
  and benchmark thresholds that keep the subsystem from drifting back
  into unmanaged snail-slow behavior.
* Docs, manifests, packaging smoke, tests, and the public API all tell
  the same truth.
