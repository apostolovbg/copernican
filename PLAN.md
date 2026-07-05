# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-07-05
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

Use this plan to close the current CMB review in two slices. The
branch already has the native declared-graph baseline and exact
curved-sky lensing remapping, but the review still requires the final
source cleanup, the remaining sector and gauge proof, and governed
workflow closure.

The target condition is narrow and final:

* `standard: true` stays on the standard backend path.
* `standard: false` stays native, CAMB-free, and CLASS-free in
  production.
* The native path keeps exact curved-sky lensing and physical hierarchy
  equations instead of acceptance-only scaffolding.
* Review parity is earned only when the source, tests, docs, and gate
  artifacts all agree that the solver is physically complete.

This is a two-slice plan. Do not add extra slices. Keep the roadmap
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
native curved-sky lensing remapping, but it does not yet have a review-
complete universal Boltzmann-hierarchy solver.

The remaining gap is the physics and proof work that lets the native
path:

* express final hierarchy physics instead of acceptance-only scaffolding;
* reproduce the full review target through native execution;
* support polarization, lensing, gauge, and neutrino completeness; and
* stay governed enough that review closure is truthful and repeatable.

This roadmap exists to close that gap in exactly two slices. Each slice
must end on a clean checkout that passes the appropriate governed tests.
A broader design that leaves the repository in a broken or misleading
state is not an acceptable outcome.

## Current Baseline

The previous CMB campaign is complete in its scope and becomes the
baseline for this roadmap.

Current facts:

* Copernican already has a working native declared-graph CMB path for
  `standard: false`.
* The exact curved-sky native lensing remapper exists.
* The native scalar route is already q-resolved and collision-aware, but
  the review still asks for stronger physics closure, proof, and
  multi-sector coverage.
* The public surface still needs the final truth checks around parity,
  gauge, initial conditions, and stacked spectrum reconstruction.
* The plan now focuses on two sequential slices, not a longer roadmap
  with hidden cleanup later.

## Overview

This plan has one job: close the review in source, then prove the
closure with governed verification. Slice One finished the remaining
source truth gaps. Slice Two proves the remaining sector, gauge, and
workflow claims and closes the plan only when the branch is clean.

## Execution Rules

* Slice One did implementation, tests, docs, and changelog work needed
  to close the remaining source truth gaps.
* Slice Two is proof-first and only repairs source truth gaps that its
  parity or sector tests expose.
* No slice may leave behind an approximate native lensing path or an
  acceptance-only scalar hierarchy.
* No slice may preserve artificial scaling, visibility injection, or
  other scaffolding that exists only to make intermediate tests pass.
* Stage completed slice changes before moving on.
* Do not commit or push unless explicitly instructed.

Task markers mean:

* [open] active work for this roadmap;
* [closed] completed and validated for this roadmap.

## Execution Slices

### [closed] Slice One - Physics completion and source truth

Purpose:

Replace the remaining review-era scaffolding with final native physics
and record the closure in tests, docs, and changelog entries.

Depends on:

* Current native declared-graph baseline.
* Existing exact curved-sky native lensing remapper.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/likelihoods/cmb/cmb.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_lensing.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/model_coder.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `README.md`
* `SECURITY.md`
* `CITATION.cff`
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
* Remove any hardcoded collision-step assumptions that should instead be
  compiled from collision metadata.
* Keep exact curved-sky remapping as the only native lensed-spectrum
  path and preserve declared primordial `BB` through `lensed_BB`.
* Fix stacked multi-spectrum reconstruction so the public CMB surface
  can index requested spectra correctly.
* Update the user-facing docs, changelog, and tests so they describe and
  guard the completed physics truth.

Tasks:

* Replace the remaining scalar source and constraint expressions with
  convention-complete forms.
* Make the q-bin moment path enforce physical integration rather than
  aggregate proxy duplication.
* Make the exact Thomson relaxation come from compiled collision
  operators instead of hardcoded state names.
* Fix the `lensed_BB` and stacked-spectrum indexing defects.
* Update the touched docs and changelog entry for the slice.

Done when:

* The source no longer contains review-visible approximation hacks for
  the scalar hierarchy or exact lensing path.
* Declared `BB` survives into the lensed outputs when it is physically
  present.
* The public spectrum reconstruction handles stacked outputs correctly.

Closed on 2026-07-05 after the BB-preserving lensed path,
collision-metadata Thomson relaxation, and row-order stacked-spectrum
reconstruction fixes.

### [closed] Slice Two - Sector completeness and proof

Purpose:

Prove the remaining sector, gauge, and proof claims that the review
still expects after the source cleanup.

Depends on:

* Slice One.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`

Scope:

* Prove vector and tensor hierarchy coverage through declared ancestry
  and observable metadata.
* Add gauge-invariant native comparisons so the observables prove the
  route is not just a label alias.
* Keep the q-resolution, source-refinement, and initial-condition proof
  tests as scientific regressions rather than activation checks.
* Update the user-facing docs and changelog so the proof slice is
  recorded clearly.

Tasks:

* Add vector and tensor sector inference tests.
* Add gauge-invariant native comparison tests.
* Preserve the source-refinement, q-resolution, and initial-condition
  proof tests.
* Update the touched docs and changelog entry for the slice.

Done when:

* Vector and tensor sector metadata are inferred and asserted.
* Gauge-invariant native outputs match the Newtonian route.
* The proof tests for source refinement, q resolution, and initial
  conditions remain green.

Closed on 2026-07-05 after explicit vector/tensor sector inference,
gauge-invariant native parity, and the maintained q-resolution and
source-refinement proof tests.

## Completion Standard

This roadmap is complete only when both slices are closed and the
repository can truthfully claim all of the following:

* Copernican ships a native Boltzmann-hierarchy CMB infrastructure whose
  `standard: false` route compiles complete declared theories upstream
  and executes them natively with no CAMB or CLASS production fallback.
* Standard-model native acceptance covers scalar, polarization, lensing,
  gauge, q-resolved massive-neutrino, and required non-scalar sector
  behavior with documented reference tolerances.
* Nonstandard declared theories can define sectors, species,
  interactions, gauges, initial conditions, and projections through
  structured contracts rather than hidden solver branches.
* Runtime is governed by bounded caches, explicit convergence controls,
  and benchmark thresholds that keep the subsystem from drifting back
  into unmanaged snail-slow behavior.
* Docs, manifests, packaging smoke, tests, and the public API all tell
  the same truth.
