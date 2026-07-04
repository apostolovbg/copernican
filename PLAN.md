# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-06-28
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

Use this plan to close the current CMB review in two slices: one
implementation slice and one verification slice. The branch already has
the native declared-graph baseline and exact curved-sky lensing
remapping, but the review still requires the last physics cleanup, the
final cross-gauge and q-resolved checks, and a governed proof run.

The target condition is narrow and final:

* `standard: true` stays on the standard backend path.
* `standard: false` stays native, CAMB-free, and CLASS-free in production.
* The native path keeps exact curved-sky lensing and physical hierarchy
  equations instead of acceptance-only scaffolding.
* Review parity is earned only when the source, tests, docs, and gate
  artifacts all agree that the solver is physically complete.

This is a two-slice plan. Do not add extra slices. Do not convert the
verification slice into a feature slice.

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

The remaining gap is the physics and proof work that lets the native path:

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
* Slice One closes the scalar acceptance physics, the q-resolved
  massive-neutrino route, the last gauge and collision-operator
  closures, and the proof that they are complete.
* The plan now focuses on one completed implementation slice and one
  verification slice, not a longer roadmap.

## Overview

This plan has one job: close the review in source, then prove the closure
with governed verification. Slice One made the physics final. Slice Two
runs the checks, clears any source-side truth gaps exposed by those
checks, and closes the gate only when the branch is clean.

## Execution Rules

* Slice One does implementation, tests, docs, and changelog work needed
  to close the review.
* Slice Two is verification-first and only repairs source truth gaps that
  verification exposes.
* No slice may leave behind an approximate native lensing path or an
  acceptance-only scalar hierarchy.
* No slice may preserve artificial scaling, visibility injection, or
  other scaffolding that exists only to make intermediate tests pass.
* Stage completed slice changes before moving on.
* Do not commit or push unless explicitly instructed.

Task markers mean:

* [open] active work for this roadmap;
* [closed] completed and validated for this roadmap;
* [blocked] not executable until named dependencies close.

## Execution Slices

### [closed] Slice One - Physics completion and source cleanup

Purpose:

Replace the remaining review-era scaffolding with final native physics and
record the closure in tests, docs, and changelog entries.

Depends on:

* Current native declared-graph baseline.
* Existing exact curved-sky native lensing remapper.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
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

* Replace scalar acceptance equations with physical Einstein-Boltzmann
  content for photons, baryons, CDM, massless neutrinos, metric closure,
  and tight coupling.
* Make massive-neutrino evolution q-resolved rather than aggregate-proxy
  based.
* Remove artificial output scaling and any PP/BB visibility hacks from the
  native exact-lensing path.
* Keep exact curved-sky remapping as the only native lensed-spectrum path.
* Make collision operators generate equation terms and conservation checks
  rather than serving as metadata only.
* Make Newtonian and synchronous gauges yield the same observables for the
  same physical model.
* Update the user-facing docs, changelog, and tests so they describe and
  guard the completed physics truth.

Tasks:

* [closed] Replace the remaining scalar acceptance physics with physical
  hierarchy and source terms.
* [closed] Remove native lensing scaffolding and derive lensed outputs from
  exact remapping only.
* [closed] Add q-resolved massive-neutrino, gauge-equivalence, and
  collision-operator regression coverage.
* [closed] Update PLAN-adjacent docs and changelog entries so the repository
  narrative matches the final source.
* [closed] Preserve the review's exact spectrum family expectations in the
  public CMB surface.

Done when:

* [closed] The native solver is physically complete enough that the review's
  remaining source concerns are closed in code and tests.
* [closed] Lensed outputs arise from exact remapping without approximation
  shortcuts or visibility injections.
* [closed] The updated docs and changelog describe the same final behavior as
  the source.

### [open] Slice Two - Verification and gate closure

Purpose:

Prove the completed source state with governed verification and close the
workflow only when the repository is clean and the review is truthfully
closed.

Depends on:

* Slice One.

Probable affected files:

* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `CHANGELOG.md`
* any source file that verification proves still needs a truth fix

Scope:

* Run `devcovenant gate --verify` against the completed implementation.
* Run `devcovenant run` on the same source state.
* Run `devcovenant gate --close` only after the implementation and
  verification agree.
* Fix only source-side truth gaps surfaced by verification.
* Keep the slice narrow: no new feature work, only completion or repair.

Tasks:

* [open] Execute the governed verification path against the completed
  implementation slice.
* [open] Repair any source truth gaps exposed by verification.
* [open] Close the gate only when the branch, tests, docs, and lifecycle
  artifacts are green.

Done when:

* [open] `gate --verify`, `run`, and `gate --close` all report green.
* [open] No review blocker remains open in source or tests.
* [open] The branch can truthfully claim the review is complete.

## Completion Standard

This roadmap is complete only when both slices are closed and the
repository can truthfully claim all of the following:

* Copernican ships a native Boltzmann-hierarchy CMB infrastructure whose
  `standard: false` route compiles complete declared theories upstream and
  executes them natively with no CAMB or CLASS production fallback.
* Standard-model native acceptance covers scalar, polarization, lensing,
  gauge, and required non-scalar sector behavior with documented reference
  tolerances.
* Nonstandard declared theories can define sectors, species, interactions,
  gauges, initial conditions, and projections through structured contracts
  rather than hidden solver branches.
* Runtime is governed by bounded caches, explicit convergence controls, and
  benchmark thresholds that keep the subsystem from drifting back into
  unmanaged snail-slow behavior.
* Docs, manifests, packaging smoke, and public API all tell the same truth.
