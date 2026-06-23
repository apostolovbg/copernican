# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-06-22
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

Use this plan to finish the CMB subsystem as a universal,
theory-agnostic, truth-audited package.

The previous optimization roadmap improved runtime and closed the first
native-baseline campaign, but it did not finish the declared-theory
acceptance boundary or the native module split. The remaining work is to
finish the native and standard solver split, make cache and runtime
ownership explicit, and close the native path around a single honest
contract: any mathematically well-posed theory that can be expressed
through Copernican's declared CMB contracts must be executable without
LCDM-specific branches or solver-family selectors.

The target condition remains non-negotiable:

* `standard: true` stays on the standard backend path.
* `standard: false` stays native, CAMB-free, and CLASS-free in production.
* The native path accepts complete declared theories rather than named theory
  families or hidden backend selectors.
* No hidden LCDM-like assumptions or scalar-only compatibility layers remain
  in the `standard: false` production path.
* The public CMB surface has one truth and one owner.
* The full governed validation path stays integral.
* There is no separate "quick development validation" lane.
* Publication-style validation remains a separate workflow and is not mixed
  into code-path validation or optimization work.
* `model_template.yml` remains documentation, not a benchmark or acceptance
  model.
* The end state must be a faster, cleaner, and more general Copernican, not
  a faster lie.

This is a forward-only plan. Do not preserve obsolete runtime bridges. Do not
reintroduce scalar-only compatibility layers. Do not introduce theory-family
selectors, `mode_families`, or backend selectors for the native path. Do not
hide remaining LCDM assumptions behind generic names.

## Table of Contents

* [Problem Preamble](#problem-preamble)
* [Current Baseline](#current-baseline)
* [Overview](#overview)
* [How Slices Are Executed](#how-slices-are-executed)
* [Execution Slices](#execution-slices)
* [Validation Routine](#validation-routine)
* [Completion Standard](#completion-standard)

## Problem Preamble

The native CMB engine now exists, but the subsystem is not finished.

The current baseline still has several concrete problems:

* the native solver file is still too large and mixes orchestration,
  background work, declared evolution, projection, cache ownership, and old
  native responsibilities;
* caches exist, but cache ownership, reset hooks, stats, and bounded-lifecycle
  rules are not yet a first-class subsystem contract;
* the declared native path still needs an explicit closure pass against hidden
  LCDM-like assumptions and scalar-only residue;
* docs and tests describe a strong native baseline, but the final acceptance
  boundary for a universal declared-theory infrastructure is not yet closed;
* the runtime is better than before, but the final benchmark and truth-audit
  evidence has not been recorded.

This plan exists to close those problems in a sequence that preserves working
slice boundaries. Every slice must end on a clean checkout that passes the
appropriate governed tests. A broader or faster design that leaves the
repository in a broken or half-truth state is not an acceptable outcome.

## Current Baseline

The previous optimization roadmap is considered complete in its own scope.
This roadmap starts from that faster baseline rather than reopening it.

Current facts:

* Copernican has a working native declared-graph CMB path for
  `standard: false`.
* `model_coder.py` already compiles a native runtime bundle and
  `engine_adapter.py` already hands that bundle to execution plugins.
* `copernican/lib/likelihoods/cmb/**` now has a split public/native package
  layout with explicit cache ownership.
* `cmb.py` now owns the public CMB facade and structured-contract dispatch.
* `camb_solver.py` now owns the standard-path CAMB helpers and imports.
* `copernican_cmb_solver.py` now owns internal native orchestration only.
* `native_background.py`, `native_evolution.py`, `native_projection.py`,
  and `native_cache.py` now own the split native internals.
* The native background resolver now accepts direct physical density inputs
  and precompiled runtimes now recognize declared background symbols.
* Declared native `k_sample_count`, `eta_sample_count`, and
  `source_grid_multiplier` are now exercised without hidden hard caps.
* Cache use, cache governance, and runtime diagnostics are explicit, but
  final benchmark evidence is still open work.
* Current tests and docs support a truthful feature baseline, not yet a final
  subsystem-closure baseline.

This roadmap therefore replaces the optimization-only framing with a smaller
number of slices that finish architecture, native generalization, and closure
evidence together.

## Overview

This campaign must preserve the existing physics and governance contracts
while finishing the subsystem boundary.

Required invariants:

* keep `standard: true` behavior intact;
* keep `standard: false` CAMB-free and CLASS-free in production;
* keep the native backend based on declared math rather than hardcoded theory
  families;
* keep failure modes explicit and fail-loud;
* keep the full governed validation suite integral;
* keep publication validation separate and unchanged;
* keep documentation, manifests, and public names truthful;
* prefer measured runtime wins over speculative rewrites.

Architecture direction:

* `cmb.py` is the only public CMB façade;
* `camb_solver.py` owns the standard path and all CAMB-only imports;
* `copernican_cmb_solver.py` becomes native orchestration only;
* native background, declared evolution, projection, and cache ownership move
  into focused internal modules under `copernican/lib/likelihoods/cmb/**`;
* `model_coder.py` owns immutable native-runtime compilation;
* `engine_adapter.py` owns plugin handoff of that runtime;
* tests may import internals directly from their internal modules instead of
  forcing private names through package `__all__`;
* the native executor must run complete declared contracts, not LCDM-shaped
  special cases hidden behind generic labels.

## How Slices Are Executed

* Each slice is a full implementation slice, not a note.
* Slices execute in dependency order.
* Each slice must end with a working checkout.
* Each slice must deliver its own tests and documentation updates where
  required.
* No slice may finish with knowingly broken CMB behavior.
* No slice may weaken validation to look faster.
* No slice may create a separate developer-only validation lane.
* No slice may hide runtime regressions behind looser tolerances.
* No slice may reintroduce CAMB or CLASS as production fallback for
  `standard: false`.
* No slice may preserve obsolete scalar-only or theory-family compatibility
  shims.
* No slice may preserve duplicate public façades when one truthful owner is
  available.
* No slice may use theory YAML edits as the main implementation vehicle.
* Prefer runtime gains early in the sequence, but not at the cost of a broken
  intermediate state.
* "Probable affected files" are guidance, not an allowlist.
* Stage completed slice changes.
* Do not commit or push unless explicitly instructed.

Task markers mean:

* [open] active work for this roadmap;
* [closed] completed and validated for this roadmap;
* [blocked] not executable until named dependencies close.

## Execution Slices

### [closed] Slice One - Public boundary closure and solver ownership reset

Purpose:

Finish the public CMB boundary so the package has one truthful API, one
dispatch path, and one owner for standard-versus-native routing.

Depends on:

* current native CMB baseline.

Probable affected files:

* `copernican/lib/likelihoods/__init__.py`
* `copernican/lib/likelihoods/cmb/__init__.py`
* `copernican/lib/likelihoods/cmb/cmb.py`
* `copernican/lib/likelihoods/cmb/camb_solver.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/engine_adapter.py`
* CMB module tests
* import and packaging smoke tests
* docs describing the public CMB surface
* `CHANGELOG.md`

Scope:

* keep `cmb.py` as the only public façade;
* remove duplicate public wrappers and duplicate likelihood classes from the
  native module;
* remove private CMB internals and `camb` from package public exports;
* keep CAMB-only helpers in `camb_solver.py`;
* replace the weak public `compute_cmb_spectrum_from_dict` naming with a
  contract-based public name;
* keep native-runtime dispatch ownership in one place.

Tasks:

* [closed] Make `copernican/lib/likelihoods/cmb/cmb.py` the only public owner
  of `CMBLike`, spectrum helpers, and dispatch logic.
* [closed] Remove duplicate public wrappers from
  `copernican_cmb_solver.py`.
* [closed] Remove duplicate public likelihood ownership from
  `copernican_cmb_solver.py`.
* [closed] Replace the public `compute_cmb_spectrum_from_dict` name with
  `compute_cmb_spectrum_from_contract`.
* [closed] Keep any temporary compatibility shim internal to the slice,
  not as a permanent public contract.
* [closed] Stop exporting underscore-prefixed CMB helpers from package
  `__all__`.
* [closed] Stop exporting `camb` from the package surface.
* [closed] Move every CAMB-only helper that still belongs on the standard path
  into `camb_solver.py`.
* [closed] Keep `model_coder.py` and `engine_adapter.py` aligned with the final
  public/native boundary.
* [closed] Rewrite module tests so internals are imported from their real
  internal modules, not through package re-exports.
* [closed] Add API-boundary tests that freeze the intended public surface.

Done when:

* [closed] `cmb.py` is the only public CMB façade.
* [closed] `copernican_cmb_solver.py` no longer exports a duplicate public
  API.
* [closed] Package `__all__` contains only intended public symbols.
* [closed] Public naming refers to structured contracts, not generic dicts.
* [closed] Relevant import, CMB, and ownership tests pass.

### [closed] Slice Two - Native module split and cache governance

Purpose:

Split the native solver into focused internal modules and make cache ownership,
reset behavior, and diagnostics explicit subsystem contracts.

Depends on:

* Slice One.

Probable affected files:

* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* new internal native modules under `copernican/lib/likelihoods/cmb/**`
* native CMB tests
* docs explaining the native module layout
* `CHANGELOG.md`

Scope:

* keep `copernican_cmb_solver.py` as native orchestration only;
* move background, recombination, and reionization logic into a focused
  internal module;
* move declared perturbation evolution into a focused internal module;
* move projection and line-of-sight logic into a focused internal module;
* move cache ownership into a focused internal module;
* ensure native modules do not import CAMB;
* make cache clear, stats, and bounded-size behavior explicit.

Tasks:

* [closed] Create a native background module for declared background,
  recombination, and reionization work.
* [closed] Create a native evolution module for declared perturbation
  execution.
* [closed] Create a projection module for transfer and line-of-sight work.
* [closed] Create a native cache module for all native cache ownership.
* [closed] Reduce `copernican_cmb_solver.py` to orchestration and native
  entry helpers only.
* [closed] Remove CAMB imports from native modules.
* [closed] Add explicit cache reset helpers for tests and long-lived runs.
* [closed] Add explicit cache stats or diagnostics helpers.
* [closed] Bound every native cache or document why it is intentionally
  bounded elsewhere.
* [closed] Add tests for cache reuse, cache separation, and cache reset.
* [closed] Add tests proving the native path remains CAMB-free in production
  execution.

Done when:

* [closed] The native solver is no longer one monolithic engine-room file.
* [closed] Cache lifecycle is explicit and testable.
* [closed] Native modules no longer own CAMB imports.
* [closed] Relevant CMB and cache tests pass.

### [closed] Slice Three - Theory-agnostic native contract closure

Purpose:

Close the native subsystem around the real goal: any mathematically
well-posed theory that can be expressed through Copernican's declared CMB
contracts must execute without LCDM-specific production branches.

Depends on:

* Slice Two.

Probable affected files:

* `copernican/lib/model_coder.py`
* `copernican/lib/engine_adapter.py`
* `copernican/lib/perturbation_contract.py`
* `copernican/lib/cmb_projection_contract.py`
* native CMB runtime modules
* native CMB tests and neutral synthetic fixtures
* docs describing contract semantics and native guarantees
* `CHANGELOG.md`

Scope:

* audit and remove remaining LCDM-like assumptions from the native production
  path;
* audit and remove remaining scalar-only compatibility layers from the native
  production path;
* keep the native route driven by declared graph roles and contracts rather
  than named theory families;
* broaden the declared contract where required so complete declared theories
  can execute end-to-end;
* justify native numerics and hard limits with explicit tests;
* define one truthful neutral acceptance fixture that is not TORG.

Tasks:

* [closed] Audit the `standard: false` production path for remaining
  LCDM-like assumptions and remove them.
* [closed] Audit the `standard: false` production path for remaining
  scalar-only compatibility layers and remove them.
* [closed] Keep generic background and observable handling role-driven, not
  theory-family-driven.
* [closed] Refuse `mode_families`, theory-family selectors, and hidden
  backend selectors as solution shapes.
* [closed] Expand or tighten contract semantics so complete declared theories
  can compile and execute end-to-end.
* [closed] Add one neutral synthetic native fixture that proves the engine
  without relying on TORG.
* [closed] Add intentionally invalid fixtures for the major declared-contract
  failure classes.
* [closed] Add convergence or sensitivity tests for `k`, `eta`, and related
  native numerics where hard limits or defaults materially affect results.
* [closed] Justify, revise, or remove native hard caps that remain from the
  optimization campaign.
* [closed] Keep `model_template.yml` documented as documentation, not as a
  benchmark or acceptance fixture.
* [closed] Document the exact native contract guarantees and failure surface
  truthfully.

Done when:

* [closed] Complete declared theories can execute through the native path
  without LCDM-specific production branches.
* [closed] No scalar-only compatibility layer remains in the native path.
* [closed] No theory-family selector or hidden backend selector exists.
* [closed] Native numerics and caps are justified by tests or revised.
* [closed] Relevant CMB, projection, perturbation-contract, model-coder, and
  engine-adapter tests pass.

### [open] Slice Four - Acceptance closure, benchmarks, and packaging truth

Purpose:

Close the CMB subsystem with measured runtime evidence, packaging evidence,
public-API freeze coverage, and truthful final docs.

Depends on:

* Slice Three.

Probable affected files:

* native CMB runtime modules
* benchmark or profiling helpers
* packaging smoke tests
* public API freeze tests
* docs
* manifests if runtime provenance changes
* `CHANGELOG.md`

Scope:

* measure before-and-after runtime for representative native CMB workloads;
* measure before-and-after runtime for representative governed CMB validation
  workloads;
* prove the installed package still works for the completed CMB surface;
* freeze the final public API boundary in tests;
* audit the completed architecture for truth and maintainability;
* document runtime expectations honestly;
* close the plan without hidden deferred structural debt.

Tasks:

* [open] Record before-and-after runtime for representative native CMB
  prediction workloads.
* [open] Record before-and-after runtime for representative governed CMB
  validation workloads.
* [open] Add or finish benchmark and profiling helpers needed to reproduce
  those measurements.
* [open] Add installed-package smoke coverage for package import, CLI
  import path, and representative standard/native CMB calls.
* [open] Add public API freeze tests for the final CMB package surface.
* [open] Audit that `standard: true` remains on the standard backend path.
* [open] Audit that `standard: false` remains CAMB-free and CLASS-free in
  production.
* [open] Audit that runtime ownership is clear between `model_coder.py`,
  `engine_adapter.py`, and the native CMB package.
* [open] Keep manifest and diagnostic truth intact after the final
  architecture cleanup.
* [open] Remove temporary scaffolding that should not remain after closure.
* [open] Ensure docs describe the final package layout, contract scope, and
  runtime expectations honestly.
* [open] Ensure changelog and closure docs record the subsystem outcome
  truthfully.

Done when:

* [open] Runtime improvements are measured rather than guessed.
* [open] Installed-package CMB smoke coverage is green.
* [open] The final public CMB API is frozen in tests.
* [open] The governed validation path remains integral and green.
* [open] Runtime expectations are documented honestly.
* [open] The subsystem can close without stale temporary scaffolding.

## Validation Routine

Run the validation routine after each completed slice.

Minimum validation:

* inspect the working tree;
* run targeted tests for touched code;
* run full native CMB tests when native CMB behavior changes;
* run perturbation contract tests when graph-compilation behavior changes;
* run projection contract tests when projection behavior changes;
* run model-coder and engine-adapter tests when runtime ownership or contract
  semantics change;
* run import and packaging smoke checks when package layout or public API
  changes;
* record benchmark evidence in the slices that promise benchmark evidence;
* run docs checks when public behavior or structure changes;
* run DevCovenant verification;
* update `CHANGELOG.md` when behavior, structure, docs, tests, validation, or
  governance changes;
* stage completed slice changes;
* do not commit or push unless instructed.

Per-slice closure validation:

* the slice ends on a working checkout;
* the slice does not leave a knowingly broken intermediate state behind;
* tests are not weakened to pass;
* runtime claims are backed by measured evidence where the slice promises
  speed;
* failures remain explicit and user-facing;
* documentation matches the implemented runtime boundary;
* generated artifacts remain generated.

Completion validation:

* `standard: true` remains CAMB-compatible;
* `standard: false` does not use CAMB or CLASS for production prediction;
* `cmb.py` is the only public CMB façade;
* no private CMB internals are exported as public package API;
* CAMB-only imports live on the standard path;
* native runtime compilation happens once per built model runtime, not once
  per likelihood call;
* native modules have focused ownership boundaries;
* cache lifecycle is explicit and testable;
* complete declared theories can execute through the native path without
  LCDM-specific production branches;
* no scalar-only compatibility layer remains;
* no theory-family selector, `mode_families`, or hidden backend selector
  exists;
* native numerics and hard limits are justified by tests or revised;
* benchmark and packaging evidence is recorded;
* the full governed CMB validation path remains integral;
* no developer-only validation lane exists;
* publication validation workflows remain separate and unchanged;
* docs and templates match the completed architecture;
* full relevant test suites pass;
* DevCovenant gate closes.

## Completion Standard

This campaign is not complete merely because the code moved again.

The campaign is complete when:

* the native CMB backend is faster in measured practice;
* the public boundary is singular and truthful;
* native and standard ownership are cleanly separated;
* the native path is truly theory-agnostic within Copernican's declared
  contract system;
* `standard: true` and `standard: false` still tell the truth;
* the full governed validation path remains the only code-validation path;
* publication validation remains separate;
* benchmark, packaging, docs, manifests, and changelog entries remain
  truthful;
* Copernican is cleaner to extend and faster to develop on without becoming
  less rigorous.
