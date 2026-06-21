# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-06-21
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

Use this plan to finish the native CMB optimization and architecture cleanup.

This roadmap replaces the completed seven-slice closure plan. The native
declared-graph backend now exists, has a truthful closed-feature baseline,
and is too slow for sane governed development. The remaining work is not
"add more physics before anything else." The remaining work is to move
responsibilities to the correct layers, stop recompiling or rebuilding math in
the runtime hot path, split the monolithic likelihood file into maintainable
modules, and reduce governed runtime without weakening validation.

The target condition remains non-negotiable:

* `standard: true` stays on the standard backend path.
* `standard: false` stays native, CAMB-free, and CLASS-free in production.
* The full governed validation path stays integral.
* There is no separate "quick development validation" lane.
* Publication-style validation remains a separate workflow and is not mixed
  into code-path optimization work.
* `model_template.yml` remains documentation, not a benchmark model.
* The end state must be a faster and cleaner Copernican, not a faster lie.

This is a forward-only plan. Do not preserve obsolete runtime bridges. Do not
reintroduce scalar-only compatibility layers. Do not introduce theory-family
selectors, `mode_families`, or backend selectors for the native path. Do not
hide LCDM-like assumptions behind the optimization work.

## Table of Contents

* [Problem Preamble](#problem-preamble)
* [Current Baseline](#current-baseline)
* [Overview](#overview)
* [How Slices Are Executed](#how-slices-are-executed)
* [Execution Slices](#execution-slices)
* [Validation Routine](#validation-routine)
* [Completion Standard](#completion-standard)

## Problem Preamble

The native CMB engine is now functionally present but structurally expensive.
The current cost is not explained by missing physics. It is explained by where
the existing physics is assembled and executed.

The investigated baseline shows several concrete problems:

* native `standard: false` prediction still routes through CAMB-style
  contract materialization that it does not need;
* `model_coder.py`, `engine_adapter.py`, and the native CMB likelihood share
  overlapping responsibility for compiling, normalizing, and carrying declared
  math;
* `copernican/lib/likelihoods/cmb.py` has become a monolith that mixes
  interface, compilation, background handling, solver execution, projection,
  caching, and validation-facing helpers;
* declared expressions are validated too late and re-interpreted too often
  inside hot loops;
* graph-context construction does too much dictionary-heavy work inside
  repeated solver stages;
* background, recombination, per-k evolution, and projection all repeat work
  that should be hoisted or cached;
* governed tests pay avoidable setup costs on top of the native-runtime cost.

This plan exists to fix those structural problems in a sequence that preserves
working slice boundaries. Every slice must end on a clean checkout that passes
the appropriate governed tests. A runtime gain that leaves the repository in a
broken mid-state is not an acceptable slice outcome.

## Current Baseline

The previous closure roadmap is considered complete in its own scope. This new
plan starts from that closed baseline rather than reopening it.

Current facts:

* Copernican has a working native declared-graph CMB path for
  `standard: false`.
* The native path already has meaningful validation and truthful docs for the
  closed feature baseline.
* The current bottleneck is architecture and runtime cost, not lack of a
  native engine.
* The largest structural overlaps are between `model_coder.py`,
  `engine_adapter.py`, and the native CMB likelihood runtime.
* The largest code-organization problem is the size and mixed ownership of
  `copernican/lib/likelihoods/cmb.py`.
* The largest workflow problem is that governed CMB-heavy runs are too slow
  for practical iteration.

This roadmap therefore restarts slice numbering and focuses only on the
optimization and refactor campaign needed after feature closure.

## Overview

The optimization campaign must preserve the existing physics and governance
contracts while moving the native backend toward a cleaner architecture.

Required invariants:

* keep `standard: true` behavior intact;
* keep `standard: false` CAMB-free and CLASS-free in production;
* keep the native backend based on declared math rather than hardcoded theory
  families;
* keep failure modes explicit and fail-loud;
* keep the full governed validation suite integral;
* keep publication validation separate and unchanged;
* keep documentation and manifests truthful;
* prefer measured runtime wins over speculative rewrites.

Architecture direction:

* native math compilation belongs upstream, near `model_coder.py`;
* `engine_adapter.py` should own transfer of compiled native runtime objects
  into execution plugins;
* the native likelihood should consume prepared runtime objects rather than
  rebuilding raw contracts;
* the standard backend path and the native backend path should live in a
  split likelihood package with clear module boundaries;
* hot-loop execution should run compiled evaluator plans instead of repeated
  mapping-heavy interpretation.

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

### [open] Slice One - Ownership reset and likelihood package split

Purpose:

Move native-runtime ownership to the correct layers and split the CMB
likelihood into maintainable modules while preserving current physics. This
slice must deliver the first meaningful runtime relief by removing needless
contract rebuilding before deeper solver optimization begins.

Depends on:

* current closed native CMB baseline.

Probable affected files:

* `copernican/lib/model_coder.py`
* `copernican/lib/engine_adapter.py`
* `copernican/lib/perturbation_contract.py`
* `copernican/lib/likelihoods/__init__.py`
* `copernican/lib/likelihoods/_protocol.py`
* `copernican/lib/likelihoods/cmb.py`
* new `copernican/lib/likelihoods/cmb/**` package modules
* import smoke tests
* native CMB tests
* docs describing likelihood layout and native runtime ownership
* `CHANGELOG.md`

Scope:

* split the monolithic CMB likelihood file into a package;
* separate standard-backend orchestration from native-backend execution;
* move native declared-math compilation ownership upstream;
* make the plugin hand off a compiled native runtime object directly;
* stop the native hot path from materializing CAMB-style contracts that it
  does not need;
* remove or rename the private `_protocol.py` circular-import shim into a
  clear shared interface module if that abstraction remains necessary.

Tasks:

* [open] Create a `copernican/lib/likelihoods/cmb/**` package.
* [open] Create `copernican/lib/likelihoods/cmb/cmb.py` as the public CMB
  likelihood entrypoint.
* [open] Create `copernican/lib/likelihoods/cmb/camb_solver.py` for the
  standard backend path.
* [open] Create `copernican/lib/likelihoods/cmb/copcmb_solver.py` for the
  native declared backend path.
* [open] Move shared helpers into focused support modules instead of one
  giant file.
* [open] Preserve the public import surface intentionally while the file move
  happens.
* [open] Move native runtime compilation ownership to `model_coder.py`.
* [open] Make `engine_adapter.py` carry the compiled native runtime bundle.
* [open] Add a native-runtime accessor separate from CAMB-contract access.
* [open] Stop `standard: false` prediction from calling
  `get_camb_contract()` in the hot path.
* [open] Stop deep-copying native contract payloads per likelihood call.
* [open] Stop recompiling declared perturbation runtime objects from raw
  mappings inside the likelihood.
* [open] Keep manifest and route provenance truthful after the ownership
  move.
* [open] Keep `standard: true` behavior intact.
* [open] Add tests proving the native path consumes the compiled runtime
  handoff.
* [open] Add or update import smoke tests for the new package layout.

Done when:

* [open] `copernican/lib/likelihoods/cmb.py` is replaced by a focused
  package layout.
* [open] Native CMB runtime ownership is no longer split across three layers
  in contradictory ways.
* [open] `standard: false` no longer materializes a CAMB-style contract in
  the prediction hot path.
* [open] The private `_protocol.py` shim is gone or replaced with a clear
  interface module.
* [open] The first runtime win is measurable before deeper optimization.
* [open] Relevant import, CMB, and ownership tests pass.

### [open] Slice Two - Compiled native executor and graph hot-path cleanup

Purpose:

Replace mapping-heavy interpreted graph execution with compiled evaluator
plans and dense runtime structures. This slice targets the core native solver
cost after Slice One removes the worst ownership mistakes.

Depends on:

* Slice One.

Probable affected files:

* `copernican/lib/model_coder.py`
* `copernican/lib/engine_adapter.py`
* `copernican/lib/perturbation_contract.py`
* native CMB runtime modules under
  `copernican/lib/likelihoods/cmb/**`
* native CMB tests
* perturbation contract tests
* docs explaining compiled-native runtime behavior
* `CHANGELOG.md`

Scope:

* compile executable evaluator plans once;
* replace repeated dictionary resolution in hot loops;
* replace state lookups by dense slot-based access where appropriate;
* hoist contract-static and k-independent work out of repeated solver stages;
* preserve existing fail-loud diagnostics.

Tasks:

* [open] Compile declared background evaluators once per built runtime.
* [open] Compile recombination and reionization auxiliary evaluators once per
  built runtime.
* [open] Compile perturbation equations, closures, constraints, and sources
  into native execution plans.
* [open] Precompute dependency order and immutable graph metadata.
* [open] Replace recursive mapping-heavy graph-context resolution in hot
  loops with indexed evaluator plans.
* [open] Replace per-stage symbol dictionaries with dense state slots where
  that reduces overhead without hiding diagnostics.
* [open] Hoist contract-static work out of Runge-Kutta stages.
* [open] Hoist k-independent work out of per-k evolution where valid.
* [open] Remove dead native CMB cache or registry plumbing that no longer
  earns its keep.
* [open] Keep missing-quantity and invalid-math failures explicit after the
  executor rewrite.
* [open] Add focused tests proving closures, equations, and source channels
  still change outputs in the intended ways.
* [open] Add focused tests proving the compiled executor preserves validated
  behavior within current tolerances.

Done when:

* [open] The native hot path no longer reinterprets declared graph structure
  through mapping-heavy helpers inside repeated solver stages.
* [open] Contract-static work is compiled once and reused.
* [open] Old dead-path registries or no-op caches are removed.
* [open] Native CMB outputs remain within the validated scope of the current
  baseline.
* [open] Relevant CMB and perturbation-contract tests pass.

### [open] Slice Three - Projection, caching, and governed-suite runtime

Purpose:

Reduce the remaining runtime cost in projection, line-of-sight work, and
governed test logistics without splitting the validation path.

Depends on:

* Slice Two.

Probable affected files:

* native CMB runtime modules under
  `copernican/lib/likelihoods/cmb/**`
* `copernican/lib/cmb_projection_contract.py`
* native CMB test helpers and fixtures
* CMB validation helpers
* docs describing runtime expectations
* `CHANGELOG.md`

Scope:

* profile the post-Slice-Two runtime to rank the remaining bottlenecks;
* optimize line-of-sight and projection work;
* add legitimate caches for immutable inputs;
* reduce repeated setup work in governed tests;
* keep the full governed suite as the single code-validation path.

Tasks:

* [open] Profile the post-Slice-Two native runtime before changing the next
  hot path.
* [open] Vectorize line-of-sight accumulation where the math permits.
* [open] Vectorize projection loops where the math permits.
* [open] Cache Bessel grids by immutable inputs.
* [open] Cache background products when inputs are identical.
* [open] Cache recombination products when inputs are identical.
* [open] Cache other immutable transfer or projection intermediates where the
  reuse case is real.
* [open] Hoist per-k and per-ell invariant work out of nested loops.
* [open] Parallelize independent k-block work only if deterministic and
  overhead-justified.
* [open] Remove repeated model compilation or plugin construction in CMB
  tests where that setup can be shared safely.
* [open] Cache immutable reference-backed validation products where
  legitimate.
* [open] Keep the full governed validation suite integral; do not invent a
  separate quick lane.
* [open] Keep the publication-style validation module separate and unchanged.
* [open] Mark intentionally expensive tests honestly without weakening them.
* [open] Document runtime-sensitive test-helper conventions for future CMB
  work.

Done when:

* [open] Projection and line-of-sight cost are materially reduced.
* [open] The governed CMB-heavy suite is materially faster.
* [open] No assertions are weakened for speed.
* [open] No separate developer-only validation path exists.
* [open] Publication validation remains unchanged.
* [open] Relevant CMB, projection, and validation-helper tests pass.

### [open] Slice Four - Runtime closure, benchmarks, and truth audit

Purpose:

Close the optimization campaign with measured runtime evidence, truthful docs,
and a clear architecture boundary.

Depends on:

* Slice Three.

Probable affected files:

* native CMB runtime modules
* benchmark or profiling helpers
* docs
* templates
* manifests if runtime provenance changes
* `CHANGELOG.md`

Scope:

* measure before-and-after runtime for representative native CMB workloads;
* audit the optimized architecture for truth and maintainability;
* document runtime expectations honestly;
* close the plan without hidden deferred structural debt.

Tasks:

* [open] Record before-and-after runtime for representative native CMB
  prediction workloads.
* [open] Record before-and-after runtime for representative governed CMB test
  workloads.
* [open] Audit that `standard: true` remains on the standard backend path.
* [open] Audit that `standard: false` remains CAMB-free and CLASS-free in
  production.
* [open] Audit that runtime ownership is clear between `model_coder.py`,
  `engine_adapter.py`, and the native CMB package.
* [open] Audit that no scalar-only compatibility layer was reintroduced.
* [open] Audit that no theory-family selector or hidden backend selector was
  introduced.
* [open] Keep `model_template.yml` documented as documentation, not as a
  benchmark model.
* [open] Keep manifest and diagnostic truth intact after optimization.
* [open] Remove temporary optimization scaffolding that should not remain.
* [open] Ensure docs describe the optimized package layout and runtime
  expectations honestly.
* [open] Ensure changelog and closure docs record the optimization outcome
  truthfully.

Done when:

* [open] Runtime improvements are measured rather than guessed.
* [open] The optimized native backend has a clear ownership boundary.
* [open] The governed validation path remains integral and green.
* [open] Runtime expectations are documented honestly.
* [open] The optimization campaign can close without stale temporary
  scaffolding.

## Validation Routine

Run the validation routine after each completed slice.

Minimum validation:

* inspect the working tree;
* run targeted tests for touched code;
* run full native CMB tests when native CMB behavior changes;
* run perturbation contract tests when graph-compilation behavior changes;
* run projection contract tests when projection behavior changes;
* run model coder and engine adapter tests when runtime ownership changes;
* run import smoke checks when package layout changes;
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
* native runtime compilation happens once per built model runtime, not once
  per likelihood call;
* native prediction no longer materializes CAMB-style contracts in the hot
  path;
* the CMB likelihood is split into a maintainable package;
* no private `_protocol.py` circular-import shim remains unless a clearly
  named shared interface replaces it;
* compiled evaluator plans replace repeated mapping-heavy graph resolution in
  hot loops;
* projection and line-of-sight costs are materially reduced;
* the full governed CMB validation path remains integral;
* no developer-only validation lane exists;
* publication validation workflows remain separate and unchanged;
* docs and templates match the optimized architecture;
* full relevant test suites pass;
* DevCovenant gate closes.

## Completion Standard

The optimization campaign is not complete merely because the code moved.

The optimization campaign is complete when:

* the native CMB backend is faster in measured practice;
* the biggest structural bottlenecks have been removed from the hot path;
* runtime ownership is clear and maintainable;
* `standard: true` and `standard: false` still tell the truth;
* the full governed validation path remains the only code-validation path;
* publication validation remains separate;
* documentation, manifests, and changelog entries remain truthful;
* Copernican is meaningfully faster to develop on without becoming less
  rigorous.
