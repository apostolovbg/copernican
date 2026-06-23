# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-06-23
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

Use this plan to finish the CMB subsystem as a universal,
theory-agnostic, native Boltzmann-hierarchy infrastructure that is honest,
governed, and operationally fast enough for real development work.

The current native declared-graph engine is the completed baseline, not the
final claim. The remaining campaign must add first-class hierarchy machinery,
native standard-model parity, complete observable sectors, and explicit
runtime governance without reintroducing CAMB or CLASS into the
`standard: false` production path.

The target condition is non-negotiable:

* `standard: true` stays on the standard backend path.
* `standard: false` stays native, CAMB-free, and CLASS-free in production.
* The native path accepts complete declared theories rather than named theory
  families, `mode_families`, or hidden solver selectors.
* No hidden LCDM-like assumptions or scalar-only compatibility layers remain
  in the `standard: false` production path.
* The public CMB surface has one truth and one owner.
* The full governed validation path stays integral.
* There is no separate "quick development validation" lane.
* Publication-style validation remains a separate workflow and is not mixed
  into code-path validation or optimization work.
* `model_template.yml` remains documentation, not a benchmark or acceptance
  model.
* Efficiency is a product requirement. Each slice must either reduce runtime
  cost or make that cost explicit, bounded, and governed.

This is a forward-only plan. Do not preserve obsolete runtime bridges. Do not
hide remaining LCDM assumptions behind generic names. Do not reopen solved
surface-ownership work unless a later slice requires a truthful adjustment.

## Table of Contents

* [Problem Preamble](#problem-preamble)
* [Current Baseline](#current-baseline)
* [Overview](#overview)
* [Execution Rules](#execution-rules)
* [Execution Slices](#execution-slices)
* [Completion Standard](#completion-standard)

## Problem Preamble

Copernican already has a real native declared-graph CMB engine, but it does
not yet have a full universal theory-agnostic Boltzmann-hierarchy solver.

The remaining gap is not one more cleanup pass. The remaining gap is the
physics and runtime substrate that lets the native path:

* express hierarchy families rather than only finite hand-declared ODE sets;
* reproduce standard-model CMB observables through native execution;
* support polarization, tensor, lensing, gauge, and neutrino completeness;
* accept custom species and interactions without hidden theory selectors; and
* stay fast enough that governed development does not collapse under
  hour-scale regression cycles.

This roadmap exists to close that gap in a sequence that preserves working
slice boundaries. Every slice must end on a clean checkout that passes the
appropriate governed tests. A broader design that leaves the repository in a
broken or misleading state is not an acceptable outcome.

## Current Baseline

The previous CMB campaign is complete in its scope and becomes the baseline
for this roadmap.

Current facts:

* Copernican has a working native declared-graph CMB path for
  `standard: false`.
* `model_coder.py` already compiles a native runtime bundle and
  `engine_adapter.py` already hands that bundle to execution plugins.
* `copernican/lib/likelihoods/cmb/**` already has a split public/native
  layout with explicit cache ownership.
* `cmb.py` owns the public CMB façade and structured-contract dispatch.
* `camb_solver.py` owns the standard-path CAMB helpers and imports.
* `copernican_cmb_solver.py` owns internal native orchestration.
* `native_background.py`, `native_evolution.py`, `native_projection.py`,
  and `native_cache.py` own split native internals.
* The native background resolver accepts direct physical density inputs and
  precompiled runtimes recognize declared background symbols.
* Declared native sampling controls are exercised without hidden hard caps.
* Cache use, cache governance, and runtime diagnostics are explicit, but the
  hierarchy layer and final runtime envelope are still open work.

This roadmap therefore starts from a truthful native engine and uses a small
number of implementation slices to reach the universal solver target.

## Overview

This campaign has one job: turn the current native declared-graph engine into
a full native Boltzmann-hierarchy subsystem without giving back truth,
governance, or runtime sanity.

The slice order is deliberate:

* Slice One finishes hierarchy-capable runtime ownership and removes
  remaining hot-path compilation.
* Slice Two establishes an efficient native scalar standard-model baseline.
* Slice Three closes observable-sector support around polarization, tensor,
  and lensing behavior.
* Slice Four closes gauges, massive neutrinos, and first-class initial
  conditions.
* Slice Five closes theory extensions, convergence governance, packaging
  truth, and production performance in the same slice.

## Execution Rules

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
* Compile symbolic contracts upstream. Do not compile math inside the
  likelihood hot path.
* The final slice must close on its own implementation, tests, docs, and
  performance evidence. There is no separate validation-only slice after it.
* "Probable affected files" are guidance, not an allowlist.
* Stage completed slice changes.
* Do not commit or push unless explicitly instructed.

Task markers mean:

* [open] active work for this roadmap;
* [closed] completed and validated for this roadmap;
* [blocked] not executable until named dependencies close.

## Execution Slices

### [open] Slice One - Universal hierarchy substrate and compile-time
ownership

Purpose:

Extend the native contract and runtime substrate so Copernican can express
Boltzmann-hierarchy structure directly and compile it upstream, outside the
likelihood hot path.

Depends on:

* current native declared-graph baseline.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/engine_adapter.py`
* `copernican/lib/run_manifest.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* CMB contract, compiler, and manifest tests
* docs describing native contract structure
* `CHANGELOG.md`

Scope:

* extend structured native CMB contracts so they can declare sectors,
  species, hierarchy families, collision operators, closures, gauges,
  initial-condition families, projection typing, and accuracy controls;
* finish moving symbolic compilation ownership into
  `perturbation_contract.py`, `model_coder.py`, and `engine_adapter.py`;
* make the native runtime payload immutable, cache-keyed, and ready for
  numeric binding only inside the likelihood hot path;
* remove any remaining math-compilation or expression-planning work from
  `copernican/lib/likelihoods/cmb/**` execution code;
* add synthetic hierarchy fixtures and fail-loud validation for malformed
  declarations;
* expose enough manifest or diagnostics data to prove that hot-path
  recompilation is gone.

Tasks:

* [open] Add contract and runtime support for sectors, species, hierarchy
  families, collision operators, closures, gauges, initial-condition
  families, and projection typing.
* [open] Compile hierarchy-capable runtimes upstream in `model_coder.py`
  rather than inside likelihood execution.
* [open] Pass immutable runtime bundles through `engine_adapter.py`.
* [open] Remove residual symbolic compile and planning work from native
  likelihood execution modules.
* [open] Add cache keys and diagnostics for compiled hierarchy runtimes.
* [open] Extend manifest surfaces so they describe the resolved native route
  truthfully.
* [open] Add compiler and runtime tests for malformed hierarchy declarations,
  missing closures, invalid gauges, and hot-path recompilation regressions.

Done when:

* [open] Native likelihood execution binds numbers and runs numerics without
  compiling symbolic math at evaluation time.
* [open] Contracts can describe hierarchy families and sector/species
  metadata without solver selectors.
* [open] Runtime bundles are immutable, cache-governed, and manifest-visible.
* [open] Governed tests prove that malformed hierarchy contracts fail early
  and that repeated evaluations reuse compiled plans.

### [open] Slice Two - Native scalar standard-model parity and efficient
scalar execution

Purpose:

Build the native scalar hierarchy needed for a truthful standard-model
acceptance target and make the scalar hot path materially faster than the
current baseline.

Depends on:

* Slice One.

Probable affected files:

* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_coder.py`
* scalar native-reference fixtures and tests
* docs describing native scalar acceptance behavior
* `CHANGELOG.md`

Scope:

* implement first-class scalar hierarchy generation for photons, baryons,
  CDM, massless neutrinos, metric equations, Thomson coupling,
  recombination, and reionization through native contracts;
* introduce a neutral native standard-model acceptance fixture under
  `standard: false` so standard scalar physics can run without CAMB in
  production;
* add tight-coupling and other stiff-regime handling needed for efficient
  scalar evolution;
* profile and remove dominant scalar hot-path waste in the solver, adapter,
  and runtime handoff;
* compare native scalar outputs to CAMB reference outputs with documented
  tolerances and runtime expectations.

Tasks:

* [open] Add scalar hierarchy generation and scalar closure machinery for the
  standard native acceptance route.
* [open] Add adiabatic scalar initial-condition generation and native scalar
  source construction.
* [open] Implement tight-coupling and scalar source batching for efficient
  early-time evolution.
* [open] Create a neutral native standard-model scalar fixture that exercises
  TT, TE, and EE through `standard: false`.
* [open] Remove dominant scalar hot-path rebuilds, including invariant table
  work that still repeats per evaluation.
* [open] Add governed parity tests and runtime checks that keep scalar
  execution from drifting back toward hour-scale regression behavior.

Done when:

* [open] A native standard-model scalar contract runs CAMB-free under
  `standard: false`.
* [open] Native scalar TT, TE, EE, and required background observables match
  CAMB within documented tolerances.
* [open] The scalar path no longer recompiles or rebuilds invariant runtime
  structures during likelihood evaluation.
* [open] Governed tests and runtime checks show a measured scalar runtime
  improvement or an explicit bounded envelope.

### [open] Slice Three - Polarization, sector, and lensing completion

Purpose:

Extend the native solver from scalar temperature acceptance to complete
observable-sector support, including polarization, non-scalar sector
plumbing, lensing, and efficient projection.

Depends on:

* Slice Two.

Probable affected files:

* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* projection and sector contract tests
* CMB runtime and scientific-reference tests
* docs describing native observable support
* `CHANGELOG.md`

Scope:

* extend native hierarchy machinery from scalar acceptance to polarization
  and non-scalar sector support;
* implement E and B source generation, tensor support, vector-aware sector
  plumbing, lensing-potential transfer, lensed spectra assembly, and
  supported cross-spectra;
* type-check source, kernel, and observable compatibility by sector, spin,
  and parity;
* tighten projection and line-of-sight runtime with batch integration,
  Bessel reuse, and bounded kernel caches;
* compare native outputs to standard references where physics overlaps and
  use analytic fixtures elsewhere.

Tasks:

* [open] Add tensor support and vector-aware sector plumbing so the native
  path is no longer architecturally scalar-only.
* [open] Implement native polarization source and projection flow for E-mode
  and B-mode outputs.
* [open] Implement lensing-potential transfer, PP output, lensed spectra,
  and supported temperature or polarization cross terms.
* [open] Enforce sector, spin, and parity compatibility in the projection
  registry and fail before runtime on invalid mixes.
* [open] Optimize line-of-sight batching, Bessel reuse, and kernel caches so
  broader observable support does not erase Slice Two runtime gains.
* [open] Add governed reference and analytic tests for polarization,
  lensing, tensor, and sector-mismatch behavior.

Done when:

* [open] The native path produces the declared temperature, polarization,
  and lensing outputs without CAMB production fallback.
* [open] Invalid sector or projection mixes fail loudly before evolution.
* [open] Runtime gains from projection batching and cache reuse are measured
  and governed.
* [open] Native scientific-reference and analytic tests support the expanded
  observable surface truthfully.

### [open] Slice Four - Massive neutrinos, gauges, and initial-condition
completeness

Purpose:

Finish the native hierarchy substrate for momentum-dependent species, gauge
completeness, and first-class initial-condition ownership.

Depends on:

* Slice Three.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* gauge, neutrino, and initial-condition tests
* docs describing gauge and species support
* `CHANGELOG.md`

Scope:

* add massless and massive-neutrino hierarchy families, momentum-grid
  runtime ownership, and background-to-perturbation coupling;
* implement conformal Newtonian and synchronous gauge support with
  gauge-equivalent observable checks;
* add a first-class initial-condition engine for adiabatic, standard
  isocurvature, tensor, and declared eigenmode or shooting cases;
* enforce conservation and constraint validation across background and
  perturbation systems;
* keep the added families efficient through shared momentum-grid caches,
  gauge-stable compiled plans, and bounded refinement controls.

Tasks:

* [open] Add hierarchy support for massless and massive neutrinos, including
  momentum-grid declarations and runtime ownership.
* [open] Add native gauge support for the required standard gauges and
  validate gauge-equivalent observables.
* [open] Add first-class initial-condition generation for standard modes and
  declared nonstandard seeding cases.
* [open] Add fail-loud conservation, constraint, and gauge-consistency
  validation.
* [open] Optimize momentum-grid and gauge-heavy runtime paths so the solver
  does not become snail-slow as completeness increases.
* [open] Add governed tests for massive-neutrino behavior, gauge parity, and
  initial-condition correctness.

Done when:

* [open] The native path supports the required gauge and neutrino machinery
  without hidden LCDM-only assumptions.
* [open] Gauge-equivalent observables agree within documented tolerances.
* [open] Initial conditions are generated and validated natively rather than
  being left to ad hoc model-side hacks.
* [open] Runtime costs for momentum-dependent species are explicit, bounded,
  and covered by governed checks.

### [open] Slice Five - Theory-extension and production performance closure

Purpose:

Close the subsystem around complete declared theories, explicit convergence
controls, truthful packaging, and production-grade runtime governance.

Depends on:

* Slice Four.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/run_manifest.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* packaging, manifest, benchmark, and installation tests
* docs describing theory-extension and performance contracts
* `CHANGELOG.md`

Scope:

* finish the extension surface for custom species, custom interactions,
  collision operators, modified conservation declarations, modular
  recombination and reionization hooks, and custom projection kernels;
* lock convergence and accuracy controls for `ell`, `k`, `eta`, hierarchy,
  and momentum-grid refinement into governed runtime contracts and fail when
  under-resolved;
* add installed-package smoke, public API freeze, manifest truth, benchmark
  thresholds, and docs for native standard-model and representative
  synthetic theories;
* tune remaining hotspots discovered during extension work so the end state
  is operationally faster than today's baseline rather than merely broader;
* close the final slice on its own tests, docs, and runtime evidence.

Tasks:

* [open] Add custom species, interaction, collision, recombination,
  reionization, and projection-extension contracts with fail-loud
  validation.
* [open] Add convergence and accuracy controls that govern refinement and
  reject under-resolved native runs.
* [open] Add installed-package smoke, public API freeze, manifest, and
  benchmark coverage for the native CMB subsystem.
* [open] Tune remaining solver hotspots and record a governed runtime
  envelope that prevents a return to unmanaged hour-scale regressions.
* [open] Finish docs so public claims, manifests, and benchmark truth match
  the shipped code.

Done when:

* [open] Complete declared theories with custom species and interactions can
  execute through `standard: false` without solver selectors or LCDM-shaped
  bridges.
* [open] Native convergence and runtime behavior are explicit, bounded, and
  enforced by governed checks.
* [open] Installed-package smoke, manifest truth, public API freeze, and
  benchmark coverage all pass from the same slice.
* [open] No separate validation-only slice remains after this one.

## Completion Standard

This roadmap is complete only when all five slices are closed and the
resulting repository can truthfully claim all of the following:

* Copernican ships a native Boltzmann-hierarchy CMB infrastructure whose
  `standard: false` route compiles complete declared theories upstream and
  executes them natively with no CAMB or CLASS production fallback.
* Standard-model native acceptance covers scalar, polarization, lensing, and
  required non-scalar sector behavior with documented reference tolerances.
* Nonstandard declared theories can define sectors, species, interactions,
  gauges, initial conditions, and projections through structured contracts
  rather than hidden solver branches.
* Runtime is governed by bounded caches, explicit convergence controls, and
  benchmark thresholds that keep the subsystem from drifting back into
  unmanaged snail-slow behavior.
* Docs, manifests, packaging smoke, and public API all tell the same truth.
