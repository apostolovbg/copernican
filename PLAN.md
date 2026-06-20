# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-06-20
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

Use this plan to bring Copernican's native CMB engine to fruition.

This roadmap replaces the previous two-megaslice structure. The old structure
was useful while the engine did not exist. It is now too coarse. The current
`new` branch has already built the declared graph foundation and the first
physical native CMB engine pass. The plan must now stop treating that work as
an almost-finished Slice One and instead mark the earned baseline honestly.

The purpose of this roadmap remains non-negotiable:

* `standard: true` remains the standard backend path for CAMB-compatible
  models.
* `standard: false` is Copernican's native path for declared math.
* Any mathematically well-posed cosmological theory should be ingestible if
  it is expressible through the model contract.
* Invalid, incomplete, contradictory, unsupported, or numerically ill-posed
  math must fail clearly.
* Copernican must not silently fall back to CAMB, CLASS, LCDM assumptions,
  fake spectra, fitted templates, hidden amplitude hacks, or theory-specific
  Python solvers when `standard: false`.

This is the target condition for the full roadmap, not a claim that the
current branch already solves every well-posed theory. Slice Five and
Slice Six must remove the specific current limits named below so complete
declared theories become executable within the model contract.

This is a forward-only plan. Do not preserve obsolete schema by adding
compatibility layers. Do not reintroduce scalar-only theory ceilings. Do not
classify whole theories into solver families. Variables, sources, equations,
and projections may carry metadata, but the whole theory is one declared
mathematical system.

This is not a tests-first plan. Each slice is an implementation, validation,
or closure pass with tests created or updated as part of the work. Do not mark
a slice complete because weak tests pass. Mark it complete only when the code,
tests, docs, and failure modes match the slice's target.

## Table of Contents

* [Problem Preamble](#problem-preamble)
* [Current Baseline](#current-baseline)
* [Overview](#overview)
* [How Slices Are Executed](#how-slices-are-executed)
* [Execution Slices](#execution-slices)
* [Validation Routine](#validation-routine)
* [Completion Standard](#completion-standard)

## Problem Preamble

Copernican already evaluates cosmological models against SNe Ia, BAO, and CMB
observations. SNe and BAO can consume broad declared expressions. The CMB path
must reach the same philosophical standard.

The current native CMB implementation now contains a declared graph contract,
declared background handling, ODE-based recombination, visibility construction,
per-k graph evolution, source evaluation, projection contracts, TT/TE/EE, BB,
lensing-potential, custom observable targets, manifest integration, and tests.

That is real progress. It is not the final target.

The final target is a native engine that can ingest a complete declared
mathematical system: background equations, perturbation variables, evolution
equations, constraints, closures, initial conditions, boundary conditions,
source terms, observable mappings, priors, validity declarations, numerical
requirements, and datasets.

The theory itself must not be classified into solver families. A theory may
contain any mixture of scalar-like, vector-like, tensor-like, spin-weighted,
ranked, parity-tagged, custom, coupled, or exotic variables. These are
properties of variables, equations, sources, and projections. They are not
separate theory families and must not become solver selectors.

The native engine must compile the declared mathematics into one internal
equation graph and solve that graph. Observable adapters may use variable and
source metadata such as rank, spin, parity, projection role, or source role,
but they must not silently choose a hardcoded theory type.

The remaining work is no longer "start building the engine." The remaining
work is to generalize projection semantics, close docs and provenance truth,
and only then optimize speed.

## Current Baseline

The current `new` branch baseline is treated as the end of Slice Four.

Current status:

* [closed] Slice One - Native declared graph foundation.
* [closed] Slice Two - First physical CMB engine implementation.
* [closed] Slice Three - Scientific validation hardening.
* [closed] Slice Four - Background and equation universality.
* [open] Slice Five - Projection and observable generalization.
* [open] Slice Six - Closure, audit, docs, and provenance truth.
* [deferred] Slice Seven - Performance and gate-speed optimization.

This baseline is intentionally not described as "almost complete Slice One."
The graph foundation and first physical engine pass are real completed
platform work. The next work begins from that platform.

Current limits that open slices must eliminate:

* Projection dispatch is finite and must become declared-kernel driven where
  mathematically safe.
* Manifest "no CAMB prediction" proof must be tied to the executed route, not
  inferred only from `standard: false`.
* Public docs, templates, and manifests still need a closure audit against
  the implemented native runtime.

## Overview

* Copernican is a Python toolkit for evaluating cosmological models against
  observational datasets.
* `standard: true` means the model intentionally uses the standard backend
  CMB path.
* `standard: false` means the model intentionally uses Copernican's native
  declared-math engine.
* The native declared-math engine must not use CAMB or CLASS as production
  prediction engines.
* CAMB and CLASS may be used only as validation references.
* The native engine must be based on one declared equation graph.
* Do not introduce `mode_families`.
* Do not introduce a solver-type selector.
* Do not introduce a scalar compatibility layer.
* Do not introduce hidden LCDM production fallback for `standard: false`.
* Convert every remaining LCDM-like default into declared math or an explicit
  fail-loud contract requirement.
* Declared variables may carry metadata needed for physics and projection:
  kind, rank, spin, parity, tensor character, gauge role, source role,
  projection role, domain, units, and notes.
* Declared equations may be differential, algebraic, constraint, closure, or
  source equations.
* Declared observables must state what they need from the solved graph.
* Projection contracts must be explicit and fail loudly when unsupported.
* The engine must check whether the graph is complete enough to solve.
* The engine must fail clearly for missing equations, missing initial
  conditions, missing background quantities, missing observable mappings,
  contradictory definitions, unsupported projections, non-finite evolution,
  singular systems, or invalid math.
* Documentation, examples, manifests, tests, and DevCovenant governance must
  all describe the implemented behavior.

## How Slices Are Executed

* Each slice means a complete implementation or validation pass, not a note.
* Slices must be executed in dependency order.
* Closed slices describe the current baseline and must not be reopened unless
  current-code audit proves a closed claim false.
* Open slices describe the next work.
* Deferred slices describe real work that must not be started yet.
* Do not turn any slice into a tests-first loop.
* Do not delete, skip, xfail, or loosen validation tests to pass.
* Do not use source-code bans as the main proof of physics.
* Do not fake spectra.
* Do not use fitted CMB templates.
* Do not use hidden CAMB or CLASS production fallback for `standard: false`.
* Do not use hidden LCDM assumptions in `standard: false`.
* Do not use arbitrary amplitude rescaling to force spectrum agreement.
* Do not hand-shift visibility peaks.
* Do not hand-rescale visibility functions.
* Do not copy reference spectra or recombination tables into production code.
* Do not create theory-specific Python solvers.
* Do not update production theory YAMLs as the implementation vehicle.
* Use synthetic declared-math fixtures for implementation and validation.
* Use reference-backed fixtures for scientific validation.
* Keep `standard: true` behavior intact.
* Keep generated artifacts generated.
* Keep repository-specific DevCovenant profile overrides narrow and factual.
* Treat token conservation as a first-class requirement.
* Use `CHANGELOG.md` to record slice outcomes when behavior, documentation,
  validation, schema, or governance changes.
* Use the configured local governance workflow around each completed slice.
* Stage completed slice changes.
* Do not commit or push unless explicitly instructed.

"Probable affected files" in a slice are guidance, not an allowlist. Change
any file required to implement the slice correctly. Do not make unrelated
changes.

Task markers inside slices mean:

* [closed] Implemented and validated enough to be treated as current baseline.
* [open] Not complete and must be considered active future work.
* [deferred] Real work, intentionally postponed until its dependency slices
  close.
* [blocked] Cannot proceed because a named dependency is missing.

## Execution Slices

### [closed] Slice One - Native declared graph foundation

Purpose:

Establish `standard: false` as a native declared-math path, not a scalar custom
path and not a CAMB or CLASS fallback.

Depends on:

* current package layout;
* current native custom CMB route;
* current safe-expression compiler;
* current model coder and validator;
* current CMB likelihood integration;
* current manifest integration.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/model_spec_validator.py`
* `copernican/lib/engine_adapter.py`
* `copernican/lib/run_manifest.py`
* `copernican/lib/likelihoods/cmb.py`
* graph and model schema tests
* perturbation contract tests
* CMB likelihood tests
* manifest tests
* docs and templates

Tasks:

* [closed] Replace scalar-only custom CMB contract with declared graph
  contract.
* [closed] Keep `standard: true` on the standard backend path.
* [closed] Keep `standard: false` CAMB-free in production prediction.
* [closed] Keep `standard: false` CLASS-free in production prediction.
* [closed] Remove theory-family classification from the native contract.
* [closed] Do not introduce `mode_families`.
* [closed] Do not introduce a solver-type selector.
* [closed] Represent the theory as one graph.
* [closed] Support declared variables.
* [closed] Support declared derived quantities.
* [closed] Support declared differential equations.
* [closed] Support declared algebraic equations.
* [closed] Support declared constraints.
* [closed] Support declared closures.
* [closed] Support declared sources.
* [closed] Support declared observables.
* [closed] Support declared initial conditions.
* [closed] Support declared boundary conditions.
* [closed] Support declared numerics.
* [closed] Support declared validity domains.
* [closed] Add variable metadata for rank, spin, parity, tensor character,
  gauge role, source role, and projection role.
* [closed] Validate referenced symbols.
* [closed] Validate derived dependencies.
* [closed] Validate evolved-variable equations.
* [closed] Validate initial-condition coverage.
* [closed] Validate observable mappings.
* [closed] Detect duplicate ambiguous declarations.
* [closed] Detect circular derived dependencies.
* [closed] Fail clearly for incomplete graph declarations.
* [closed] Add synthetic complete-graph tests.
* [closed] Add hybrid declared graph tests.
* [closed] Add coupled-equation graph tests.
* [closed] Add invalid graph failure tests.
* [closed] Record graph provenance in manifest output.

Done when:

* [closed] `standard: false` compiles one declared equation graph.
* [closed] No theory-family selector exists.
* [closed] No `mode_families` schema exists.
* [closed] No obsolete scalar compatibility layer exists.
* [closed] Complete declared graph fixtures compile.
* [closed] Invalid declared graph fixtures fail clearly.
* [closed] `standard: true` remains intact.
* [closed] `standard: false` remains CAMB-free in production.
* [closed] Targeted graph, schema, and manifest tests pass.

### [closed] Slice Two - First physical CMB engine implementation

Purpose:

Make the declared graph drive a native CMB prediction path with physical
background, recombination, source, and projection machinery.

Depends on:

* Slice One;
* current CMB likelihood path;
* current declared graph compiler;
* current safe-expression runtime;
* current model schema handling.

Probable affected files:

* `copernican/lib/likelihoods/cmb.py`
* `copernican/lib/cmb_projection_contract.py`
* `copernican/lib/perturbation_contract.py`
* CMB likelihood tests
* projection contract tests
* perturbation contract tests
* docs and templates
* manifest tests
* `CHANGELOG.md`

Tasks:

* [closed] Require declared background mapping for native CMB execution.
* [closed] Fail clearly when native CMB background mapping is missing.
* [closed] Fail clearly when background declarations are malformed.
* [closed] Remove loose hidden LCDM background fallback.
* [closed] Resolve declared `H(a,z)` from background declarations.
* [closed] Build eta grid from declared background expansion.
* [closed] Build distance grid from declared background expansion.
* [closed] Build visibility from recombination and reionization history.
* [closed] Replace empirical recombination transition shortcut.
* [closed] Implement hydrogen recombination as an ODE treatment.
* [closed] Use case-B hydrogen recombination coefficient.
* [closed] Use detailed-balance photoionization.
* [closed] Use Peebles-style C factor.
* [closed] Include physical helium electron contribution.
* [closed] Preserve reionization optical-depth handling.
* [closed] Remove hand-shifted visibility peak logic.
* [closed] Remove hand-rescaled visibility logic.
* [closed] Evaluate declared perturbation equations during runtime evolution.
* [closed] Evaluate declared constraints and closures during runtime.
* [closed] Evaluate declared sources from solved graph context.
* [closed] Fail clearly on non-finite derivatives.
* [closed] Fail clearly on non-finite sources.
* [closed] Fail clearly on non-finite evolved state.
* [closed] Add explicit projection contracts.
* [closed] Build TT from declared graph quantities.
* [closed] Build TE from declared graph quantities.
* [closed] Build EE from declared graph quantities.
* [closed] Build BB from declared `polarization_b` source.
* [closed] Do not map BB to E-mode plumbing.
* [closed] Build lensing from declared potential source.
* [closed] Fail clearly for unsupported projections.
* [closed] Fail clearly for missing projection source roles.
* [closed] Fail clearly for wrong projection source roles.
* [closed] Add tests proving equations change intended outputs.
* [closed] Add tests proving closures change intended outputs.
* [closed] Add tests proving sources change intended outputs.
* [closed] Add tests proving BB responds to declared B source.
* [closed] Add tests proving lensing responds to declared potential source.

Done when:

* [closed] Native CMB predictions use declared graph runtime.
* [closed] Native background is declared, not silently inferred from LCDM.
* [closed] Recombination is physical rather than empirical transition fitting.
* [closed] TT, TE, EE, BB, and lensing run from declared source contracts.
* [closed] Unsupported or incomplete projection requests fail clearly.
* [closed] Synthetic runtime tests demonstrate graph-driven response.
* [closed] `standard: true` remains intact.
* [closed] `standard: false` remains CAMB-free in production.

### [closed] Slice Three - Scientific validation hardening

Purpose:

Move from proving that the engine runs to proving that its physics is
credible. This slice must strengthen validation without weakening or deleting
tests.

Depends on:

* Slice One;
* Slice Two.

Probable affected files:

* `tests/copernican/lib/likelihoods/test_cmb.py`
* CMB validation helpers
* CMB reference fixture helpers
* `copernican/lib/likelihoods/cmb.py`
* projection contract tests if validation exposes defects
* docs describing validation status
* `CHANGELOG.md`

Scope:

* Audit existing CMB tests as scientific validation, not just runtime checks.
* Preserve synthetic runtime tests but stop treating them as physics proof.
* Add or harden reference-backed validation.
* Tighten loose tolerances where implementation can support it.
* Make failures name the physical discrepancy.
* Fix implementation defects exposed by stronger validation.
* Do not optimize speed in this slice except to keep validation executable.
* Do not broaden background universality in this slice unless validation
  exposes a direct defect in current declared-background behavior.

Tasks:

* [closed] Separate runtime-response tests from scientific-reference tests.
* [closed] Label slow scientific validation clearly.
* [closed] Tighten recombination validation beyond current loose thresholds.
* [closed] Validate recombination history against CAMB or CLASS reference.
* [closed] Validate visibility peak against CAMB or CLASS reference.
* [closed] Validate visibility width against CAMB or CLASS reference.
* [closed] Validate eta0 against CAMB or CLASS reference.
* [closed] Validate sound horizon against CAMB or CLASS reference.
* [closed] Replace weak normalized TT shape-only validation.
* [closed] Validate TT over a meaningful ell range.
* [closed] Validate TE over a meaningful ell range.
* [closed] Validate EE over a meaningful ell range.
* [closed] Validate peak positions.
* [closed] Validate acoustic peak spacing.
* [closed] Validate TE zero crossings.
* [closed] Validate low-ell behavior only where numerically meaningful.
* [closed] Validate tensor BB where declared and reference-supported.
* [closed] Validate lensing-potential behavior where reference-supported.
* [closed] Validate custom source-channel perturbations against
  reference-backed or analytic observable expectations beyond current
  synthetic response tests.
* [closed] Validate custom closures against reference-backed or analytic
  observable expectations beyond current synthetic response tests.
* [closed] Validate custom equations against reference-backed or analytic
  observable expectations beyond current synthetic response tests.
* [closed] Remove or physically justify remaining empirical numerical scale
  factors.
* [closed] Ensure failed reference comparisons report named quantities,
  tolerances, and measured error.
* [closed] Ensure validation never uses source-code string bans as the main
  proof of physics.
* [closed] Ensure validation never accepts copied reference spectra as
  production behavior.
* [closed] Ensure CAMB or CLASS are validation references only.

Done when:

* [closed] Recombination reference validation is credible.
* [closed] Visibility reference validation is credible.
* [closed] Eta0 and sound-horizon validation are credible.
* [closed] TT, TE, and EE validation are stronger than smoothed shape checks.
* [closed] BB validation exists where declared and reference-supported.
* [closed] Lensing validation exists where declared and reference-supported.
* [closed] Synthetic tests remain but are not used as scientific proof.
* [closed] Tolerances are documented and not loosened to pass.
* [closed] Defects exposed by validation are fixed.
* [closed] Relevant CMB validation tests pass.

### [closed] Slice Four - Background and equation universality

Purpose:

Remove the remaining standard-like named-parameter ceiling and make declared
background equations first-class native engine inputs.
This slice is where the universal goal stops being aspirational for
background and equation execution.

Depends on:

* Slice Three.

Probable affected files:

* `copernican/lib/likelihoods/cmb.py`
* `copernican/lib/perturbation_contract.py`
* model coder and validator code
* background validation helpers
* CMB tests
* model schema tests
* docs and templates
* manifest tests
* `CHANGELOG.md`

Scope:

* Treat declared background math as part of the native equation system.
* Reduce hard dependency on fixed LCDM-style parameter names.
* Keep physical requirements explicit where CMB physics needs them.
* Fail clearly when a declared theory omits quantities required by the CMB
  engine.
* Remove current one-independent-variable runtime restriction or replace it
  with a declared coordinate transform that preserves arbitrary declared
  equations.
* Remove current start-only boundary-condition restriction or implement a
  declared boundary solver that supports non-start boundary data.
* Convert CDM, radiation, dark-energy, and primordial defaults into declared
  quantities or explicit fail-loud requirements.
* Do not create a hidden LCDM fallback.
* Do not create theory-family selectors.
* Do not move to performance optimization.

Tasks:

* [closed] Audit all named physical parameter requirements in native CMB code.
* [closed] Classify each requirement as physically required, derivable, or
  obsolete.
* [closed] Audit current one-independent-variable runtime restriction.
* [closed] Support multiple declared independent variables or a proven declared
  coordinate transform.
* [closed] Audit current start-anchored-only boundary-condition restriction.
* [closed] Support non-start boundary conditions through a declared boundary
  solver.
* [closed] Make declared background equations first-class graph inputs.
* [closed] Allow declared background outputs to provide expansion quantities.
* [closed] Allow declared background outputs to provide density quantities.
* [closed] Allow declared background outputs to provide pressure quantities.
* [closed] Allow declared background outputs to provide equation-of-state
  quantities.
* [closed] Allow declared background outputs to provide curvature quantities.
* [closed] Fix CDM detection so declared background quantities are considered,
  not only parameter names.
* [closed] Fix baryon detection so declared background quantities are
  considered where possible.
* [closed] Fix radiation detection so declared background quantities are
  considered where possible.
* [closed] Fix dark-energy detection so declared background quantities are
  considered where possible.
* [closed] Remove zero-CDM default when CDM is absent from parameter names.
* [closed] Remove dark-energy `w0=-1` and `wa=0` defaults unless declared.
* [closed] Replace formula-derived photon and neutrino densities with declared
  quantities or explicit physical requirements.
* [closed] Replace named primordial amplitude and tilt requirements with
  declared primordial-power inputs or explicit CMB requirements.
* [closed] Preserve physically required scalars when they cannot be derived.
* [closed] Fail clearly when a required physical quantity is missing.
* [closed] Fail clearly when declared background arrays have invalid shape.
* [closed] Fail clearly when declared background values are non-finite.
* [closed] Fail clearly when declared background domains are invalid.
* [closed] Ensure perturbation graph can consume background graph outputs.
* [closed] Add non-LCDM synthetic background fixtures.
* [closed] Add tests where changing declared background changes CMB outputs.
* [closed] Add tests where incomplete background declarations fail early.
* [closed] Add manifest provenance for declared background quantities.

Done when:

* [closed] Native CMB background is graph-driven where expressible.
* [closed] Native graph execution is not limited to one evolution variable.
* [closed] Boundary conditions are not limited to start anchors.
* [closed] Hidden LCDM background fallback does not exist.
* [closed] CDM, radiation, dark-energy, and primordial quantities are declared
  or explicitly required; none are silently defaulted.
* [closed] Remaining named requirements are physically justified.
* [closed] Declared non-LCDM background fixtures run.
* [closed] Invalid background declarations fail before CMB runtime.
* [closed] Perturbations can consume declared background outputs.
* [closed] Manifest output records background provenance.
* [closed] Relevant background, CMB, schema, and manifest tests pass.

### [open] Slice Five - Projection and observable generalization

Purpose:

Move from a finite projection adapter toward a more general declared
observable system while preserving fail-loud behavior for unsupported
projection math.

Depends on:

* Slice Three;
* Slice Four.

Probable affected files:

* `copernican/lib/cmb_projection_contract.py`
* `copernican/lib/likelihoods/cmb.py`
* `copernican/lib/perturbation_contract.py`
* projection contract tests
* CMB observable tests
* model schema tests
* docs and templates
* manifest tests
* `CHANGELOG.md`

Scope:

* Generalize projection semantics without theory-family selectors.
* Keep projection requirements explicit.
* Strengthen BB parity and source validation.
* Strengthen lensing source and projection validation.
* Add custom observable projection support where declared math can provide it.
* Fail clearly when a projection is not supported.
* Do not fake unsupported observables.
* Do not tune outputs with post-hoc amplitude hacks.
* Do not optimize performance in this slice.

Existing Slice Two baseline already proves wrong-parity BB rejection, missing
`polarization_b` rejection, E-mode source-substitution rejection, missing
lensing-potential rejection, and wrong lensing source-role rejection. Slice
Five must extend that baseline instead of restating it as new work.

Tasks:

* [open] Audit current projection vocabulary.
* [open] Document current projection limits as implementation facts.
* [open] Distinguish source roles from projection kernels.
* [open] Distinguish observable targets from projection machinery.
* [open] Allow observable mappings to declare required source roles.
* [open] Allow observable mappings to declare required projection roles.
* [open] Allow observable mappings to declare kernel requirements.
* [open] Support custom projection kernels where safe and mathematically
  declared.
* [open] Keep unsupported projection math fail-loud.
* [open] Extend BB validation beyond current parity and source-role baseline.
* [open] Validate BB behavior for custom projection kernels when supported.
* [open] Validate multi-source BB declarations cannot hide E-only sources.
* [open] Extend lensing validation beyond current potential-role baseline.
* [open] Validate lensing behavior for custom kernels when supported.
* [open] Validate custom observable mappings consume solved graph outputs.
* [open] Validate custom projection kernels change intended observables.
* [open] Ensure no hidden source substitutions occur.
* [open] Ensure no post-hoc amplitude tuning occurs.
* [open] Ensure projection provenance appears in manifest output.

Done when:

* [open] Projection contracts are explicit and general where implemented.
* [open] Unsupported projections fail clearly.
* [open] BB validation covers current baseline and new custom-kernel cases.
* [open] Lensing validation covers current baseline and new custom-kernel
  cases.
* [open] Custom observable projections consume solved graph quantities.
* [open] Projection provenance is recorded.
* [open] Docs describe implemented projection behavior honestly.
* [open] Relevant projection, CMB, schema, and manifest tests pass.

### [open] Slice Six - Closure, audit, docs, and provenance truth

Purpose:

Close the native CMB engine as a trustworthy feature. This slice is not for
new physics expansion unless audit exposes a defect that blocks truthful
closure.

Depends on:

* Slice Three;
* Slice Four;
* Slice Five.

Probable affected files:

* CMB implementation files
* graph contract files
* projection contract files
* validation tests
* docs
* templates
* manifests
* `CHANGELOG.md`
* DevCovenant profile or governance docs if needed

Scope:

* Audit code against this plan.
* Audit tests against implementation.
* Audit docs against implementation.
* Audit templates as documentation, not benchmarks.
* Audit manifests against runtime behavior.
* Remove stale claims.
* Remove stale scalar wording.
* Close feature truthfully.
* Do not optimize performance in this slice.

Tasks:

* [open] Audit all closed-slice claims against code.
* [open] Reopen any closed task that code disproves.
* [open] Audit docs against implemented behavior.
* [open] Audit templates as documentation only.
* [open] Audit examples against current schema.
* [open] Audit manifest provenance for graph identity.
* [open] Audit manifest provenance for background quantities.
* [open] Audit manifest provenance for recombination settings.
* [open] Audit manifest provenance for projection contracts.
* [open] Audit manifest provenance for solver/runtime settings.
* [open] Prove manifest no-CAMB claims from executed route metadata, not only
  the `standard` flag.
* [open] Audit `standard: true` behavior.
* [open] Audit `standard: false` CAMB-free production behavior.
* [open] Verify one-independent-variable and start-boundary limits are removed
  before closure.
* [open] Audit failure messages for missing graph pieces.
* [open] Audit failure messages for invalid background declarations.
* [open] Audit failure messages for invalid projection declarations.
* [open] Remove stale scalar-engine wording.
* [open] Remove docs that imply hidden CAMB fallback.
* [open] Remove docs that imply unsupported universal projection behavior.
* [open] Ensure docs explain what Copernican can solve.
* [open] Ensure docs explain what Copernican refuses to solve.
* [open] Ensure docs explain failure diagnostics.
* [open] Ensure docs explain validation status honestly.
* [open] Ensure docs explain slow reference validation honestly.
* [open] Ensure generated artifacts remain generated.
* [open] Ensure changelog records closure accurately.
* [open] Ensure the final gate validates the feature truthfully.

Done when:

* [open] Code, docs, tests, templates, and manifest behavior agree.
* [open] No stale scalar-only public wording remains.
* [open] No unsupported behavior is promised.
* [open] `standard: true` is intact.
* [open] `standard: false` is native and CAMB-free in production.
* [open] Complete declared theories are executable within the model contract.
* [open] Invalid declared theories fail clearly.
* [open] Scientific validation status is explicit.
* [open] Manifest provenance is truthful.
* [open] Full relevant validation passes.
* [open] DevCovenant gate closes.

### [deferred] Slice Seven - Performance and gate-speed optimization

Purpose:

Make the now-correct native CMB engine fast enough to use sanely. This slice
must not start until Slice Six closes.

Depends on:

* Slice Six.

Probable affected files:

* CMB implementation files
* CMB tests
* validation fixtures
* test configuration
* profiling helpers
* docs describing test tiers
* `CHANGELOG.md`

Scope:

* Profile before optimizing.
* Split quick development validation from full scientific validation.
* Keep scientific validation strong.
* Do not optimize by weakening physics.
* Do not optimize by deleting validation.
* Do not optimize by hiding failures.
* Prefer measured bottleneck removal over speculative rewrites.

Tasks:

* [deferred] Profile full CMB validation.
* [deferred] Profile native CMB prediction runtime.
* [deferred] Identify recombination bottlenecks.
* [deferred] Identify background-grid bottlenecks.
* [deferred] Identify Bessel-grid bottlenecks.
* [deferred] Identify line-of-sight integration bottlenecks.
* [deferred] Identify graph-compilation bottlenecks.
* [deferred] Identify k-mode evolution bottlenecks.
* [deferred] Split quick gate from full scientific validation.
* [deferred] Mark slow scientific reference tests explicitly.
* [deferred] Cache CAMB or CLASS reference products where legitimate.
* [deferred] Cache recombination products where inputs are identical.
* [deferred] Cache background products where inputs are identical.
* [deferred] Cache Bessel grids.
* [deferred] Cache compiled graph objects.
* [deferred] Remove duplicate work in CMB tests.
* [deferred] Vectorize line-of-sight hot paths.
* [deferred] Parallelize independent k-mode evolution.
* [deferred] Consider numba, cython, or compiled kernels only after profiling
  proves the target.
* [deferred] Document test tiers and expected runtime.
* [deferred] Keep full scientific validation available.

Done when:

* [deferred] Quick validation is fast enough for normal development.
* [deferred] Full scientific validation remains strong.
* [deferred] Runtime bottlenecks are measured, not guessed.
* [deferred] Optimizations preserve validated physics.
* [deferred] No validation is weakened for speed.
* [deferred] Test tiers are documented.
* [deferred] Performance changes are recorded.

## Validation Routine

Run the validation routine after each completed slice.

Minimum validation:

* inspect the working tree;
* run targeted tests for touched code;
* run CMB tests when CMB behavior changes;
* run perturbation contract tests when graph schema or compiler behavior
  changes;
* run projection contract tests when projection behavior changes;
* run model coder and model validator tests when model schema behavior
  changes;
* run manifest tests when provenance behavior changes;
* run docs checks when public behavior or templates change;
* run import smoke checks for `copernican`;
* run DevCovenant verification;
* update `CHANGELOG.md` when behavior, structure, docs, tests, validation, or
  governance changes;
* stage completed slice changes;
* do not commit or push unless instructed.

Per-slice closure validation:

* closed tasks have code, tests, docs, or manifest evidence;
* open tasks are not silently skipped;
* deferred tasks are not started early;
* tests are not weakened to pass;
* failures are precise and user-facing;
* generated artifacts remain generated.

Completion validation:

* `standard: true` remains CAMB-compatible;
* `standard: false` does not use CAMB or CLASS for production prediction;
* the native engine uses one declared equation graph;
* no `mode_families` schema exists;
* no solver-type selector exists;
* no obsolete scalar compatibility layer exists;
* declared background behavior is explicit and fail-loud;
* multiple declared independent variables, or declared coordinate transforms,
  pass validation;
* non-start boundary conditions pass validation;
* hidden LCDM-like defaults are gone;
* recombination reference validation passes;
* visibility reference validation passes;
* scalar TT, TE, and EE reference validation passes;
* BB validation passes where declared and reference-supported;
* lensing validation passes where declared and reference-supported;
* custom observable validation passes where declared and supported;
* invalid theory failure tests pass;
* source-channel, closure, equation, initial-condition, boundary-condition,
  constraint, projection, and observable tests pass;
* manifests record graph, background, recombination, projection, solver, and
  observable provenance;
* docs and templates match the implemented declaration system;
* full relevant test suite passes;
* DevCovenant gate closes.

## Completion Standard

The CMB engine is not complete merely because it runs.

The CMB engine is complete when:

* the declared graph is the native production path for `standard: false`;
* the background is declared or explicitly required and never hidden;
* recombination is physical and reference-validated;
* observables are built from solved graph quantities;
* projections are explicit and fail-loud;
* TT, TE, EE, BB, lensing, and custom observables are validated within their
  implemented scope;
* invalid theories fail before fake numerical output;
* docs distinguish the current baseline, remaining plan, and final achieved
  scope;
* manifests tell the truth;
* `standard: true` remains intact;
* `standard: false` remains CAMB-free and CLASS-free in production;
* scientific validation is credible;
* performance optimization is deferred until after closure.
