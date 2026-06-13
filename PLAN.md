# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-06-13
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

Use this plan to turn Copernican's custom CMB path into a universal
declarative cosmological-theory executor.

The purpose of this roadmap is non-negotiable:

* `standard: true` remains the standard backend path for CAMB-compatible
  models.
* `standard: false` becomes Copernican's native path for declared math.
* Any mathematically well-posed cosmological theory should be ingestible if
  it is expressible through the model contract.
* Invalid, incomplete, contradictory, unsupported, or numerically ill-posed
  math must fail clearly.
* Copernican must not silently fall back to CAMB, CLASS, LCDM assumptions,
  fake spectra, fitted templates, hidden amplitude hacks, or theory-specific
  Python solvers when `standard: false`.

This is a forward-only plan. Do not preserve obsolete schema by adding
compatibility layers. Migrate the current custom CMB contract into the new
native declaration system.

This is not a tests-first plan. Slice One is the implementation slice.
Validation tests must be created as part of that implementation. Slice Two is
the dedicated verification and validation slice.

Keep slices dependency-ordered, concrete, current, and runtime-focused.

## Table of Contents

* [Problem Preamble](#problem-preamble)
* [Overview](#overview)
* [How Slices Are Executed](#how-slices-are-executed)
* [Execution Slices](#execution-slices)
* [Validation Routine](#validation-routine)

## Problem Preamble

Copernican already evaluates cosmological models against SNe Ia, BAO, and CMB
observations. SNe and BAO can already consume broad declared expressions. The
CMB path must now reach the same philosophical standard.

The current custom CMB implementation is not the final target. It contains a
native non-standard route, custom background construction, recombination
history, visibility function, per-k mode evolution, line-of-sight projection,
TT/TE/EE spectra, source channels, declared perturbation contracts, and
manifest integration. That is the baseline.

The target is broader.

Copernican must become able to ingest a complete declared mathematical
system: background equations, perturbation variables, evolution equations,
constraints, closures, initial conditions, boundary conditions, source terms,
observable mappings, priors, validity declarations, numerical requirements,
and datasets.

The theory itself must not be classified into solver families. A theory may
contain any mixture of scalar-like, vector-like, tensor-like, spin-weighted,
ranked, parity-tagged, custom, coupled, or exotic variables. These are
properties of variables, equations, and observable projections. They are not
separate theory families and must not become solver selectors.

The native engine must compile the declared mathematics into one internal
equation graph and solve that graph. Observable adapters may use variable
metadata such as rank, spin, parity, projection role, or source role, but they
must not classify the whole theory or silently choose a hardcoded theory type.

The purpose of this roadmap is to remove the scalar-only ceiling and replace
it with a forward-only universal declaration system for `standard: false`.

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
* Do not introduce a compatibility layer for obsolete scalar-only schema.
* Current custom scalar fixtures, docs, and tests must be migrated into the
  new forward-only declaration system.
* Declared variables may carry metadata needed for physics and projection:
  kind, rank, spin, parity, tensor character, gauge role, source role,
  projection role, domain, units, and notes.
* Declared equations may be differential, algebraic, constraint, closure, or
  source equations.
* Declared observables must state what they need from the solved graph.
* The engine must check whether the graph is complete enough to solve.
* The engine must fail clearly for missing equations, missing initial
  conditions, missing observable mappings, contradictory definitions,
  unsupported projections, non-finite evolution, singular systems, or invalid
  math.
* Documentation, examples, manifests, tests, and DevCovenant governance must
  all describe the implemented behavior.

## How Slices Are Executed

* Each slice means a complete implementation pass, not a note.
* Slice One implements the new declared-math engine and creates validation
  tests as part of implementation.
* Slice Two verifies, validates, hardens, documents, and audits the result.
* Do not turn Slice One into a tests-first loop.
* Do not mark implementation complete merely because weak tests pass.
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

## Execution Slices

* [open] Slice One - Implement the universal declared-math CMB engine.

  Depends on:

  * current `new` branch baseline;
  * current custom CMB route for `standard: false`;
  * current perturbation contract compiler;
  * current CMB likelihood tests;
  * current manifest integration.

  Probable affected files:

  * `copernican/lib/likelihoods/cmb.py`
  * `copernican/lib/perturbation_contract.py`
  * `copernican/lib/model_coder.py`
  * `copernican/lib/model_spec_validator.py`
  * `copernican/lib/engine_adapter.py`
  * `copernican/lib/run_manifest.py`
  * `tests/copernican/lib/likelihoods/test_cmb.py`
  * perturbation contract tests
  * model schema and model coder tests
  * manifest tests
  * docs and model templates
  * `CHANGELOG.md`

  Scope:

  * Replace scalar-only custom CMB execution with one unified declared-math
    equation graph for `standard: false`.
  * Remove the assumption that the custom CMB path is a scalar engine.
  * Do not introduce `mode_families`.
  * Do not classify theories as scalar, vector, tensor, or anything else.
  * Represent the theory as one graph containing variables, equations,
    constraints, closures, sources, initial conditions, boundary conditions,
    validity declarations, numerical requirements, and observable mappings.
  * Allow variables to carry metadata such as kind, rank, spin, parity,
    tensor character, gauge role, source role, projection role, domain, units,
    and notes.
  * Allow equations to declare their left-hand side, right-hand side,
    derivative order, independent variable, equation role, domain, and
    dependencies.
  * Allow algebraic constraints and closure equations to participate in graph
    validation and evolution.
  * Allow initial and boundary conditions to be declared explicitly.
  * Allow observable mappings to declare what they consume from the solved
    graph.
  * Support scalar-like, vector-like, tensor-like, spin-weighted, ranked,
    parity-tagged, and custom variables as metadata on graph nodes, not as
    separate theory families.
  * Migrate current scalar custom CMB declarations into the new graph format.
  * Remove obsolete scalar-only schema rather than preserving it through a
    compatibility layer.
  * Keep `standard: true` CAMB-compatible behavior intact.
  * Keep `standard: false` CAMB-free in production prediction.
  * Use CAMB and CLASS only as references in validation tests.

  Implementation details:

  * Build an internal immutable equation-graph representation.
  * The graph must include model name, backend name, `standard` flag,
    variables, derived quantities, differential equations, algebraic
    equations, constraints, closures, source terms, initial conditions,
    boundary conditions, observable mappings, validity domains, numerical
    requirements, dependency graph summary, and manifest summary.
  * Compile declared expressions through the existing safe-expression system.
  * Validate all referenced symbols.
  * Validate all independent variables.
  * Validate that every evolved variable has enough evolution information.
  * Validate that required initial or boundary conditions exist.
  * Validate that observable mappings reference solved or derived quantities.
  * Detect duplicate definitions where they would make the graph ambiguous.
  * Detect circular derived dependencies.
  * Detect unsupported projection requests before runtime.
  * Detect invalid algebraic equations where possible.
  * Detect incomplete systems before numerical evolution where possible.
  * Fail with precise errors naming the missing or invalid graph component.
  * Keep runtime non-finite checks during evolution.
  * Fail clearly on non-finite state, non-finite derivative, singular solve,
    or impossible numerical state.
  * Replace empirical recombination transition shortcuts with a physical
    recombination implementation.
  * Implement hydrogen recombination through an ODE-based physical treatment:
    case-B recombination coefficient, photoionization from detailed balance,
    and Peebles C factor or equivalent escape-probability correction.
  * Keep helium electron contribution physical.
  * Keep reionization optical depth handling.
  * Do not hand-set visibility peak shifts.
  * Do not hand-rescale visibility.
  * Replace scalar RHS and metric-potential heuristic behavior with graph
    equations, constraints, and physical defaults where the graph requests
    the standard scalar CMB observable.
  * Implement tight-coupling handling where needed for stable early-time
    photon-baryon evolution.
  * Build CMB source terms from declared graph quantities and observable
    mappings.
  * Remove arbitrary transfer amplitude multipliers and source normalization
    hacks.
  * Improve k-grid, quadrature, interpolation, and spherical-Bessel handling
    as required for stable spectra.
  * Support TT, TE, EE, BB, lensing-potential, and custom observable targets
    when their required graph quantities and projections are declared.
  * If a requested observable is not mathematically declared or not supported
    by the current projection machinery, fail clearly.
  * Record graph, solver, observable, and validation provenance in the run
    manifest.
  * Update docs and templates so the public model contract matches the
    implemented forward-only schema.
  * Update existing synthetic tests and fixtures to use the new graph schema.

  Validation tests to create as part of implementation:

  * A complete synthetic `standard: false` graph compiles and runs.
  * A hybrid graph containing scalar-like, vector-like, tensor-like, and
    custom variables compiles as one graph.
  * Coupled equations across differently tagged variables compile as one
    graph.
  * Observable mappings consume graph quantities rather than hidden engine
    variables.
  * Missing evolved-variable equations fail clearly.
  * Missing initial conditions fail clearly.
  * Missing observable mappings fail clearly.
  * Unsupported projections fail clearly.
  * Duplicate incompatible definitions fail clearly.
  * Circular derived dependencies fail clearly.
  * Non-finite expression results fail clearly.
  * Non-finite evolution states fail clearly.
  * `standard: true` still uses the standard backend path.
  * `standard: false` does not use CAMB or CLASS as production prediction.
  * Recombination history is compared against CAMB or CLASS reference.
  * Visibility peak and width are compared against CAMB or CLASS reference.
  * Eta0 and sound horizon are compared against CAMB or CLASS reference.
  * Scalar TT, TE, and EE spectra are compared against CAMB or CLASS for an
    LCDM-equivalent declared graph.
  * Tensor BB support is validated where the declared graph and reference
    data support it.
  * Lensing-potential behavior is validated where declared and supported.
  * Custom source-channel perturbations change the intended observables.
  * Custom closures change the intended observables.
  * Custom equations change the intended observables.
  * Manifest output records graph, solver, observable, and validation
    provenance.
  * The source file does not contain old fake CMB paths, acoustic templates,
    hidden amplitude hacks, source rescaling hacks, hand-shifted visibility,
    hand-rescaled visibility, or copied reference spectra.

  Done when:

  * `standard: false` uses a unified declared-math equation graph.
  * No theory-family selector exists.
  * No `mode_families` schema exists.
  * No obsolete scalar compatibility layer exists.
  * Current scalar custom CMB behavior is migrated into the new graph system.
  * Complete declared theories run.
  * Invalid declared theories fail clearly.
  * Recombination is physical rather than an empirical transition fit.
  * CMB observables are built from solved graph quantities.
  * `standard: true` remains CAMB-compatible.
  * `standard: false` remains CAMB-free in production prediction.
  * Implementation validation tests exist.
  * Docs and templates describe the new declaration system.
  * Changelog records the implementation.
  * Targeted tests pass.
  * DevCovenant verification passes.

* [open] Slice Two - Verify, validate, harden, and close the universal engine.

  Depends on:

  * Slice One.

  Probable affected files:

  * tests
  * validation helpers
  * CMB likelihood code if validation exposes defects
  * perturbation contract code if validation exposes defects
  * model coder and validator code if validation exposes defects
  * manifest code if validation exposes defects
  * docs and templates
  * `CHANGELOG.md`

  Scope:

  * Audit the Slice One implementation against the roadmap purpose.
  * Verify that `standard: true` still uses the standard backend path.
  * Verify that `standard: false` uses only Copernican native production
    prediction.
  * Verify that no CAMB or CLASS production fallback exists for
    `standard: false`.
  * Verify that no theory-family schema or solver selector was introduced.
  * Verify that no obsolete scalar compatibility layer remains.
  * Verify that graph validation is real and fail-loud.
  * Verify that invalid theories do not silently run.
  * Verify that complete theories run from declared math.
  * Verify that observables are built from solved graph quantities.
  * Verify that docs, examples, templates, tests, and manifests describe the
    same behavior.
  * Harden implementation defects found by validation.
  * Do not broaden scope beyond defects required to make the declared-math
    engine correct.

  Required validation:

  * Run full CMB tests.
  * Run perturbation contract tests.
  * Run model coder and model spec validator tests.
  * Run manifest tests.
  * Run relevant CLI, import, and package smoke tests.
  * Run DevCovenant verification.
  * Run any available slow CAMB or CLASS reference tests.
  * Confirm that generated artifacts are generated, not hand-patched.

  Scientific validation:

  * Compare custom recombination history to CAMB or CLASS.
  * Compare visibility peak to CAMB or CLASS.
  * Compare visibility width to CAMB or CLASS.
  * Compare eta0 to CAMB or CLASS.
  * Compare sound horizon to CAMB or CLASS.
  * Compare scalar TT, TE, and EE spectra to CAMB or CLASS over a meaningful
    ell range.
  * Validate peak positions and TE zero crossings.
  * Validate tensor BB behavior where declared and reference-supported.
  * Validate lensing-potential behavior where declared and reference-supported.
  * Validate custom observable mappings with synthetic graphs.
  * Validate failure modes for invalid graphs.

  Failure-mode validation:

  * Missing variable definition fails clearly.
  * Missing differential equation fails clearly.
  * Missing algebraic dependency fails clearly.
  * Missing initial condition fails clearly.
  * Missing boundary condition fails clearly when required.
  * Missing observable mapping fails clearly.
  * Unsupported projection fails clearly.
  * Contradictory declarations fail clearly when detectable.
  * Circular derived dependency fails clearly.
  * Non-finite expression fails clearly.
  * Non-finite derivative fails clearly.
  * Non-finite state fails clearly.
  * Singular solve fails clearly.
  * Invalid `standard` declaration fails clearly.
  * `standard: false` with undeclared CMB requirements fails clearly.

  Documentation validation:

  * Docs explain the purpose of `standard: true`.
  * Docs explain the purpose of `standard: false`.
  * Docs explain the declared equation graph.
  * Docs explain variables, equations, constraints, closures, sources,
    initial conditions, boundary conditions, observables, validity, and
    numerical requirements.
  * Docs explain what Copernican can solve.
  * Docs explain what Copernican refuses to solve.
  * Docs explain failure diagnostics.
  * Docs do not describe mode families.
  * Docs do not describe a scalar compatibility layer.
  * Docs do not promise hidden CAMB fallback for `standard: false`.
  * Examples match the implemented schema.

  Done when:

  * Full targeted validation passes.
  * Scientific reference validation passes within documented tolerances.
  * Failure-mode validation passes.
  * Documentation validation passes.
  * `standard: true` is intact.
  * `standard: false` is native and CAMB-free in production.
  * Complete declared theories are edible.
  * Invalid declared theories fail clearly.
  * No family classifier exists.
  * No solver-type selector exists.
  * No obsolete compatibility layer exists.
  * Changelog records validation closure.
  * DevCovenant gate closes.

## Validation Routine

Run the validation routine after each completed slice.

Minimum validation:

* inspect the working tree;
* run targeted tests for touched code;
* run CMB tests when CMB behavior changes;
* run perturbation contract tests when schema or compiler behavior changes;
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

Completion validation:

* `standard: true` remains CAMB-compatible;
* `standard: false` does not use CAMB or CLASS for production prediction;
* the native engine uses one declared equation graph;
* no `mode_families` schema exists;
* no solver-type selector exists;
* no obsolete scalar compatibility layer exists;
* recombination reference validation passes;
* visibility reference validation passes;
* scalar TT, TE, and EE reference validation passes;
* tensor, BB, lensing, and custom observable validation pass where declared
  and supported;
* invalid theory failure tests pass;
* source-channel, closure, equation, initial-condition, boundary-condition,
  constraint, and observable tests pass;
* manifests record full theory provenance;
* docs and model templates match the implemented declaration system;
* full relevant test suite passes;
* DevCovenant gate closes.
