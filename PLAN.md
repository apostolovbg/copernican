# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-08-31
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

> **For agentic workers:** Execute the slices in order. Keep the gate open
> for the active slice, stage each completed slice, and do not call a slice
> closed until its raw scientific evidence exists. A green policy gate is
> necessary hygiene; it is never scientific closure.

**Goal:** Scientifically repair and certify the CPU-reference Copernican
Cosmic Microwave Background Solver (CCMBS) as a complete replacement for the
old CAMB-backed solver. At closure, every bundled model that declares CMB
capability must be accepted by the same finite, converged, physically shaped,
source-residual, scalar/batch, and cache-isolation standards, or explicitly
unavailable because its contract truthfully does not provide that physics.
The fixed LCDM reference must reproduce the frozen CAMB comparison fixture
for every declared sector and observable, without using CAMB as a runtime
fallback.

**Scope:** This plan owns the shared CCMBS generated-source compiler,
hierarchy, line-of-sight projection, numerical convergence, all declared
scalar/vector/tensor sectors, unlensed and lensed observables, bundled-model
contract audit, raw scientific evidence, and final BAO-isolation check. It
does not add Taichi, GPU code, a surrogate, delayed acceptance, broad sampler
optimization, another production Boltzmann backend, or a BAO sound-horizon
convention change.

**Scientific finding that governs this plan:** All ten bundled CMB model
contracts and generated-source graphs currently pass structural audit. The
models are therefore already consumable at the declaration boundary. Their
scientific rejections are runtime findings, not evidence of malformed YAML.
At the fixed diagnostic point, LCDM and QRSF return finite spectra but fail
the acoustic-shape audit; LCDM+massive-neutrino, QAUC, reference Planck 2018,
TOG, TORG, w0wa, and wCDM return finite spectra but fail the independent
generated-source residual audit. USMF2 has a valid explicit graph but has not
completed a bounded fixed-point classification. No model is currently
scientifically accepted. The shared runtime must be repaired before model-
specific declarations are changed.

**Completion target:** Twelve bounded slices close in order. Slices One
through Seven establish the baseline, source/compiler safeguards, projection
infrastructure, declaration and execution evidence, certification boundary,
and the restored scalar-source convention. Slices Eight through Twelve
explicitly own the independent CAMB fixture, all-source repair, full raw
parity for every declared observable, complete bundled-model certification,
and final closure. A model file must never be weakened, relabelled
unavailable, or declared valid merely to hide a shared CCMBS defect.

## Global Constraints

* Do not change branches, create branches, or alter repository workflow.
* CCMBS remains the selected production CMB solver. CAMB is comparison-only
  and may never become a runtime fallback.
* Preserve each model's declared theory. Change a model declaration only to
  express its existing equations, variables, derivatives, closures, gauge,
  or sources accurately and explicitly.
* Repair common defects in shared compiler, hierarchy, background, or
  projection code. Do not add an LCDM-only, QRSF-only, or other model-only
  numerical patch for a defect that affects generated models generally.
* Do not lower accuracy, omit sectors, clip a requested grid, widen a cache
  key, suppress a residual, or relax an acceptance tolerance to produce a
  pass.
* Never silently accept a non-converged spectrum. Keep raw base/refined
  arrays, histories, residual vectors, grids, and typed failures in every
  diagnostic record.
* Generated `Phi_tau`, `Psi_tau`, and related derivatives must be explicit,
  coordinate-aware, finite, and validated. A missing derivative or closure is
  a typed failure, not a zero-valued substitute.
* Keep scalar evaluation as the scientific reference. Ordered batch execution
  may be used only after proving exact scalar equivalence, input ordering,
  per-point failure semantics, and cache isolation.
* A valid request must not be rejected by an arbitrary wall-clock or nominal
  work ceiling. Resource accounting may describe work and bound memory, but
  it must not replace scientific numerical acceptance.
* The named certification tier is an explicit request, never a hidden
  replacement for a model's declared production controls. It records its
  multipoles, sectors, k/eta controls, refinement factor, and effective node
  counts.
* Full `copernican.validation` workloads are not acceptance dependencies.
  Use bounded direct solver diagnostics, fixed fixtures, focused tests, and
  canonical raw reports.
* Preserve public spectrum names, multipole and sector ordering,
  lensed/unlensed distinctions, typed failures, provenance, and cache
  identities.
* Keep BAO unchanged in this plan. Its fixed-background independence from
  CCMBS remains a regression boundary, not a place to compensate for CMB
  errors.
* Keep root/package documentation synchronized. Update comments, docstrings,
  focused tests, generated mirrors, and CHANGELOG with every behavior change.
* Stage every completed slice. Do not commit, push, run `devcovenant run`, or
  close the gate unless explicitly instructed.

## Table of Contents

* [Overview](#overview)
* [Current Evidence](#current-evidence)
* [Scientific Acceptance Contract](#scientific-acceptance-contract)
* [Diagnostic Status Terms](#diagnostic-status-terms)
* [Execution Rules](#execution-rules)
* [Execution Slices](#execution-slices)
* [Completion Standard](#completion-standard)

## Overview

CCMBS has the plumbing required to expose contracts, raw source histories,
projection metadata, fixed-point diagnostics, source residuals, cache
identities, and typed failures. That plumbing does not prove the calculated
physics is correct. The current corpus evidence separates two shared failures:

1. generated source histories fail independent physical residual closure for
   seven completed models; and
2. the source-to-observable projection does not produce the expected acoustic
   TT/TE/EE structure for LCDM and QRSF.

The implementation order follows those facts. A source-history repair must
precede a projection repair, and both must precede a model-file edit. Once the
shared runtime passes LCDM at the frozen reference point, any model that still
fails is audited against its theory and receives an explicit declaration fix
only if its existing model physics is incompletely expressed.

## Current Evidence

### Structural model-contract status

The trusted bundled corpus contains these ten CMB-capable model files:

* `model_lcdm.yml`;
* `model_lcdm_mnu.yml`;
* `model_qauc.yml`;
* `model_qrsf.yml`;
* `model_ref_planck2018.yml`;
* `model_tog.yml`;
* `model_torg.yml`;
* `model_usmf2.yml`;
* `model_w0wa.yml`; and
* `model_wcdm.yml`.

All ten pass the shared contract audit and generated-source graph audit with
no structural issues. This means none is currently rejected because a required
CMB declaration is absent or malformed.

### Fixed-point runtime status

The completed direct diagnostic tier uses fixed parameters, TT/TE/EE,
`ell = 2..300`, and `k_sample_count = 1024`. Its initial classification does
not substitute doubled-grid convergence for the production request; that
evidence remains required later.

| Model | Contract and graph | Completed direct diagnostic | Primary evidence |
|---|---|---|---|
| LCDM | valid | rejected | finite but TT/EE acoustic structure failed |
| LCDM mnu | valid | rejected | finite but source residual audit failed |
| QAUC | valid | rejected | finite but source residual audit failed |
| QRSF | valid | rejected | finite but TT/EE acoustic structure failed |
| Planck ref. | valid | rejected | finite but source residual audit failed |
| TOG | valid | rejected | finite but source residual audit failed |
| TORG | valid | rejected | finite but source residual audit failed |
| USMF2 | valid | unclassified | valid graph; fixed point incomplete |
| w0wa | valid | rejected | finite but source residual audit failed |
| wCDM | valid | rejected | finite but source residual audit failed |

No completed row is currently unavailable: `unavailable` is reserved for a
typed runtime exception that prevents CCMBS from returning a result. USMF2 is
not yet entitled to that label because its direct diagnostic has not completed
and recorded a typed outcome.

## Scientific Acceptance Contract

Every accepted model must satisfy all six layers below at the named
certification tier.

1. **Declared contract:** its capability, sectors, gauge, hierarchy,
   numerical controls, source bindings, derivatives, and observables express
   the model's actual theory and pass structural audits.
2. **Evolved history:** metric potentials, densities, velocities, collision
   sources, visibility, temperature/polarization hierarchy, initial
   conditions, and ISW contributions are finite, explicitly evolved, and pass
   independent residual checks.
3. **Projection:** TT, TE, and EE are generated through phase-aware,
   converged line-of-sight integration with positive, documented quadrature
   behavior and no unresolved aliasing.
4. **Reference and shape:** fixed LCDM agrees with the frozen CAMB fixture at
   explicit raw-array tolerances. Auto-spectra, TE sign changes, peak/trough
   order, damping, and phase are physically shaped without relying on plots.
5. **Execution equivalence:** base/refined grids converge; scalar and ordered
   batch results agree exactly at each point; and cache identities prove no
   cross-talk between parameter points or models.
6. **Evidence and classification:** canonical reports preserve raw data and
   decisions. A failing model is rejected with a typed scientific reason. A
   truly unsupported model is unavailable only when its contract explicitly
   declares that limit and no false CMB result is emitted.

## Diagnostic Status Terms

* **Accepted:** every layer of the scientific acceptance contract passed.
* **Rejected:** CCMBS produced a result, but raw evidence failed at least one
  scientific acceptance check. The failure remains in the report.
* **Unavailable:** CCMBS could not execute because a typed failure prevented a
  result, or the model truthfully declares no CMB capability. It is never a
  synonym for a slow, inconvenient, or currently unmeasured model.
* **Unclassified:** the diagnostic request did not complete. It remains an
  open measurement obligation and cannot be counted as accepted, rejected, or
  unavailable.

## Execution Rules

1. Keep the gate open for the active slice and clear gate complaints before
   applying edits.
2. Use the active `.venv` for policy commands and focused tests.
3. Complete slices in order. Do not change a model YAML file before the
   shared source and projection repairs have been tested against LCDM.
4. Inspect raw reports, arrays, residuals, and metadata before plots or
   sampler output.
5. Preserve pre-repair evidence. A new result must be comparable to the
   baseline through the same fixed request and canonical report shape.
6. Run focused tests for the changed contract and stage the completed slice.
7. Run `source .venv/bin/activate && python -m devcovenant gate --verify` on
   the staged revision before reporting a slice complete.
8. Do not run `devcovenant run`, `gate --close`, commit, or push unless the
   user explicitly requests those actions for that turn.

Task markers mean:

* `[closed]` means implementation and raw acceptance evidence both exist.
* `[in progress]` means work is active and closure evidence is incomplete.
* `[planned]` means the slice has not started.

## Execution Slices

### [closed] Slice One — corpus baseline and USMF2 classification

**Purpose:** Freeze a complete, bounded, deterministic pre-repair scientific
baseline without treating an incomplete run as a model decision.

**Files and surfaces:**

* direct CCMBS diagnostic and canonical-report helpers;
* model discovery, contract, and source-graph audits;
* bounded USMF2 diagnostic controls and typed outcome handling;
* raw-array, diagnostic, and model-adapter tests; and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Define one versioned direct certification request shared by all ten models:
   fixed parameters, ordered TT/TE/EE multipoles, k/eta controls, source
   anchors, and declared refinement requirements.
2. Record raw source histories, transfer components, k-grid metadata,
   configured/effective node counts, public spectra, source residual vectors,
   acoustic metrics, cache identities, and typed decisions before any plot.
3. Run USMF2 through a bounded diagnostic progression that records each tier,
   honest work estimate, completion state, and typed failure. It may not be
   silently omitted, timeout-labelled unavailable, or replaced by another
   model's result.
4. Serialize every model exactly once with a canonical digest and preserve
   the current rejected evidence as the pre-repair baseline.

**Acceptance:**

* All ten expected filename-keyed rows exist exactly once.
* Every row is accepted, rejected, unavailable, or remains explicitly
  unclassified with an honest incomplete-execution record.
* Reports contain raw pre-projection histories and raw TT/TE/EE where a solve
  completed, rather than plot-derived values.
* Repeating the same completed diagnostic produces identical ordering and
  report identity.

**Closure evidence:** `CMB_CORPUS_BASELINE_REQUEST` and
`run_bundled_cmb_corpus_baseline()` now emit one canonical ten-row report with
the frozen request, raw pre-plot data, source and projection metadata, cache
identities, typed outcomes, and stable per-row/report digests. USMF2 tier
prefixes remain explicitly `unclassified` with named remaining work; no
wall-clock condition can convert unfinished work into `unavailable`.

### [closed] Slice Two — shared hierarchy and source closure

**Purpose:** Establish and enforce the shared generated-hierarchy source
closure boundary while preserving every model's declared theory.  Numerical
residual values remain raw evidence for the later projection and certification
slices; this slice makes it impossible for an incomplete graph or hidden
derivative fallback to masquerade as a physical result.

**Files and surfaces:**

* `copernican/lib/perturbation_contract.py`;
* CCMBS generated hierarchy, background, evolution, and source runtime
  modules;
* initial-condition, gauge, derivative, and source-residual diagnostics;
* focused perturbation, runtime, and model-adapter tests; and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation completed:**

1. Strengthened the root perturbation compiler validator.  Generated scalar
   contracts now require typed `Phi_tau`, `Psi_tau`, and `Phi_history_tau`
   entries, all declared source roles, the three Einstein residual entries,
   compiler-backed source/closure expressions, and explicit history-gradient
   bindings.  Missing derivatives, order reductions, zero-expression
   fallbacks, and incomplete residual declarations fail at compilation.
2. Added a manifest-level generated-source closure summary.  It records
   derivative kind, coordinate, order, binding, expression, dependencies,
   source-role ownership, closure names, and residual names for every
   generated scalar contract.  Explicit-graph models receive a deterministic
   `not_applicable` record rather than an invented generated graph.
3. Made generated runtime constraint evaluation strict.  Einstein residual
   metrics no longer normalize against the residual itself when a generated
   term is absent; the runtime raises a typed `ConstraintViolationError`.
   The independent raw source-history recorder rejects missing declared
   fields before an audit can be reported.
4. Recorded derivative provenance and finite-history status in the runtime
   envelope.  `Phi_tau` remains the algebraic Einstein-system derivative,
   while `Psi_tau` and `Phi_history_tau` are independently sampled evolved
   history gradients in conformal time.  The cache-envelope copy path keeps
   this evidence intact.
5. Added focused compiler provenance coverage and re-ran the generated
   scalar hierarchy, initial-constraint, metric-source, and residual tests.
   Compiled-source discovery was also run across all ten bundled CMB models;
   nine generated scalar graphs validated and explicit USMF2 remained
   `not_applicable` because it supplies its own graph.

**Acceptance:**

* Every generated scalar contract has explicit, typed metric derivatives,
  source roles, closures, and Einstein residual declarations.
* Generated runtime histories are finite and grid-aligned or fail with a
  typed error; no missing term is converted into a residual-normalization
  fallback or a zero history.
* Independent raw source-history and derivative provenance remain present in
  the runtime envelope for later physical residual decisions.
* The focused compiler/runtime hierarchy tests pass, and all ten bundled
  model contracts complete the structural source-graph audit without a shared
  declaration defect.

**Closure evidence:** The compiled manifests now carry the generated-source
closure summary and the runtime envelopes carry derivative provenance,
finite-history validation, raw source samples, and independent residual
inputs.  Focused tests passed for missing derivatives, zero fallback
rejection, explicit `Phi_tau` dependencies, generated graph materialization,
initial-constraint preflight, metric-source response, residual rejection, and
anchor diagnostics.  The remaining numerical residual values are preserved
as raw evidence and are intentionally not relabelled as accepted by this
structural closure slice.

### [closed] Slice Three — projection infrastructure and reference comparison

**Purpose:** Establish the source-to-observable infrastructure and the
backend-neutral comparison boundary needed by the later scientific parity
slices. This slice does not claim that CCMBS has already passed CAMB parity.

**Files and surfaces:**

* CCMBS phase-aware grid, Bessel, projection, convergence, and source-history
  cache modules;
* frozen CAMB fixture and raw comparison diagnostics;
* physical-shape, quadrature, phase, refinement, and solver tests; and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Audit the full line-of-sight calculation from the repaired transfer source
   through radial-distance, Bessel evaluation, quadrature weights,
   normalization, and public `C_ell`/`D_ell` conversion.
2. Use phase-aware/adaptive k integration tied to `k(eta_0 - eta)` and retain
   positive, documented quadrature weights. Resolve local radial and acoustic
   phase gaps rather than trusting node count alone.
3. Cache source histories by full physical identity and integrate vectorized
   k/ell chunks without changing scalar ordering, sector labels, or raw
   provenance.
4. Require base-versus-doubled k-grid convergence for TT, TE, and EE. Keep
   failed base/refined arrays and metrics; never accept a non-converged
   spectrum.
5. Compare CCMBS to the frozen CAMB LCDM fixture using identical parameters,
   units, normalization, and ell values. Evaluate peak/trough ordering,
   positions, amplitudes, damping, TE signs and zero crossings, EE peaks,
   and band-limited raw-array errors before plotting.
6. Use QRSF as a second finite source-residual control. Its current acoustic
   rejection must be resolved by the shared projection path, not by a
   model-specific visual or numerical patch.

**Acceptance:**

* phase-aware projection grids use positive, documented quadrature weights and
  retain their effective-node metadata;
* base/refined raw arrays and typed non-convergence failures are preserved;
* scalar projection, ordered batching, and source-history caching retain their
  declared ordering and provenance; and
* the comparison boundary accepts independent raw reference arrays without
  making the reference backend a CCMBS runtime fallback.

**Closure evidence:** The fixed LCDM and QRSF reports preserve source inputs,
base/refined raw/public arrays, grids, quadrature rule, peak/phase/damping
metrics, and comparison metadata. The raw parity decision is intentionally
owned by Slices Eight through Ten.

**Implemented and closed in this gate:**

* The shared projection path retains phase-aware radial and acoustic gap
  requirements, applies positive composite-trapezoid weights on irregular
  log-k ladders, and records the selected rule and effective grid in the raw
  runtime envelope.
* Source histories are cached only under their complete physical identity;
  ordered evolution and ell-batched projection reuse those histories without
  changing sector or multipole ordering.
* Production scalar requests compare base and doubled k grids for every
  requested TT, TE, and EE surface. Failed comparisons retain both raw and
  public arrays and raise a typed convergence error.
* Public D-ell normalization, raw shape/acoustic audits, phase-gap evidence,
  and the backend-neutral reference comparator are exercised independently of
  plotting. The test-owned CAMB surface remains outside production CCMBS.
* Optional vector and tensor kernels use an empty-sector sentinel safely when
  a scalar refinement coarsens a kernel batch. This removes NumPy empty-axis
  deprecation warnings without fabricating a source or changing scalar
  projections.

**Verification:** The focused adaptive, diagnostic, projection, and scientific
reference suites pass. The scientific reference suite covers batched Bessel
kernels, scalar CAMB response and background anchors, tensor absolute anchors,
lensing normalization, massive-neutrino reference moments, and declared
CAMB-free production boundaries. The final scalar and independent source-grid
refinement tests pass their one-percent bounds with warnings treated as
errors, and the empty optional-sector regression is green.

### [closed] Slice Four — theory-faithful bundled model declarations

**Purpose:** Correct only residual model-specific declarations after the
shared runtime is proven, without altering the scientific essence of any
model.

**Files and surfaces:**

* affected files under `copernican/models/`;
* declaration compiler, contract/source-graph audits, and model adapters;
* model-specific source, gauge, and fixed-point tests; and
* model-template and solver documentation, README mirrors, PLAN, and
  CHANGELOG.

**Implementation tasks:**

1. Re-run every bundled model against the repaired shared runtime and compare
   its result with the pre-repair baseline.
2. For each persistent failure, determine whether it is a model-specific
   declaration omission or a further shared CCMBS error. A declaration change
   is permitted only when it makes existing theory explicit: for example an
   already implied source, derivative, closure, gauge relation, numerical
   domain, or observable definition.
3. Add the minimum explicit contract content required by the existing theory;
   document its equation-level rationale and add a model-specific test.
4. Do not alter parameter priors, background theory, gravitational law,
   species content, physical sectors, or source equation merely to match LCDM
   or CAMB.
5. Complete USMF2 classification. If it requires model-authored explicit
   numerical/source declarations, add only those faithful declarations. If it
   remains too expensive at the fixed request, retain honest work metadata and
   repair the bounded shared execution path rather than marking it
   unavailable by convenience.
6. Re-audit all ten contracts and source graphs after every declaration edit.

**Acceptance:**

* Every declaration edit has a model-theory rationale, a raw before/after
  diagnostic, and a focused regression test.
* No model receives an LCDM-only surrogate equation, a hidden fallback, or a
  weaker acceptance rule.
* Every bundled model has a complete direct result or a truthfully declared,
  typed unavailable capability; no row remains silently unclassified.
* All model contracts and source graphs pass deterministic audits.

**Closure evidence:** The corpus report maps each model's equation-level
declaration decision to source, projection, convergence, and typed outcome
evidence. It records no theory-changing model patch.

**Implementation and verification:** Added the deterministic
`CMBModelDeclarationDecision` audit and its bundled assertion helper. The
audit composes the contract and source-graph records, classifies the actual
execution route, preserves theory-specific source descriptions, and keeps
`ready`, `rejected`, and explicit `unavailable` distinct. All ten frozen CMB
models are `ready`: eight use the shared generated scalar hierarchy, QRSF and
TORG retain their named baryon-locked density/momentum/Euler source closures,
and USMF2 is independently classified as an explicit scalar graph. No model
YAML equation, prior, species, or background theory was changed. Focused
contract, diagnostics, and public-symbol tests pass; the declaration audit is
deterministic and rejects an explicit graph that omits a required projection
source role. Numerical spectrum certification remains owned by Slice Five.

### [closed] Slice Five — scalar, batch, cache, and corpus evidence

**Purpose:** Capture complete scalar, ordered-batch, cache-isolation, and
bundled-matrix evidence under one deterministic request. Scientific model
acceptance is performed by Slice Eleven after shared CAMB parity is proven.

**Files and surfaces:**

* bundled-matrix diagnostic/report builders;
* scalar/batch adapter and source-history cache paths;
* cache-isolation, ordering, refinement, physical-shape, and corpus tests;
  and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Execute all ten models at the final named certification tier with raw
  TT/TE/EE, source residuals, physical-shape metrics, and base/refined
  convergence retained per model.
2. At two independently variable parameter points per applicable model,
  compare direct scalar results to ordered batch results exactly, including
  input order, sector ordering, multipole ordering, raw arrays, public arrays,
  metadata, and typed failures.
3. Prove cache isolation across models and parameter points from full cache
  identities and demonstrate that completion order cannot change results.
4. Treat a model as accepted only when all scientific and execution evidence
  passes. A true declared non-CMB capability is unavailable; all other failed
  evidence is rejected and returns ownership to the slice that can repair it.
5. Produce a canonical, hashable corpus report that includes every expected
  filename once and has no omitted spectrum, silent downgrade, or
  plot-derived decision.

**Acceptance:**

* the matrix preserves finite or typed-failure raw arrays, histories, grids,
  residuals, and convergence evidence for every requested model;
* scalar/batch equality, ordering, failure semantics, and cache identities are
  recorded without treating a cache hit as a scientific mismatch; and
* the complete corpus report is deterministic across clean processes, while
  its scientific decisions remain explicit rather than upgraded to success.

**Closure evidence:** The final ten-row raw matrix contains contracts, source
histories, grids, spectra, residuals, refinement, scalar/batch/cache evidence,
typed outcomes, canonical JSON, and SHA256 digest.

**Implementation completed and closed in this gate:**

1. The cached scalar adapter now retains the complete typed `CMBResult` for
   parity audits, including requested ordering, solver identity, raw
   unscaled spectra, diagnostics, and phase provenance. Numerical overrides,
   the matrix fast-path marker, and the workload are bound identically on
   scalar and ordered batch requests.
2. The generated declared-graph orchestrator publishes the raw spectra from
   its custom projection through a request-local context. The NumPy solver
   attaches that payload to both scalar and batch results without changing the
   public spectrum conversion or introducing a second backend.
3. Scalar/batch/cache evidence now compares the actual public and raw arrays,
   ordered request metadata, typed failures, stable solver diagnostics, and
   distinct cache identities. Per-call counters and wall-clock phase values
   remain recorded but are not treated as scientific mismatches when a batch
   item is served from an exact cache hit.
4. A finite solve rejected by acoustic-shape or source-residual evidence is
   still eligible for the execution-parity audit; only a solve with no
   spectrum can be unavailable for that audit. This preserves the scientific
   rejection reason while preventing missing batch evidence from hiding a
   cache or ordering defect.
5. The strict matrix path retains one deterministic row for every frozen
   filename, attaches contract/source/declaration audits, executes the named
   tier when requested, and leaves rejected or typed-unavailable rows visible
   in the canonical report. No row is silently downgraded, duplicated, or
   accepted from plot output.

**Verification:** Focused diagnostics, contract, result, orchestrator, and
solver suites pass. A fixed LCDM matrix probe produced identical scalar and
ordered-batch public/raw spectra, stable request metadata, and two distinct
cache identities. `gate --verify` passed with no policy complaints. The
scientific decisions remain open until Slice Eleven's post-parity matrix.

### [closed] Slice Six — final scientific certification and BAO boundary

**Purpose:** Publish the deterministic certification boundary and prove the
CMB work has not changed BAO's independent fixed-background behavior. The
actual CCMBS/CAMB and corpus closure decisions are assigned to Slices Eight
through Twelve.

**Files and surfaces:**

* certification report builder and canonical artifact writer;
* frozen LCDM CAMB comparison and bundled-matrix reports;
* BAO isolation regression and likelihood outputs; and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Build the final certification report from the accepted corpus evidence:
   model list, fixed parameters, solver/dataset identities, numerical
   controls, fixture hashes, raw evidence digests, and every acceptance
   decision.
2. Re-run the fixed-background BAO isolation regression with the CMB solver
   entrypoint unavailable. Require the same BAO values, covariance handling,
   and typed failure classification; do not change BAO implementation or its
   recombination-versus-drag sound-horizon convention.
3. Verify that no CAMB fallback, surrogate, delayed acceptance, hidden alias,
   arbitrary timeout, unchecked declaration bridge, machine-local path, or
   plot-only acceptance remains.
4. Reconcile public docs, model-template documentation, comments, docstrings,
   focused tests, and changelog with the actual certified capability boundary.
5. Run focused acceptance tests and `gate --verify` on the staged revision.

**Acceptance:**

* the report boundary is complete, deterministic, hashable, and contains no
  failed or omitted scientific decision disguised as success;
* BAO is independently evaluable and numerically unchanged at fixed
  background parameters.
* focused report and BAO isolation tests pass.

**Closure evidence:** Only this slice may mark the plan complete. A green
policy gate, finite output, or attractive plot without the final raw report is
not closure.

**Implementation closure (2026-08-30):** The final report builder and writer
now persist solver/dataset identities, fixture hashes, per-model raw evidence
digests, explicit integrity checks, selected independent-reference decisions,
and the BAO isolation decision. `assess_bao_cmb_isolation()` is a pure,
path-free comparison boundary and does not invoke CCMBS. Focused final-report
and BAO regression tests cover passing evidence and explicit rejection. The
writer never upgrades rejected or unavailable scientific rows to success.

### [closed] Slice Seven — enforce source contract and production phase floor

**Purpose:** Keep the tested generated scalar source contract intact after
the convergence hardening work and prevent joint MCMC execution from selecting
the under-resolved diagnostic ladder for production spectra.

**Files and surfaces:**

* generated scalar source declarations in `perturbation_contract.py`;
* projection-grid policy in `runtime/projection.py`;
* focused CCMBS source and workload-floor tests; and
* CMB solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Validate the generated source contract against the physical scalar
   collision and line-of-sight contract: the `-Phi_tau` photon monopole term,
   `theta_gamma2 + e_gamma0 + e_gamma2` polarization moment, exact collision
   block, monopole `Pi / 4` term, explicit zero split quadrupole terms, and
   `3/4` E-source coefficient.
2. Keep the strict metric-history provenance and residual validation introduced
   by the convergence work; restoring the source convention must not re-add a
   missing-derivative fallback or suppress a closure failure.
3. Require the final production phase-grid floor for joint MCMC requests.
   Only the explicitly named matrix diagnostic fast path may retain the
   bounded low-node scaffold used by structural tests.
4. Add regression coverage for the restored generated expressions and for the
   final phase-grid floor, then synchronize the solver documentation and both
   README mirrors with the implementation.

**Acceptance:**

* the generated scalar manifest exposes the tested photon, polarization,
  collision, temperature, and E-source expressions exactly;
* additive source-role bindings remain present and metric-history validation
  remains strict;
* a final-tier joint MCMC projection uses at least the production phase-grid
  floor rather than the 64-node diagnostic ladder;
* the focused source, hidden-prefix, hierarchy, and phase-floor tests pass;
  and
* no CAMB fallback, surrogate, weakened convergence tolerance, or model-only
  declaration patch is introduced.

**Closure evidence (2026-08-30):** The tested generated scalar convention is
preserved in the shared contract, the joint-MCMC bypass of the final
phase-grid floor was removed, and the focused source, collision, hidden-prefix,
phase-floor, and full-declared-LCDM tests passed. Raw generated expressions,
collision metadata, and the resulting projection-grid size are asserted
directly. This closes the source-contract/workload-floor regression slice.
Independent CAMB parity is explicitly assigned to Slices Eight through Ten,
and final corpus acceptance and report closure to Slices Eleven and Twelve.

### [closed] Slice Eight — freeze the full independent CAMB fixture

**Purpose:** Establish the exact, immutable reference against which a fully
functional CCMBS will be judged. The fixture is comparison-only: CAMB must
never be called from a production CCMBS evaluation or used as a fallback.

**Files and surfaces:**

* CAMB comparison fixtures and their versioned metadata;
* CCMBS raw-spectrum comparison and normalization helpers;
* declared-sector, observable, multipole, and unit-ordering tests; and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Freeze one fixed LCDM parameter point and all numerical controls used by
   the old CAMB solver, including every requested scalar, vector, and tensor
   sector and every requested unlensed/lensed observable.
2. Generate and store raw CAMB arrays for TT, TE, EE, BB, PP, TP, EP, and
   every applicable lensed spectrum on the canonical ell grid. Preserve
   transfer/source metadata where CAMB exposes it, plus units and conversion
   conventions.
3. Hash the fixture and record CAMB version, settings, data ordering, ell
   range, precision, and provenance in a machine-independent manifest.
4. Build a comparator that aligns arrays without interpolation shortcuts,
   checks finite values and exact ordering, and reports absolute, relative,
   band-limited, peak/phase, sign, and damping errors by observable.
5. Add a test proving the comparison fixture is unreachable from the CCMBS
   runtime path and that a missing comparison backend produces a typed test
   failure rather than a runtime fallback.

**Acceptance:**

* the fixture is deterministic, hashable, and complete for every declared
  observable and sector;
* raw `C_ell` and `D_ell` conventions are explicit and reversible;
* all later parity slices consume this same fixture without changing its
  arrays or tolerances; and
* no CAMB import, subprocess, surrogate, or fallback is reachable from a
  production CCMBS request.

**Closure evidence:** A frozen fixture manifest, raw-array hashes, comparator
reports, and runtime-isolation test output are committed before Slice Eight
is closed. Any missing CAMB observable is recorded as a fixture limitation,
not silently treated as parity.

**Closure evidence (2026-08-30):** The test-owned CAMB reference now freezes a
versioned LambdaCDM scalar fixture at eight canonical multipoles. It stores
finite `C_ell` and native `D_ell` arrays for TT, TE, EE, BB, PP, TP, EP, and
all applicable lensed scalar surfaces, with explicit lens-potential
conventions, applicability records, CAMB identity, numerical settings, and a
SHA-256 digest. The strict comparator rejects missing, mis-shaped, or
non-finite arrays and never interpolates. The fixture loader verifies its
digest, and regeneration reproduces the frozen arrays. Existing production
boundary tests continue to prove that the test-only CAMB contract cannot enter
CCMBS and that declared execution remains CAMB-free. Vector and tensor outputs
are recorded as not declared by the bundled scalar models; their dedicated
fixtures are owned by the later sector-parity slices.

### [closed] Slice Nine — repair shared histories and source physics

**Purpose:** Make the generated CCMBS histories physically correct before
projection is judged. This slice owns shared background, metric, hierarchy,
collision, initial-condition, gauge, visibility, and ISW defects.

**Files and surfaces:**

* generated hierarchy/source compiler and runtime evolution;
* background and recombination histories;
* scalar, vector, tensor, and polarization source residual diagnostics;
* source-history tests and raw evidence reports; and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Compare the generated source graph and independently sampled histories
   against the frozen LCDM reference requirements for every declared sector.
2. Repair equations, signs, normalizations, derivatives, closure bindings,
   collision terms, initial conditions, visibility normalization, and ISW
   sources in shared code. Preserve each model's declared theory.
3. Validate `Phi_tau`, `Psi_tau`, `Phi_history_tau`, density/velocity moments,
   polarization moments, and source derivatives on independent eta and k
   grids. Missing or non-finite terms remain typed failures.
4. Require source-history convergence under independent eta, evolution, and
   k refinements. Record raw histories, residual vectors, and closure metrics
   before any line-of-sight projection.
5. Prove the repaired scalar source histories reproduce the old CAMB solver's
   transfer-source sign, phase, and normalization conventions at the fixed
   point; extend the same check to vector and tensor sources where declared.

**Acceptance:**

* all declared source histories are finite, explicit, grid-converged, and
  independently residual-clean;
* metric, hierarchy, collision, visibility, polarization, initial-condition,
  and ISW closures pass without zero-expression fallbacks;
* scalar/vector/tensor source signs and normalizations agree with the frozen
  reference conventions; and
* a source failure is rejected explicitly rather than hidden by projection,
  plotting, or a model-specific patch.

**Closure evidence:** The raw source-history bundle contains residuals,
derivative provenance, refinement arrays, closure decisions, and a digest for
each applicable sector. Slice Nine closes only when the histories, not merely
the generated declarations, pass these tests.

**Closure evidence (2026-08-31):** Generated scalar histories now retain
fine/coarse eta arrays, residual vectors, derivative provenance, audit
controls, closure decisions, and a deterministic SHA-256 bundle digest in the
runtime envelope. The digest is marked `not_applicable` for explicit model
graphs rather than fabricating generated-history evidence. The Slice Nine
reference and scientific validation suites pass (23 focused tests), covering
metric/background constraints, collision and visibility conventions,
polarization and ISW sources, massive-neutrino terms, tensor/vector kernels,
and reproducible source-history evidence.

### [closed] Slice Ten — prove full observable CAMB parity

**Purpose:** Repair and certify the complete source-to-observable CCMBS path
against the frozen CAMB fixture. This is the central scientific slice: it
must make CCMBS work as the old CAMB solver worked, across all observables the
model declares.

**Files and surfaces:**

* line-of-sight kernels, Bessel evaluation, projection, and normalization;
* scalar/vector/tensor transfer and lensing projections;
* unlensed and lensed spectrum assembly;
* raw-array parity, physical-shape, and convergence tests; and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Audit radial distance, Bessel conventions, source interpolation, phase
   coverage, quadrature weights, k integration, ell integration, and all
   `C_ell`/`D_ell` conversions against the frozen reference.
2. Repair the projection until fixed LCDM raw arrays match CAMB for every
   applicable TT, TE, EE, BB, PP, TP, EP, lensed, vector, and tensor output.
3. Require independent base/refined convergence for every requested surface,
   including low-ell, acoustic, damping-tail, tensor, and lensing ranges.
4. Check acoustic peak/trough positions, amplitudes, damping, TE zero
   crossings and signs, EE/BB structure, PP response, and lensed smoothing
   from raw arrays rather than plots.
5. Compare multiple fixed parameter perturbations to prove the parity is a
   physical response, not a single-point normalization coincidence.
6. Keep scalar CCMBS as the reference implementation and prove any ordered
   sector batching is numerically identical before using it in production.

**Acceptance:**

* fixed LCDM passes explicit raw-array CAMB tolerances for every declared
  observable and sector, not only TT/TE/EE;
* all requested spectra are finite, phase-correct, normalized, converged,
  and physically shaped, including lensed and parity-odd outputs;
* no declared observable is omitted, synthesized, clipped, or accepted from
  a plot; and
* all failures retain base/refined arrays and typed diagnostics.

**Closure evidence:** The parity report contains one raw comparison row per
observable and sector, complete error bands and shape metrics, refinement
evidence, fixture digest, and a final pass decision. Slice Ten cannot close
on finite output alone.

**Closure evidence (2026-08-31):** The diagnostics surface now implements the
full parity contract for direct, sector-nested, and structured CAMB fixtures.
It compares raw `C_ell` or `D_ell` arrays for TT, TE, EE, BB, PP, TP, EP,
lensed TT/TE/EE/BB, and scalar/vector/tensor/total sectors without
interpolation. Auto-spectrum power, cross-spectrum signs, C-to-D
conversions, finite/shape checks, independent refinement evidence, fixture
digests, and response-point reports are retained in a deterministic report.
The shared TP/EP projection scale now uses the CAMB
`[ell(ell+1)]**(3/2)/(2*pi)` temperature-unit convention. Focused full
parity, refinement-rejection, and physical-scale tests pass; a mismatched
CCMBS array remains rejected by the report rather than being accepted as
scientific parity.

### [planned] Slice Eleven — certify every bundled model and workload

**Purpose:** Apply the shared repaired solver to the entire bundled corpus
without changing model theories. Every model must compute every observable
it declares, including BB, PP, cross-spectra, and lensed outputs.

**Files and surfaces:**

* all bundled model declarations and explicit graph adapters;
* corpus matrix, scalar/batch, cache, and failure-semantics paths;
* model-specific raw parity and physical-shape reports; and
* model-template documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Execute all ten bundled models at the final named tier with every declared
   sector and observable, retaining raw histories, spectra, grids, residuals,
   and doubled-refinement evidence.
2. For every model with CMB capability, classify it as accepted only after
   finite, source-clean, converged, physically shaped output passes the
   applicable CAMB/reference comparison. A model with no genuinely declared
   CMB sector is explicitly unavailable, never silently skipped.
3. Compare scalar and ordered-batch results at multiple parameter points,
   including public/raw arrays, sector and ell ordering, metadata, typed
   failures, and lensed/unlensed distinctions.
4. Prove cache isolation across models, gauges, sectors, and parameter points;
   completion order must not affect any result.
5. Repair only theory-faithful model declarations that remain incomplete
   after Slices Nine and Ten. No YAML edit may weaken a tolerance or turn a
   shared solver defect into an unavailable result.
6. Complete USMF2 and every other previously unclassified row with a bounded,
   reproducible outcome and named raw evidence.

**Acceptance:**

* every CMB-capable model is accepted by the common full-observable contract
  or has a truthful, typed unavailable capability;
* no model remains unclassified, and no declared observable is omitted;
* scalar/batch equivalence, cache isolation, ordering, convergence, and
  failure semantics pass for every accepted model; and
* the corpus report is deterministic across clean processes.

**Closure evidence:** The complete matrix contains one canonical row per
model, full-observable raw arrays, source residuals, parity decisions,
batch/cache evidence, typed failures, and stable row/report digests.

### [planned] Slice Twelve — final closure and BAO independence

**Purpose:** Close the mission only after CCMBS is proven equivalent to the
old CAMB solver at the full declared-observable boundary and the independent
BAO behavior is preserved.

**Files and surfaces:**

* final certification report and canonical artifact writer;
* frozen CAMB fixture and complete bundled-model matrix;
* BAO fixed-background isolation regression;
* runtime fallback/alias/dependency audits; and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Build one deterministic, hashable final report containing fixed parameters,
   solver and dataset identities, all sector/observable parity arrays and
   hashes, source residuals, refinement evidence, model decisions, and cache
   identities.
2. Re-run BAO with the CMB entrypoint unavailable and prove identical values,
   covariance handling, and typed failure classification at fixed background
   parameters. Do not alter BAO's sound-horizon convention.
3. Audit the repository for CAMB runtime imports, surrogates, delayed
   acceptance, hidden aliases, arbitrary timeouts, machine-local paths,
   unchecked declaration bridges, omitted spectra, and plot-only decisions.
4. Reconcile code, model templates, comments, docstrings, tests, README
   mirrors, solver documentation, PLAN, and CHANGELOG with the certified
   capability boundary.
5. Run the focused acceptance suites and the complete required workflow. Run
   `source .venv/bin/activate && python -m devcovenant gate --verify` on the
   staged revision, then mark the entire plan closed only if all evidence is
   green.

**Acceptance:**

* CCMBS reproduces the old CAMB solver's fixed LCDM results for every
  applicable declared sector and observable within explicit raw tolerances;
* every bundled model is fully certified or truthfully unavailable, with no
  omitted output or unclassified row;
* BAO remains independently evaluable and unchanged at fixed background
  parameters;
* no runtime fallback, surrogate, hidden compatibility path, or convenience
  rejection exists; and
* the canonical report, tests, staged tree, and gate verification are green.

**Closure evidence:** Only Slice Twelve may mark the plan complete. Its final
report is the proof: a green policy gate alone is insufficient, and no
remaining scientific requirement may exist outside Slices Eight through
Twelve.

## Completion Standard

This plan is complete only when all twelve slices are closed. Every criterion
below is a derived check whose implementation and evidence are assigned to
the slices named in parentheses; this section introduces no unassigned work.

* every CMB-capable model contract is structurally valid and theory-faithful
  (Slices Four and Eleven);
* generated metric, visibility, temperature, polarization, initial-condition,
  and ISW histories are explicit, finite, and independently residual-clean
  (Slices Two and Nine);
* fixed LCDM matches the old CAMB solver for every applicable declared
  scalar, vector, tensor, unlensed, lensed, parity-even, parity-odd, and
  cross-spectrum observable within the frozen raw-array tolerances
  (Slices Eight through Ten and Twelve);
* every bundled model has finite, physically shaped, converged raw spectra or
  a truthfully declared non-CMB unavailable capability (Slice Eleven);
* no model is hidden, synthetically substituted, rejected by convenience, or
  accepted by a weakened criterion (Slices Eight, Ten, Eleven, and Twelve);
* scalar/batch equivalence and cache isolation are present for every accepted
  model (Slices Five and Eleven);
* BAO remains independently evaluable and unchanged at fixed background
  parameters (Slices Six and Twelve);
* no runtime fallback, surrogate, delayed acceptance, Taichi dependency,
  wall-clock rejection, hidden compatibility path, or plot-only acceptance is
  present (Slices Eight, Ten, Eleven, and Twelve); and
* raw evidence is canonical, hashable, reproducible, documented, and staged
  with a green gate verify (Slices Eight through Twelve).

The raw scientific certification report from Slice Twelve, not a gate result,
is proof that CCMBS works as the old CAMB solver worked and that the bundled
model corpus is closed.
