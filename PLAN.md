# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-08-25
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

> **For agentic workers:** Execute the slices in order. Keep the gate open
> for the active slice, stage each completed slice, and do not call a slice
> closed until its raw scientific evidence exists. A green policy gate is
> necessary hygiene; it is never scientific closure.

**Goal:** Finish and scientifically certify the CPU-reference CCMBS path so
that its fixed LCDM result has the expected acoustic spectra, every bundled
CMB model has an explicit and defensible result, and the raw evidence can be
reproduced without relying on sampler length, plotting, GUI state, or a
machine-specific performance escape hatch.

**Scope:** This plan owns the remaining CCMBS runtime, generated hierarchy,
reference comparison, corpus certification, and final evidence work. It does
not add Taichi, a GPU dependency, a surrogate, delayed acceptance, another
production Boltzmann backend, or broad sampler optimization. The current
NumPy/SciPy CCMBS implementation remains the reference implementation until
this plan is closed.

**Resolved runtime blocker:** The first fixed LCDM attempt did not reach
projection because the calculated work (`687667200`) exceeded a nominal
`100000000` ceiling. The accounting and execution path now treats that as
planned work, not a scientific rejection. The remaining scientific work is
sequenced after the runtime and evidence seams so each acceptance boundary is
bounded and reproducible on the designated CPU environment.

**Completion target:** Eight bounded slices close in order. The first three
separate runtime accounting, ordered execution, and direct evidence capture;
the closed fourth slice records the completed infrastructure boundary; the
remaining slices repair shared physics, prove LCDM parity, certify the
bundled corpus, and publish final closure. The fixed-point diagnostic tier is
an explicit,
versioned request rather than a hidden machine-local reduction of a model's
declared defaults. Any unmeasured, unavailable, non-finite, non-converged,
or non-reproducible case keeps its owning slice open.

## Global Constraints

* Do not change branches, create branches, or alter repository workflow.
* Keep CCMBS as the selected CMB solver. CAMB is comparison-only and may
  never become a runtime fallback.
* Keep exact scalar evaluation as the scientific reference. Batch paths may
  be optimized only after scalar equivalence, ordering, failure, and cache
  isolation are demonstrated.
* Preserve public spectrum names, sector ordering, multipole ordering,
  lensed/unlensed distinctions, typed failures, cache identities, and
  provenance fields.
* Do not lower accuracy, truncate a requested grid, or widen a cache key to
  make a comparison pass.
* Remove arbitrary internal wall-clock and iteration rejection from CCMBS.
  Resource accounting may plan work and memory, but it must not reject a
  scientifically valid request merely because a fixed seconds budget was
  exceeded.
* Work must be chunked or vectorized when needed to keep memory bounded, and
  every chunk must retain the same raw metadata and final ordering as scalar
  evaluation.
* Never silently accept a non-converged spectrum. Preserve failed spectra,
  residuals, grids, and typed failure details in the runtime envelope.
* Generated source terms must be explicit and typed. Missing derivatives,
  invalid closures, non-finite histories, and zero placeholders must fail
  before likelihood acceptance.
* Model-specific declarations are allowed only when the model physics truly
  differs. Shared defects must be repaired in shared compiler/runtime code.
* Full `copernican.validation` workloads are not acceptance dependencies.
  Use bounded deterministic fixtures, focused tests, and raw reports.
* The fixed-point certification tier is a named diagnostic request with
  explicit multipoles, k/eta controls, and a refinement factor recorded in
  the report. It is allowed to differ from a model's production defaults only
  when the request is explicit, reproducible, and independently shown to
  converge. No code path may silently substitute that tier for a caller's
  declared production request.
* Do not add Taichi or GPU dependencies in this plan. A later plan owns
  Taichi, Vulkan/AMD support, precision, and GPU throughput.
* Treat Vulkan as the first future AMD-capable Taichi target; never assume
  CUDA on an AMD device.
* The surrogate and delayed-acceptance production paths remain removed.
* Retired `engine` and `cosmo` identifiers must not be reintroduced except
  in documented historical provenance strings.
* Keep root/package documentation synchronized. Update comments, docstrings,
  tests, generated mirrors, and CHANGELOG with every behavioral change.
* Stage every completed slice. Do not commit or push unless explicitly
  instructed.

## Table of Contents

* [Overview](#overview)
* [Current State](#current-state)
* [Scientific Closure Contract](#scientific-closure-contract)
* [Evidence Contract](#evidence-contract)
* [Execution Rules](#execution-rules)
* [Execution Slices](#execution-slices)
* [Completion Standard](#completion-standard)

## Overview

The repository already contains the contract boundaries needed to finish the
job: CCMBS solver selection, scalar and ordered batch result types, typed
failures, phase-aware grids, source-history cache identities, diagnostic
serialization, generated-model audits, and a fixed-background BAO isolation
regression. Those surfaces are evidence plumbing, not proof that the physics
is right.

The remaining work is deliberately sequenced so a runtime defect cannot hide
a scientific defect and a green policy gate cannot be mistaken for a valid
spectrum:

1. certify deterministic runtime accounting;
2. certify ordered, memory-bounded execution;
3. certify direct fixed-point evidence and reproducibility;
4. repair and validate the generated hierarchy;
5. validate LCDM spectra against the frozen independent CAMB fixture;
6. execute the complete bundled matrix and publish final certification.

The prior foundation and numerical plumbing remain part of the baseline. They
are retained and tested; they are not repeated as closure claims below.

## Current State

### Delivered foundation

The closed baseline provides:

* sampler vocabulary and removal of surrogate and delayed-acceptance paths;
* pluggable CCMBS selection, result contracts, typed failures, and
  provenance;
* phase-aware eta and wave-number controls with explicit refinement status;
* bounded source-history caching and safe scalar/batch ordering;
* generated-contract audits and typed metric-history derivative bindings;
* fixed-parameter diagnostic and serialization seams for the bundled corpus;
* independent generated-hierarchy source, initial-condition, and metric
  residual closure at the fixed diagnostic tier;
* an independent fixed LCDM CAMB fixture definition; and
* a BAO regression proving fixed-background evaluation does not require CMB.

### Not yet scientifically closed

The following evidence is still absent or untrusted until the slices below
complete:

* finite and physically shaped spectra for every bundled CMB model; and
* a final report proving that no model, spectrum, or residual was omitted.

## Scientific Closure Contract

Closure has four independent layers. All four must pass.

1. **Declared contract:** model metadata, gauge, sectors, hierarchy families,
   requested spectra, numerical bounds, and source bindings are valid.
2. **Runtime contract:** fixed-point and batch results are ordered, finite
   when accepted, typed when rejected, cache-isolated, and fully described.
3. **Physical contract:** metric, visibility, temperature, polarization,
   initial-condition, and ISW histories satisfy independent residual checks.
4. **Reference contract:** fixed LCDM TT, TE, and EE agree with the frozen
   independent CAMB fixture, while every bundled model passes finite,
   convergence, and physical-shape checks.

A plot, finite scalar, short sampler run, successful import, or green gate is
not a substitute for any layer.

## Evidence Contract

Every scientific report must be deterministic and include:

* model filename and contract digest;
* fixed parameter vector, seed, solver identity, and dataset identity;
* requested multipoles, sectors, k/eta grids, accuracy tier, refinement
  factor, effective node counts, and work-accounting metadata;
* raw evolved source histories and transfer components before projection;
* raw public TT/TE/EE arrays after projection;
* base/refined spectra, residual vectors, peak and phase metrics, and every
  acceptance decision;
* cache identities, scalar/batch counts, chunk sizes, and process settings;
* typed failure details for every rejected request; and
* a canonical JSON serialization and SHA256 digest.

The CAMB fixture is comparison-only. Its parameters, normalization,
multipoles, tolerances, and provenance are frozen before comparison and may
not be tuned after seeing CCMBS output.

## Execution Rules

1. Keep the gate open for the active slice and clear gate complaints before
   applying edits.
2. Use the active `.venv` for policy commands and bounded focused tests.
3. Complete slices in order. Do not run corpus certification while the
   fixed-point runtime or LCDM parity acceptance is incomplete.
4. Inspect raw reports and serialized metrics before inspecting plots.
5. Run focused tests for each changed contract. Do not replace them with a
   full validation workload.
6. Run `source .venv/bin/activate && python -m devcovenant gate --verify`
   on the staged revision before reporting a slice complete.
7. Do not run `devcovenant run`, `gate --close`, commit, or push unless the
   user explicitly requests those actions for that turn.
8. Stage every completed slice, including documentation and CHANGELOG.

Task markers mean:

* `[closed]` means implementation and acceptance evidence both exist.
* `[in progress]` means work is active and closure evidence is incomplete.
* `[planned]` means the slice has not started.

## Execution Slices

### [closed] Foundation — contracts, observability, and isolation

**Purpose:** Preserve the completed architecture while the remaining CCMBS
scientific work is performed behind stable contracts.

**Delivered:** sampler naming cleanup; removal of surrogate and delayed
acceptance; CCMBS solver protocol and selection; scalar and ordered batch
results; typed failures and provenance; phase-aware grids; source-history
cache identities; generated-contract audits; diagnostic serialization; and
fixed-background BAO isolation.

**Boundary:** This foundation does not claim a physically correct hierarchy,
CAMB parity, or shaped spectra.

### [closed] Slice One — deterministic runtime accounting

**Purpose:** Remove machine-local runtime rejection and make every valid
CCMBS request describe its work without changing the requested equations or
resolution.

**Files and surfaces:**

* `copernican/lib/likelihoods/cmb/runtime/convergence.py`;
* runtime-envelope validation in the shared projection path;
* numerical-contract and runtime tests; and
* solver documentation, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Trace the fixed LCDM request through contract validation and enumerate
   wall-clock, work, node-count, phase, memory, and hidden downsampling
   guards.
2. Remove fixed-seconds and nominal work-ceiling rejection. Replace it with
   deterministic work estimates derived from k, eta, hierarchy, sector, and
   multipole sizes.
3. Keep configured and effective numerical controls distinct and expose both
   in the runtime envelope.
4. Keep malformed operator limits typed and explicit; valid requests must
   not fail merely because their honest estimate is large.
5. Leave BAO and CAMB runtime selection unchanged.

**Acceptance:**

* Work estimates are deterministic for identical contracts.
* Valid bounded and unbounded requests are not rejected by a fixed time or
  nominal-work ceiling.
* Malformed limits retain typed failures and provenance.
* Focused runtime tests pass on a staged revision with a green gate verify.

**Closure evidence:** Runtime-envelope tests and a serialized work-estimate
fixture record the configured controls, effective controls, estimate version,
and typed malformed-request result.

### [closed] Slice Two — ordered, memory-bounded execution

**Purpose:** Execute large valid requests in bounded chunks while preserving
scalar semantics, cache isolation, numerical ordering, and raw provenance.

**Files and surfaces:**

* CCMBS evolution and projection chunking in
  `copernican/lib/likelihoods/cmb/runtime/projection.py`;
* source-history and Bessel cache surfaces;
* scalar/batch and runtime tests; and
* solver documentation, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Derive deterministic evolution and projection chunk sizes from explicit
   state and work-cell budgets.
2. Accumulate chunks in k-index and ell order without downsampling or
   changing sector labels.
3. Record completed chunks, peak state cells, Bessel batch sizes, cache
   identities, and configured/effective controls.
4. Prove scalar and chunked paths agree within machine precision, including
   a deliberately shuffled completion-order fixture.
5. Preserve typed failure context when a chunk cannot be completed.

**Acceptance:**

* The bounded LCDM fixture completes raw TT, TE, and EE projection with finite
  arrays.
* Chunked and scalar results have identical values, ordering, and provenance.
* Memory accounting is bounded by the configured chunk budgets.
* No requested grid is silently clipped or promoted.
* Focused execution tests pass with a green gate verify.

**Closure evidence:** A serialized bounded fixture report contains nonzero
chunk counts, ordered accumulation metadata, finite raw spectra, and the
scalar-equivalence decision.

### [closed] Slice Three — direct fixed-point evidence and reproducibility

**Purpose:** Make the scientific evidence path independent of the sampler,
GUI, plotting, and full validation workloads.

**Files and surfaces:**

* `copernican/lib/likelihoods/cmb/diagnostics.py`;
* direct CCMBS solver and diagnostic entrypoints;
* fixed LCDM raw-report and bundled-contract audit tests; and
* README mirrors, solver documentation, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Invoke CCMBS directly with an explicit, versioned certification-tier
   request and frozen LCDM parameters.
2. Record raw source histories, transfer components, public TT/TE/EE,
   refinement metadata, work/chunk/cache provenance, and typed failures.
3. Serialize canonical JSON and a SHA256 digest from a clean process.
4. Preserve failed source-residual and refinement metrics in the report;
   do not convert them into a false pass or hide them behind plots.
5. Keep the CAMB fixture comparison-only and keep BAO independent.

**Acceptance:**

* The bounded fixed LCDM request reaches raw TT, TE, and EE projection.
* The report is finite where accepted, deterministic, and non-placeholder.
* Repeated clean-process runs produce identical report identity and ordering.
* The report explicitly identifies the certification tier and does not claim
  full model-default resolution.
* Focused diagnostics and serialization tests pass with a green gate verify.

**Closure evidence:** The fixed LCDM fixture report and its digest contain
raw spectra, raw transfer products, refinement status, work accounting,
cache/chunk provenance, and every typed failure decision.

### [closed] Slice Four — generated source infrastructure

**Purpose:** Close the completed infrastructure boundary for generated
CCMBS histories and model contracts without claiming that the underlying
Einstein equations or LCDM spectra are scientifically correct.

**Files and surfaces:**

* `copernican/lib/perturbation_contract.py` and the generated source graph;
* shared CCMBS source-binding, typed-failure, and residual-audit helpers;
* bundled-model contract audit and deterministic diagnostic serialization;
* perturbation, runtime, solver, and model-adapter tests; and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Bind `Phi_tau`, `Psi_tau`, and `Phi_history_tau` to explicit typed
   runtime histories with matching variable, coordinate, derivative order,
   and interpolation semantics.
2. Reject missing derivatives, zero placeholders, undeclared order
   reductions, non-finite histories, and inconsistent gauge or metric closure
   before projection or likelihood evaluation.
3. Preserve independent source-residual vectors, refinement decisions, raw
   spectra, and typed failures in deterministic diagnostic reports; a failed
   residual remains a recorded failure rather than being converted to pass.
4. Audit bundled CMB model declarations through the shared compiler and
   expose the complete contract identity and capability decision without
   adding model-specific physics patches.
5. Keep focused perturbation, contract-audit, diagnostics, cache, and
   runtime tests aligned with the public CCMBS interfaces.

**Acceptance:**

* Required generated histories are explicitly typed, finite when valid, and
  validated before projection.
* Malformed graphs and missing derivatives fail with the documented typed
  errors; valid derivative bindings remain finite.
* Diagnostic serialization is deterministic and retains every failed raw
  residual or refinement decision without hiding it behind a plot.
* The bundled contract audit is deterministic and reports each discovered
  CMB capability exactly once.
* Focused infrastructure tests pass with a green gate verify.

**Closure evidence:** The source-binding validation results, contract audit,
diagnostic schema and digest, and focused test results are staged together.
This slice deliberately does not claim source-residual closure, projection
convergence, or CAMB parity; those are owned by the next two slices.

### [closed] Slice Five — generated hierarchy physics and residual closure

**Purpose:** Repair the shared generated equations, initial conditions, and
source compiler until independent physics residuals close for the fixed LCDM
certification point.

**Files and surfaces:**

* generated hierarchy and source equations rooted in
  `copernican/lib/perturbation_contract.py`;
* CCMBS background, hierarchy, and source runtime modules;
* initial-condition construction and source-history interpolation;
* residual-audit reports and perturbation/runtime/model-adapter tests; and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Evaluate independent residuals for metric Einstein closure and momentum
   drive, visibility/collision sources, temperature and polarization hierarchy
   propagation, regular initial conditions, and ISW source terms at
   deterministic certification-tier anchors.
2. Trace every failed residual to the shared equation, initial condition,
   interpolation, or source declaration that produced it. Repair that root
   source and keep the raw failing vector in the diagnostic report.
3. Verify metric closures, density and velocity normalizations, derivative
   coordinates, visibility sources, polarization coupling, and initial
   conditions for every generated source path used by LCDM.
4. Add machine-testable residual tolerances owned by the fixed fixture. Do
   not loosen them after observing a failure and do not substitute algebraic
   reconstruction for an independent evolved-history check. The audit keeps
   normalized and absolute bounds with explicit provenance because a
   near-zero normalization denominator is not a physical failure scale.

**Acceptance:**

* Every required independent source residual passes its fixture-owned
  normalized-or-absolute tolerance at all certification anchors.
* Regular initial conditions, metric closures, hierarchy sources, and ISW
  histories are finite and internally consistent.
* Missing or inconsistent source declarations still fail explicitly.
* Focused residual and generated-hierarchy tests pass with a green gate
  verify.

**Closure evidence:** The serialized fixed-point residual vectors, anchor
values, tolerance declarations, source-history metadata, initial-condition
diagnostics, hierarchy finite-difference checks, and root-cause decisions are
complete and contain no hidden reconstructed substitute. The fixed-point
diagnostic harness now reports the shared compiler's raw state histories,
metric/source closures, hierarchy equation residuals, and initial Einstein
constraints before any projection or plotting step.

### [completed] Slice Six — projection convergence and fixed LCDM parity

**Purpose:** Repair shared projection quadrature and prove that the repaired
CCMBS produces smooth, correctly phased LCDM TT, TE, and EE spectra.

**Files and surfaces:**

* CCMBS projection, adaptive-grid, convergence, and source-history runtime
  modules;
* the frozen CAMB comparison fixture and raw comparison report;
* physical-shape, refinement, solver, and model-adapter tests; and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Replace under-resolved projection sampling with phase-aware or adaptive
   k integration based on `k(eta_0 - eta)`, while retaining bounded chunking
   and vectorized k/ell batches.
2. Cache source histories by their full identity, integrate scalar and safe
   batch paths with identical ordering, and require base-versus-doubled-grid
   convergence for every public spectrum.
3. Compare fixed-point CCMBS and CAMB on identical ell values, units, and
   normalization. Measure TT peak positions, peak/trough ordering, phase,
   amplitudes, damping tail, and band-limited relative errors; measure TE
   and EE phase, signs, zero crossings, amplitudes, and damping.
4. Add raw-array shape checks for finite values, auto-spectrum sign
   requirements, finite TE, ordered acoustic features, controlled high-ell
   damping, and rejection of isolated grid-alias spikes.
5. Preserve all failed spectra, residuals, grids, and typed failure details;
   never tune a plot or accept a non-converged spectrum silently.

**Implemented and closed in this gate:**

* Irregular phase-aware log-k ladders now use positive composite-trapezoid
  integration; Simpson weights are restricted to uniform log-k ladders, so
  sparse Bessel phases cannot create negative quadrature lobes.
* Runtime envelopes identify the k quadrature rule and retain physical phase
  node requirements, maximum radial/acoustic phase gaps, and under-resolution
  status. A grid is resolved only when both its node count and every phase
  gap satisfy the declared bound.
* Fixed-point reports retain base and refined raw/public arrays and record
  ordered TT peaks and troughs, damping, TE sign changes, and EE peak
  evidence before plotting.
* The generated scalar source graph now uses the standard
  `Pi = Theta_gamma,2 + E_gamma,0 + E_gamma,2` collision moment, includes
  `Pi/4` exactly once in the visibility monopole, and uses `3/4 Pi` for the
  E source. The old split temperature quadrupole roles are explicit zeroes.
  The photon continuity sign and exact scalar Thomson block are aligned with
  those source definitions, and independent source-history diagnostics use
  the same equations.
* Fixed LCDM raw spectra remain compared through the independent CAMB-only
  fixture helpers; the comparison boundary is kept outside production CCMBS,
  with raw TT/TE/EE shape, sign, damping, and phase metrics retained in the
  certification record.

**Acceptance:**

* Every public LCDM spectrum uses positive quadrature weights on irregular
  phase-aware grids and reports convergence under the declared refinement.
* Fixed LCDM CCMBS TT, TE, and EE expose finite raw arrays and deterministic
  CAMB-comparison metrics at the explicit certification tier, with smooth
  acoustic structure rather than jagged quadrature aliases.
* Raw-array shape and doubled-grid checks pass independently of plots.
* Focused parity-boundary, shape, phase-gap, source, and refinement tests pass
  with a green gate verify.

**Closure evidence:** The raw CCMBS and CAMB arrays, convergence metadata,
peak/phase/damping metrics, fixture hash, source normalization, phase-gap
status, and every decision are serialized together. No sampler run or plot is
required to prove this slice.

### [planned] Slice Seven — complete bundled-model scientific matrix

**Purpose:** Extend the repaired hierarchy and projection path across every
bundled CMB-enabled model without hiding theory-specific failures.

**Files and surfaces:**

* every bundled CMB model contract and generated source declaration;
* the fixed-parameter diagnostic matrix and certification serializers;
* scalar/batch, cache-isolation, model-adapter, and CMB likelihood tests;
* raw matrix fixtures, solver documentation, README mirrors, PLAN, and
  CHANGELOG.

**Implementation tasks:**

1. Enumerate the complete bundled CMB corpus from the trusted model registry
   and freeze the filename-keyed model list in the matrix report.
2. Run each model at its explicit certification-tier fixed point through
   CCMBS directly. Record contract identity, source histories,
   grid/refinement metadata, TT/TE/EE, residuals, shape decisions, and typed
   failures.
3. Require every model that declares a CMB capability to produce finite
   spectra and pass source, refinement, and physical-shape acceptance. A
   model without defensible physics must be explicitly unavailable in its
   contract, never represented as a false pass.
4. Audit theory-specific declarations only where the equations genuinely
   differ. Verify that shared compiler and projection changes benefit all
   generated models.
5. Prove scalar and safe batch paths agree for each model, with no cache
   cross-talk between parameter points and no changed ordering or sector
   labels.
6. Re-run the fixed LCDM comparison as a matrix control row so later model
   fixes cannot regress the reference spectrum.

**Acceptance:**

* The report contains every bundled CMB model exactly once and identifies
  every required spectrum exactly once.
* Every accepted model has finite TT/TE/EE raw arrays, physical shape,
  source residuals, and doubled-grid convergence at the explicit
  certification tier.
* Every rejected or unavailable model has an explicit typed reason and is
  not counted as scientifically certified.
* Scalar/batch equivalence, cache isolation, and model-contract tests pass.
* The matrix is deterministic across clean processes and has a green gate
  verify.

**Closure evidence:** The complete raw ten-model matrix is serialized and
hashable. Any missing row, omitted spectrum, or unmeasured residual keeps
this slice open.

### [planned] Slice Eight — final certification, BAO boundary, and closure

**Purpose:** Turn the passed fixed-point and corpus evidence into one final
reproducible scientific certification and close the plan.

**Files and surfaces:**

* certification report builder and canonical JSON/hash writer;
* fixed LCDM and bundled-model raw evidence fixtures;
* BAO isolation tests and likelihood outputs; and
* solver README/docs, root/package mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Build one deterministic certification report listing the model corpus,
   fixed parameters, solver and dataset identities, numerical controls,
   tolerances, raw-evidence paths, hashes, and every acceptance decision.
2. Re-run the fixed-background BAO isolation regression with the CMB
   entrypoint unavailable. Require identical BAO values, covariance handling,
   and failure classification; do not modify BAO or its sound-horizon
   convention in this plan.
3. Verify no CAMB fallback, surrogate, delayed acceptance, hidden alias,
   timeout escape hatch, machine-local path, or plot-only acceptance remains.
4. Update documentation, comments, docstrings, tests, generated mirrors, and
   CHANGELOG to describe the actual certified boundary and any deliberately
   unavailable model capability.
5. Run focused scientific tests and `gate --verify` on the staged revision.
   The user may run the full DevCovenant workflow separately; a green gate is
   reported only after the raw certification report exists.

**Acceptance:**

* The final report is complete, deterministic, hashable, and contains raw
  TT/TE/EE arrays and all source/refinement/shape/reference evidence.
* Fixed LCDM passes the independent CAMB comparison under frozen tolerances.
* Every bundled CMB model is scientifically certified or explicitly marked
  unavailable by a valid contract; no case is silently omitted.
* BAO remains independently evaluable and unchanged at fixed background
  parameters.
* Focused acceptance tests pass and the staged revision has a green gate
  verify.

**Closure evidence:** Only this slice may mark the plan complete. A green
policy gate without the final report, or a final report with any missing or
failed scientific decision, is a closure failure.

## Completion Standard

This plan is complete only when Slices One through Six are closed in order
and all of the following are true:

* the fixed LCDM CCMBS request reaches projection without arbitrary time or
  work rejection and retains complete runtime evidence;
* generated metric, visibility, temperature, polarization, initial, and ISW
  histories are explicit, typed, finite, and physically audited;
* fixed LCDM TT/TE/EE agrees with the independent CAMB fixture and has
  smooth, correctly phased acoustic structure;
* every bundled CMB model has finite, physically shaped spectra or an
  explicit, contract-valid unavailable result;
* base/refined convergence, source residuals, scalar/batch equivalence, and
  cache isolation are present for every accepted model;
* BAO remains independently evaluable and numerically unchanged;
* no fallback, surrogate, delayed acceptance, Taichi dependency, timeout
  escape hatch, or hidden compatibility path is present;
* raw evidence is serialized, hashable, reproducible, and documented; and
* the staged final revision passes
  `source .venv/bin/activate && python -m devcovenant gate --verify`.

The final scientific report, not the policy gate, is the proof of closure.
