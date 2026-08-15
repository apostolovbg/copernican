# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-08-15
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

Use this plan to make the manifest-driven native CMB workflow correct,
observable, performant, capability-driven, and scientifically complete.
The roadmap begins with the production run boundary, repairs both scalar
constraint surfaces, establishes a practical MCMC execution budget, and then
extends the declared theory contract and model corpus.

This roadmap has twelve slices: eleven implementation slices and one final
acceptance slice. Every slice has a bounded purpose, explicit dependencies,
targeted tests, and closure criteria suitable for one continuous work session.
No slice may hide unfinished work behind a green policy gate or defer required
correctness to an unnamed follow-up.

The target condition is final:

* CLI and GUI launches execute one shared manifest workflow.
* One worker owns the canonical run log and output directory.
* Dataset discovery, loading, validation, and hashing occur once per run.
* Native likelihood failures have typed, scientifically meaningful outcomes.
* Generated initial data and evolved scalar histories satisfy documented
  Einstein constraints across the accepted parameter domain.
* Native CMB evaluations are fast enough for ensemble sampling.
* Solver behavior is selected by declared mathematical capabilities rather
  than theory names or borrowed model-family assumptions.
* Every CMB-valid model executes through the native path without a production
  CAMB or CLASS fallback.
* `model_usmf2.yml` becomes CMB-valid only through a sourced, explicit, and
  independently tested perturbation closure.
* The exact reference GUI and CLI run completes within 1800 seconds.

## Table of Contents

* [Overview](#overview)
* [Problem Preamble](#problem-preamble)
* [Reference Workload](#reference-workload)
* [Scientific and Runtime Baseline](#scientific-and-runtime-baseline)
* [Acceptance Invariants](#acceptance-invariants)
* [Execution Rules](#execution-rules)
* [Execution Slices](#execution-slices)
* [Completion Standard](#completion-standard)

## Overview

Slices One through Three establish one trustworthy run boundary, typed native
failure semantics, complete performance evidence, and process-local structural
reuse. Slices Four and Five repair generated and evolved scalar constraints.
Slice Six brings successful native spectra inside the ensemble-sampling
budget. Slices Seven and Eight audit and complete the declared capability
surface. Slices Nine and Ten specify and implement the USMF2 CMB closure.
Slice Eleven certifies the model corpus, and Slice Twelve performs exact CLI,
GUI-worker, scientific, performance, documentation, and policy acceptance.

## Problem Preamble

The production workflow combines GUI orchestration, a CLI worker, trusted
dataset loaders, model plugins, ensemble samplers, and the native CMB engine.
Correctness therefore depends on more than a finite spectrum from one isolated
declared-graph solver call.

The reference run exposes six independent failure classes:

* The GUI monitor and CLI worker both write representations of the same
  records, producing duplicate messages, severity changes, and conflicting
  output-directory text.
* The compound BAO dataset deliberately uses independent diagonal errors, but
  runtime text describes that declared statistical model as a fallback.
* Native CMB evaluations take long enough that a small ensemble run becomes a
  multi-hour workload.
* MCMC handling converts broad solver exceptions to `-inf` and emits an error
  for each rejected point, without separating an invalid proposal from a
  solver invariant failure.
* Generated scalar initial data can violate their normalized Einstein
  constraint before evolution.
* Evolved scalar histories can exceed a separate model-declared Einstein
  residual tolerance.

The model-extension objective is related but distinct. The perturbation
contract contains declarations for sectors, species, gauges, interactions,
collision operators, initial-condition families, projection typing, and
observables. The roadmap audits those primitives before changing the runtime,
implements only demonstrated capability gaps, and uses the resulting contract
to specify and implement the USMF2 CMB closure.

The roadmap does not treat a tolerance increase, a swallowed exception, a
test-only shortcut, or a production backend fallback as a solution.

## Reference Workload

The production acceptance fixture is one saved manifest with these exact
selections:

* Control model: `copernican/models/model_lcdm.yml`.
* Test model: `copernican/models/model_torg.yml`.
* SNe dataset: `union3_2025`.
* BAO dataset: `compound_bao_set`.
* CMB dataset: `planck_2018_lite`.
* Sampler: `copernican.engines.engine_mcmc`.
* Seed: `0`.
* Burn-in steps: `5`.
* Production steps: `10`.
* Walkers: `32`.
* Worker pool: `3`.

The workload requires roughly 960 model likelihood evaluations before any
reseed or retry work. With three workers and a hard 1800-second end-to-end
limit, a warm native CMB evaluation must remain near five seconds at the
95th percentile on the reference host. A 60-second or 180-second evaluation
budget is not a practical MCMC budget.

The runtime acceptance envelope is:

* One cold full-spectrum request completes within 180 seconds.
* One warm parameter-rebound request completes within five seconds at the
  95th percentile of the deterministic acceptance sample.
* One exact cache hit performs no evolution or projection work and completes
  within a subsecond cache-hit budget recorded by the performance test.
* The reference CLI run completes within 1800 seconds.
* The reference GUI-worker run completes within 1800 seconds.
* The complete repository test workflow completes within 1800 seconds on the
  governed reference environment.
* No individual targeted test is allowed to run for more than 180 seconds.

Timing tests must record hardware, process count, cold or warm state, cache
statistics, requested spectra, multipole range, numerical tier, and phase
timings. A wall-time assertion without that provenance is not sufficient.

## Scientific and Runtime Baseline

The native solver is the production CMB backend. CAMB remains available only
to independent test references and parity fixtures. Production modules must
not import, invoke, or silently fall back to CAMB or CLASS.

The following model manifests form the CMB regression corpus:

* `copernican/models/model_lcdm.yml`
* `copernican/models/model_lcdm_mnu.yml`
* `copernican/models/model_qauc.yml`
* `copernican/models/model_qrsf.yml`
* `copernican/models/model_ref_planck2018.yml`
* `copernican/models/model_tog.yml`
* `copernican/models/model_torg.yml`
* `copernican/models/model_w0wa.yml`
* `copernican/models/model_wcdm.yml`

Schema validity or a low-resolution smoke spectrum does not establish
production acceptance. Each CMB-valid model must satisfy its declared physics,
convergence, performance, and observable contracts.

`copernican/models/model_usmf2.yml` remains `valid_for_cmb: false` until its
background, perturbation variables, gauge relations, initial conditions,
closures, sources, observables, and numerical controls are explicit and
validated.

The scalar failure surface has two independent contracts:

* Generated initial data use normalized Einstein residuals before ODE
  evolution.
* Evolved histories use model-declared residual diagnostics across the source
  grid.

These contracts may use different normalization and acceptance bounds only
when the distinction is explicit, dimensionally sound, and supported by
convergence evidence.

The runtime exposes both `full_spectrum` and `joint_mcmc` performance budgets.
Every caller must propagate its workload identity, and enforcement must cover
successful and failed requests. Measuring only the end of a successful
full-spectrum call does not govern MCMC execution.

## Acceptance Invariants

The following invariants apply to every slice:

* The confirmed manifest is the single source of run configuration.
* GUI and CLI use the same executor, plugin loading, dataset loading, and
  sampler pipeline.
* The worker is the only canonical run-log file writer.
* GUI monitoring consumes worker events without appending duplicate worker
  records to the canonical file.
* Every selected dataset parser and hash collector executes once per run.
* A dataset with declared independent errors uses diagonal covariance by
  contract, not by exception fallback.
* Parameter-domain rejection returns deterministic `-inf` without an error
  storm.
* Contract, implementation, and numerical-invariant failures abort the run
  with typed diagnostics; they are not converted into posterior exclusions.
* Accuracy controls are not weakened to fit one observed failure.
* Static runtime assets are reused only when their complete structural
  identity matches.
* Parameter-dependent state is never reused across unequal parameter points.
* Performance work preserves the declared numerical accuracy tier and physics
  outputs.
* Capability routing uses declared primitives and fails before expensive work
  when a requested combination is unsupported.
* CAMB parity references remain test-only and independent of native results.
* Root and package README files remain synchronized.

## Execution Rules

* Execute slices strictly in order.
* Complete each slice in one continuous work session.
* Do not create hidden sub-slices, partial closure claims, or deferred cleanup.
* Reproduce each defect with a bounded targeted test before changing behavior.
* Stop a targeted test that reaches 180 seconds and repair the underlying
  runtime path before rerunning it.
* Preserve the CMB regression corpus after every solver-facing slice.
* Run focused LCDM and TORG regressions after every scalar or runtime change.
* Keep USMF2 outside CMB execution until Slice Ten closes.
* Update code, tests, docs, comments, model prose, templates, and changelog in
  the same slice when their contract changes.
* Update generated or mirrored artifacts through their owning source.
* Stage all changes after each completed slice.
* Do not commit or push unless explicitly instructed.
* A green `devcovenant gate --verify` proves repository discipline only; each
  slice must separately satisfy its implementation acceptance criteria.
* Stop at a green `devcovenant gate --verify` for the operator-owned full
  `devcovenant run` and `devcovenant gate --close` unless explicitly directed
  otherwise.

Task markers mean:

* `[open]` identifies active roadmap work.
* `[closed]` identifies work completed in substance and validation.

## Execution Slices

### [closed] Slice One - Canonical run logging and dataset contracts

Purpose:

Make a manifest run produce one authoritative record stream, one output
location, and one dataset-ingestion pass before changing solver behavior.

Depends on:

* The reference manifest and managed environment.

Probable affected files:

* `copernican/lib/logger.py`
* `copernican/lib/console_output.py`
* `copernican/lib/gui/app.py`
* `copernican/lib/gui/run_worker.py`
* `copernican/lib/run_executor.py`
* `copernican/lib/dataset_registry.py`
* `copernican/datasets/bao/compound/cosmo_parser_compound.py`
* `copernican/datasets/bao/compound/metadata_compound.yml`
* `tests/copernican/lib/test_logger.py`
* `tests/copernican/lib/gui/test_app.py`
* `tests/copernican/lib/test_run_executor.py`
* `tests/copernican/lib/test_dataset_registry.py`
* `tests/copernican/datasets/bao/compound/test_cosmo_parser_compound.py`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Assign canonical run-log ownership to the worker process.
* Keep the GUI application log, in-memory monitor, progress channel, and
  canonical run log as distinct destinations.
* Pass one resolved run directory, timestamp, and log identity to the worker.
* Prevent stdout forwarding from writing a second copy into the worker file.
* Preserve worker severity when the GUI displays an event.
* Remove path rewriting that turns the selected output directory into `.`.
* Ensure selected datasets are loaded and hashed once.
* Represent compound BAO errors as declared diagonal covariance.

Tasks:

* Define a structured worker event or parseable transport record for GUI
  monitoring.
* Remove duplicate `print`, stream-proxy, logger, and monitor capture paths.
* Add call-count tests for parser loading and file hashing.
* Add log-content tests that reject duplicate records and severity changes.
* Add output-path tests for absolute external run directories.
* Replace the compound BAO fallback warning with one informational statement
  describing the declared diagonal likelihood.
* Keep CLI-only logging complete without a GUI monitor.
* Update run and dataset documentation.

Done when:

* One canonical run log exists in the selected run directory.
* Every logical event appears once with its original severity.
* The log contains one output directory and never reports `.` for an external
  output directory.
* Each selected dataset loads and hashes exactly once.
* Compound BAO uses diagonal errors without a fallback warning.
* CLI and GUI tests prove the same manifest reaches the same executor.

### [open] Slice Two - Native failure taxonomy and performance evidence

Purpose:

Create a typed likelihood boundary and complete timing evidence so expected
proposal rejection, model incompatibility, and solver failure cannot be
confused.

Depends on:

* Slice One.

Probable affected files:

* `copernican/lib/likelihoods/cmb/cmb.py`
* `copernican/lib/likelihoods/cmb/native_performance.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/engines/engine_mcmc.py`
* `copernican/lib/run_pipeline.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_performance.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_cache.py`
* `tests/copernican/engines/test_engine_mcmc.py`
* `tests/copernican/lib/test_run_pipeline.py`
* `copernican/docs/cmb_solver.md`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Define typed errors for parameter-domain rejection, unsupported capability,
  contract invalidity, convergence failure, non-finite evolution, constraint
  violation, and performance-budget violation.
* Return `-inf` only for scientifically valid parameter-domain rejection.
* Abort on contract, implementation, or numerical-invariant failures.
* Rate-limit or aggregate expected proposal diagnostics.
* Propagate `full_spectrum` or `joint_mcmc` workload identity from the caller.
* Record phase timing and work-unit accounting for success and failure.
* Capture cache state and the point at which a failed request stopped.

Tasks:

* Replace broad exception conversion in `CMBLike.loglike` with typed handling.
* Add an initial-point preflight before walker creation.
* Record compilation, background, initial-data, evolution, projection,
  lensing, and likelihood-assembly timings through `finally`-safe accounting.
* Make the runtime envelope identify cold, warm, and exact-cache-hit requests.
* Add deterministic tests for every error category and MCMC response.
* Add a bounded reproducer for both reference scalar failures.
* Record parameter values, k mode, eta location when available, gauge,
  numerical tier, requested spectra, and tolerance provenance.
* Update failure and performance documentation.

Done when:

* Expected out-of-domain proposals return `-inf` without repeated errors.
* Solver invariant failures stop the run instead of changing the posterior.
* The initial model point is checked before multiprocessing begins.
* Both scalar failure surfaces have stable, distinct diagnostics.
* Failed requests retain complete phase and work-unit timing.
* The declared `joint_mcmc` budget is exercised by production likelihood code.

### [open] Slice Three - Runtime lifecycle and structural reuse

Purpose:

Ensure every worker builds structural solver assets once and performs only
parameter-dependent work for each MCMC proposal.

Depends on:

* Slice Two.

Probable affected files:

* `copernican/lib/model_coder.py`
* `copernican/lib/engine_adapter.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/engines/engine_mcmc.py`
* `tests/copernican/lib/test_model_coder.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_cache.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/engines/test_engine_mcmc.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Separate structural compilation from scalar parameter binding.
* Initialize process-local immutable runtime assets once per model and worker.
* Reuse expression plans, topology, hierarchy metadata, index maps, quadrature
  topology, and parameter-independent projection geometry.
* Recompute every parameter-dependent background, source, and spectrum value.
* Preserve complete cache identity across model, gauge, sector, observable,
  numerical tier, requested multipoles, and parameter inputs.
* Prevent multiprocessing spawn from rebuilding static assets per proposal.

Tasks:

* Inventory every cache family as structural, parameter-dependent, or result.
* Add worker initialization for control and test model runtime bundles.
* Remove contract recompilation and graph materialization from proposal loops.
* Make cache ownership explicit and bounded.
* Add miss, hit, eviction, process-isolation, and parameter-invalidation tests.
* Add work-count tests that fail if structural compilation repeats.
* Compare cold and warm spectra bit-for-bit at identical numerical controls.
* Update runtime lifecycle documentation.

Done when:

* Each worker compiles each model structure once.
* Warm parameter changes perform no structural recompilation.
* Exact cache hits perform no evolution or projection work.
* Unequal parameters cannot receive stale backgrounds or spectra.
* LCDM and TORG retain finite, responsive native spectra.
* Structural reuse produces a measured warm-request speedup.

### [open] Slice Four - Generated scalar initial-condition constraints

Purpose:

Repair the regular scalar mode generator so every requested k mode begins on
the declared Einstein constraint surface before expensive evolution.

Depends on:

* Slice Three.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/models/model_lcdm.yml`
* `copernican/models/model_torg.yml`
* `tests/copernican/lib/test_perturbation_contract.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Reproduce the normalized energy residual at the high-k reference mode.
* Audit requested k-grid construction against declared numerical limits.
* Derive generated scalar seeds from the complete regular-mode constraints.
* Preserve gauge-specific relations without hidden Newtonian variables.
* Validate all initial contexts before evolving the first mode.
* Keep the normalized initial-condition tolerance fixed unless an independent
  convergence derivation changes its definition.

Tasks:

* Record every term and normalization scale in each initial residual.
* Test superhorizon and high-k seed behavior separately.
* Solve coupled initial algebraic constraints rather than patching one state.
* Verify adiabatic and supported isocurvature families independently.
* Add full-k-grid preflight with deterministic failure ordering.
* Compare Newtonian, synchronous, and gauge-invariant routes where supported.
* Prove seed changes alter physical mode content rather than only diagnostics.
* Update initial-condition documentation.

Done when:

* The reference `0.012948... > 0.01` failure is eliminated at its source.
* Every generated scalar mode satisfies its normalized initial constraints.
* Invalid declarations fail before any ODE solve.
* No tolerance-only patch or skipped high-k mode is present.
* LCDM and TORG initial-condition regressions pass within 180 seconds each.

### [open] Slice Five - Evolved scalar constraint convergence

Purpose:

Repair or rebaseline evolved Einstein residuals through convergence evidence,
with one documented normalization and provenance contract per residual.

Depends on:

* Slice Four.

Probable affected files:

* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/models/model_lcdm.yml`
* `copernican/models/model_torg.yml`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/project/lib/camb_reference.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Reproduce every reference energy-residual breach near the declared bound.
* Record the eta position and physical regime of each maximum.
* Establish whether each residual is absolute, relative, or dimensionless.
* Sweep eta resolution, k resolution, hierarchy truncation, ODE tolerance,
  tight-coupling transitions, and source-grid refinement independently.
* Separate discretization error from an inconsistent equation or closure.
* Derive the accepted bound from converged behavior without naming a desired
  replacement value in advance.

Tasks:

* Add deterministic convergence fixtures for LCDM and TORG.
* Compare coarse, intermediate, and reference tiers using the same cosmology.
* Test representative interior and boundary points from the accepted priors.
* Correct equations, transition matching, interpolation, or normalization
  wherever residuals fail to converge.
* Store tolerance source, normalization source, maximum location, and
  refinement evidence in the runtime envelope.
* Keep independent CAMB comparisons test-only and secondary to constraint
  closure.
* Update scalar constraint documentation.

Done when:

* The reference `0.003`-class failures do not occur for valid converged points.
* Every enforced residual has a dimensionally coherent definition.
* Under-resolved requests are identified before being judged as physical
  failures.
* The accepted tolerance follows measured convergence rather than the log.
* LCDM and TORG meet their scalar and observable acceptance thresholds.

### [open] Slice Six - Native numerical throughput

Purpose:

Bring successful parameter-rebound spectra and the reference ensemble run
inside the practical MCMC budget without lowering scientific accuracy.

Depends on:

* Slice Five.

Probable affected files:

* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/native_lensing.py`
* `copernican/lib/likelihoods/cmb/native_performance.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* `copernican/engines/engine_mcmc.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/lib/test_engine_adapter.py`
* `tests/copernican/engines/test_engine_mcmc.py`
* `copernican/docs/cmb_solver.md`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Use Slice Two phase evidence to optimize the dominant successful hot path.
* Batch or vectorize independent k-mode and projection work where numerically
  equivalent.
* Remove repeated allocations, interpolation setup, decomposition, and
  transform construction from inner loops.
* Reuse bounded work products across spectra requested from the same evolution.
* Keep adaptive refinement responsive to declared error controls.
* Govern cold, warm, cache-hit, and full ensemble workloads separately.

Tasks:

* Profile the exact Planck Lite multipole and spectrum request.
* Optimize phases in measured descending cost order.
* Add numerical equivalence tests before replacing each kernel path.
* Verify worker-pool scaling and prevent oversubscription.
* Make performance-budget enforcement occur at the correct workload boundary.
* Add a deterministic warm-parameter sample and report median and p95 time.
* Run the exact reference CLI fixture under the 1800-second limit.
* Update performance and operator documentation.

Done when:

* Cold full-spectrum, warm p95, and cache-hit budgets all pass.
* The reference CLI run completes within 1800 seconds.
* No targeted test exceeds 180 seconds.
* Optimized spectra satisfy the same convergence and parity thresholds.
* Performance tests fail on repeated static work or material regressions.

### [open] Slice Seven - Capability audit and compatibility specification

Purpose:

Define the exact expressible theory surface from the implemented contract and
identify concrete capability gaps without speculative refactoring.

Depends on:

* Slice Six.

Probable affected files:

* `copernican/lib/cmb_contract.py`
* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_spec_validator.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/engine_adapter.py`
* `copernican/docs/cmb_solver.md`
* `copernican/docs/model_template.yml`
* `docs/model_template.yml`
* `tests/copernican/lib/test_cmb_contract.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `tests/copernican/lib/test_model_spec_validator.py`
* `tests/copernican/lib/test_model_coder.py`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Inventory background, sector, species, gauge, hierarchy, interaction,
  collision, closure, initial-mode, projection, lensing, and observable
  primitives.
* Identify every theory-name, filename, model-family, and assumed-species
  branch in production CMB execution.
* Distinguish legitimate generated standard hierarchies from hidden routing.
* Define capability completeness for each requested observable.
* Define unsupported combinations and their early failure messages.
* Produce a model-by-capability matrix for the full corpus.

Tasks:

* Trace each manifest field from validation through compilation and execution.
* Add tests for declared capabilities that exist but are ignored at runtime.
* Add tests for runtime assumptions not represented in the schema.
* Specify the minimum capability set for TT, TE, EE, BB, PP, TP, and EP.
* Specify gauge and sector compatibility rules.
* Record only demonstrated implementation gaps for Slice Eight.
* Synchronize both model templates and solver documentation.

Done when:

* Every production routing decision maps to declared data or a documented
  universal numerical rule.
* The expressible theory surface and unsupported combinations are explicit.
* The model corpus has a machine-testable capability matrix.
* Slice Eight contains no inferred or open-ended refactor work.

### [open] Slice Eight - Implement proven capability gaps

Purpose:

Implement the finite gap set from Slice Seven so any capability-complete
contract expressible by the documented primitives follows one native route.

Depends on:

* Slice Seven.

Probable affected files:

* `copernican/lib/cmb_contract.py`
* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_spec_validator.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/engine_adapter.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `tests/copernican/lib/test_cmb_contract.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Add only capabilities justified by the Slice Seven audit.
* Remove name-based routing when equivalent contract data exists.
* Bind background, evolution, source, projection, and observable assembly from
  compiled capability data.
* Preserve generated hierarchy helpers as explicit opt-in materializers.
* Reject unsupported combinations before background or mode evolution.
* Keep native production execution independent of test reference engines.

Tasks:

* Implement each recorded schema, compiler, and runtime delta.
* Add renamed-variable tests so semantic roles do not depend on identifiers.
* Add representative scalar, vector, tensor, collisionless, and interacting
  analytic limits where declared.
* Add unsupported-capability tests with exact actionable diagnostics.
* Re-run LCDM and TORG scientific and performance acceptance.
* Remove superseded routing rather than preserving compatibility bridges.
* Update model author documentation and templates.

Done when:

* Capability-complete contracts execute without theory-name routing.
* Semantic role renaming preserves results.
* Unsupported combinations fail before expensive numerical work.
* The CMB regression corpus retains scientific and performance acceptance.
* No production backend fallback or compatibility bridge remains.

### [open] Slice Nine - Specify the USMF2 CMB closure

Purpose:

Create a scientifically sourced, internally complete USMF2 perturbation
specification before enabling production CMB execution.

Depends on:

* Slice Eight.

Probable affected files:

* `copernican/models/model_usmf2.yml`
* `copernican/docs/cmb_solver.md`
* `copernican/docs/model_template.yml`
* `docs/model_template.yml`
* `tests/copernican/lib/test_model_spec_validator.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Identify authoritative equations and conventions for the USMF2 background
  and perturbations.
* Declare independent variables, dynamical variables, algebraic relations,
  gauge roles, species, interactions, and closures.
* Declare regular initial-condition families and their normalization.
* Declare source terms, projection typing, primordial inputs, and observables.
* Declare conservation identities, limiting cases, and numerical controls.
* Keep `valid_for_cmb: false` throughout this specification slice.

Tasks:

* Map every theory equation to one contract node with units and provenance.
* Resolve gauge freedom and constraint closure explicitly.
* Define analytic limits that can be tested without another production solver.
* Identify all required capability primitives from Slice Eight.
* Reject any missing physical relation instead of filling it with LCDM math.
* Add schema and dependency-graph tests for the proposed closure.
* Document the theory-facing model-author contract.

Done when:

* The USMF2 closure is complete on paper and in declarative contract shape.
* Every evolved degree of freedom has an equation or explicit closure.
* Initial conditions and observables have sourced definitions.
* Analytic identities and limiting cases are testable.
* No borrowed LCDM species, alias, equation, or source is unexplained.

### [open] Slice Ten - Implement and validate the USMF2 CMB path

Purpose:

Encode the Slice Nine specification and promote USMF2 only after its native
physics, convergence, observables, and performance pass.

Depends on:

* Slice Nine.

Probable affected files:

* `copernican/models/model_usmf2.yml`
* `copernican/lib/cmb_contract.py`
* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_spec_validator.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `tests/copernican/lib/test_model_spec_validator.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Implement the complete USMF2 background and perturbation graph.
* Use only declared capability primitives or explicit Slice Eight extensions.
* Validate constraints, conservation rules, initial modes, source histories,
  observables, and parameter response.
* Establish numerical convergence and practical runtime controls.
* Flip `valid_for_cmb` only after every acceptance test passes.
* Preserve USMF2 mathematics without a model-family fallback.

Tasks:

* Encode variables, equations, closures, initial conditions, and observables.
* Add finite and responsive spectrum tests across representative parameters.
* Add analytic-limit and conservation tests from Slice Nine.
* Add coarse-to-reference convergence tests.
* Add negative tests for incomplete or contradictory USMF2 declarations.
* Verify native execution contains no CAMB or LCDM route substitution.
* Add USMF2 to the bounded runtime and cache-identity tests.
* Update user and solver documentation.

Done when:

* USMF2 compiles and executes through the native declared graph.
* Its spectra are finite, structured, parameter-responsive, and convergent.
* Its theory-specific identities and constraint bounds pass.
* Its warm execution meets the governed model budget.
* `valid_for_cmb: true` is justified by the complete acceptance surface.

### [open] Slice Eleven - Migrate and certify the model corpus

Purpose:

Bring every model manifest and template into the final capability contract and
publish one explicit compatibility state for the entire corpus.

Depends on:

* Slice Ten.

Probable affected files:

* `copernican/models/*.yml`
* `copernican/docs/model_template.yml`
* `docs/model_template.yml`
* `copernican/lib/model_spec_validator.py`
* `copernican/lib/model_coder.py`
* `tests/copernican/lib/test_model_spec_validator.py`
* `tests/copernican/lib/test_model_coder.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Audit every model against the final capability and observable requirements.
* Remove unexplained LCDM descriptions, species, aliases, and equations from
  theories whose physics does not contain them.
* Preserve theory-neutral declarations in LCDM where no model-specific wording
  is required.
* Keep non-CMB models explicitly excluded with precise capability reasons.
* Validate all intended CMB models through the native path.
* Keep the two model-template files synchronized through their owning source.

Tasks:

* Generate and review the complete model compatibility matrix.
* Migrate manifests that require explicit capability declarations.
* Add corpus tests for validation, compilation, native smoke execution,
  parameter response, and unsupported-state reporting.
* Re-run scalar, tensor, vector, neutrino, gauge, lensing, and observable
  regressions for applicable models.
* Verify model prose describes each theory accurately.
* Verify CAMB appears only in independent test-reference surfaces.
* Update package-facing and repository-facing documentation separately.

Done when:

* Every model is CMB-valid or explicitly excluded for a documented reason.
* Every CMB-valid model executes only through the native solver.
* No manifest silently borrows another theory's physics.
* TORG and USMF2 satisfy their own declared closures.
* Templates, docs, comments, tests, and manifests agree on the final contract.

### [open] Slice Twelve - End-to-end acceptance and closure

Purpose:

Prove the complete product workflow, scientific matrix, performance envelope,
documentation, and repository discipline on one final staged revision.

Depends on:

* Slices One through Eleven.

Probable affected files:

* `PLAN.md`
* `CHANGELOG.md`
* Any source, test, model, template, or documentation file requiring a final
  substantive correction.

Scope:

* Run bounded targeted tests for every slice acceptance contract.
* Run the exact reference manifest through CLI and GUI-worker paths.
* Verify one canonical log, one output directory, and one dataset-ingestion
  pass in each path.
* Verify both control and test chains execute and produce comparison outputs.
* Verify scientific constraints, parity metrics, capability failures, and
  corpus status.
* Verify all runtime budgets and the complete 1800-second workflow limit.
* Audit code, docs, comments, docstrings, tests, configuration, managed assets,
  mirrors, consistency, performance, and architecture.

Tasks:

* Run the complete targeted unit, integration, scientific, and performance
  matrix without an individual test exceeding 180 seconds.
* Run the reference CLI workflow and inspect its manifest, log, results, and
  timing artifacts.
* Run the reference GUI-worker workflow and compare its effective manifest and
  results with CLI.
* Confirm expected proposal rejection is summarized and solver failures are
  absent.
* Confirm no production CAMB or CLASS import or fallback exists.
* Confirm root and package README synchronization and model-template sync.
* Mark all slices closed only after implementation acceptance is complete.
* Run `devcovenant gate --verify` until green.
* Stage the complete final state for the operator-owned workflow run and gate
  close.

Done when:

* The reference CLI and GUI-worker runs both complete within 1800 seconds.
* Their selected models, datasets, numerical controls, outputs, and results
  agree.
* Logs contain no duplicates, severity changes, conflicting paths, constraint
  failures, or unexpected fallbacks.
* Every CMB-valid model passes the final native corpus matrix.
* USMF2 is enabled only with its complete theory-specific closure.
* The complete repository test workflow passes within 1800 seconds.
* `devcovenant gate --verify` is green on the staged final state.

## Completion Standard

The roadmap is complete only when all twelve slices are closed in order and
the final revision satisfies the same manifest, physics, performance, logging,
dataset, model-corpus, documentation, and policy contracts.

A green policy gate, a finite smoke spectrum, or an isolated parity fixture is
not sufficient by itself. Completion requires the exact production comparison
workflow to finish both model chains through the native declared-graph solver
within the governed runtime envelope and without hidden fallback behavior.
