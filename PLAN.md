# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-08-16
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

> **For agentic workers:** Execute the slices in order and use the
> repository gate workflow. Each slice has independent implementation,
> correctness, scientific, and performance acceptance evidence.

**Goal:** Reduce native CMB MCMC runtime while preserving the exact scalar
sampler as the scientific reference and adding only validated acceleration
paths.

**Architecture:** The exact scalar native CMB path remains the default and
reference implementation. First remove observable runtime overhead and make
measurements trustworthy. Then add an ordered batch/vectorized evaluation
contract. Finally add an explicit opt-in surrogate/delayed-acceptance path
whose second-stage exact evaluation preserves the target distribution.

**Tech Stack:** Python 3.11, NumPy, SciPy, emcee, multiprocessing, Tk, native
declared-graph CMB execution, focused unittest suites, and DevCovenant.

## Global Constraints

* Do not change branches, create branches, or alter the repository workflow.
* Keep the native declared-graph solver as the only production CMB backend.
* Keep the exact scalar sampler as the default and the reference path.
* Do not broaden parameter-dependent cache keys across unequal parameter
  points.
* Do not lower a declared accuracy tier or physical resolution to manufacture
  a timing result.
* Do not introduce a production CAMB or CLASS fallback.
* Batch evaluation must preserve input ordering and per-item diagnostics.
* A failed batch item must not corrupt or hide the result of another item.
* Surrogate and delayed-acceptance execution must be explicit opt-in.
* A surrogate may reject a proposal cheaply only when the exact correction
  remains mathematically valid.
* Correctness and scientific validation are separate acceptance surfaces from
  performance measurement.
* The `copernican.validation` package and its full-corpus workload are not an
  acceptance dependency for these slices.
* Use bounded focused tests and scientific reference fixtures; do not impose
  an unmeetable full-run wall-clock gate on this host.
* Record hardware, process count, numerical threads, model, dataset, seed,
  cache state, requested spectra, and phase timings for every benchmark.
* Preserve root and package documentation synchronization when user-facing
  behavior changes.
* Update code, tests, comments, docstrings, documentation, and changelog in
  the same slice when their contracts change.
* Stage all changes after each completed slice.
* Do not commit or push unless explicitly instructed.

## Table of Contents

* [Overview](#overview)
* [Problem Preamble](#problem-preamble)
* [Baseline and Measurement Contract](#baseline-and-measurement-contract)
* [Scientific Safety Contract](#scientific-safety-contract)
* [Execution Rules](#execution-rules)
* [Execution Slices](#execution-slices)
* [Completion Standard](#completion-standard)

## Overview

This plan replaces the previous open-ended performance work with three
numbered slices. The slices are intentionally the smallest safe decomposition
that keeps the two algorithmic changes independently reviewable:

* Slice Thirteen establishes trustworthy measurements and applies
  behavior-preserving runtime optimizations.
* Slice Fourteen introduces batch/vectorized native evaluation and validates
  it independently against scalar execution and scientific references.
* Slice Fifteen introduces an opt-in surrogate/delayed-acceptance sampler and
  validates its mathematical and scientific behavior independently.

The slices must remain separate even though they share benchmark fixtures.
Batch evaluation changes execution structure. Delayed acceptance changes the
proposal decision process. Combining them would make numerical regressions,
posterior bias, and performance changes impossible to attribute cleanly.

The plan does not require a full production MCMC run on this development
machine. It requires bounded, reproducible workloads that measure the same
phases and compare the same scientific outputs. A host-qualified end-to-end
run may be recorded separately when suitable hardware is available.

The target condition is:

* Progress logs and GUI telemetry report completed MCMC steps, walker work,
  elapsed time, throughput, and ETA using consistent counters.
* GUI rendering and progress persistence do not materially contend with the
  native worker pool.
* Worker startup and proposal scheduling avoid avoidable discovery and
  straggler overhead.
* Scalar and batch native CMB evaluation produce equivalent spectra,
  likelihoods, diagnostics, and failure classifications.
* The batch path reuses only parameter-independent structure and preserves
  parameter-dependent isolation.
* The exact scalar sampler remains available as a switchable reference for
  every model and dataset.
* The delayed-acceptance path is exact in distribution when enabled, rejects
  uncertain surrogate predictions safely, and records every exact correction.
* Performance improvements are reported with phase-level evidence rather than
  misleading cumulative rates or unqualified wall-clock claims.

## Problem Preamble

The current live-run evidence identifies four separate costs.

### Misleading progress accounting

`copernican/engines/engine_mcmc.py` reports each MCMC iteration with the
walker count as both `processed` and `total`. `BatchProgressBar` then divides
that constant count by cumulative stage time. The displayed rate therefore
falls by construction, and the displayed ETA is zero even when the stage has
many iterations remaining.

This is an observability defect, not a solver optimization. It must be fixed
before timing comparisons are trusted.

### GUI and progress contention

The GUI polls the progress file every 0.5 seconds and schedules monitor and
validation redraws for each changed record. A second periodic refresh also
runs at the same cadence. Monitor refreshes rewrite the entire visible log
tail, causing Tk text and label rendering to compete with native CMB workers.
Progress persistence also flushes and fsyncs every update.

The GUI must remain responsive, but it must not redraw or sync more often than
the operator-visible state requires.

### Worker scheduling and startup overhead

The MCMC engine uses a spawned pool. Worker initialization repeats dataset
discovery, and the main emcee path uses ordered pool mapping with default
chunking. Red-blue proposal waves contain expensive, variable-duration CMB
evaluations, so default chunks can leave workers idle behind a straggler.

These changes are behavior-preserving and belong in the first slice.

### Cold parameter-dependent CMB work

The native runtime plan and structural graph assets are reusable per worker,
but continuous MCMC proposals rarely hit exact spectrum, transfer, or
background cache keys. Each unique proposal can rebuild recombination,
reionization calibration, perturbation evolution, and line-of-sight
projection for the requested spectra.

Batch evaluation and delayed acceptance may reduce repeated work, but neither
may reuse unequal parameter-dependent state without an explicit numerical
error contract. This is why both algorithmic paths require their own
correctness and scientific validation surfaces.

## Baseline and Measurement Contract

Every slice uses the same baseline identity and records it in its evidence.

### Exact reference path

The reference path is:

* `copernican/engines/engine_mcmc.py`;
* `copernican/lib/likelihoods/cmb/cmb.py`;
* the native declared-graph solver and its existing cache identities;
* the selected model and datasets from the confirmed manifest;
* scalar, one-contract-at-a-time likelihood evaluation;
* no surrogate, delayed acceptance, or approximate cache reuse.

The reference path remains the default when no new execution mode is declared.

### Bounded runtime fixtures

Use existing focused tests and fixtures wherever possible. The benchmark
matrix must include:

* one cold native CMB spectrum request;
* one repeated exact request;
* one changed-parameter request with the same structural contract;
* one bounded MCMC step over the current walker shape;
* one bounded walker-initialization phase;
* one headless worker run and one GUI-worker smoke run;
* one representative LCDM reference point and one non-LCDM reference point;
* all requested spectra used by the selected CMB dataset.

The current production-style controls may be used for identity, but a local
benchmark must not require the entire multi-hour chain to complete. Every
measurement must state whether it is cold, warm, exact-cache, scalar, batch,
surrogate, or delayed-acceptance execution.

### Required evidence fields

Each timing record must include:

* model manifest identity and manifest hash;
* dataset identifiers and dataset hashes;
* sampler, seed, active parameter names, walker count, and pool size;
* Python, NumPy, SciPy, emcee, and operating-system versions;
* CPU count and numerical-thread environment;
* requested spectra, observed multipoles, k-grid, eta-grid, and accuracy tier;
* cold/warm/cache state and cache hit/miss counts;
* compilation, background, evolution, source, projection, likelihood, and
  serialization phase durations when available;
* scalar or batch item count and per-item status;
* GUI enabled or headless execution mode.

No speed claim is accepted from an elapsed-time number without this context.

## Scientific Safety Contract

The exact scalar result is the comparison authority.

### Numerical equivalence

For every new execution mode, compare scalar and accelerated outputs at the
same parameter points. Reuse the solver's existing absolute and relative
tolerances; do not introduce looser tolerances solely for the new path.
Compare:

* every requested CMB spectrum and its multipole support;
* finite values, shape, ordering, and spectrum availability metadata;
* background and perturbation diagnostics;
* native failure type, phase, and parameter attribution;
* likelihood and chi-squared values;
* cache identity and provenance fields.

### Scientific validation

Correctness validation checks API behavior and numerical equivalence.
Scientific validation separately checks that the accelerated path preserves
the observables and inferences used by the project. It must include:

* representative LCDM and non-LCDM models;
* TT, TE, and EE where available, plus every additional declared observable;
* finite and parameter-responsive spectra;
* peak and trough locations at the sampled multipoles;
* residuals against the exact native scalar reference;
* chi-squared and derived distance or acoustic observables;
* short independent chains with posterior summary and convergence
  comparisons when a sampler path changes proposal decisions;
* exact-call count, acceptance, and effective-sample-size evidence for any
  performance claim.

These checks are bounded focused scientific fixtures. They are not a request
to run the long `copernican.validation` corpus as a hidden acceptance gate.

### Failure and fallback behavior

The scalar path must remain available when an accelerated mode is disabled,
unsupported, uncertain, or diagnostically invalid. A surrogate prediction
must never be silently promoted to an exact result. A batch item must carry
its own typed failure without changing the classification of neighboring
items. A delayed-acceptance stage must perform the exact second-stage test
whenever the surrogate stage accepts a candidate.

## Execution Rules

* Execute Slices Thirteen, Fourteen, and Fifteen strictly in order.
* Do not begin an algorithmic slice until the preceding slice's exact-output
  baseline and acceptance record are complete.
* Keep exact scalar execution as the default throughout the roadmap.
* Keep batch and delayed-acceptance modes behind explicit configuration until
  their separate acceptance records are complete.
* Reproduce each defect with a bounded focused test before changing behavior.
* Stop a focused test that exceeds 180 seconds and repair or narrow the
  fixture before rerunning it.
* Do not use a full validation-suite run as a substitute for the acceptance
  matrix defined here.
* Preserve all existing native CMB, scalar-constraint, capability, and model
  corpus tests relevant to the touched code.
* Update user-facing documentation for observable CLI, GUI, sampler, or
  output changes.
* Update `CHANGELOG.md` for every completed slice and record only touched
  paths in its Files block.
* Stage all changes after each completed slice.
* Run `source .venv/bin/activate && python -m devcovenant gate --verify`
  before the operator-owned workflow run.
* Stop at a green gate verification for the operator-owned `devcovenant run`
  and `gate --close` unless explicitly directed otherwise.

Task markers mean:

* `[open]` identifies active roadmap work.
* `[closed]` identifies work completed in substance and acceptance evidence.

## Execution Slices

### [open] Slice Thirteen - Baseline and safe acceleration

**Purpose:** Make performance evidence truthful and remove behavior-preserving
overhead from progress reporting, GUI rendering, worker startup, and proposal
scheduling.

**Depends on:** Existing exact scalar native CMB and MCMC paths.

**Probable affected files:**

* `copernican/engines/engine_mcmc.py`
* `copernican/lib/progress.py`
* `copernican/lib/progress_state.py`
* `copernican/lib/gui/app.py`
* `copernican/workflow.py`
* `tests/copernican/engines/test_engine_mcmc.py`
* `tests/copernican/lib/test_progress.py`
* `tests/copernican/lib/test_progress_state.py`
* `tests/copernican/lib/gui/test_app.py`
* `tests/copernican/lib/gui/test_run_worker.py`
* `tests/copernican/lib/test_workflow.py`
* `docs/cli_guide.md`
* `docs/gui_guide.md`
* `docs/gui_overview.md`
* `copernican/docs/cli_guide.md`
* `copernican/docs/gui_guide.md`
* `copernican/docs/gui_overview.md`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`
* `PLAN.md`

**Interfaces and invariants:**

* MCMC stage progress reports completed sampler iterations separately from
  walker evaluations.
* Walker initialization continues to report completed walker evaluations.
* Progress JSON, CLI text, and GUI labels derive from the same counters.
* A progress update may be coalesced for display but must not reorder or lose
  the final stage state.
* GUI refresh work is coalesced onto the Tk thread and does not rewrite an
  unchanged log tail.
* Pool mapping preserves result order and exception semantics.
* Worker startup does not rediscover immutable datasets once per proposal.

**Tasks:**

1. Add focused regression tests that distinguish MCMC iteration progress from
   walker progress, including elapsed, rate, remaining work, and ETA.
2. Change `_run_stage_with_progress` to pass iteration totals for burn-in and
   production stages while retaining walker totals for initialization.
3. Verify progress listeners and `gui_progress_*.json` expose consistent
   `step_index`, `step_total`, `walker_processed`, `walker_total`, elapsed,
   rate, and ETA fields.
4. Add tests for repeated identical progress snapshots and prove that the GUI
   does not schedule redundant redraw work for unchanged state.
5. Coalesce monitor and validation refresh callbacks while retaining the
   operator-visible stage, status, controls, and final log tail.
6. Update the monitor log widget incrementally or only when its visible tail
   changes; preserve tail locking, scrolling, and severity rendering.
7. Keep progress-file persistence atomic and verify that bounded update
   frequency does not lose the final `batch_finish` record.
8. Separate worker-pool proposal mapping from initialization mapping and
   benchmark ordered `chunksize=1` scheduling against the current mapping.
9. Keep the scheduling variant only if it preserves ordered results,
   exception propagation, seeded scalar likelihoods, and improves the
   bounded proposal benchmark.
10. Remove repeated dataset discovery from worker hot paths without changing
    dataset identity, hashes, or parser ownership.
11. Run the bounded scalar, MCMC, CLI, and GUI-worker fixtures in headless and
    rendered modes and record the required evidence fields.
12. Update CLI and GUI documentation to describe truthful progress fields and
    the distinction between initialization, burn-in, and production.

**Correctness acceptance:**

* Existing scalar spectra, likelihoods, failures, seeds, and sampler state
  remain unchanged within current tolerances.
* Progress tests show iteration counts and walker counts in their correct
  fields, with nonzero remaining work before stage completion.
* GUI tests prove no duplicate canonical log records and no lost final state.
* Worker startup still loads each selected dataset with the same hash and
  parser result.
* Ordered pool results and typed failures match the pre-change behavior.

**Performance acceptance:**

* The benchmark record includes phase timings and CPU/process evidence.
* The progress rate is a measured rate for completed work rather than
  `constant_count / cumulative_elapsed`.
* GUI rendering and progress persistence no longer dominate the sampled
  worker interval in the rendered benchmark.
* The selected pool scheduling change improves median or p95 proposal time;
  an ineffective variant is removed rather than recorded as an optimization.
* No local full-chain wall-clock claim is required.

**Done when:**

* All focused tests pass with the exact scalar native path selected.
* Headless and GUI-worker evidence are present and explain their overhead.
* The progress log no longer reports impossible `32/32 steps` updates at
  partial stage percentages.
* The measured safe optimizations are staged and the slice is marked closed.

### [open] Slice Fourteen - Batch and vectorized native evaluation

**Purpose:** Add an ordered batch evaluation contract that reuses safe static
structure and vectorizes parameter-independent numerical work without
changing the exact scalar scientific result.

**Depends on:** Slice Thirteen and its scalar baseline evidence.

**Probable affected files:**

* `copernican/lib/likelihoods/cmb/cmb.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* `copernican/engines/engine_mcmc.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/lib/likelihoods/cmb/test_copernican_cmb_solver.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_background.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_evolution.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_projection.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_cache.py`
* `tests/copernican/engines/test_engine_mcmc.py`
* `tests/copernican/lib/test_engine_adapter.py`
* `docs/cmb_solver.md`
* `copernican/docs/cmb_solver.md`
* `docs/cli_guide.md`
* `copernican/docs/cli_guide.md`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`
* `PLAN.md`

**Batch contract:**

Add a native batch entry point with the following contract:

```python
compute_cmb_spectrum_batch(
    contracts: Sequence[Mapping[str, Any]],
    ells: Iterable[int],
    *,
    background_provider: Any | None = None,
    requested_spectra: Iterable[str] | None = None,
) -> tuple[NativeCMBBatchResult, ...]
```

Each `NativeCMBBatchResult` contains the input index, either one native
spectrum result or one typed failure, the performance envelope, and cache
provenance. Results are returned in input order regardless of worker
completion order. A single-item batch must be numerically equivalent to the
existing scalar call.

**Interfaces and invariants:**

* Structural graph assets, fixed grids, and projection kernels are shared
  only when their complete structural identity matches.
* Background, perturbation, source, and transfer state that depends on a
  parameter value remains isolated per batch item unless a documented exact
  vectorized representation is used.
* A batch item may return a domain rejection, typed solver failure, or success
  without changing neighboring items.
* Cache statistics distinguish shared structural reuse from per-item result
  reuse.
* The scalar public path remains the default until this slice closes.

**Tasks:**

1. Add result and failure datatypes with stable serialization and input-index
   provenance.
2. Add a scalar-to-batch adapter so the new contract is testable before any
   vectorized kernel is introduced.
3. Add tests for one, two, and multiple-item batches, preserving order and
   exact per-item diagnostics.
4. Add mixed valid, domain-invalid, and solver-failing batch fixtures and
   verify isolation of all outcomes.
5. Identify structural graph, grid, Bessel, and projection data that can be
   shared without parameter-dependent approximation.
6. Implement vectorized operations only over those proven shared dimensions;
   keep parameter-dependent state in an explicit batch axis or per-item
   record.
7. Add cache-identity tests proving unequal cosmologies cannot retrieve one
   another's parameter-dependent background or transfer data.
8. Integrate bounded batch calls into the MCMC evaluation adapter behind an
   explicit execution setting, preserving scalar fallback.
9. Compare scalar and batch spectra at representative LCDM and non-LCDM
   points for every supported requested spectrum.
10. Compare scalar and batch likelihoods, chi-squared values, diagnostics,
    failure types, and manifest provenance.
11. Add scientific fixtures for finite response, acoustic peak locations,
    TT/TE/EE residuals, and every additional declared observable used by the
    selected models.
12. Benchmark scalar versus batch cold, warm, and repeated-structure cases
    with the evidence fields in this plan.
13. Keep the batch setting disabled by default until all acceptance sections
    pass and the exact scalar comparison record is complete.
14. Update solver, CLI, and package documentation with the batch contract,
    ordering guarantee, failure behavior, and opt-in setting.

**Correctness acceptance:**

* Batch size one matches scalar output, metadata, cache state, and failures.
* Batch sizes greater than one match independent scalar calls within the
  existing numerical tolerances and preserve input order.
* Mixed failures are isolated and retain typed diagnostics.
* Repeated batches do not leak parameter-dependent state between items.
* Serial and worker-backed batch execution agree.
* The scalar default path and all existing CMB regression tests remain green.

**Scientific acceptance:**

* Representative LCDM and non-LCDM spectra are finite and responsive.
* TT, TE, EE, and every selected additional observable agree with scalar
  native references within the existing tolerance contract.
* Peak and trough locations do not move beyond the declared comparison
  tolerance.
* Chi-squared and derived observables agree with scalar references.
* Any changed numerical phase has a recorded residual and convergence record.

**Performance acceptance:**

* The batch benchmark reports per-item and total phase timings.
* A batch speedup is claimed only when it improves measured throughput for a
  repeated-structure workload without degrading scalar-equivalent output.
* Cache and vectorization effects are reported separately from process-pool
  effects.
* No full production-chain timing gate is required on this host.

**Done when:**

* The batch contract is implemented, independently tested, scientifically
  compared, and documented.
* The opt-in batch MCMC fixture completes with equivalent scalar results and
  a measured throughput record.
* The scalar default remains available and unchanged.
* The slice is marked closed only after correctness and scientific evidence
  are both complete.

### [open] Slice Fifteen - Surrogate and delayed-acceptance sampling

**Purpose:** Add one explicit opt-in surrogate-assisted delayed-acceptance
path that reduces exact CMB calls while preserving the target distribution
through exact second-stage correction.

**Depends on:** Slices Thirteen and Fourteen, including their exact-output and
scientific reference records.

**Probable affected files:**

* `copernican/engines/engine_mcmc.py`
* `copernican/engines/surrogate.py`
* `copernican/lib/engine_capabilities.py`
* `copernican/lib/run_config.py`
* `copernican/lib/run_manifest.py`
* `copernican/lib/likelihoods/cmb/cmb.py`
* `copernican/lib/likelihoods/cmb/native_performance.py`
* `tests/copernican/engines/test_engine_mcmc.py`
* `tests/copernican/engines/test_surrogate.py`
* `tests/copernican/lib/test_engine_capabilities.py`
* `tests/copernican/lib/test_run_config.py`
* `tests/copernican/lib/test_run_manifest.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_performance.py`
* `tests/project/lib/test_core.py`
* `docs/cli_guide.md`
* `docs/cmb_solver.md`
* `docs/design_overview.md`
* `copernican/docs/cli_guide.md`
* `copernican/docs/cmb_solver.md`
* `copernican/docs/design_overview.md`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`
* `PLAN.md`

**Surrogate contract:**

The surrogate is a deterministic local interpolant over normalized active
parameters, built only from exact native evaluations. It reports a prediction,
an uncertainty or support diagnostic, and the exact-sample provenance used to
make that prediction. It must force an exact evaluation when the candidate is
outside its declared domain or lacks sufficient local support.

The default sampler remains exact. The surrogate setting must be explicit in
the confirmed manifest and must appear in run provenance, cache identity, and
the output manifest.

**Delayed-acceptance contract:**

Stage one evaluates the surrogate and applies only the declared cheap-stage
screen. A candidate that survives stage one receives an exact native CMB
evaluation. The second-stage decision uses the delayed-acceptance correction
for the same proposal density and target log probability. A surrogate value
is never written as an exact native likelihood.

Every proposal record must identify whether it was screened, exactly
corrected, rejected for insufficient support, or rejected by the exact stage.
Surrogate construction must not consume hidden random state or alter the
exact sampler's seed stream when the mode is disabled.

**Tasks:**

1. Add a surrogate result type containing prediction, uncertainty, support,
   training-sample identities, and domain status.
2. Add deterministic normalized-parameter support checks and exact fallback
   for unsupported or uncertain candidates.
3. Add an explicit delayed-acceptance configuration and manifest provenance;
   reject unknown or incomplete settings before sampling begins.
4. Implement the stage-one screen and exact stage-two correction without
   changing the scalar path when the setting is disabled.
5. Add exact-call, proposal, screen, correction, and rejection counters to
   sampler and native performance records.
6. Add analytic target tests covering Gaussian, correlated, bounded, and
   invalid-domain proposals.
7. Add tests proving that surrogate-disabled execution matches the exact
   sampler's seeded scalar evaluations and acceptance decisions.
8. Add tests proving uncertain support, surrogate failure, and exact solver
   failure fall back or classify deterministically without silent acceptance.
9. Add native CMB fixtures comparing exact and delayed-acceptance spectra,
   likelihoods, diagnostics, and correction records at fixed points.
10. Run independent bounded chains for representative LCDM and non-LCDM
    models and compare posterior summaries, correlations, acceptance,
    convergence, and effective sample size with exact chains.
11. Confirm that delayed acceptance reduces exact CMB calls or improves ESS
    per second without changing the scientific comparison results.
12. Keep the feature opt-in until all mathematical and scientific acceptance
    evidence is complete.
13. Document the approximation boundary, exact fallback, correction rule,
    provenance fields, and limitations in package-facing and repository-facing
    documentation.

**Correctness acceptance:**

* Surrogate-disabled results match the exact scalar sampler under the same
  seed and manifest.
* A forced-exact surrogate produces the exact scalar result.
* Unsupported, uncertain, and failed surrogate predictions trigger the
  declared exact fallback or typed rejection.
* Stage-two correction uses the exact target and preserves proposal-density
  accounting.
* Analytic target chains recover the known distribution within predefined
  bounded-fixture tolerances.
* Every proposal decision is attributable in the sampler and native records.

**Scientific acceptance:**

* Native spectra and likelihoods agree with exact scalar references at fixed
  parameter points.
* Representative exact and delayed-acceptance chains agree in posterior
  summaries, correlations, acceptance behavior, convergence diagnostics, and
  derived observables within predefined comparison tolerances.
* No model-specific constraint, conservation, or observable regression is
  hidden by surrogate screening.
* The exact scalar chain remains the comparison authority for every report.

**Performance acceptance:**

* Reports include exact-call reduction, surrogate cost, exact correction cost,
  wall time, acceptance, and effective sample size per second.
* A speedup is accepted only when scientific and mathematical validation has
  already passed.
* A surrogate that is faster but scientifically biased is rejected and the
  opt-in mode remains disabled.
* No local full-chain wall-clock threshold is used as a substitute for these
  records.

**Done when:**

* The opt-in delayed-acceptance path is mathematically specified,
  implemented, tested against analytic targets, and compared with the exact
  native CMB sampler.
* Its scientific and performance records are complete.
* Exact scalar execution remains the documented default and fallback.
* The slice is marked closed only after both correctness and scientific
  validation are independently green.

## Completion Standard

The roadmap is complete only when all three slices are closed in order.

Completion requires:

* truthful progress and phase-level timing evidence;
* safe runtime improvements with exact scalar equivalence;
* an independently validated batch/vectorized contract;
* an independently validated opt-in surrogate/delayed-acceptance sampler;
* separate correctness and scientific acceptance records for both algorithmic
  slices;
* preserved native CMB capability, constraint, observable, and failure
  contracts;
* no production CAMB or CLASS fallback;
* no hidden full-validation-suite dependency;
* documentation, comments, tests, manifests, mirrors, and changelog aligned;
* a green `devcovenant gate --verify` on the staged revision.

A green policy gate, a faster benchmark, a finite spectrum, or a matching
single-point likelihood is not sufficient by itself. The work is complete
only when the exact scalar reference and each accelerated execution mode have
their own evidence and the accelerated modes satisfy both correctness and
scientific acceptance.
