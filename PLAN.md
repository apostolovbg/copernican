# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-10-03
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

## Overview

**Goal:** Complete the Copernican Cosmic Microwave Background Solver
(CCMBS), with demonstrated numerical and physical correctness on the bounded
acceptance matrix below, before the human launches the final application
comparison. Completion means a working engine with retained evidence. A
finished reporting interface, a green policy gate, or closed slice headings
cannot establish that result.

This plan replaces the nineteen-slice plan at commit `148a175`. The previous
plan and its implementation history remain in Git and `CHANGELOG.md`. Preserve
the useful implementation; do not restart CCMBS or repeat completed work
without a failing acceptance criterion that requires it. Old slice numbers in
historical records refer to the previous plan, not to this one.

There are eight implementation slices. Execute them in order. Each slice owns
concrete changes, focused tests, evidence, and closure conditions. Numerical
repair precedes actual LCDM parity; actual LCDM parity precedes broader
certification. Do not move an unmet requirement to a later handoff or mark a
slice closed because a command capable of testing it now exists.

The human's full scientific production campaign is outside this plan. Bounded
real CCMBS forward calculations, independent reference comparisons, and
adversarial numerical tests are required development work inside this plan.
The final handoff prepares an ordinary Copernican comparison of LCDM against
Planck2018ref with **5 burn-in steps and 10 production steps**. The human
launches that demonstration after engine acceptance; neither that short chain
nor a long production campaign substitutes for the acceptance evidence.

**Initial state:** all eight slices below are pending. Replacing this document
does not close Slice One or certify the current engine.

## Scope and invariants

The scope includes declarations and compilation, automatic numerical planning,
background and recombination, perturbation hierarchies and collisions, source
histories, line-of-sight projection, lensing and observable assembly, caches,
sampler integration, BAO independence, GUI/CLI output, and retained evidence.

1. CCMBS remains the single public production CMB solver. CAMB is an
   independent test reference only. Never use it to supply production spectra
   or repair a failed CCMBS request.
2. Every complete declaration expressible by the grammar follows the same
   engine route. No model-name branches, privileged LCDM implementation,
   hidden standard-cosmology defaults, or model-specific numerical patches.
   A missing compiler/runtime capability is an engine defect to repair.
3. Preserve each theory's equations and parameter semantics. Models declare
   physics, domains, species, gauges, initial data, interactions, sources, and
   observables. The engine owns numerical counts, tolerances, hierarchy
   depths, integration starts, and refinement decisions. Do not restore
   numerical recipes in YAML or rename a physical mismatch as unavailable
   merely to make a reference row pass.
4. Normal requests, including explicit ell arrays, use automatic production
   planning. Diagnostic overrides are allowed for component tests, but their
   results cannot certify the ordinary route. No hidden reduced-grid path in
   acceptance, likelihood, sampler, graph, or export execution.
5. A valid request returns complete, finite, accepted requested products or
   a typed failure with the numerical or mathematical evidence. An empty
   graph, omitted surface, fabricated physical zero, swallowed exception, or
   replacement reference curve is not an accepted outcome.
6. All applicable scalar, vector, tensor, unlensed, lensed, auto, cross, and
   total surfaces remain in scope. A physical zero or non-applicability
   decision requires a declaration-based reason. Engine absence is neither.
7. Preserve distinct nested k refinement, measured phase spacing, explicit
   CAMB neutrino inputs, and incomplete sampler timeout semantics already
   implemented. Regressions in these boundaries block closure.
8. Retain scalar/CPU correctness as the reference for batching, caching, and
   any acceleration. An optional device is never required to complete this
   plan. Device work is allowed only to resolve a measured acceptance-work
   bottleneck while preserving numerical results and failure behavior.
9. Fix the owning implementation. Do not widen tolerances, remove difficult
   points, disable tests, suppress failures, alter observations, or change
   CI/environment policy to obtain a pass. Do not add unrelated cleanup,
   dependency changes, version bumps, or general architecture projects.
10. Stay on the current branch and use the existing managed `.venv`. Do not
    recreate or upgrade it. Follow `AGENTS.md`, including managed-block and
    changelog rules. No side agents, commits, or pushes without instruction.

## Starting defects and ownership

These findings are the starting work inventory, not assumptions of completion.
Reproduce them against the active source and preserve the repair regressions.

* **G1, actual parity:** the previous final slices delivered commands and
  mocked acceptance tests without an accepted actual CCMBS/CAMB row. The
  previous completion standard nevertheless required bounded parity. Slices
  One, Three, and Four own the executable matrix and its actual results.
* **G2, graph coverage:** `tests/project/lib/test_slice_seven.py` currently
  stops at ell 100, supplies numerical overrides, and no longer asserts the
  earlier TT trough-to-peak behavior. Slices Three and Four restore meaningful
  ordinary-route wave coverage for LCDM and Planck2018ref.
* **G3, source acceptance:** production source tolerance is 2.0 and evolution
  tolerance is 0.2. The history comparator samples three fractional positions;
  a source sign reversal and an intervening spike can pass its source check.
  Slice Two replaces this assurance with justified error control and negative
  controls, then Slice Three verifies its effect on observables.
* **G4, numerical evidence:** q/hierarchy controls are labelled
  `validated_bound` from configured floors, and physical limits are labelled
  `measured` from grid information. Those labels do not demonstrate the
  associated errors. Slice Two owns real estimates, bounds, and enforcement.
* **G5, inference:** `joint_mcmc` still defers doubled-k acceptance without
  first requiring a valid certificate for that exact calculation. Slice Six
  closes this gap without reintroducing timeout-to-posterior rejection.
* **G6, persistence:** the graph subprocess reads inside an outer temporary
  directory which is then removed. Slices One and Three retain actual
  evidence from the outset; Slice Seven verifies its final durability.

The latest inspected baseline run at the reset passed 871 tests in about
56 minutes. This records repository health, not physical acceptance. Preserve
its useful coverage while replacing inadequate acceptance assertions.

## Acceptance contract

### Model and observable matrix

Slice One must materialize this exact required inventory in the existing
acceptance machinery. Missing, failed, stale, or unclassified rows prevent
completion. A report that successfully records rejection is still rejection.

| Declaration | Required evidence |
| --- | --- |
| `model_lcdm.yml` | Matched CAMB parity and ordinary-route graph |
| `model_ref_planck2018.yml` | Matched CAMB parity and ordinary-route graph |
| `model_lcdm_mnu.yml` | Matched CAMB parity and q/hierarchy convergence |
| `model_wcdm.yml` | Matched CAMB parity and LCDM-limit behavior |
| `model_w0wa.yml` | Matched CAMB parity and wCDM/LCDM-limit behavior |
| `model_qauc.yml` | Theory-specific invariants and full declared output |
| `model_qrsf.yml` | Theory-specific invariants and full declared output |
| `model_tog.yml` | Theory-specific invariants and full declared output |
| `model_torg.yml` | Theory-specific invariants and full declared output |
| `model_usmf2.yml` | Theory-specific invariants and full declared output |

Every row enumerates its declared observables and active sectors before
execution. Cover TT, TE, EE, BB, PP, TP, EP, lensed products, and sector totals
where declared. A request for every declared surface must match that inventory
exactly. Add explicit nonzero tensor/vector/lensing fixtures where the bundled
scalar models cannot exercise those paths. Do not certify a nonzero path by
running a scalar declaration whose answer is physically zero.

The comparable matrix includes more than each initial point: a shifted
primordial amplitude, more than one nonzero neutrino mass plus the zero-mass
limit, and non-Lambda dark-energy points plus their common limits. Freeze
exact physical inputs and expected applicability in Slice One. Compatible
points may share compilation, but distinct physical points must receive their
own physical evolution and acceptance evidence.

Add representative complete novel declarations with unrelated names, changed
recombination/opacity, and an extra fluid or interaction. Renaming alone proves
name independence, not support for different physics. Malformed, singular,
non-finite, or contradictory declarations require separate negative tests;
they cannot replace successful execution of valid novel declarations.

### Reference identity and quantitative acceptance

Compare identical physics and conventions: all resolved parameters and
defaults, species and neutrino masses/distributions, primordial normalization,
recombination and reionization, gauge/source definitions, units, lensing,
sector definitions, and the exact ell grid. Matching a few cosmological
parameters is insufficient. Assert CAMB's resolved inputs, including zero
massive neutrinos for the fixed massless LCDM row, after defaults apply.

Retain the existing per-surface relative ceilings:

| Surface | Relative ceiling |
| --- | --- |
| TT, EE, BB | 2% |
| TE, PP | 3% |
| TP, EP | 5% |
| Lensed TT, lensed EE | 2% |
| Lensed TE | 3% |
| Lensed BB | 5% |

No unlisted applicable surface has an implicit pass. In Slice One, specify
its metric, units, absolute floor, relative ceiling, and reference before
examining CCMBS residuals. Require both raw C_ell and public D_ell checks and
verify their conversion independently. Fix normalization or sign errors at
the source; never fit away amplitude or phase before deciding parity.

Cross-spectra and near-zero surfaces require explicit absolute-plus-relative
criteria. Use physically justified reference scales, including the matching
auto spectra for cross-spectrum normalization, with sign and zero-crossing
checks retained separately. An arbitrary denominator floor, dominant TT
amplitude, or a best-fit rescaling cannot hide an EE/cross-spectrum failure.
Freeze these metric definitions and numerical floors with their physical
justification. They are not tunable parameters for making a row green.

The canonical sparse coverage remains
`(2, 20, 100, 200, 500, 800, 1200, 1500, 2000, 2500)`.
These anchors establish range coverage, not wave-shape coverage by themselves.
Add contiguous or demonstrably adequate sampled windows around reference
peaks, troughs, TE sign changes, and the damping regime. Resolve window choices
and maximum spacing from the independent reference in Slice One, before
looking at CCMBS output. Assert feature locations, amplitudes, damping, and
absence of unexplained spikes from raw arrays before plotting.

Frozen CAMB products are permitted when generated independently, versioned,
physically matched, and digest-verified. CCMBS output cannot be an independent
oracle for itself. Never replace an actual comparison with a mock, a
CAMB-versus-CAMB row, synthetic observations copied from CCMBS, or a graph
whose two theory curves are the same cached array.

### Numerical and execution acceptance

Every applicable numerical axis needs either measured independent refinement
or an implemented error bound validated over its stated domain. Cover:

* background, recombination, visibility, and drag;
* momentum/q resolution and distribution support;
* hierarchy truncation in every active sector;
* evolution accuracy, initial-time handling, and tight-coupling transitions;
* source-history sampling and interpolation;
* k integration and line-of-sight/projection quadrature; and
* physical integration endpoints and lensing support/padding.

Node counts, minimum-depth rules, finite values, enabled flags, and status
strings are metadata, not error bounds. Allocate numerical error budgets from
the observable acceptance criteria and validate their combined effect. A
200% source difference or 20% evolution difference cannot be accepted merely
because those values were configured. Any looser internal bound needs an
independent demonstrated bound on the resulting observable error.

Refinement must change the effective calculation after all floors and caches.
Retain both products, node/endpoint identities, measured errors, and actual
work. Independently evolve source/history validation samples; interpolation
of one history is not an independent evolution. A finer k quadrature over a
fixed interval does not establish that the interval itself is sufficient.

Measure request-subset consistency: overlapping multipoles must agree across
low-to-high ell requests and observable subsets within the declared numerical
budget. Measure scalar/batch and cold/warm/exact-repeat equivalence, ordering,
and failure isolation. Validate cross-parameter cache separation and any
partial reuse against the complete physical and numerical identity.

### Evidence and test workload

Use existing diagnostics, reference helpers, and report readers. Extend them
only where acceptance requires it; do not create another certification layer.
Each required row records actual execution status and retains:

* source revision and source-content identity for uncommitted work;
* declaration, resolved physical inputs, reference version and digest;
* exact request, solver/device, numerical plan, and cache identity;
* raw/public spectra, grids, weights, source/transfer histories, and both
  refinement products needed to reproduce its decisions;
* absolute/relative/shape residuals, thresholds, and accepted/rejected result;
* phase timing, effective work, cache reuse, and refinement work; and
* reloadable artifact locations and content hashes.

Choose one persistent evidence location under the configured output root in
Slice One. Store relative artifact paths in manifests and keep compact
regression fixtures under the existing test fixture conventions. Evidence must
survive the producer's exit and ordinary test cleanup. A hash without its
payload is not a retained artifact. Reopen any accepted row invalidated by a
source, theory, reference, or numerical-contract change; rerun affected rows
and their dependents instead of trusting a stale manifest.

Ordinary tests retain bounded real solver coverage and adversarial tests.
Share compatible forward products across observables, assertions, and graph
rendering. Expensive acceptance cases may run through an explicit focused
command instead of every discovery run, but the implementing agent must
actually execute them and retain accepted results before closing their slice.
They may not be reassigned to the human's later scientific production run.

Do not hide full observational covariance calculations, long chains, or
repeated high-resolution CAMB generation in ordinary discovery. Use bounded
synthetic observations for application/likelihood plumbing and frozen
independent references for numerical comparisons. Keep real parser/hash tests
separate. Every accepted numerical result still uses its required ordinary
production controls, regardless of where its test is scheduled.

Measure per-test and per-request cost. Use the current approximately one-hour
full suite as the development budget to improve or preserve through shared
work. If required acceptance exposes a runtime blocker, fix that blocker in
the current slice, preserving accuracy; report evidence of the cost rather
than weakening acceptance or deferring the physics indefinitely.

## Execution and closure rules

Read `AGENTS.md` and the active slice before editing. Inspect a dirty worktree
once and continue from that state. Use the active repository's policy and
profile owners; never edit managed blocks or unrelated metadata.

For each authorized slice, use the managed entrypoint:

```bash
source .venv/bin/activate && python -m devcovenant gate --open
```

Run the slice's focused tests and actual acceptance commands, inspect their
retained evidence, repair failures at source, then run:

```bash
source .venv/bin/activate && python -m devcovenant gate --verify
```

The human runs the expensive `devcovenant run` in this working arrangement.
Leave the gate open at the verified handoff; do not run `gate --close` before
required workflow evidence is fresh and successful. Do not run
`devcovenant check --nofix`, commit, push, or launch the final demonstration
unless specifically instructed. Stage all changes after each completed slice.

The implementing agent records the focused commands, actual results, artifact
paths/digests, and any pending human full-suite confirmation under that slice.
Keep implementation acceptance, policy verification, and full-suite workflow
confirmation distinct. Mark a slice closed only when its stated acceptance
has passed and the required full-suite confirmation is recorded. A pending
human run is not an engine defect, but it is not a completed workflow either.

Any newly found blocker belongs to the earliest slice whose obligation it
violates. Repair only what is needed to satisfy that obligation. If a later
change invalidates earlier evidence, reopen that acceptance and regenerate it.
Do not invent additional slices, redefine success, turn failures into accepted
non-applicability, or silently carry unresolved requirements forward.

## Slice One — establish the executable completion contract

**Status:** implementation acceptance complete; workflow confirmation pending.
**Entry:** the existing CCMBS implementation at this reset.

**Owners:** `PLAN.md`, the CMB diagnostics and certification functions,
`tests/project/lib/camb_reference.py`,
`tests/project/lib/scientific_acceptance.py`, and their existing tests.

**Work:**

1. Materialize the required model/point/surface matrix, exact reference
   identities, ell windows, error metrics, and applicability decisions from
   this document. Map G1 through G6 to named regression tests and owners.
2. Make final certification reject missing real execution, incomplete
   surfaces, unresolved axes, failed parity, stale source/reference identity,
   and unavailable artifacts. Distinguish successful report generation from
   accepted engine output. Keep mock-based wiring tests explicitly identified.
3. Establish persistent artifact output using the existing writer/reader
   contract. Run the smallest meaningful ordinary LCDM baseline against its
   matched independent reference, retaining actual results or typed failure.
   This baseline diagnoses work; rejection cannot count as engine acceptance.
4. Record the concrete focused test selectors and commands for each later
   slice using existing modules. Do not leave placeholders for critical
   comparisons or move unresolved physical-input matching to the final run.

**Focused tests:** diagnostic and final-report rejection, reference resolved
inputs, exact inventories, stale/tampered/missing artifacts, and command
wiring.
The primary existing modules are
`tests.copernican.lib.likelihoods.cmb.test_diagnostics`,
`tests.project.lib.test_camb_reference`, and
`tests.project.lib.test_scientific_acceptance`.

**Closure:** an executable acceptance matrix and honest baseline exist;
negative tests prove that incomplete or synthetic-only evidence cannot certify
CCMBS. This closes the contract slice only. All physics rows remain pending
until their actual comparisons pass in the owning slices.

### Frozen Slice One decisions and regression ownership

The executable inventory contains 17 bundled physical points and six auxiliary
cases. LCDM amplitude is multiplied by 1.1. Massive-neutrino response masses
are 0, 0.06, and 0.50 eV, in addition to the declared 0.25 eV initial point.
The dark-energy responses are w=-0.9, (w0,wa)=(-0.9,0), and (-1,0.2).
Initial wCDM/w0wa points retain their common massive-neutrino limit; they are
not silently compared with the massless LCDM declaration.

`tests/project/fixtures/ccmbs_completion_contract.json` freezes every case
identity, reference digest, surface, applicability decision, ell list, and
feature location. The persistent `completion-contract.json` additionally
retains full physical declarations and resolved CAMB defaults. Twelve CAMB
reference rows cover the five comparable declarations and their response
points. The five other theories require their named source, conservation,
initial-condition, and limiting-case invariants; six auxiliary recipes cover
renaming, changed recombination, an interaction, and nonzero sectors.

Reference-selected windows contain every integer ell within eight multipoles
of the first three TT peaks/troughs, EE peaks, and TE zero crossings above
ell 50. Damping windows are 1988–2012 and 2480–2500. Feature displacement may
not exceed two multipoles. The existing per-surface relative ceilings apply
at every retained sample in both native C_ell and D_ell units. Cross absolute
allowances are 0.001 times the square root of the corresponding reference
auto-spectrum product, plus 64 binary64 epsilons of that surface. Auto floors
use only that surface's scale; exactly zero BB has a zero floor. Independent
C/D conversion must agree within 1e-10 relative error. CCMBS dimensionless
raw C_ell values retain their original units in artifacts. The comparator
converts them to CAMB native units using the frozen physical temperature:
(Tcmb*1e6)^2 for TT/TE/EE/BB, one temperature factor for TP/EP, and none for
PP. This unit conversion is independent of residuals and performs no fit.
The combined numerical
budget is 0.5%, allocated across applicable axes, leaving margin below the
strictest 2% parity ceiling. These are frozen acceptance bounds, not evidence
that the current runtime already meets them.

Physical-input matching remains explicit: species/distributions, primordial
inputs, recombination, reionization, gauge/sources, units, lensing, and sectors
must have retained quantitative evidence. In particular the declared
collapse-source reionization is not CAMB's default tanh history merely because
both have tau=0.054. Slice Three owns that comparison and any necessary source
repair for LCDM; Slice Four owns the remaining comparable points. Neither may
claim parity based on parameter maps alone or defer matching to Slice Eight.

All selectors below are existing modules; use the full listed module for new
regressions added in that slice. Prefix each invocation with
`source .venv/bin/activate && python -m unittest`.

* Two: `tests.copernican.lib.likelihoods.cmb.runtime.test_planner`,
  plus `test_adaptive`, `test_convergence`, `test_background`,
  `test_evolution`, and `test_projection` in that same runtime package.
* Three: `tests.project.lib.test_slice_seven`,
  `tests.project.lib.test_camb_reference`, and
  `tests.copernican.lib.likelihoods.cmb.test_cmb`.
* Four: `tests.project.lib.test_camb_reference`,
  `tests.copernican.lib.likelihoods.cmb.test_diagnostics`, and
  `tests.copernican.lib.likelihoods.cmb.runtime.test_lensing`.
* Five: `tests.copernican.lib.likelihoods.cmb.test_cmb`,
  `tests.copernican.lib.likelihoods.cmb.test_contracts_audit`, and
  `tests.copernican.lib.likelihoods.cmb.test_diagnostics`.
* Six: `tests.copernican.samplers.test_sampler_mcmc`,
  `tests.copernican.lib.likelihoods.cmb.runtime.test_adaptive`, and
  `tests.copernican.lib.likelihoods.cmb.runtime.test_cache`.
* Seven: `tests.copernican.lib.likelihoods.cmb.runtime.test_performance`,
  `tests.copernican.lib.likelihoods.cmb.runtime.test_cache`, and
  `tests.project.lib.test_scientific_acceptance`.
* Eight: `tests.project.lib.test_scientific_acceptance` and
  `tests.project.lib.test_slice_seven`.

G1 is guarded by `CompletionContractTestCase` in
`tests/copernican/lib/likelihoods/cmb/test_diagnostics.py` and the three
explicitly synthetic final-report tests in that module. G2 uses
`CompletionReferenceWindowTestCase` in `test_camb_reference.py`, then the
ordinary graph regression in `test_slice_seven.py` during Slice Three. G3/G4
are rejected at the final contract by
`test_completion_rejects_synthetic_missing_stale_and_unresolved_rows`;
Slice Two must add actual source-sign/spike and refinement regressions in
`runtime/test_adaptive.py` and `runtime/test_convergence.py`. G5 remains the
Slice Six sampler obligation in `test_sampler_mcmc.py`; no Slice One result
claims that bypass has been repaired. G6 uses
`test_retained_artifacts_reject_tampering_missing_and_escape` and
`CompletionPersistenceTestCase`, plus separate-process baseline readback.

The baseline command is
`python -m tests.project.lib.scientific_acceptance --baseline` in the active
venv. It records ell=(2,20,100), TT/TE/EE on the ordinary full-spectrum route,
without numerical overrides. It deliberately exits 1 for a diagnostic-only
or failed result; artifact verification uses `--verify-baseline PATH` and
exits 0 only after all referenced files and current source identity validate.

### Slice One evidence — 2026-10-03

The three primary focused modules passed 74 tests in 103.581 seconds.
After the unit-conversion and provenance adjustments, the completion,
reference-window, persistence, and command regressions passed 12 tests in
0.478 seconds. Their logs are retained at
`devcovenant/registry/runtime/slice-one-focused.log` and
`devcovenant/registry/runtime/slice-one-completion-tests.log`.
The original synthetic final-report tests now explicitly assert rejection;
they retain their orchestration assertions without claiming physical success.

The actual ordinary LCDM baseline completed in 252.256 seconds, including
512-to-1024 k refinement, against `camb:1.6.0`. It was **rejected** for both
C_ell and D_ell parity in TT, TE, and EE. The independent unit-conversion
checks passed. Maximum public relative errors over ell=(2,20,100) were
176.7303% TT, 98.5186% TE, and 90.4770% EE. These are unadjusted diagnostic
residuals, not matched-history scientific certification. Slices Two and
Three own the numerical acceptance and physical-input/source comparisons
needed to resolve them. No amplitude rescaling or tolerance relaxation was
used to obtain this result.

Persistent evidence is under the configured output root at
`ccmbs-acceptance/slice-one/`. The producer exited before a separate
`--verify-baseline` invocation successfully reloaded `baseline.json`,
`completion-contract.json`, and all twelve reference payloads. All 23
materialized cases match the tracked frozen inventory. Portable identities:

* Baseline report SHA-256:
  `d1ccb525da6f2d9a68f34eaf564bcc1b46febffd1d884534ab88e1f96a3c70e1`
* Completion contract SHA-256:
  `d62cd15f933eb1787138cfce7e819576d264fde4b51af13a7a3c6483ad97fbbe`
* Source content SHA-256, including uncommitted implementation:
  `ea038e9e5adc5d37ba2b03ffc4a7b682b6fd5017857a144f330722968fbd6229`

The contract slice's implementation acceptance is satisfied. This does not
close any physics row. Required full-suite workflow confirmation remains
pending; the gate stays open until that run succeeds and is reviewed.

## Slice Two — repair automatic numerical acceptance

**Status:** pending. **Entry:** Slice One's metrics and failure inventory.

**Owners:** CMB `runtime/planner.py`, `runtime/adaptive.py`,
`runtime/convergence.py`, `runtime/background.py`, `runtime/evolution.py`,
`runtime/projection.py`, and the corresponding runtime tests.

**Work:**

1. Replace permissive source/evolution acceptance with physically justified
   error budgets. Sample actual recombination/visibility and interaction
   features, not three fixed fractions labelled as physical epochs. Use
   independent histories at informative k values and account for errors
   between sampled anchors and near zeros.
2. Make source refinement materially change resolution. Do not certify it
   with effectively identical grids or an N-versus-N-minus-one comparison
   without a validated estimator. Separate time-history, k-history
   interpolation, and projection-quadrature errors.
3. Implement actual q and hierarchy refinement or validated truncation bounds
   for the active declarations. Bind each bound to its assumptions and domain;
   configured floors alone cannot produce `validated_bound`.
4. Test and enforce changes to k/eta endpoints, integration start, thermal
   tails, and lensing support where applicable. Preserve background and drag
   independence and all raw refinement evidence.
5. Ensure ordinary planning selects the intended strict refinement algorithm.
   Remove accidental selection of a diagnostic interpolation comparison by
   production tolerance values. Refinement continues to its justified bound
   or returns a typed non-convergence result with the failed products.
6. Preserve effective nested k refinement and measured-spacing termination.
   Cold refinement proves new work; warm reuse proves a matching accepted
   finer calculation. Neither can claim work solely from a nominal count.

**Focused tests:** the existing planner, adaptive, convergence, background,
evolution, and projection test modules. Add adversarial sign reversals,
localized source spikes between anchors, oscillatory aliasing, insufficient
hierarchy/q support, omitted endpoint contributions, identical effective grids,
and genuinely finer warm reuse. Each corrupted calculation must fail the
relevant numerical assertion; well-resolved controls must pass it.

**Closure:** all applicable axes have enforceable evidence rather than status
labels. Real bounded ordinary requests exercise the revised checks. Their
measured errors meet justified budgets and request-subset consistency holds.
No global CAMB parity claim is made before Slice Three.

## Slice Three — close actual LCDM physics and graph recovery

**Status:** pending.
**Entry:** enforceable numerical acceptance from Slice Two.

**Owners:** the generic background, hierarchy, source, projection, and
post-processing code responsible for measured disagreement; independent CAMB
helpers, diagnostics, and `tests/project/lib/test_slice_seven.py`.

**Work:**

1. Execute actual fixed massless-LCDM CCMBS through the normal public route
   and compare with the physically matched CAMB reference. Cover the canonical
   sparse range and Slice One's acoustic/zero-crossing/damping windows. Use
   production planning without numerical overrides or diagnostic bypasses.
2. Find the earliest disagreement in background/recombination, initial data,
   metric/species evolution, collisions, polarization/ISW sources, transfer
   projection, normalization, or assembly. Retain aligned intermediate arrays
   and fix that generic owner. Do not compensate for an upstream defect with
   downstream scaling, smoothing, clipping, or model-specific rules.
3. Require accepted TT/TE/EE raw and public comparisons, physically located
   peaks/troughs, TE signs/zeros, and damping behavior. Restore a meaningful
   real graph regression reaching acoustic structure and the high-ell regime.
   Finite, smooth, nonnegative output alone is insufficient.
4. Feed the same accepted arrays into plotting, likelihood/export checks,
   and exact-repeat evidence. Use independent synthetic observations for
   plumbing assertions; never use equality with self-generated observations
   as physical parity evidence.
5. Retain the graph and all numerical products outside temporary directories.
   After the producer exits and temporary workspaces are removed, reload the
   manifest and its actual payloads in another process and verify decisions.

**Focused tests:** the fixed-LCDM acceptance command, real public graph test,
reference-input checks, and the source/runtime regressions needed for each
repair. Retain one cold solve, compatible refinements, and exact-repeat work
accounting; do not perform another cold solve for each graph assertion.

**Closure:** an actual accepted LCDM comparison and durable graph demonstrate
correct quantitative TT/TE/EE and wave structure. The comparison has executed
successfully; a command definition or mocked accepted row cannot close this
slice. Any runtime blocker to obtaining this result is repaired here.

## Slice Four — close comparable models and complete observables

**Status:** pending. **Entry:** accepted actual LCDM evidence from Slice Three.

**Owners:** generic model/background/species/source/lensing/assembly paths,
the existing bounded parity matrix, reference fixtures, and related tests.

**Work:**

1. Execute the required Planck2018ref, massive-neutrino, wCDM, and w0wa rows
   plus the frozen parameter-response points. Match resolved CAMB physics,
   including mass distributions, radiation bookkeeping, and dark-energy
   perturbation conventions, rather than relying on model labels.
2. Validate amplitude response, zero/nonzero mass continuity, and common
   dark-energy limits. Changing parameters must change the appropriate
   physical histories and cache identities. Do not borrow another point's
   refinement or acceptance certificate.
3. Extend real comparisons to every applicable declared observable: BB, PP,
   TP, EP, lensing, active sector contributions, and totals. Validate absolute
   units, signs, physical zeros, remapping support, and covariance bounds.
   Reference-inapplicable sectors need independent analytic or convergence
   evidence under Slice Five; they cannot simply disappear from the matrix.
4. Produce the ordinary Planck2018ref graph from its accepted arrays and
   verify that comparison rendering retains both model curves. Extend shape
   assertions and range/subset consistency to the same physical points.
5. Share compatible work and retain all accepted and rejected attempts.
   Repair generic defects and rerun affected earlier rows after each fix.

**Focused tests:** actual parity-matrix execution, independent CAMB fixtures,
full-observable diagnostics, massive-neutrino/background tests, lensing and
post-processing tests, and public graph/result tests.

**Closure:** every required comparable point/surface has an actual accepted
comparison, with complete resolved reference identity and numerical evidence.
The fixed LCDM and Planck2018ref public graphs both pass quantitative and shape
acceptance. No matrix row is pending because it was called expensive.

## Slice Five — close universal declarations and all active sectors

**Status:** pending. **Entry:** comparable physics and observables accepted.

**Owners:** validator, expression compiler, model adapter, generic CCMBS
runtime, corpus/full-observable diagnostics, and their existing tests.

**Work:**

1. Execute all ten bundled models through the same ordinary engine route.
   Reuse accepted comparable rows from Slice Four only when their complete
   identities remain current. Validate each non-CAMB theory against its own
   declared equations and invariants, not resemblance to LCDM.
2. Exercise unrelated names, altered recombination/opacity, additional
   species/interactions, and nonzero vector/tensor declarations. Validate
   source/kernel routing, gauge and initial-condition conventions, and all
   declared auto/cross/total outputs. Repair missing generic compiler/runtime
   support rather than converting valid declarations to unavailable models.
3. Validate residuals and conservation, continuous physical limits, initial
   and boundary conditions, positive auto spectra and allowed cross spectra,
   and independently refined histories and observables for these requests.
   Use analytic/manufactured solutions where they provide independent truth.
4. Keep malformed/incomplete, non-finite, and singular mathematics tests
   distinct from valid-model execution. Error context identifies the equation,
   field, axis, and request without hiding failures in discovery or graphs.
5. Confirm scalar/batch ordering, failure isolation, renamed-model identity,
   and cache separation across different theories and physical inputs.

**Focused tests:** model validator/coder/adapter tests, declared-contract and
source-graph tests, scalar/vector/tensor runtime cases, and actual bundled and
novel-declaration matrix execution with full declared-surface accounting.

**Closure:** every valid required declaration executes its complete requested
surfaces with accepted internal evidence. The corpus contains no unresolved
engine-capability defect, unclassified outcome, or fabricated physical zero.
Finite representative coverage supports the completion claim; automatic
numerical enforcement remains mandatory for future declarations and points.

## Slice Six — close sampler and application integration

**Status:** pending. **Entry:** accepted model and numerical matrix.

**Owners:** CMB public API/cache/error boundaries, `sampler_mcmc.py`,
likelihood composition, model comparison, plot/export paths, GUI/CLI, and
related tests.

**Work:**

1. Remove unconditional `joint_mcmc` convergence deferral. A proposal obtains
   its required numerical acceptance or reuses a certificate matching its
   complete physical point, declaration, request, resolved plan, and execution
   identity. Validate actual scalar/full-spectrum and proposal-route agreement.
2. Preserve timeout/cancellation as incomplete execution with evidence, never
   numerical failure disguised as an ordinary `-inf` posterior. Distinguish
   genuine prior/physical rejection from engine inability to compute. Prevent
   incomplete chains or stale results being exported as successful inference.
3. Test exact, partial, and cross-request reuse without duplicate evolution
   or cross-parameter certification. Share only products whose dependencies
   match; preserve result ordering and failure context in batch execution.
4. Verify independent BAO drag-ruler evaluation with the CMB entrypoint
   unavailable, and correct SNe/BAO results when CMB fails. CMB failure must
   remain visible in the combined result and in both GUI and CLI displays.
5. Drive likelihood, comparison, graph, export, and manifest assembly from
   the accepted canonical products. Retain real model identities and raw/public
   conventions. Test the free LCDM and fixed Planck2018ref execution paths;
   do not invent a sampled chain for a model with no free parameters.

**Focused tests:** sampler timeout/incomplete and proposal-path tests, CMB
public/cache tests, independent BAO and combined-likelihood tests, model
comparison, plotting/export, manifest, and GUI/CLI failure tests. Use a small
real proposal sequence plus bounded fixtures for broad dispatch coverage;
a long posterior run is not required.

**Closure:** inference cannot bypass the engine's numerical contract; accepted
proposal values agree with the corresponding forward calculation. Failures
remain typed and cannot bias a completed posterior through silent numerical
rejection. Both model roles and all application consumers use accepted data.

## Slice Seven — consolidate durable evidence and execution cost

**Status:** pending. **Entry:** all required physics and integration rows pass.

**Owners:** existing evidence writers/readers, test output lifecycle, runtime
work accounting and measured bottlenecks, relevant docs and exact mirrors.

**Work:**

1. Consolidate the accepted rows into one final manifest linked to the
   current source/reference/declaration identities. Verify every required
   matrix entry and retained graph, array, history, grid, and refinement
   product after producer exit and normal test cleanup. Reject tampering,
   missing payloads, stale inputs, and disconnected hashes.
2. Benchmark accepted ordinary cold, warm, exact-repeat, partial-cache, and
   nearby-parameter requests. Record per-test and per-phase timing, memory
   where material, actual evolution/projection work, and cache hits/misses.
   Explain every repeated cold calculation required by independent acceptance.
3. Remove demonstrated duplicate work and resolve bottlenecks that make the
   bounded acceptance or final short comparison impractical. Optimize owning
   kernels/schedules/caches without changing physics or acceptance. Rerun the
   affected numerical comparisons after performance changes.
4. Ensure ordinary discovery remains bounded and preserves meaningful real
   coverage. Retain an explicit reproducible command for any less-frequent
   acceptance case and its current accepted artifact; do not replace it with
   an unexecuted handoff. Report measured cost against the development budget.
5. Reconcile code comments/docstrings, test claims, configuration prose,
   public docs, and managed mirrors with the demonstrated behavior. Remove
   stale parity, durability, automatic-device-selection, or old slice-closure
   claims. Follow canonical doc ownership and keep package docs package-facing.
6. Review code, tests, docs, config, managed assets/mirrors, consistency,
   performance, and architecture against this plan. Repair remaining defects
   required for CCMBS completion; avoid unrelated repository improvement.

**Focused tests:** persistence/reload/tamper and cleanup regressions,
performance/cache tests, all affected numerical rows, and doc/mirror checks.
Record the human's complete successful `devcovenant run` and required gate
results against the final accepted implementation.

**Closure:** a durable, current accepted manifest proves the entire matrix,
with no unowned requirement or unexplained execution regression. Required
checks are green; these checks accompany, rather than replace, engine evidence.

## Slice Eight — prepare the 5/10 Copernican demonstration

**Status:** pending. **Entry:** Slice Seven's accepted final evidence.

**Owners:** existing run-manifest/configuration and application launch paths,
comparison validation, operator documentation, and bounded integration tests.

**Work:**

1. Prepare a reproducible ordinary Copernican configuration comparing
   `model_lcdm.yml` with `model_ref_planck2018.yml`, using CCMBS and the
   human's configured observational inputs. Resolve dataset identities,
   parser hashes, model roles, seed, supported walker/resource settings,
   output location, and **5 burn-in / 10 production steps**. Do not invent
   dataset choices if the repository and human configuration do not supply
   them; obtain that missing selection before declaring the handoff complete.
2. Preserve the fixed Planck2018ref parameter semantics. The 5/10 sampler
   settings govern the free-model sampling path; the fixed reference follows
   its normal evaluation path. Validate this distinction in the final summary.
3. Validate the configuration, model/dataset loading, parser integrity,
   application dispatch, and output contract through existing supported
   interfaces without launching the human's observational comparison. Any
   preflight must be proven not to start that solve. Bounded integration
   tests use already accepted products and controlled observations.
4. Record the exact tested launch command and equivalent GUI selections,
   prerequisites, configuration path, expected output directory, and runtime
   estimate based on measured accepted forward/proposal work. Include all
   walker/proposal evaluations in the estimate; 15 steps do not mean 15
   forward solves. Do not invent a CLI flag or label a dry run as execution.
5. Define what the human should see: both labelled theory curves, complete
   requested CMB products, coherent SNe/BAO/CMB summaries where enabled,
   exported canonical arrays, manifest and numerical provenance, and a
   correctly classified complete/incomplete run. A missing curve, stale
   result, hidden convergence failure, or unexplained spiky output is failure.
6. Record engine acceptance and the demonstration's not-yet-run status
   separately. The handoff does not require a long scientific production run
   or a converged posterior from ten samples. Do not launch the demonstration
   until the human explicitly instructs it.

**Focused tests:** manifest/config validation, fixed/free model dispatch,
bounded end-to-end comparison/output tests, and exact documented launch-path
validation. If application code changes, rerun its affected numerical and
integration tests and obtain fresh required workflow confirmation.

**Closure:** the engine evidence remains accepted and current, the final
configuration and launch procedure are concrete and validated, and the only
remaining action is the human's explicit launch of the short demonstration.
No unresolved engine requirement may be relabelled as a production-run input.

## Final completion checklist

CCMBS is complete under this plan only when all of these are demonstrated:

* every required comparable point and surface has accepted actual independent
  parity and quantitative shape evidence through the ordinary engine;
* every required bundled and novel declaration has complete, theory-faithful
  accepted output, including the tested nonzero sector and lensing paths;
* every applicable numerical axis is enforced with meaningful measured errors
  or validated bounds, and adversarial failures are detected;
* production, sampler, batch, cache, graph, and export paths share the accepted
  numerical contract, with correct independent BAO and typed failure behavior;
* graphs and raw/refined products are retained, reloadable, hash-verified, and
  tied to the source and reference identities that produced them;
* runtime evidence accounts for required work and demonstrates reuse without
  accuracy reductions or arbitrary wall-clock acceptance;
* docs, tests, configuration, and mirrored assets agree with the
  implementation;
* all focused acceptance and required full-suite/gate checks have passed,
  with no outstanding defect hidden by a closed slice or deferred command; and
* the validated LCDM/Planck2018ref 5/10 configuration and exact launch handoff
  are ready for the human, with its unexecuted status stated honestly.

Only after this checklist is satisfied may the final handoff say:

> CCMBS implementation and bounded engine acceptance are complete. Now we have
> to run Copernican with 5 burn-in steps and 10 production steps on LCDM versus
> Planck2018ref, so we can inspect what we built together. That comparison is
> ready for the human to launch; it has not been run as part of this plan.
