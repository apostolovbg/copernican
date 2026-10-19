# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-10-18
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

## Overview

**Goal:** Complete CCMBS as Copernican's ordinary production CMB engine:
complete theories expressible by the declaration grammar execute their own
physics, numerical resolution is selected and justified by the engine, and
independent comparisons demonstrate correctness. Deliver the surrounding
application with consistent SNe/BAO/CMB likelihoods, model comparisons,
samplers, GUI/CLI behavior, plots, exports, and reproducible evidence.

The final demonstration compares the free LCDM control with the fixed
Planck2018 reference using **5 burn-in and 10 production steps**. CAMB parity
means comparing each CCMBS physical point with its own matched independent
CAMB calculation. It does not mean forcing the free and fixed models to have
identical spectra. The human's later scientific campaigns and posterior
convergence studies remain separate from bounded engine acceptance.

This roadmap replaces the preceding nine-slice plan. Completed Slices One,
Two, and Three are preserved, including the human's confirmation that Slice
Three is complete. Their useful code, numerical regressions, and ownership
boundaries remain the starting point. They do not establish CAMB parity.
Historical plans and evidence remain in Git and CHANGELOG.md. Unpushed
experiments on another machine are not part of this accepted baseline.

**Current state:** Slices One through Three are complete. Slice Four is
implementation-complete, awaiting the human's fresh full-workflow result.
Slices Five through Ten remain pending; implementation proceeds one
explicitly authorized slice at a time. A green policy gate, a generated
report, or a successful mock cannot close a physical acceptance requirement.

## Execution principles

1. One public production solver, `ccmbs`, evaluates the declared graph. CAMB
   is an independent test reference only. No reference spectra, empirical
   amplitude/phase fits, smoothing, clipping, or surrogate theories repair
   production output.
2. Universality means every complete declaration within the supported grammar
   follows the same engine contract. No model-name dispatch, privileged LCDM
   path, hidden standard-cosmology physics, or numerical recipes in YAML.
   Missing compiler/runtime capability for valid mathematics is engine work;
   malformed mathematics and a theory that explicitly declares no observable
   remain distinct cases.
3. Models own equations, species/distributions, interactions, gauges, initial
   and boundary data, physical domains, and observables. The engine owns
   grids, hierarchy depths, integration starts, step sizes, tolerances,
   quadrature, refinement, and hardware choice. Initial heuristic estimates
   must earn acceptance through measurements or validated bounds.
4. Preserve the completed numerical safeguards: distinct nested refinement,
   measured phase spacing, independently evolved histories, explicit neutrino
   inputs, typed numerical failures, and incomplete sampler timeout semantics.
   A numerical algorithm may change only with direct equivalence/refinement
   tests and fresh affected acceptance evidence.
5. Repair the earliest failing physical or numerical owner. Background errors
   are not projection corrections; equation errors are not tolerance changes.
   Derive sign, gauge, unit, and normalization expectations independently of
   the implementation under test.
6. Keep Slice Three's one-way boundaries. Physics owns physical evolution and
   sources; numerical execution owns planning, quadrature, reuse, and work;
   diagnostics measure completed products; evidence serializes and verifies
   them. Public orchestration composes these owners. Do not restart the
   architecture, grow another monolith, or add a certification framework.
7. A production result contains every requested accepted surface or a typed
   failure with usable context and failed products. Finite arrays, physical
   zeros, unsupported labels, and status strings are not substitutes for
   mathematical or numerical evidence.
8. Ordinary explicit ell requests use production planning. Bounded diagnostic
   overrides are useful component probes, visibly classified as diagnostic.
   They cannot certify the ordinary route or enter likelihoods unnoticed.
9. Preserve the scalar CPU correctness reference. Optimize measured kernels,
   schedules, memory, and compatible work before adding devices or
   dependencies. No accelerator is required by this roadmap.
10. Keep the scope of each slice explicit. No unrelated cleanup, version
    bumps, dependency changes, CI workarounds, blanket test suppression, or
    metadata changes to obtain a pass. Keep forward-only contracts and repair
    generated artifacts through their owning sources.

## Starting findings and regression obligations

These are concrete investigation inputs, not claims that every root cause is
already understood. Reproduce against the current source before repairing.

* **Physics identity (reproduced and repaired in Slice Four):** the starting
  generated photon monopole RHS used `-Phi_tau`,
  while baryon and neutrino continuity use positive metric derivatives.
  With the documented variables and equal photon/baryon velocities, the
  declared equations give `(delta_b - 3*Theta_gamma0)' = 6*Phi'`. Establish
  the independent continuity/entropy invariant and repair the owning
  materializer, comments, documentation, and regression together.
* **Reference matching:** the LCDM declarations specify a collapse-source
  reionization history; CAMB 1.6.0's default is tanh. Equal `tau` and parameter
  maps do not establish equal opacity, visibility, or source histories.
* **Parameter consistency:** audit distance/sound-horizon and CMB background
  uses of shared parameters, including photon density, temperature, matter,
  neutrino bookkeeping, and recombination versus drag endpoints. Preserve
  intended theory semantics and make accidental inconsistencies explicit.
* **Work growth:** the radial-phase rule at ell 2500 implies approximately
  9,600 base k nodes before production doubling. Sparse output ell values do
  not remove those evolution modes. This is a formula-derived estimate, not
  a measured runtime or a justified minimum for every future algorithm.
* **Repeated work:** outer k refinement re-enters execution with internal
  checks. Adaptive production controls disable transfer-cache reuse, and
  source histories have a 1,024-entry/128 MiB cache. Measure actual reuse and
  recomputation; do not assume matching nodes survived or remained reusable.
* **Diagnostic fidelity:** stage diagnostics can select different integration
  behavior. An informative diagnostic comparison must be followed by an
  ordinary-route regression before it can support production acceptance.
* **Acceptance coverage:** the existing fixed-LCDM `--output` command targets
  the historical sparse-only request. The graph regression stops at ell 100,
  supplies numerical overrides, compares a model with itself, and deletes
  its outer artifact directory. Neither closes the new acceptance contract.
* **Application coverage:** the shipped Planck parser keeps the TT covariance
  block, and the wide-table CMB likelihood selects TT while post-processing
  also discovers TE/EE. The manifest named `reference_planck2018.yml` selects
  TORG. Fix these inconsistencies in the owning application slice; a graph
  label or filename is not proof of the underlying likelihood or model pair.
* **Behavioral coverage:** direct run-pipeline tests check imports/callables,
  while executor tests mock that pipeline. Add bounded behavioral coverage
  for the actual shared application path, with independently controlled data.
* **Evidence freshness:** completion identity currently includes broad app
  and test sources. Record which changes invalidate physical products,
  comparisons, or rendering; whole-repository provenance must remain honest
  without forcing unrelated consumers to repeat identical physical work.

The preceding plan records an ordinary low-ell LCDM rejection in 252.256
seconds and a bounded graph test in 691.749 seconds. Its full-suite baseline
was about 56 minutes. These are historical observations, not current
benchmarks, accepted spectra, or acceptable development-loop targets.

## Test workloads and practical cost contract

The deterministic forward solver and the stochastic sampler need different
proofs. A full chain is unnecessary to detect a continuity error, a bad
collision block, an incorrect likelihood vector, or a stale numerical cache.
Use the following workloads, each with an explicit claim and reproducible
command. Extend existing test owners and the existing acceptance runner.

| Workload | Required evidence | Working cost target |
| --- | --- | --- |
| Focused development | Independent math and app behavior | At most 60 s |
| Bounded real physics | Histories, kernels, refinement | At most 5 min |
| Ordinary discovery | Broad tests and production smoke | At most 15 min |
| Cold LCDM acceptance | Full frozen physical parity request | At most 15 min |
| Cold Planck acceptance | Matched massive-neutrino point | At most 30 min |

These are implementation targets on one recorded reference machine using the
managed CPU environment, not claimed present measurements or physical
validity cutoffs. Record hardware, interpreter, numerical libraries, thread
settings, cold/warm state, per-test and per-phase timings, peak memory when
material, and actual evolved/projected work. Record fixed budgets for the
remaining required cases from measured component costs before launching them.

An exceeded target is a performance blocker to investigate in the slice that
needs the calculation. Do not repeatedly launch a multi-hour matrix to learn
that one component is still wrong. Profile representative bounded work,
retain rejected attempts, and repair the measured cause. A valid slow theory
is not scientifically rejected by a clock. Any revision of these completion
budgets requires an explicit human decision with measurements; do not weaken
physics or silently redefine completion to meet them.

Ordinary discovery must keep real solver composition and adversarial coverage.
Move expensive exact acceptance to explicit focused execution only while
retaining its assertions, commands, and actual current evidence. Do not hide
hard cases with skip flags or replace every physical test with mocks. Freeze
independently generated, versioned, digest-verified CAMB products for recurring
comparisons; regenerate them explicitly when their physical/reference inputs
change, not on every test run.

Use small independent synthetic observations and the existing synthetic solver
for application dispatch, row ordering, failures, and output tests. Such tests
prove application behavior, never physical parity. Analytic/manufactured
solutions and frozen independent histories supply physical truth for bounded
component tests. Include negative controls that expose wrong signs, shifted
phases, source spikes, missed tails, and invalid reuse.

## Scientific acceptance contract

### Required physical cases

Retain the existing 23-case inventory: 17 bundled physical points and six
auxiliary declarations. The tracked
`tests/project/fixtures/ccmbs_completion_contract.json` owns the frozen exact
requests, reference identities, windows, surfaces, and applicability. Extend
existing diagnostics and readers to execute that contract; do not add a
parallel report system or silently rebaseline its hashes.

| Declaration | Required points and proof |
| --- | --- |
| `model_lcdm.yml` | Initial, As multiplied by 1.1; CAMB and real graph |
| `model_ref_planck2018.yml` | Fixed initial point; CAMB and real graph |
| `model_lcdm_mnu.yml` | Initial 0.25 eV; 0, 0.06, 0.50 eV; CAMB and q |
| `model_wcdm.yml` | Initial and w=-0.9; CAMB and common limits |
| `model_w0wa.yml` | Initial, (-0.9,0), (-1,0.2); CAMB and common limits |
| `model_qauc.yml` | Declared attractor/normalization invariants |
| `model_qrsf.yml` | Declared baryon-locked density/momentum/Euler laws |
| `model_tog.yml` | Declared temporal normalization and early limits |
| `model_torg.yml` | Declared relational sources and photon drag |
| `model_usmf2.yml` | Shrink-field dynamics, constraints, physical limits |

The six auxiliary cases cover an unrelated name, changed recombination and
opacity, an extra fluid/interaction, and nonzero vector, tensor, and lensing
paths. A rename alone proves name independence. An extra-fluid test must
actually change the physical state/equations. Scalar zeros cannot certify
nonzero sectors. Malformed, singular, contradictory, and non-finite inputs
need separate negative tests and cannot substitute for valid execution.

Enumerate every declaration's sectors and observables before execution.
Retain TT, TE, EE, BB, PP, TP, EP, lensed products, and sector totals wherever
applicable. No undeclared surface is invented; no required surface silently
disappears. Preserve exact ell ordering, duplicates, units, signs, physical
zeros, and non-applicability reasons through all consumers.

### Matched independent physics and metrics

Freeze complete reference identities before inspecting residuals: parameters
and resolved defaults, density closure, mass counts and distributions,
primordial conventions, recombination, reionization, gauge transformations,
source definitions, temperature, units, sectors, and lensing. Assert resolved
massless CAMB defaults for the massless LCDM point. Compare physical histories
quantitatively, not only parameter maps or class names.

If a standard reference and a declaration describe different physical
histories, locate and resolve the difference explicitly. Match the reference
to the declared physics where supported, or correct a demonstrated declaration
error with its derivation. Do not silently replace a theory's law, use an
unmatched reference, relax the metric, or call an engine defect unavailable.

Preserve the per-surface ceilings below, in both raw C_ell and public D_ell.
Use the existing independent dimensional conversion; never fit amplitudes or
phases before deciding parity.

| Surface | Relative ceiling |
| --- | --- |
| TT, EE, BB, lensed TT, lensed EE | 2% |
| TE, PP, lensed TE | 3% |
| TP, EP, lensed BB | 5% |

Cross absolute allowances remain 0.001 times the square root of the matching
reference auto-spectrum product plus 64 binary64 epsilons of that surface.
Auto floors use that surface's scale; exactly zero BB keeps a zero floor.
Retain sign/zero-crossing checks separately. Independent C/D conversion must
agree within 1e-10 relative error. No unlisted applicable surface has an
implicit pass; define its independent metric before viewing its residuals.

Keep sparse anchors `(2,20,100,200,500,800,1200,1500,2000,2500)` plus the
reference-selected unit-spaced windows within eight multipoles of the first
three TT peaks/troughs, EE peaks, and TE zero crossings above ell 50. Retain
damping windows 1988-2012 and 2480-2500 and a maximum feature displacement of
two multipoles. Assess locations, amplitudes, damping, and unexplained spikes
from raw arrays before plotting. Sparse anchors alone do not prove shape.

### Numerical acceptance and reuse

Preserve the combined 0.5% observable numerical-error budget. Allocate it
across applicable background, q/support, hierarchy, evolution/start/coupling,
source representation, k/projection, endpoint, and lensing axes. Do not add
unrelated dimensionless history errors and call the sum an observable bound.
Different local tolerances require a validated propagation bound or an
independent comparison showing their combined observable effect.

Each axis needs independent measured refinement or a validated bound with
assumptions, applicability domain, and negative controls. Counts, floors,
finite values, and metadata labels alone cannot establish acceptance. Resolve
between-anchor features and near-zero behavior, actual background/interaction
features, thermal tails, and physical integration endpoints.

Independently evolve histories when measuring evolution error. Measure source
interpolation with that evolution held fixed. Refine k and eta calculations
materially after all floors; changing a nominal count is insufficient. Keep
both products, grids, weights, identities, errors, and actual new work.
Finer quadrature on a fixed interval does not prove adequate interval support.

Separate the resolution needed for physical source evolution from the
resolution needed for oscillatory projection if measurements justify that
algorithm. Every interpolated/approximated product needs its own error check,
new independent physical samples where necessary, and ordinary-route parity.
Never cap a phase grid or reuse interpolation as independent evolution proof.

Reuse only products with matching actual dependencies: declaration, physical
point, parameter dependencies, request, resolved numerical contract, grids,
solver/backend, and source implementation. Structural compilation, background,
source histories, transfers, complete spectra, and acceptance decisions have
different dependency sets. A nearby parameter point cannot borrow a numerical
certificate. Primordial-only changes may reuse evolution only when the
compiled dependencies prove independence; their spectra and decisions remain
point-specific.

Keep cold, valid-warm, invalid-reuse, exact-repeat, request-subset, and ordered
scalar/batch tests. Independent refinement must retain new work; a verified
exact hit reports no new evolution/projection. Keep a demonstrated cold solve
for every required physical execution path and regenerate affected evidence
after relevant changes. Do not require a new cold solve for each plot panel.

### Persistent evidence and selective revalidation

Use the existing diagnostics, acceptance runner, writers/readers, and one
persistent location under the configured output root:
`ccmbs-acceptance/completion-evidence/`. Retain compact reference fixtures
under the existing test-fixture conventions. Manifests use relative artifact
paths; tracked files contain no machine-local absolute paths.

Each accepted or rejected attempt records source revision plus uncommitted
content identity, declaration and full physical/reference inputs, exact
request, numerical plan, backend, cache identity, raw/public spectra,
source/transfer histories, grids/weights/refinement products, numerical and
physical residuals, decisions, phase timings, actual work, and payload hashes.
An integrity hash without its available payload is not retained evidence.

Record whole-checkout provenance and the dependency identity of each product.
A rendering-only change requires fresh consumer verification; it does not
justify repeating unchanged physical evolution. A changed equation, plan,
background, reference, or metric invalidates all dependent products/decisions.
Implement and test that dependency distinction before relying on it. Never
rewrite an old artifact's identity to claim execution on changed physics.

Persist one completed case at a time with explicit missing/failed/stale
states. Resume only validated unaffected work. After producer exit and normal
test cleanup, reload actual graphs and numerical payloads in a separate
process; reject missing files, tampering, escaped paths, and disconnected
hashes. Successful report generation or verification is distinct from an
accepted scientific result.

## Workflow and closure

Follow AGENTS.md and the active slice. Use the existing branch and managed
`.venv`; do not recreate or upgrade it. Read current repository instructions
again after a repository switch. Inspect a dirty worktree once, preserve
existing work, and keep changes scoped to the authorized slice. Review any
imported staged experiment against its actual diff and evidence first.

Open before edits:

```bash
source .venv/bin/activate && python -m devcovenant gate --open
```

Run the focused tests and actual calculations required by that slice, inspect
summary/tail/log artifacts, and repair blockers in their owning source. Then:

```bash
source .venv/bin/activate && python -m devcovenant gate --verify
```

Stage all completed slice changes and hand off with the gate open. In this
working arrangement the human runs `devcovenant run`; do not run or close it
on the human's behalf without instruction. A subsequent close requires the
fresh successful workflow evidence. Never commit, push, run
`devcovenant check --nofix`, or launch the final demonstration without explicit
instruction for that action.

Record exact focused commands, measured results/costs, artifact identities,
and outstanding human workflow confirmation under the active slice. A slice
may be implementation-complete while awaiting that confirmation. Close it
only when its stated behavioral/scientific conditions and workflow evidence
are satisfied. Do not close an expensive physical obligation with command
wiring or transfer it to the human's later scientific campaign.

A new blocker belongs to the earliest unmet obligation it violates. Later
changes reopen affected earlier evidence rather than repeating unrelated
work. No automated stage may quietly expand to the entire observational
campaign. This plan replacement requires policy verification only; it makes
no claim that the implementation slices below have executed.

## Preserved completed foundation

### Slice One - executable completion contract

**Status:** complete, carried forward.

Keep the frozen 23-case inventory, independent reference conventions, exact
windows/metrics, persistent-evidence contract, and rejection of missing,
synthetic, stale, or tampered evidence. The recorded low-ell rejection is a
baseline, not accepted physics. Reuse current independent references when
their identities still match.

### Slice Two - automatic numerical acceptance

**Status:** complete, carried forward.

Preserve independently evolved history comparisons, source sign/spike
controls, q and hierarchy checks, endpoint coverage, meaningful nested grids,
measured work, and request-subset regressions. New algorithms must satisfy
those behaviors and the combined observable-error budget; do not restart the
slice or assume a passing regression certifies all unseen declarations.

### Slice Three - solver and test ownership

**Status:** complete, as confirmed by the human.

Keep the extracted physics, numerical, diagnostics, and evidence owners,
behavior-named tests, public composition boundary, and identity-safe reuse.
Its ordinary graph evidence preserves behavior but is not quantitative CAMB
parity. Further changes repair measured defects in those owners, without a
second architecture rewrite.

## Slice Four - establish independent, practical development checks

**Status:** implementation-complete on 2026-10-18; awaiting the human's
fresh `devcovenant run` confirmation. **Entry:** the completed foundation
above. Slice Five has not started.
The full-workflow follow-up below records the rejected run and its repairs;
that rejected run is not a full-suite confirmation.

**Owners:** `perturbation_contract.py`, direct runtime test owners,
`tests/project/lib/camb_reference.py`, the existing synthetic fixture,
application tests, and the existing scientific acceptance runner.

**Work:**

1. Map every starting finding and acceptance obligation to a behavior-named
   test and source owner. Inspect other-machine staged work if supplied;
   preserve useful repairs only after reproducing their intended behavior.
2. Establish independently derived continuity/adiabatic-entropy, gauge/sign,
   initial-series, collision conservation/damping, polarization-normalization,
   and unit-conversion regressions. Reproduce the photon metric inconsistency
   and repair its source plus aligned comments/docstrings/docs. Expectations
   must not simply copy strings or formulas from the implementation.
3. Add analytic/manufactured small real evolution and projection cases, with
   corrupted-sign, shifted-phase, spike, tail, and cache negative controls.
   Keep equation correctness distinct from the solver's own residual checks.
4. Freeze compact independent CAMB background/history data with explicit
   gauge, variable, normalization, version, physical-input, and digest maps.
   Compare selected k modes and epochs directly; recurring tests do not
   regenerate full CAMB spectra. Retain an explicit regeneration path.
5. Add actual bounded application behavior tests using independent small
   observations and controlled solver outputs: model roles, covariance row
   order/cross terms, complete requested spectra, typed failures, fixed/free
   dispatch, summary/CSV/plot consumption, and manifest outcome. Keep parser
   trust/hash tests separate from large covariance inversions and sampling.
   Use supported explicit spectrum tables for this initial application
   surface; reproduce and repair the shipped wide-table defect together in
   Slice Nine. Do not leave intentionally failing discovery tests between
   slices.
6. Give each workload above an exact reproducible test command by selecting
   existing modules/methods and moving expensive assertions to their owning
   focused modules where necessary. Measure per-test costs before and after;
   remove demonstrated duplicate work without dropping physical coverage.

**Focused owners:** `test_perturbation_contract`, `test_cmb_output`,
`runtime.test_collisions`, `runtime.test_evolution`,
`runtime.test_mode_evolution`, `runtime.test_spectrum_projection`,
`test_run_pipeline`, `test_run_executor`, `test_likelihoods`,
`test_sampler_mcmc`, and `tests.project.lib.test_camb_reference`.
Select methods explicitly when a broad module still includes costly solves.

**Closure:** independent negative controls detect the named defects; their
resolved controls pass; the photon inconsistency has a derived repair; actual
application composition is covered; and the focused/bounded workload commands
meet their measured targets. This slice establishes a useful development
loop, not final CAMB parity. Record the human's full-suite confirmation.

### Slice Four findings, source owners, and acceptance mapping

No other-machine staged experiments were supplied. Changes were derived
from the current checkout and independently reproduced failures. The named
tests below exist in their indicated owners; pending scientific execution
remains assigned to its original later slice.

1. **Continuity, initial data, and gauge/sign.**
   `perturbation_contract.py` owns the repaired photon monopole sign.
   `test_perturbation_contract` checks
   `test_photon_baryon_adiabatic_entropy_has_no_metric_source` and
   `test_adiabatic_initial_series_has_common_density_and_velocity` against
   independent continuity/regular-series expectations. The wrong sign
   generates `6 Phi'` in adiabatic entropy. Real manufactured RK4 evolution
   in `runtime.test_evolution` rejects a flipped metric drive and shifted
   oscillator phase. `runtime.test_mode_evolution` checks
   `test_generated_stage_hubble_rate_matches_declared_expansion`.
   The generated stage background mixed declared H(a) with a different
   interpolated Hconf; its 0.00563949 inconsistency is now below 1e-12.
   The ordinary superhorizon bound and
   `test_declared_scalar_hierarchy_synchronous_matches_newtonian` pass
   unchanged. `runtime/mode_evolution.py` owns that stage repair.
2. **Collision conservation, damping, and polarization normalization.**
   `runtime.test_collisions` checks
   `test_scalar_thomson_action_conserves_momentum_and_damps_slip` against
   analytic total momentum, exponential slip, and both quadrupole
   eigenmodes, with a reversed-sign negative control. Existing vector
   momentum/damping controls remain in `test_perturbation_contract`.
   `test_polarization_source_matches_frozen_camb_normalization` and
   `runtime.test_spectrum_projection`'s
   `test_integrated_temperature_quadrupole_matches_angular_thomson_source`
   independently reproduced a factor-of-four source deficit. The owning
   materializer now uses 1/4 and 3/4 for temperature and 3/4 for E with
   temperature-normalized Pi. `diagnostics.py`'s closure audit and its
   regression use the same corrected convention. The angular integral,
   native CAMB source, spin-2 kernel, and dimensional checks are independent
   of that audit. Derivations and primary references are in
   `copernican/docs/cmb_solver.md`, under Development Checks.
3. **Reference matching and shared physical parameters.**
   `tests/project/lib/camb_reference.py` owns the new compact fixture's
   exact massless inputs, native gauge/normalization, thermal histories,
   and digest.
   `test_selected_modes_match_frozen_histories_without_angular_spectra`
   compares selected modes/epochs directly, while the source test above
   evaluates actual CCMBS expressions on frozen native quadrupoles.
   `test_fixed_lcdm_parity_rejects_mismatched_physical_inputs` and
   `test_fixed_lcdm_parity_rejects_massive_camb_defaults` reject mismatched
   parameter/default controls. They do not prove matched thermal histories.
   Collapse-source versus tanh reionization, absolute common-gauge histories,
   and temperature/density/neutrino consistency remain Slice Five work in
   model declarations, adapter/compiler, and `runtime/background.py`.
   Existing owner regressions include
   `test_lcdm_declared_cmb_uses_derived_photon_density`,
   `test_planck_reference_separates_cdm_and_massive_neutrinos`,
   `test_sound_horizon_uses_supplied_hubble_curve`, and
   `test_bao_sound_horizon_uses_drag_epoch_helper`. Slice Five must add
   physically matched quantitative comparisons, including drag endpoints;
   these contract tests do not close that obligation.
4. **Analytic projection, source features, support, and units.**
   `runtime/spectrum_projection.py`, `background.py`, and the generated
   sources are exercised by `runtime.test_spectrum_projection`: the
   sine-integral j0 projection, direct angular Thomson integral, spin-2
   origin limit, and constant-transfer logarithmic support integral.
   Wrong signs, a localized spike, and a missing integration tail reject.
   `test_cmb_output`'s
   `test_public_units_follow_temperature_and_potential_dimensions` uses
   independently dimensional TT/TE/EE/BB, PP and TP/EP C/D factors,
   preserving duplicates/order. No amplitude or phase fit decides parity.
5. **Automatic resolution, work growth, and numerical error.**
   The adaptive/convergence owners remain responsible for the 0.5%
   observable budget, physical tails, hierarchy, evolution/start/coupling,
   source representation, k/eta grids, endpoints, and lensing.
   `runtime.test_adaptive` checks
   `test_nested_phase_grid_refines_measured_spacing_after_count_floor`,
   `test_phase_requirements_report_uncapped_physical_resolution`,
   `test_history_convergence_rejects_spikes_between_named_anchors`, and
   `test_history_convergence_rejects_resolved_sign_reversal`.
   `runtime.test_convergence` checks
   `test_q_bound_measures_count_and_both_thermal_tails`,
   `test_q_bound_rejects_insufficient_support`, and
   `test_hierarchy_bound_requires_assumptions_domain_and_error`.
   These now run in the cheap focused command. Actual combined observable
   bounds, radial work growth, and practical cold production resolution
   remain Slice Six work, with real parity evidence in Slice Seven.
6. **Repeated work, exact identities, and diagnostic fidelity.**
   `runtime.test_cache` checks dependency separation, parameter invalidation,
   request accounting, and deterministic warm metrics. The bounded command
   runs `test_declared_spectrum_cache_identity_covers_runtime_inputs`,
   `test_runtime_cache_state_requires_matching_request_shape`, and
   `test_batched_declared_evolution_matches_scalar_modes` in `test_cmb`.
   Its real scalar gauge, metric, and vector refinement cases preserve
   ordinary-route coverage alongside analytic probes. Production histories,
   planner and cache owners still owe measured cold/warm/subset reuse,
   independent new refinement work, and pressure/chunking behavior in
   Slice Six. Reports retain raw executor counters; some projection phase
   counters exceed request wall time. Correct counter attribution belongs
   to Slice Six profiling. Those counters do not decide this slice's cost
   pass: independently measured workload/test wall clocks do.
7. **Exact inventory, surfaces, metrics, and sparse/dense acceptance.**
   `test_diagnostics.CompletionContractTestCase` checks
   `test_frozen_case_inventory_includes_physical_response_points`,
   `test_completion_rejects_synthetic_missing_stale_and_unresolved_rows`,
   `test_surface_metric_rejects_sign_spikes_and_bad_conversion`, and
   `test_raw_units_convert_potential_and_mixed_correlations`.
   `test_camb_reference.CompletionReferenceWindowTestCase` checks dense
   feature-window selection and all frozen physical points. They preserve
   the 23-case/12-comparable-point contract, all eleven applicable angular
   surfaces, raw/public units, signs/zeros, two-ell feature displacement,
   damping windows, and the unchanged 2/3/5% ceilings. The actual dense
   ordinary-route executions, accepted LCDM/Planck graphs, and complete
   payloads remain Slice Seven; the historical sparse handoff is not proof.
8. **Universal equations and active sectors.**
   Compiler/validator, generic background/evolution and projection own
   declaration capability. Existing behavior tests in
   `test_perturbation_contract` and `test_cmb` retain gauge-role validation,
   source/kernel routing, real modified equations/interactions, scalar
   versus batch ordering, nonzero vector/tensor/lensing, and malformed
   contract rejection. The bounded vector refinement test remains real.
   Theory-faithful execution of the five non-CAMB theories and six auxiliary
   cases, including extra physical state and intended initial/boundary laws,
   remains Slice Eight. Neither renaming nor manufactured scalar data
   certifies the unexecuted 23-case scientific matrix.
9. **Application covariance, roles, dispatch, and complete outputs.**
   The existing synthetic solver fixture supplies controlled signed
   TE/TT/EE predictions; independently chosen long-form observations and
   cross-covariance give chi-square 45.4. `test_likelihoods` verifies row
   permutations/cross terms, rejects wrong ordering/missing EE, and
   propagates a typed numerical failure. The actual
   `test_real_pipeline_preserves_free_fixed_roles_and_cmb_outputs` checks
   real free/fixed MCMC, summaries, CSV values/order and CMB plotted curves.
   `test_manifest_runs_real_pipeline_and_separates_failure_provenance`
   invokes the same pipeline from the actual manifest executor and checks
   successful artifacts/provenance versus early numerical failure.
   Sampler fixed-bound, ordered-batch and timeout/incomplete controls remain
   in `test_sampler_mcmc`. Application source owners remain likelihoods,
   samplers, `run_pipeline`, `run_executor`, plotting and CSV/summary writers.
10. **Shipped data and application completion.**
    Slice Four deliberately uses supported explicit spectrum tables.
    Shipped Planck TT-only covariance/wide-table selection and the
    reference-manifest TORG selection remain Slice Nine owner repairs,
    together with parser trust/hash separation, complete/incomplete
    lifecycle, both samplers, GUI/CLI and complete consumer alignment.
    Existing parser/hash and dispatcher tests remain in discovery; the new
    explicit-row application checks give an independent consumer control
    for the later shipped-data regression. No known broken shipped path
    has been relabeled as passing scientific acceptance here.
11. **Persistent evidence, tampering, and freshness.**
    Diagnostics/readers and the existing acceptance runner own artifacts.
    `test_retained_artifacts_reject_tampering_missing_and_escape`,
    `test_stale_reference_and_metadata_only_axes_are_rejected`, and
    `CompletionPersistenceTestCase` reject stale/missing/forged evidence,
    escaped paths and source changes. They run in the focused command.
    Development reports explicitly set `scientific_acceptance: false`.
    Selective physical/comparison/rendering dependency invalidation, resume
    and persistent full graph payloads remain Slice Nine/Seven obligations;
    whole-checkout identity does not yet implement selective reuse.
12. **Broader review and final handoff.**
    Slice Ten retains the whole-project code/docs/comments/docstrings/tests/
    configuration/assets/mirror/performance/architecture review and the
    ordinary LCDM/Planck2018ref 5-burn/10-production handoff. The application
    behavior test above checks the actual free/fixed roles; it does not
    replace the final accepted forward-engine inputs or launch a scientific
    campaign. This slice leaves all later closure criteria unchanged.

### Slice Four reproducible commands and measured cost

From the checkout, run sequentially in the managed CPU environment:

```bash
source .venv/bin/activate
python -m tests.project.lib.scientific_acceptance \
  --development-checks focused
python -m tests.project.lib.scientific_acceptance \
  --development-checks bounded
python -m unittest tests.project.lib.camb_reference_acceptance -v
```

The first command covers independent equations, all named negative controls,
compact selected CAMB histories, numerical/contract controls and bounded
actual application behavior. The second selects real scalar stage/gauge/
metric and vector refinement plus cache/batch checks. The third explicitly
rebuilds angular references; it is outside recurring discovery and its six
original builder/array assertions are preserved. Compact regeneration is:

```bash
source .venv/bin/activate
python -m tests.project.lib.scientific_acceptance \
  --freeze-development-reference
```

Per-test before/after costs and logs are retained under the configured
output root's `ccmbs-acceptance/completion-evidence/`:
`development-before.json`, `development-before.log`,
`development-before.py`, `development-focused.json`,
`development-bounded.json`, their sibling logs, and
`development-cost-comparison.json`. Per-test costs for the explicit six-test
reference rebuild are in `development-reference.json` and its sibling log;
`development-reference-costs.py` reproduces that measurement from the checkout.
The original profiling script preserves
the initial exact unittest module selections. That 183-test timing sample
was recorded after the isolated photon repair, before the remaining new
behavior tests; it is not an unchanged pre-slice scientific baseline.
The final runner's explicit selections make each current workload
reproducible without running the expensive matrix.

Measured on the reference machine (Apple M5, macOS 26.3 arm64, ten CPU cores,
Python 3.11.14, NumPy 1.26.4, SciPy 1.12.0, CAMB 1.6.0; recorded numerical
thread variables unset):

| Measurement | Tests | Wall seconds | Target seconds |
| --- | --- | --- | --- |
| Initial selected-owner timing sample | 183 | 9.397 | Comparison only |
| Final focused development | 255 | 9.194 | 60 |
| Final bounded real physics | 10 | 48.300 | 300 |
| Explicit angular reference rebuild | 6 | 0.886 | Explicit maintenance |

All measured current workload tests passed. Reports record per-test costs,
peak process RSS, environment, source/reference identities, and actual
parent-process request contexts/work counters. Caches are cleared at each
workload's entry; individual tests exercise and may clear their own cold,
warm and invalid states. Cold execution remains present. Moving six angular
reference builder/rebuild checks out of recurring reference discovery and
using a frozen array for the comparator negative control removes measured
duplicate reference work without discarding its assertions. The common-test
cost comparison and moved-test timings are in the retained comparison JSON;
the two different suite totals are not an overall speedup measurement.
The six moved checks accounted for 1.018 seconds in the initial timing
sample. Peak process RSS was 408.69 MiB focused and
1230.41 MiB bounded.
The compact history fixture's SHA-256 is:

```text
4addcd1d844590c9b95f495a5705275c113ef5b2b0f64a57d9ce8e008ad97aad
```

Reproductions retain the original rejected metric-stage and polarization
logs alongside resolved controls under `development-reproductions/`.
The affected broad CMB module also passed all 224 tests in 388.230 seconds
after the metric-stage repair; subsequent polarization changes are covered
by the final focused/bounded checks and source-audit regression. This is
distinct from a fresh complete discovery run.

**Workflow handoff:** stage the complete slice after a green
`gate --verify`, leave the gate open, and await the human's fresh full
`devcovenant run`. The previous pre-slice full run is not this confirmation.
No Slice Five implementation, cold full-parity campaign, gate close, commit
or push is included. Full slice closure remains pending that human result.

### Full-workflow follow-up, 2026-10-18

The human's run `20261018T143051672415Z-run` executed 896 tests in
1257.919 seconds and reported three errors: scalar hierarchy EE refinement,
QRSF diagnostic Einstein reconstruction, and production projection TT
refinement. Its original logs remain under `devcovenant/logs/`; no accepted
workflow result is inferred from the previous focused checks.

The repairs preserve the existing numerical acceptance thresholds:

* The generated momentum combination substitutes its declared Phi'
  relation before evaluation, avoiding subtraction/addition cancellation
  of much larger metric terms. An independent large-dynamic-range test
  reproduced the lost source and now preserves it. QRSF's original small
  declared-surface request then succeeds without widening its constraint
  tolerance or changing the theory.
* Generated scalar temperature projects the angular Thomson quadrupole
  through the existing ordinary and second radial Bessel kernels. It no
  longer twice differentiates sampled visibility/moment histories. The
  direct angular integral now tests the actual compiled source-role dispatch,
  including nonzero endpoint moments. Production coarse/fine projection
  and adaptive transfer projection use the existing positive Simpson rule.
  Source audits, source-role validation, comments, and docs use that same
  direct angular representation.
* Generated scalar exact-collision splitting now caps its phase step at
  0.1 radians. Exact collision exponentials had masked a streaming/splitting
  error at the former 0.5-radian ceiling. A coupled oscillator benchmark
  checks the actual Strang/RK4 integrator against an independent matrix
  exponential, rejects coarse steps, and verifies second-order refinement.
  The original 10/12 hierarchy-depth regression keeps its unchanged 1%
  bound and now also runs in the bounded development selection. No hierarchy
  closure, reference spectrum, source amplitude, or acceptance bound was
  adjusted to obtain that pass.

The broader CMB/source-audit sweep passed 229 checks and found an older
prefix test's invalid minimum-amplitude proxy. That test required at least
a 5% monopole displacement, although no adiabatic law requires such a
minimum. The corrected solution's displacement is 0.0226566. The test now
observes real RHS evaluations before the visible source grid and requires
their initial `k eta < 0.1`; it no longer rewards numerical amplitude drift.
The separate early-start history comparison retains both its 5% median and
maximum error ceilings. Both prefix tests pass together after this correction.

Exact commands used for affected coverage and the original failing owners:

```bash
source .venv/bin/activate
python -m unittest \
  tests.copernican.lib.likelihoods.cmb.test_cmb \
  tests.copernican.lib.likelihoods.cmb.test_contracts_audit -v
cmb_owner=tests.copernican.lib.likelihoods.cmb.test_cmb
cmb_owner+=.CMBCustomRuntimeBehaviorTestCase
prefix_name=test_declared_scalar_adiabatic_
python -m unittest \
  "$cmb_owner.${prefix_name}sources_use_hidden_superhorizon_prefix" \
  "$cmb_owner.${prefix_name}hidden_prefix_tracks_early_start" -v
adapter_owner=tests.copernican.lib.test_model_adapter.ModelInterfaceTestCase
adapter_case=test_declared_cmb_models_execute_finite_declared_surfaces
python -m unittest "$adapter_owner.$adapter_case" \
  tests.project.lib.test_ccmbs_production_graph -v
python -m tests.project.lib.scientific_acceptance \
  --development-checks focused
python -m tests.project.lib.scientific_acceptance \
  --development-checks bounded
```

Follow-up results: the broader sweep ran 230 tests in 787.016 seconds;
229 passed before correcting the prefix proxy described above. The two
prefix regressions then passed together in 8.110 seconds. The original
model-corpus and production-graph regressions passed together in 257.723
seconds. Final focused checks passed all 257 tests in 15.501 seconds against
their 60-second target; bounded checks passed all 11 tests in 110.581
seconds against their 300-second target. Both development reports match
the final tested source identity. The new focused/bounded JSON reports
and sibling logs retain per-test costs, request work, and environment
metadata at the same configured evidence root.
Pre-fix reproducers and the affected-owner logs are retained under
`development-reproductions/`. Earlier timings above remain historical
observations; these follow-up measurements use the repaired source.

The complete slice remains staged with a green policy verify and an open
gate for a fresh human workflow run. No full-suite pass, Slice Five work,
gate close, commit, or push is claimed by this follow-up.

## Slice Five - align physical inputs and locate first disagreement

**Status:** pending. **Entry:** independent development checks pass.

**Owners:** model declarations, adapter/compiler, `runtime/background.py`,
initial data, hierarchy/source owners, and independent reference helpers.

**Work:**

1. Resolve the exact LCDM and fixed Planck reference points, including
   massless versus massive species/distributions, temperatures/densities,
   primordial amplitudes, recombination/reionization, and physical units.
   Audit the corresponding distance and drag-ruler definitions as well.
2. Compare H(a), conformal time, thermal moments, sound horizons, ionization,
   opacity, visibility shape/normalization, and the full reionization history
   against physically matched independent references. Resolve the
   collapse-source/tanh mismatch explicitly. No tau-only matching claim.
3. Follow selected modes from regular initial conditions through horizon
   entry and recombination into metric/species histories and source terms.
   Compare with independent histories in a common gauge and normalization.
   Repair the earliest mismatch before downstream projection or amplitude.
4. Test source decomposition, derivatives, collision transitions, and radial
   kernels independently. Exercise the same repaired behavior through the
   ordinary executor so diagnostic integration differences cannot mask it.
5. Derive and record each correction and its observable consequence using
   bounded reference points and negative controls. Preserve theories' intended
   equations and parameter semantics; do not turn physics into labels that
   select reference-like behavior.

**Focused owners:** `test_model_coder`, `test_model_adapter`,
`test_perturbation_contract`, runtime background/evolution/collision/constraint
and projection tests, and `tests.project.lib.test_camb_reference`.

**Closure:** matched physical histories and selected mode/source comparisons
pass their independent bounds for both starting reference models. Remaining
full-spectrum questions are identified by stage and metric, not hidden by
finer grids. Exact physical inputs and fresh affected regressions are retained.

## Slice Six - make automatic production planning accurate and practical

**Status:** pending. **Entry:** matched component physics is established.

**Owners:** planner, adaptive/convergence controls, background, mode evolution,
projection, caches, numerical work accounting, and their direct tests.

**Work:**

1. Profile representative actual production work before another full matrix:
   compilation, background/calibration, initial data, evolution, projection,
   internal/outer refinements, diagnostics, serialization, memory, and reuse.
   Account for every repeated cold mode and distinguish counts from timings.
2. Resolve scales from the declared background, interaction rates, source
   features, sectors, kernels, and physical support. Fixed standard-cosmology
   estimates may initialize the search; measurements must correct them.
   Use declared roles/distributions, not species-name substrings, to determine
   momentum support and numerical requirements for novel declarations.
3. Allocate the combined observable-error budget and validate its propagation.
   Refine the failing axis/material region, with informative independently
   evolved samples and explicit endpoints. Test oscillatory aliasing,
   localized features, near-zero crossings, and insufficient support.
4. Remove measured duplication between internal checks and outer refinement.
   Retain compatible base histories during their own refinement lifetime,
   rather than relying on global cache capacity. Preserve independent new
   work and every necessary cold/invalid-reuse test; do not merely enlarge
   caches or borrow certificates from different physical/numerical points.
5. Optimize the measured owning kernels and schedule partitions. Where
   justified, separate source-evolution and oscillatory projection grids with
   independently tested interpolation/phase errors and fresh physical
   samples. Preserve full effective projection resolution and the declared
   equations; do not introduce a hidden reduced production path.
6. Measure ordinary cold, compatible warm, exact-repeat, changed-parameter,
   changed-request, subset, and batch behavior. Verify useful work survives
   chunking/cache pressure without changing ordering, failures, or acceptance.
7. Meet the recorded workload targets on the reference machine. Plan and run
   individual exact reference cases as needed to validate cost and numerical
   closure; retain results rather than launching duplicate acceptance jobs.

**Focused owners:** all direct runtime numerical/physics tests, public
`test_cmb` method selectors, cache/performance tests, and the existing
acceptance runner. Keep scalar CPU results as the comparison authority.

**Closure:** ordinary requests select and enforce adequate resolution with
measured/validated bounds, request-subset consistency, and identity-safe reuse.
Actual cold LCDM and Planck requests meet their practical budgets and retain
usable products; errors or runtime blockers are repaired here. Finite output
and a cheap diagnostic path alone cannot close this slice.

## Slice Seven - demonstrate actual matched CAMB parity

**Status:** pending. **Entry:** ordinary execution is accurate and practical.

**Owners:** extracted physics/numerical owners for residual defects, existing
CAMB fixtures, diagnostics, acceptance runner, graph tests, and readers.

**Work:**

1. Extend the existing runner to execute and persist named frozen cases and
   their complete windows. Resolve sparse-only handoff and completion-contract
   differences in that owner. Validate exact requests and references before
   expensive execution; persist each attempt so unaffected work can resume.
2. Execute massless LCDM first through the ordinary route without numerical
   overrides. Require quantitative TT/TE/EE raw/public parity, acoustic
   locations/amplitudes, TE signs/zeros, damping, numerical acceptance, and
   request-subset consistency. Repair first divergences in their owner.
3. Execute fixed Planck2018ref next, with its actual massive-neutrino contract
   and matched physical histories. Keep the two models' curves and identities
   distinct. Extend to the massive-neutrino and dark-energy response points,
   amplitude response, zero/nonzero mass continuity, and common limits.
4. Compare every applicable declared surface, including PP/TP/EP, BB, lensing,
   sector totals, and lensed products. Retain physical zeros with reasons,
   cross signs, covariance bounds, and exact C/D conventions. A successful
   TT/TE/EE row does not certify other declared products.
5. Feed each accepted execution's arrays into real graph, likelihood/export,
   and repeat-cache assertions without separate cold solves per assertion.
   Keep independent observations for plumbing, and persist actual artifacts
   beyond producer exit and ordinary temporary-directory cleanup.

**Focused owners:** `tests.project.lib.scientific_acceptance`,
`test_camb_reference`, `test_ccmbs_production_graph`, CMB diagnostics tests,
public CMB tests, and direct regressions for every repaired owner.

**Closure:** all twelve comparable physical points have actual accepted
matched comparisons for their required surfaces/windows. LCDM and Planck
have durable quantitative ordinary-route graphs. Independent numerical axes,
physical matching, raw products, and reload verification pass. Commands,
mocked reports, and accepted CAMB-versus-CAMB rows cannot close this slice.

## Slice Eight - demonstrate universal declarations and active sectors

**Status:** pending. **Entry:** comparable physical acceptance passes.

**Owners:** validator/compiler/adapter, generic background/hierarchy/source
and projection owners, corpus diagnostics, and existing declaration fixtures.

**Work:**

1. Execute the five non-CAMB bundled theories using their own equations and
   named invariants. Reuse comparable accepted products only under complete
   current dependency identity. Every required bundled row remains present.
2. Execute the six auxiliary cases: unrelated names, changed recombination
   and opacity, extra physical state/interaction, and nonzero vector, tensor,
   and lensing. Verify conservation, initial/boundary conditions, independent
   analytic/manufactured limits, physical support, and refined output.
3. Repair valid grammar/compiler/runtime gaps generically. Do not supply
   missing model mathematics, compare a novel theory with borrowed LCDM
   spectra, manufacture a zero, or classify unfamiliar valid physics as
   unavailable. Keep malformed/singular mathematics negative tests separate.
4. Verify declaration/parameter dependency separation, changed rates and
   distributions, nonstandard clocks, gauge/source/kernel routing, scalar
   versus batch ordering, failure isolation, and cross-theory cache identity.
5. Preserve all applicable raw/public/sector/lensed surfaces and numerical
   axes. Document the demonstrated grammar and tested domain honestly:
   representative coverage supports a general engine, not a theorem that
   arbitrary mathematical systems are well posed or numerically tractable.

**Focused owners:** model validator/coder/adapter and perturbation-contract
modules, runtime source/sector/projection tests, contract audits, diagnostics,
and the existing completion matrix execution/readback paths.

**Closure:** all 23 required physical/auxiliary cases have complete accepted
current evidence, with meaningful nonzero sector execution and no unresolved
engine capability defect. New complete declarations retain automatic
numerical enforcement; successful renaming alone is insufficient.

## Slice Nine - close application integration and final readiness

**Status:** pending. **Entry:** accepted forward physics and universal cases.

**Owners:** CMB API/likelihood/cache boundaries, both samplers, dataset parser
and registry, shared executor/pipeline, GUI/CLI, outputs, evidence, and docs.

**Work:**

1. Remove unconditional `joint_mcmc` convergence deferral. Every proposal
   obtains numerical acceptance or reuses a certificate for its exact
   dependency identity. Verify real forward/proposal and scalar/batch
   equivalence on a small sequence; no long chain is needed for this proof.
2. Preserve domain rejection versus engine/numerical failure and incomplete
   cancellation/timeout. Verify worker and serial paths, partial outputs,
   manifest state, and exported success cannot turn inability to compute into
   an ordinary rejected posterior or a successful incomplete inference.
3. Make Planck spectrum rows, full covariance/cross blocks, likelihood
   selection, diagnostics, graphs, and CSVs agree. Preserve table ordering and
   repeated/noncontiguous ell semantics. Test with small independent data and
   separate real parser/hash checks; do not alter observational tables.
4. Preserve independent SNe intercept treatment and BAO drag-ruler evaluation
   with CMB unavailable. Keep component failure visible and test both sampler
   backends through the actual shared pipeline. Reuse accepted physical
   products for broad output tests with explicit tested identities.
5. Verify fixed Planck evaluation versus free LCDM sampling, priors/transforms,
   model role labels, progress, GUI worker dispatch, summary/NetCDF/CSV/plot
   products, and lifecycle outcomes. Do not describe fixed traces as sampled
   evidence or assume a validation filename implies the selected model.
6. Complete dependency-aware product invalidation and durable readback tests.
   Assemble one final current manifest; rerun affected physical acceptance
   after implementation changes, while retaining verified unaffected work.
   Reopen every raw graph/history/refinement after producer exit and cleanup.
7. Measure full ordinary discovery, affected exact acceptance, nearby proposal
   work, memory, and cold/warm reuse against the cost contract. Resolve
   remaining bottlenecks without reducing accuracy, products, or coverage.
8. Audit code, docs, comments/docstrings, tests, config, managed assets,
   mirrors, consistency, performance, and architecture against this roadmap.
   Reconcile stale numerical recipes, historical slice claims, reference
   descriptions, and documentation audiences through their owning sources.
   Record the human's successful full workflow and required gate evidence.

**Focused owners:** sampler/public CMB tests, SNe/BAO/isolation tests, real
pipeline/executor/worker tests, dataset parsing, comparison/plot/export tests,
persistence/tamper tests, and all numerical cases invalidated by repairs.

**Closure:** inference and application consumers share the accepted numerical
contract, correct datasets and model roles, explicit failures, coherent
outputs, practical cost, and durable current evidence. The complete audit
finds no unresolved requirement; green workflow checks accompany that result
and do not substitute for it.

## Slice Ten - prepare the ordinary LCDM/Planck 5/10 demonstration

**Status:** pending. **Entry:** final engine/application readiness accepted.

**Owners:** existing manifests/configuration, comparison validation, launcher,
validation documentation, bounded application tests, and final evidence.

**Work:**

1. Prepare a concrete supported configuration for `model_lcdm.yml` control and
   `model_ref_planck2018.yml` fixed test, CCMBS, seed, dataset identities and
   hashes, worker/walker settings, output location, and 5 burn-in/10 production
   steps. Use the human's actual configured observational choices; obtain
   missing choices rather than inventing them or silently using TORG.
2. Preserve fixed reference semantics. The free model samples; the reference
   follows its ordinary fixed evaluation path. Include initialization and all
   walker/proposal evaluations when estimating runtime; 15 iterations do not
   mean 15 forward solves. Base estimates on measured accepted work.
3. Validate configuration, model/data loading, parser trust, dispatch, and
   output contracts without starting the human's observational comparison.
   Prove that the preflight cannot start that solve. Existing bounded
   integration tests may consume accepted products and controlled data.
4. Record the exact verified CLI launch command, equivalent GUI selections,
   configuration path, prerequisites, expected output paths, and measured
   runtime estimate. Do not invent a dry-run flag or call preflight execution.
5. State expected observations: two correctly labelled theory curves, complete
   requested spectra, coherent SNe/BAO/CMB summaries, exported canonical
   arrays, numerical/source provenance, and honest complete/incomplete state.
   Define missing curves, stale results, unreported failures, or unexplained
   spikes as failures for follow-up, not a completed demonstration.
6. Hand off the already accepted engine separately from the unexecuted
   demonstration. Launch it only when the human explicitly instructs it.
   Longer inference/scientific campaigns and subsequent polish remain human
   work; no unresolved bounded engine acceptance is deferred to those runs.

**Closure:** the final accepted evidence remains current; configuration and
launch instructions are concrete and validated; the only pending action is
the human's explicit launch of the short observational demonstration.

## Final completion checklist

CCMBS and the application handoff are complete under this roadmap only when:

* independent inexpensive tests catch the demonstrated physics and
  application defects, and ordinary discovery remains meaningfully physical;
* every comparable point has actual matched raw/public CAMB parity and
  quantitative wave/damping evidence through the ordinary engine;
* every bundled/auxiliary declaration and nonzero required sector has
  complete theory-faithful accepted output, without hidden substitutions;
* automatic engine planning meets the combined observable numerical budget
  with measured refinement or validated bounds over its declared domain;
* execution and the development suite meet their measured cost targets,
  identity-safe reuse preserves necessary cold proof, and no costly job is
  repeated merely to test another consumer of identical products;
* samplers, likelihoods, GUI/CLI, manifests, and all result consumers share
  accepted numerical products and correct complete/incomplete semantics;
* retained source/reference identities, raw/refined payloads, graphs, and
  decisions survive process exit and cleanup and reject tampering/staleness;
* the complete repository review, documentation/mirror reconciliation,
  focused acceptance, human full workflow, and required gates are successful;
* the ordinary LCDM/Planck2018ref 5/10 configuration and exact launch handoff
  are ready, with the demonstration honestly recorded as not yet run.

The final report may then state that CCMBS implementation and bounded engine
acceptance are complete and that the short LCDM/Planck2018ref observational
comparison is ready for the human to launch. It must not claim a converged
posterior from ten production steps, universal proof for all mathematics, or
execution of a demonstration that has not run.
