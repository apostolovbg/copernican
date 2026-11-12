# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-11-12
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

**Current state:** Slices One through Five are complete. Their closure
records identify the successful human workflows and tested states. The
repaired Slice Six full workflow passed on 2026-11-08 (1144 tests; 5978.307
seconds), and the human confirmed gate closure. Its ordinary angular
acceptance cost remains unresolved. Slice Seven is in progress. Its Planck
reference mismatch has been independently checked and explicitly repaired;
actual angular parity and the complete durable case matrix remain open.
Slices Eight through Eleven remain pending. Implementation proceeds one
explicitly authorized slice at a time. A green policy gate, a generated report,
or a successful mock cannot close a physical acceptance requirement.

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
* **Reference matching:** the starting LCDM declarations specified a
  collapse-source reionization history; CAMB 1.6.0 uses tanh. Slice Five now
  explicitly declares tanh for the two standard reference models and tests
  their full late ionization histories. Equal `tau` alone is insufficient.
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
* **Diagnostic fidelity (reproduced and repaired in Slice Four):** diagnostic
  collection implicitly selected a different integration method. Integration
  choice is now explicit, and independent refined RK/BDF prefixes exercise
  the small-state error control. Preserve ordinary-route regressions beside
  diagnostic comparisons; neither alone establishes full physical parity.
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

### Bundled-model and convention investigation (2026-11-10)

A read-only initial-point screen compared the distance-route expansion with
CCMBS for every bundled model. The table reports
`100 * (H_CMB / H_distance - 1)` in percent at the stated redshifts:

| Model | z = 2 | z = 1100 |
| --- | ---: | ---: |
| LCDM | 0.000 | 0.000 |
| Planck2018ref | 0.000 | 0.000 |
| LCDM+Mnu | -0.014 | -1.293 |
| wCDM | -0.013 | -0.955 |
| w0wa | -0.013 | -0.955 |
| QAUC | -0.197 | -31.657 |
| QRSF | -0.025 | -14.063 |
| TOG | -0.021 | +0.357 |
| TORG | +0.813 | +0.130 |
| USMF2 | 0.000 | 0.000 |

These are diagnostic values at selected points, not an independent proof of
any model or a completed parameter-space or perturbation-equation audit.
LCDM and Planck2018ref partly share the background route, so agreement there
is not independent evidence. Trace every discrepancy to its intended theory
before choosing an owner or prescribing a correction.

Additional bounded checks identified the following obligations:

* QAUC's sampled total radiation changes its distance expansion but not its
  CMB expansion. Changing its declared recombination redshift changed its BAO
  sound horizon but not the CMB visibility peak. A bounded background check
  also found different sound horizons at the same recombination endpoint;
  establish intended parameter semantics and convergence before correction.
* TORG permits a total radiation density below its photon contribution, while
  its CMB residual is clipped to zero. One allowed point differed by about
  35% in expansion at z=1100. Its BAO and CMB sound-speed equations also use
  different radiation inputs. Establish the theory and valid domain rather
  than silently clipping or applying an arbitrary universal constraint.
* wCDM and w0wa allow curvature, but their generated BAO and luminosity
  distances use a radial integral as the transverse distance. At z=2 and
  Omega_k=0.1 the diagnostic discrepancy was about 2%; verify the intended
  geometry independently and repair the owning distance route.
* LCDM+Mnu's zero and 1e-8 eV neutrino mass cases differed in expansion by
  about 0.84% at z=1100. Audit the shared massive and massless normalization
  paths and the other declarations using them. A finite-result test alone
  does not establish continuity.
* Consistently renaming QRSF normalization symbols changed the q-resolved
  background. Replace name-dependent physics with declared dependencies.
  A valid phase-space temperature symbol in `model_parameters` compiled but
  was unavailable during background evaluation; propagate it consistently.
* The observable path currently uses a power-law primordial spectrum. Review
  what the convention can express without assuming this suffices for every
  intended theory. A necessary convention change requires the human's explicit
  decision before implementation.

Do not infer that independent photon density and temperature are always wrong
or replace each theory's physics with LCDM. Slice Seven repairs the affecting
comparable-case prerequisites; Slice Eight owns the full corpus and
convention review before the universal execution work in Slice Nine.

## Test workloads and practical cost contract

The deterministic forward solver and the stochastic sampler need different
proofs. A full chain is unnecessary to detect a continuity error, a bad
collision block, an incorrect likelihood vector, or a stale numerical cache.
Use the following workloads, each with an explicit claim and reproducible
command. Extend existing test owners and the existing acceptance runner.

| Workload | Required evidence | Historical cost target |
| --- | --- | --- |
| Focused development | Independent math and app behavior | 60 s |
| Bounded real physics | Histories, kernels, refinement | 5 min |
| Ordinary discovery | Broad tests and production smoke | 15 min |
| Cold LCDM acceptance | Full frozen physical parity request | 15 min |
| Cold Planck acceptance | Matched massive-neutrino point | 30 min |

The human has relaxed fixed time limits while retaining the requirement for
reasonable execution times. The numbers above preserve the original
reference-machine optimization targets, not active deadlines or scientific
validity cutoffs. Record hardware, interpreter, numerical libraries, thread
settings, cold/warm state, per-test and per-phase timings, peak memory when
material, and actual evolved/projected work. Estimate remaining cases from
measured component work before launching them; identify the estimate's scope
and uncertainty instead of promising an unmeasured total runtime.

Correctness and performance have separate outcomes. Passing a clock cannot
excuse a failed assertion; exceeding an old target cannot reject valid
physics or alone prevent slice closure. Unreasonable recurring cost still
requires profiling and a repair in its owning slice. Do not repeatedly
launch a multi-hour matrix to discover the same early defect. Preserve
negative controls, independent cold/refinement work, and all physical and
numerical thresholds while removing demonstrated duplicate work.

Slice Five removed the development runner's time-only failure: passing tests
return success even above the historical 60/300-second targets. Actual
failures, skips, empty selections and rejecting controls retain their
semantics, while timing remains telemetry. Its tests and documentation now
agree with that behavior.
Slice Six owns measured solver/suite performance improvements; Slice Ten
verifies final application and suite cost. A completed Slice Four does not
claim that the full discovery suite is already reasonably fast on Intel.

Ordinary discovery must keep real solver composition and adversarial coverage.
Scientific acceptance remains explicit and outside `devcovenant run` and
ordinary unittest discovery. Keep expensive exact cases in their explicit
execution owners, retaining assertions, commands and actual current evidence.
Do not hide
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
campaign. The 2026-10-21 amendment records the already tested Slice Four
implementation and reconciles the remaining roadmap. It changes no engine,
test, reference, metric, or acceptance payload; its policy verification is
separate from the successful implementation workflow recorded below.

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

**Status:** complete, closed on 2026-10-21 after the human's successful
full workflow and gate close. **Entry:** the completed foundation above.
Slice Five has not started. The dated follow-ups retain rejected attempts,
their repairs, focused evidence, and the final successful closure record.

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
   Slice Ten. Do not leave intentionally failing discovery tests between
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
application composition is covered; and focused/bounded commands retain
their passing behavioral evidence and measured hardware-specific costs.
The successful full-suite run and gate close are recorded below. Historical
time targets are no longer hard closure limits. This slice establishes the
development checks; final CAMB parity and production/suite performance remain
explicit later obligations.

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
   Slice Six. The October 20-21 follow-up also covers explicit diagnostic
   integration, tiny-state scaling, batch damping writeback, compatible
   schedules, once-only evolution, and refined RK/BDF physical prefixes.
   Reports retain raw executor counters; some projection phase
   counters exceed request wall time. Correct counter attribution belongs
   to Slice Six profiling. Measure workload/test wall costs independently;
   raw counters alone cannot establish either practical cost or correctness.
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
   remains Slice Nine. Neither renaming nor manufactured scalar data
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
    reference-manifest TORG selection remain Slice Ten owner repairs,
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
    Dependency validation, safe resume and persistent physical/graph
    payloads are prerequisites within Slice Seven, before relying on reused
    acceptance cases. Slice Ten extends that distinction to final consumer
    and rendering invalidation. Whole-checkout identity alone does not yet
    implement selective reuse.
12. **Broader review and final handoff.**
    Slice Ten retains the whole-project code/docs/comments/docstrings/tests/
    configuration/assets/mirror/performance/architecture review. Slice Eleven
    prepares the ordinary LCDM/Planck2018ref 5-burn/10-production handoff.
    The application behavior test above checks the actual free/fixed roles;
    it does not
    replace the final accepted forward-engine inputs or launch a scientific
    campaign. The later physical thresholds and case inventory remain intact.

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
The runner's explicit selections make each workload
reproducible without running the expensive matrix.

Historical implementation measurements on 2026-10-18 used the reference
machine (Apple M5, macOS 26.3 arm64, ten CPU cores,
Python 3.11.14, NumPy 1.26.4, SciPy 1.12.0, CAMB 1.6.0; recorded numerical
thread variables unset):

| Measurement | Tests | Wall seconds | Historical target seconds |
| --- | --- | --- | --- |
| Initial selected-owner timing sample | 183 | 9.397 | Comparison only |
| Initial implementation focused development | 255 | 9.194 | 60 |
| Initial implementation bounded real physics | 10 | 48.300 | 300 |
| Explicit angular reference rebuild | 6 | 0.886 | Explicit maintenance |

All tests in those measured workloads passed at their recorded source state.
Those reports are not new measurements of the October 21 implementation.
Reports record per-test costs, peak process RSS, environment,
source/reference identities, and actual
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

**Historical implementation handoff, 2026-10-18:** the slice was staged
after policy verification and handed to the human for a fresh full workflow.
The dated follow-ups below supersede that pending state. No Slice Five work
or cold full-parity campaign was included in that handoff.

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
their October 18 tested source identity, not the later October 21 repairs.
The then-current focused/bounded JSON reports
and sibling logs retain per-test costs, request work, and environment
metadata at the same configured evidence root.
Pre-fix reproducers and the affected-owner logs are retained under
`development-reproductions/`. Earlier timings above remain historical
observations; these follow-up measurements use the repaired source.

At that follow-up's handoff the slice remained staged with a green policy
verify and an open gate for a fresh human workflow run. That historical
state did not claim a full-suite pass or gate close. The closure record below
supersedes it; Slice Five remains unstarted.

### Intel numerical and workflow follow-up, 2026-10-20 and 2026-10-21

Run `20261020T140907362173Z-run` exposed a regular-mode failure on the Intel
checkout: the early scalar metric Phi was -0.3903786167, violating the
existing positive regular-mode bound. The previously passing M5 checks did
not establish portability of that numerical path. Reproduction identified
an implicit diagnostic switch to BDF and raw absolute error control that
under-resolved tiny shear/velocity states with significant metric coupling.

The owning `runtime/mode_evolution.py` repairs preserve the declared graph:

* Diagnostic collection no longer changes integration selection. The
  explicit `continuous_collision_solver` accuracy control selects supported
  coupled BDF integration; scalar and batch capability checks agree on it.
* BDF evolves normalized coordinates derived from initial amplitudes and
  physical drives. Scale propagation includes initially zero, indirectly
  driven moments; a centered numerical Jacobian uses the same normalized
  coordinates. Histories return in physical units. There is no Intel-only
  branch or fixed replacement tolerance chosen for this one request.
* Batched high-moment collision damping writes the updated selected moments
  back into the accepted state array. An independent exponential-law test
  checks selected moments and preservation of all other entries.
* Schedules include both streaming and conformal-expansion requirements.
  Small generated batches retain identical scalar schedules, including
  one-row batches; larger requests retain bounded schedule compatibility.
  Cache identity covers the full evolution grid and expansion history.
* Batch interval-boundary projections match the scalar implementation.
  Accepted batch histories are retained without unconditional scalar replay.
  Independently computed scalar/batch spectra retain their existing
  comparison tolerances; neither result is replaced just to obtain a pass.

Run `20261021T034248903198Z-run` then passed all other tests and exposed a
stale assertion requiring exactly one batch. Its request now correctly
partitions 18 modes into 17 compatible batches. The corrected regression
does not hardcode that partition count: it requires one contract/model/
request preparation, positive bounded batch count equal to schedule groups,
complete once-only mode evolution, zero small-request schedule inflation,
zero corrective scalar replay, finite spectra, and exact cached repetition.

Direct regression ownership is `runtime.test_mode_evolution` and the public
`test_cmb.CMBCustomRuntimeBehaviorTestCase`. The analytic tests verify unit
changes across 42 orders of magnitude, zero and indirectly driven states,
the normalized RHS/Jacobian, and exact damping. The real physical prefix
test runs the compiled hierarchy and collision operators through the first
visible source epoch, independently refines RK phase steps and BDF error
tolerances, and checks their regular-mode agreement. It stops after that
measured prefix, before unnecessary late evolution and projection. This
does not replace the full ordinary-route regular-mode regression, whose
positive sign and minimum-amplitude assertions remain unchanged.

Reproduce the affected checks without launching full discovery:

```bash
source .venv/bin/activate
runtime_owner=tests.copernican.lib.likelihoods.cmb.runtime
mode_owner=$runtime_owner.test_mode_evolution.ModeEvolutionTestCase
cmb_owner=tests.copernican.lib.likelihoods.cmb.test_cmb
cmb_owner+=.CMBCustomRuntimeBehaviorTestCase
python -m unittest -v \
  "$mode_owner.test_batched_damping_updates_selected_accepted_moments" \
  "$mode_owner.test_scalar_early_metric_agrees_between_refined_rk_and_bdf" \
  "$runtime_owner.test_execution" \
  "$cmb_owner.test_generated_scalar_superhorizon_metric_seed_stays_regular"
callback_case=test_continuous_callbacks_match_independent_normalized_generator
python -m unittest -v \
  "$mode_owner.test_continuous_history_is_invariant_under_state_unit_changes" \
  "$mode_owner.$callback_case" \
  "$mode_owner.test_continuous_history_scales_initially_zero_driven_states" \
  "$mode_owner.test_continuous_history_resolves_indirectly_driven_tiny_states"
python -m unittest -v \
  "$cmb_owner.test_batched_declared_evolution_matches_scalar_modes" \
  "$cmb_owner.test_declared_generated_modes_use_declared_graph_evolution"
```

The ordinary bounded development selection already includes the direct
`runtime.test_mode_evolution` module, the full regular-mode test and the
scalar/batch spectrum comparison. The explicit commands above also isolate
the updated generated-mode accounting test. Do not rerun them merely to
rewrite this closure record; the successful full workflow below includes
all of their test owners.

Recorded local environment: macOS 26.2 x86_64, Python 3.11.14,
NumPy 1.26.4 and SciPy 1.12.0 with OpenBLAS. The following measurements
overlap; they must not be summed or compared with M5 totals as a speedup:

| Measurement | Tests | Wall seconds | Outcome |
| --- | --- | --- | --- |
| Damping, RK/BDF prefix, execution and ordinary metric | 8 | 159.706 | Pass |
| Scalar/batch spectrum equivalence | 1 | 141.956 | Pass |
| Generated-mode accounting and exact-repeat cache | 1 | 79.453 | Pass |
| Final ordinary discovery | 897 | 8360.995 | Pass |

The complete workflow command wall time was 8366.539 seconds. Full discovery
is still about two hours nineteen minutes on this machine. Slice Four's
closure records its behavioral success, not a claim that this recurring
cost is satisfactory. Slice Six must profile the measured test/solver work
and remove duplication while preserving coverage; Slice Ten verifies the
final suite cost. Earlier M5 workload reports remain historical evidence
for their own source identities; no new current-source M5 run is claimed.

### Slice Four closure evidence, 2026-10-21

The human confirmed the successful full workflow and gate close. Their
repository-relative artifact paths are:

* Policy verification:
  `devcovenant/logs/20261021T144013477213Z-gate/summary.txt`.
* Full workflow, exit code 0:
  `devcovenant/logs/20261021T144348665452Z-run/summary.txt`.
  Its `tail.txt` records all 897 tests passing in 8360.995 seconds;
  `workflow_profile.txt` records the full command wall time.
* Gate close, exit code 0:
  `devcovenant/logs/20261021T174128595777Z-gate/summary.txt`.

These records describe the implementation tested before this documentation
amendment, including its then-uncommitted runtime/test/doc changes. Preserve
that provenance; do not relabel old reports as executions of a later source
state. The amendment's new policy verification records only its own checks.
Local logs are evidence references, not a promise that ignored artifacts are
available in every clone. Retain the measured outcomes here independently of
local runtime-registry availability; do not fabricate replacement workflow
or scientific records.

Slice Four is closed for independent development checks, repaired numerical
regressions and bounded application composition. At that handoff Slice Five
was in progress; its closure below supersedes that checkpoint. Slices Six
through Ten remain pending. Full matched CCMBS/CAMB histories and
dense spectra, the 23-case accepted scientific matrix, automatic production
performance and the final application/demo handoff have not been certified
by this run.

## Slice Five - align physical inputs and locate first disagreement

**Status:** closed (2026-10-25). **Entry:** Slice Four is closed and its
independent development checks pass. No new full discovery run is required
merely to begin bounded diagnosis from the recorded successful
implementation state.

**Owners:** model declarations, adapter/compiler, `runtime/background.py`,
initial data, hierarchy/source owners, independent reference helpers, and
the existing development runner and its direct tests/documentation.

**Work:**

The development runner's legacy time-only failure is removed. Passing tests
return success even above historical cost targets; actual failures, skips,
empty selections and negative controls retain their semantics. Timing and
performance-target telemetry remain in reports. Scientific acceptance stays
outside ordinary discovery and `devcovenant run`.

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
   Preserve the tiny-state scaling and independently refined regular-prefix
   controls. Check actual state histories before an expensive spectrum solve.
4. Test source decomposition, derivatives, collision transitions, and radial
   kernels independently. Exercise the same repaired behavior through the
   ordinary executor so diagnostic integration differences cannot mask it.
   Collecting diagnostics must retain the selected integration method;
   explicit alternative methods need separate refined physical comparisons.
5. Derive and record each correction and its observable consequence using
   bounded reference points and negative controls. Preserve theories' intended
   equations and parameter semantics; do not turn physics into labels that
   select reference-like behavior.

**Focused owners:** `test_model_coder`, `test_model_adapter`,
`test_perturbation_contract`, runtime background/evolution/collision/constraint
and projection tests, and `tests.project.lib.test_camb_reference`.
Include `tests.project.lib.test_scientific_acceptance` for runner outcomes.

**Closure:** matched physical histories and selected mode/source comparisons
pass their independent bounds for both starting reference models. Remaining
full-spectrum questions are identified by stage and metric, not hidden by
finer grids. Exact physical inputs and fresh affected regressions are retained.
Development command correctness is independent of elapsed time, with costs
reported separately. No later physical ceiling is relaxed to obtain closure.

### Implementation evidence retained so far

The human approved explicit tanh histories for `model_lcdm.yml` and
`model_ref_planck2018.yml`, with temperature as the authoritative blackbody
input. Both declarations now fix 2.7255 K and derive photon density from
Planck's law and H0. Their independent photon-density parameters are removed.
Distance, sound-horizon and CMB helpers now consume the same native
background through `geometry: cmb.background`. Alternative theories retain
their declared radiation and rate laws;
there is no model-label selection or hidden reference substitution.

The tanh hydrogen width is 0.5 in redshift, with second helium midpoint
3.5 and width 0.4. The declared midpoint is calibrated to tau=0.054 and
retained in background evidence. A generic total-electron-fraction
expression supports this prescription; simultaneous rate/fraction inputs
and unphysical fractions are rejected. Rate-based collapse histories remain
supported. Independent CAMB checks compare the complete late ionization
and opacity histories over redshifts 0--25 for both actual physical points,
not merely their integrated optical depths. Electron fractions have a 0.01
absolute bound; opacity has a 1% relative bound where reference x_e>0.02.
These checks do not certify precise residual-ionization thermal histories.

The drag owner now integrates opacity/R, with R=3*rho_b/(4*rho_gamma),
through the complete remaining recombination tail. The former inverse
loading and visibility-peak truncation were both wrong. Independent drag
rulers are checked within 0.2% and drag redshifts within 0.5% at both points.
Equivalent scale-factor/redshift ionization laws now agree because endpoint
and midpoint evaluations receive their actual stage redshifts.

The tight-coupling regime now compares opacity with both streaming and
expansion rates. A streaming-only decision kept long wavelength E
polarization on instantaneous equilibrium after decoupling. Direct regime
checks and an independent-opacity scalar/batch polarization control isolate
this repair. The latter retains a 2% E-quadrupole bound and rejects the old
criterion; its supplied reference opacity is explicit test input, not
production evidence or a certification of native recombination.

The independent helper converts CAMB's native synchronous densities and
velocities to the Newtonian state convention, with an explicit fixed
curvature sign and no fitted amplitude. No frozen reference was regenerated.

The initial physical-input checkpoint did not close this slice. Its
exploratory native long-wavelength quadrupoles still differed by roughly
3.6%/2.3%, and the controlled-opacity test did not certify native thermal
physics. The following native closure work supersedes that pending state;
no historical report is relabeled as a later source execution.

### Focused verification of the physical-input changes

The 80-test focused physics/reference/background/evolution suite passed in
687.883 seconds on this checkout, including the prior scalar-seed and
scalar/batch/accounting regressions with their existing assertions.
The 27 prior, contract and automatic-planner consumer checks passed in
4.447 seconds. Subsequent owning checks passed for reference temperature
(7 tests), history/provenance (17 tests), and independent photon inputs
(1 test). Repeated owner checks are not additional scientific cases.

Logs and a source-hashed development manifest are retained under the
configured output root at
`ccmbs-acceptance/completion-evidence/physical-inputs-20261023/`.
The manifest records actual unittest IDs and per-group counts/times. These
are focused development results, with `scientific_acceptance=false`.
No full discovery or scientific acceptance was launched for this handoff.


### Native closure repairs and acceptance, 2026-10-23

The remaining work was executed against the actual LCDM and Planck model
files, not the reduced synthetic declaration or supplied reference opacity.

1. **One expansion and ruler source.** `model_coder.py` supports the explicit
   `geometry: cmb.background` declaration. Both standard models use it for
   H(z), distances, recombination/drag rulers and drag redshift. Their
   duplicate approximate H(z)/sound-horizon expressions and independent
   recombination-redshift parameters are removed. Scalar, ordered, permuted
   and parameter-rebound expansion calls retain the same native density
   normalization; the final requested epoch cannot stand in for today.
   Pickling and both physical ruler endpoints are covered by owner tests.
2. **Native atomic and gas evolution.** `runtime/recombination.py` implements
   an explicitly selected effective three-level H/He prescription, including
   helium escape/continuum/triplet channels, hydrogen escape corrections and
   coupled Compton gas-temperature evolution. It uses the declared H(a),
   atomic hydrogen mass and helium/hydrogen number ratio. Published effective
   atomic coefficients are physical prescription inputs, not fitted spectrum
   corrections. Custom rate/role declarations retain their separate routes.
   Native matter temperature is retained and used in baryon pressure.
3. **Actual kinetic Planck neutrinos.** Automatic momentum-grid naming now
   resolves before scalar graph materialization. Previously the planner
   supplied a grid while the graph silently omitted its kinetic states.
   Generated quadrature sums obey the existing parser size/depth limits.
   Physical velocity, anisotropic stress, Einstein weights and regular q-bin
   seeds obey the relativistic moment identities. The explicit 3.044 heating
   convention matches the reference's three degenerate masses totaling
   0.06 eV. The Fermi-Dirac density keeps kinetic energy in addition to the
   declared rest-mass density; treating rest density as total energy had
   suppressed early radiation. Independent quadrature expectations now
   normalize to rest energy, including the very light-mass anchor.
4. **Stable metric evolution with independent constraints.** The generated
   scalar graph advances Phi using Einstein 00, avoiding the 0i velocity
   division by k squared. The momentum equation remains an independent
   diagnostic. The initial photon-density correction forms 00 using that
   independent momentum source, so it closes both constraints while retaining
   the declared metric seed. Source graph validation, source derivative
   audits, comments and cancellation controls follow this equation choice.
   Wrong velocity data still reject through the momentum residual. No
   post-processed metric or reference histories decide native parity.
5. **Fresh selected histories and sources.** The explicit test-owned
   `native_history_acceptance` runner exercises the actual executor before
   angular projection, with ordinary collision integration and source
   reconstruction disabled. It compares two k values at 105 epochs from
   eta=10 through 1100 Mpc, integrating the early prefix and continuing to the
   test's eta<=1250 Mpc boundary. Both metric potentials, baryon/CDM density
   and velocity, photon monopole/quadrupole, E quadrupole and active neutrino
   density are compared in one gauge and fixed curvature normalization.
   Native E/Weyl and temperature monopole, quadrupole, Doppler and ISW sources
   have independent comparisons. Massive density uses CAMB's own background
   equation of state for the gauge shift. No amplitude or phase is fitted.

The independent native background checks cover H(a), conformal time,
ionization, opacity, visibility, gas temperature, late tanh reionization and
rulers at both points, with a doubled background grid. Fixed bounds include
0.02% expansion, 0.05% conformal time, 0.1% recombination electron fraction,
0.5% gas temperature/visibility, and the previously retained late-ionization
and drag bounds. Native modes/sources retain a fixed 2% peak-normalized bound;
independent finer temporal/hierarchy controls must change each measured
component by <=0.5%. Time is recorded as telemetry.

The fresh native component command passed with `accepted=true`,
`refinement_accepted=true`, and `unchanged_sources=true`, in 360.838 seconds.
Its actual scalar state counts grew from 89 to 169 for LCDM and from 1145 to
2249 for Planck, demonstrating that the finer graphs were executed.

| Native comparison | LCDM | Planck2018 reference |
| --- | --- | --- |
| Largest finer component error | 1.218% E2 | 0.417% E2 |
| Largest finer source discrepancy | 0.132% Weyl | 0.141% Doppler |
| Largest coarse/fine change | 0.187% massless density | 0.218% Doppler |

These are selected component measurements; dense angular spectra and the
full 23-case scientific matrix remain Slices Seven through Nine work. Passing
the component check does not certify late-time free streaming, full support,
lensing remapping or combined production error budgets.
Slice Six must establish those numerical/performance obligations using the
repaired physical graph, before the unchanged later angular ceilings apply.

The native command is reproducible from the checkout:

```bash
source .venv/bin/activate
python -m tests.project.lib.native_history_acceptance \
  --refine --output ../ccmbs-native-components.json
```

Choose a report destination outside the checkout. Scientific/component
acceptance stays outside ordinary discovery and `devcovenant run`. Its cheap
verdict tests reject incomplete/nonfinite arrays, spikes, forged zero-error
claims, changed physical inputs and insufficient refinement agreement.
The actual native report, with full sampled arrays, physical inputs,
refinement results, source identity and runner digest, is retained under the
configured output root at
`ccmbs-acceptance/completion-evidence/native-components/` as
`native-components-20261023.json`. Kernel source identity is
`afac36a692ff58cea87f2e8f55d22fe2dca89c13a38a30c572210e45bc3ed876`.
The preceding 354.445-second accepted measurement is preserved as
`native-components-20261023-before-policy.json`; it is not relabeled with
the final source identity. Earlier diagnostic probes are not closure evidence.

Affected verification also passed: 157 focused compiler, geometry, atomic,
evolution, collision, constraint, background, verdict and physical-binding
checks in 94.507 seconds; 30 native reference, runner, refined RK/BDF-prefix
and preserved-metric controls in 197.033 seconds; and the corrected
scalar/batch regression in 158.660 seconds. Nine spawn/callback/binding
follow-ups and two ruler/verdict checks also passed. The massive-density
regression now checks the kinetic correction against an independent
Fermi-Dirac integral rather than incorrectly equating total and rest energy.
An earlier runtime-owner selection exposed the initial
projection's dependence on the old metric equation; that defect was fixed
and the unchanged failing regression then passed. Repeated tests and
counterfactual diagnostic probes are not additional scientific cases.
The final `verification-20261023.json` manifest records retained logs and
reproducible commands without claiming a new full discovery or DevCovenant
workflow run.

### 2026-10-24 full-run regression repairs

The first subsequent human workflow ran 908 tests and failed with four
failures and three errors. A focused local reproduction retained exactly
those seven failures in 547.847 seconds. The October 23 focused evidence
above did not cover these failing assertions and did not establish a green
full workflow.

Velocity-isocurvature initialization had become circular after the metric
energy-equation change: its photon density seed depended on the density it
was calculating. It now uses the independent momentum source. A state-free
initialization matrix checks all five scalar families in all three gauges,
with and without massive neutrinos. A direct compiler regression checks
that this seed retains the independent source.

The bundled graph audit now uses the runtime's generated graph validator
instead of requiring the old momentum-driven metric equation. Six missing
energy-input controls must fail in both audit and runtime validation. The
repository integrity audit therefore retains its declaration checks.
Massive-neutrino regressions independently integrate density, pressure,
momentum and shear across four epochs; public velocity and shear normalize
by rho plus pressure, distinct from normalized q moments. BOSS comparisons
use the native baryon drag ruler, retaining the original chi-square and
residual bounds and checking the covariance quadratic form independently.

The refreshed native component and refinement command passed in 383.069
seconds with unchanged sources. Its report is
`native-components-20261024.json` in the same configured evidence directory.
The accepted kernel identity is
`1020ff5d1144144d9f652eab2bdea2280201ace67cdf940ba36c77214ac30fd9`.
The previous dated reports remain intact. This is bounded native component
evidence, separate from full angular scientific acceptance.

The affected regression selection ran 124 tests in 763.465 seconds: 123
passed, while a further obsolete dipole-seed assertion was exposed behind
the original q-materializer failure. That assertion now evaluates the
regular Fermi-Dirac dipole numerically across momenta and both potential
signs; its complete regression passed on rerun in 0.088 seconds. The final
state-free matrix and rejecting audit controls also passed after policy
formatting. All seven originally failing cases therefore have passing
results. Import-only discovery collected 909 tests without errors.
`verification-20261024.json` retains the failed selection, successful
reruns, latest per-test verdict, current component evidence and gate result;
it does not relabel the selection as a single green full workflow.

### 2026-10-25 final closure after the successful full workflow

The human's subsequent `devcovenant run` passed with exit code 0. Ordinary
discovery executed 909 tests in 12,740.633 seconds and reported `OK`.
The complete workflow took 12,754.66 seconds; its successful result was
recorded on 2026-10-24 at 20:25 UTC. The retained run directory is
`devcovenant/logs/20261024T165231254579Z-run/`, containing `summary.txt`,
`summary.json`, `tail.txt`, full logs and workflow profiling artifacts.

Final Slice Five closure is confirmed on 2026-10-25 using that passing full
workflow together with the source-bound native component and refinement
acceptance recorded above. The October 23 focused handoff, October 24 failed
workflow and subsequent repairs remain historical evidence; none is
relabeled as the successful full run. The closure date at this slice's
heading supersedes the premature October 23 status.

The measured full-suite cost is approximately 3.5 hours on this machine.
Correctness passed, but that cost is the repaired-graph baseline for Slice
Six profiling and reuse work. No physical or test threshold is relaxed.
Dense angular spectra, the full scientific matrix and later production
obligations remain in their existing slices; this workflow pass does not
claim that those later tasks are complete.

## Slice Six - make automatic production planning accurate and practical

**Status:** open for ordinary angular acceptance cost; repaired full workflow
passed on 2026-11-08. The earlier closure claim below is superseded.
**Entry:** matched component
physics and the repaired full-discovery workflow both pass.

Use the accepted native component report above as the physical baseline.
Profile the restored q-resolved Planck graph and coupled atomic background;
old Planck timings from the omitted kinetic graph are not a valid cold-cost
baseline. Include shared distance/ruler background reuse, scalar/batch
execution and compilation costs. Keep the independent momentum constraint
and native source histories intact while optimizing. The bounded component
command is available for early failure diagnosis before angular work.

Use the successful Slice Five run above as the current full-suite cost
baseline: 909 tests in 12,740.633 seconds on this machine. Its retained
workflow profiles supplement the earlier Slice Four timing evidence.
Measure solver work and test orchestration separately before removing
duplication; retain necessary independent cold and refinement checks.

**Owners:** planner, adaptive/convergence controls, background, mode evolution,
projection, caches, numerical work accounting, and their direct tests.

**Work:**

1. Profile representative actual production work before another full matrix:
   compilation, background/calibration, initial data, evolution, projection,
   internal/outer refinements, diagnostics, serialization, memory, and reuse.
   Account for every repeated cold mode and distinguish counts from timings.
   Include schedule fragmentation, repeated graph preparation, collision
   work, diagnostic overhead and duplicate test solves. Profile the Intel
   discovery baselines recorded under Slices Four and Five as well as forward
   requests.
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
   Preserve scaled-state and collision-damping controls, boundary projection
   semantics and scalar/batch parity. A fixed batch count is not acceptance;
   compatible grouping must justify its accuracy and measured work savings.
6. Measure ordinary cold, compatible warm, exact-repeat, changed-parameter,
   changed-request, subset, and batch behavior. Verify useful work survives
   chunking/cache pressure without changing ordering, failures, or acceptance.
7. Demonstrate reasonable measured workload costs on each recorded machine;
   historical targets guide optimization without imposing time-only failure.
   Plan and run individual exact reference cases as needed to validate cost
   and numerical closure; retain results instead of duplicating acceptance.
8. Repair heartbeat attribution and remaining-time estimates. Distinguish
   phase, subtask and request elapsed time; count actual scheduled work once
   and update denominators when adaptive work grows. Base estimates on
   measured weighted work for the stated scope, not RK interval count or a
   previous phase's elapsed time. Mark unresolved future adaptive work as
   unknown instead of presenting a phase estimate as total request time.
   Test changing totals, unequal interval costs, batching, cache hits and
   phase transitions; completed work must not exceed its stated total.

### Measured baseline and work in progress, 2026-10-27

The unmodified ordinary public LCDM request for ell=(2,10,20), TT/TE/EE,
completed under cProfile in 532.556 seconds, including its production
refinement. It evolved 512 modes and then a 1024-mode refinement on 3072
source samples. The native Planck hierarchy/source prefix completed under
cProfile in 169.332 seconds. The two profiles ran concurrently; these are
instrumented baseline costs, not uninstrumented speedup measurements.
Raw products, source identities and cumulative profiles are retained under
the configured output root in
`ccmbs-acceptance/completion-evidence/numerical-planning-20261027/`.

The Planck profile identifies repeated condition-symbol set construction,
scalar state-context binding, and recursive cache payload accounting as
avoidable Python work. Kernel changes remove those repetitions without
changing their equations or scalar conversions. Request-owned raw source
and selected native/reference histories now survive nested refinements and
global LRU pressure, with bounded resident arrays and checked spill files.
Independent comparisons are recomputed; certificates are not retained there.

The actual-batch checks exposed false earlier refinement evidence. The hidden
early prefix consumed the late-time sampling controls, and reference meshes
could overwrite the shared fine grid. The repaired grids add that prefix,
resolve measured opacity/expansion/visibility features and actual Fourier
phase, retain rate transitions, and keep independent controls distinct.
Selected native states and their compiled sources are checked before the
remaining modes are scheduled. Unresolved engine-owned evolution controls
are promoted; measured stagnation remains a failure.

An analytic forced-damping regression demonstrates why exact collision
exponentials alone can be stable and inaccurate after tight coupling ends.
Scalar and batch schedules now resolve that collision phase. Native states
use smooth shape-preserving Hermite sampling before source differentiation;
this preserves the evolved metric instead of substituting an algebraic one.
Per-state amplitude normalization retains tiny states independently, applies
each absolute allowance to its own samples, and reports local zero-crossing
errors separately. State and physical-source checks remain independent.

Phase accounting no longer adds repeated projection prefixes or nested
intervals twice. Outer refinement shares the request clock and contributes
its timings. Heartbeat estimates use known mode-row RK work or ell/eta
quadrature cells for their named scope. Unplanned future adaptive work is
unknown. Cached work does not complete unscheduled chunks, and native RK
history grids determine memory chunking instead of projection-grid lengths.

At this earlier checkpoint, implementation and verification were incomplete.
The remaining obligations included declared role/distribution planning, a
combined observable
error budget with validated propagation, final current-identity LCDM/Planck
cold/warm/subset/batch evidence and the complete focused regression handoff.
The earlier diagnostic or interim production passes do not close Slice Six.

The measured ordinary LCDM ell=(2,10,20), TT/TE/EE request completes its real
production refinement in 454.420 seconds under concurrent verification work.
Its exact repeat takes 0.013 seconds; the reordered subset ell=(20,2) takes
252.237 seconds and agrees within 7.6e-16, 4.8e-16 and 7.9e-17 of the sampled
TT, TE and EE amplitudes. The retained production-refinement errors are
1.06e-7 (TT), 8.26e-8 (TE) and 2.50e-9 (EE), within the unchanged final
thresholds. The finer calculation reuses 513 already evolved modes. Raw
retention records 239 spills with a 128 MiB resident-array budget.

The measured full-time-domain, three-mode Planck native/source diagnostic
completes in 158.407 seconds. Its largest native-state error is 0.00212 and
its largest compiled-source error is 0.00913 under independent evolution
comparison. This is not a full Planck angular request or CAMB-parity report.
Both records retain unchanged producer identities and products under the
progress evidence directory above; `numerical-progress-summary.json` links
their digests. All 172 direct runtime tests pass in 214.600 seconds, and nine
public-route/hardware regressions pass in 378.090 seconds. These were focused
checks, not a replacement for the human-run full workflow. Slice Six remained
open at that checkpoint against its unfulfilled closure obligations.
The measurements precede the final docstring/layout cleanup; their producer
hashes are preserved rather than relabelled as a later source revision.

### Additional numerical implementation

Physical hierarchy roles and phase-space laws now flow through declarations,
compilation, support planning, bound momentum moments and reuse identities.
The bundled kinetic declarations explicitly preserve their thermal law and
present-temperature convention. Numerical q support and density are measured
separately, including a nonthermal peak beyond the initial search interval.

Ordinary final execution independently refines actual background grids,
hierarchy depth and applicable q quadrature on informative modes including
both endpoints. Source probes evolve real equations and return before angular
projection. Changed background comparisons must use a distinct accepted grid;
kinetic/depth comparisons must materialize additional states. Tail estimates
remain candidates, not measured certificates. Native source discrepancies
propagate through absolute LOS operators and signed spline weights; geometry
changes, quadratic transfer errors, projection, Fourier refinement and
cross-spectrum denominators participate in one additive observable budget.
The between-wave source envelope is the maximum measured normalized source
discrepancy on that explicitly sampled domain, not an analytic universality
claim. The unchanged acceptance thresholds still apply.

Grouped arithmetic preserves each validated equation's operands and operation
order. Actual declared collision operators set split phase schedules in a
coordinate-scale-invariant way. Full native trajectories can use scoped
file-backed buffers; live RK working memory determines batching, while mapped
history cells and the OS page cache remain distinct accounting categories.
Stiff operators without a fast manifold use controlled local full-equation
integration, preserving boundary and conservation semantics. The old scalar
substep ceiling is removed. Schedule evidence bounds extra compatible-group
work against the actual scalar work instead of requiring zero extra steps.
Kinetic binding keys exclude unrelated amplitudes while retaining every
actual distribution, temperature and mass dependency.
Production projection grids measure visibility/Fourier features and compare
nested grids instead of filling the allowed search budget. Native evolution
samples the full physical background independently of projection thinning.

`tests.project.lib.numerical_planning_acceptance` is an explicit ordinary
LCDM/Planck angular workload runner. It persists cold, exact-repeat, compatible
warm, changed-request, reordered-subset, changed-Hubble-parameter and ordered
batch outcomes, including actual global-LRU pressure. Its assessment requires
numerical budgets and consistent products; timings alone cannot earn closure.
It remains outside discovery, as does scientific reference acceptance.
The final measured closure record below supplies current-identity workload
evidence and the focused regression handoff.

The workload reader compares common raw and public multipoles within both
calculations' propagated absolute error budgets. Accepted adaptive subsets
may choose different meshes. The initial new reader's arbitrary 1e-7
roundoff-style cutoff incorrectly rejected small, bounded subset differences;
the producer reports retain that assessment. Separate readback reports bind
the original payload digest, original runner identity and current reader
identity, without rerunning or relabeling the physical products. Negative
controls reject raw or public differences outside the two recorded budgets.
The numerical acceptance thresholds themselves are unchanged.

The subsequent algebra review found missing cross-axis quadratic transfer
terms and a coarse-only final axis bound. The corrected budget includes all
transfer-error products and the larger coarse/finer-grid measured bound.
Independent zero-transfer and disjoint-axis tests expose the omitted terms.
Additional regressions retain newly resolved signals from exactly zero
histories and include geometry changes multiplying other source errors.
Earlier independent-axis reports remain profiling/history records and cannot
close the corrected budget. Parameter-matched resolution hints retain only
starting controls, with physical-key isolation and fresh-acceptance tests.
Grouped scalar/one-row arithmetic retains an independent ungrouped CPU
reference rather than duplicating slow row evaluation in refinement probes.
The interrupted corrected-budget attempt also exposed a cache-pressure
execution switch: one remaining physical probe bypassed the compatible RK
executor and re-entered the scalar adaptive solver. A one-row compatible
schedule now remains eligible, including singleton independent-reference
partitions. The kinetic single/pair regression also exercises the actual
coarse-reference override and verifies its distinct retained native grid.
The first one-row repair missed a second guard on reference partitions;
the expensive retained Planck attempt exposed that incomplete repair.
The retained LCDM budget also showed repeated source-grid increases with
little reduction in the source contribution. Joint-budget refinement now
records measured axis contributions, avoids repeating an unproductive axis
while another material contributor can be reduced, and retries a material
axis when the remaining small contributors cannot close the deficit.
Stagnation comparisons use the same failure kind and the total joint-budget
violation, including normalized cross errors. All acceptance checks and
physical tolerances remain unchanged; these observations only choose the
next independent trial.
Source/projection promotion now changes its own measured LOS node minima,
not the background-resolution seed. Native event searches use the accepted
physical background instead of source sampling. A real-grid regression
checks increased source sampling against identical background and native
inputs. Selected pilot/reference histories are retained under identities
covering physical inputs, compiled equations, q weights, integration controls
and exact native background/coordinate/rate arrays. Source-only changes
resample that raw work and re-evaluate sources, constraints and numerical
comparisons; no acceptance certificate crosses the refinement. Kinetic
regressions force eviction and integrity-checked spills, prohibit another
ODE on a compatible source-only change, require fresh integration for new
controls, and compare refreshed sources with an independent cold solve.
Reference-state diagnostics do not overwrite the primary state statistics.
Background promotion now starts from its accepted internal refinement seed
through a separate `background_minimum_nodes` control. Doubling the original
input could otherwise select the same already-refined background. A real
planning regression changes the background from 3,796 to 13,657 samples while
preserving the 192-node LOS seed; source-only promotion changes 695 to 1,615
samples while preserving actual background/native inputs. These are numerical
counts, not seconds. Both directions retain independent fresh comparisons.
New transfer-refinement nodes also use chunked compatible evolution, retaining
separate reference work counts and the scalar fallback for incompatible
contracts. A kinetic TT/TE/EE comparison uses the original ungrouped scalar
equations under actual global-cache pressure.

The subsequent resolved-axis measurement accepted the ordinary LCDM cold
request in 4,936.365 seconds, its exact repeat in 0.016 seconds and its warm
request in 1,321.027 seconds. The cold combined TT/TE/EE relative bounds were
0.002737, 0.002581 and 0.000208; normalized TE was 0.006712. Evolution consumed
4,348.445 seconds. Those costs do not satisfy practical closure. The Planck
attempt remained unaccepted when both workers were interrupted for the
profiled repair; neither incomplete producer is relabelled as closure.
Their original identity and partial outcomes remain in the resolved-axis
evidence files. Additional bounded profiles are explicitly partial records.

The batch profile locates over half its cost in repeated scalar background
evaluation at already known stage coordinates, including repeated kinetic
moments. The stage-background owner now evaluates the exact same coordinates,
interpolants and declared graph in bounded vector blocks. Momentum bindings
separate constant fields from varying columns once per batch. These changes
require fresh measured workload evidence; no tolerance or physical equation
is changed, and the independent scalar executor remains the authority.
The vector-stage LCDM attempt subsequently accepted the same numerical
request in 3,333.709 seconds with nine resolution retries. Its exact repeat
took 0.021 seconds. The expensive cold search still prevented practical
closure, so that retained producer remains a partial cost record.
Initial ordinary scalar planning now proposes active evolution/LOS densities
from the accepted physical background's actual sample count, rather than
repeatedly starting far below its measured feature resolution. This is only
a candidate: every full independent numerical check still runs, failed
acceptance still propagates, finer inputs and disabled controls are retained,
and later background/LOS promotions remain independent. The final record
below supplies fresh workload evidence after this planning change.
The first measured candidate completed all seven LCDM operations, with cold
execution in 894.059 seconds. The Planck cold request accepted in 4,697.143
seconds, but that cost still required work; its later workload was interrupted
and cannot close the slice. The retained producer identities remain intact.
The historical final proposal started evolution at measured background
density and LOS products at paired density. Slice Seven removed that
coupling after profiling its excessive work; those historical measurements
do not describe the current starting proposal. Mixed equation operands gather
interior state rows directly and bind only boundary/rebound replacements;
RHS contexts retain only actual dependencies. Full diagnostic, collision,
custom-coordinate and independent scalar contexts retain their semantics.
Grid logs explicitly label conformal-time node counts as `eta_samples` and
`source_eta_samples`, distinct from elapsed or remaining wall-clock seconds.

### Historical sparse numerical evidence, 2026-10-29

At the recorded source revision, both ordinary public workloads complete
all seven operations with their
engine-selected final controls and unchanged numerical thresholds. The cold
and changed-parameter requests cover the full time domain, with TT/TE/EE at
ell=(2,10,20); the changed request adds ell=3 and the reordered subset uses
ell=(20,2). No user numerical overrides or diagnostic execution are used.
Each independent background, evolution, source, hierarchy, projection and
Fourier comparison runs; the kinetic Planck graph additionally compares q
quadrature. Both producer assessments and separate digest-bound readbacks
pass, with identical initial/final/current source identity:

`6065e65fefae78086c4e957a2e8b25bd72d9342c98bd627deae9a658d42c6880`.

| Operation | LCDM seconds | Planck seconds |
| --- | ---: | ---: |
| Cold, with global-cache pressure | 617.956 | 3053.376 |
| Exact repeat | 0.015 | 0.172 |
| Warm, complete result caches cleared | 651.770 | 2767.460 |
| Changed request | 611.965 | 2577.990 |
| Reordered subset | 570.090 | 2547.245 |
| Changed Hubble parameter | 678.115 | 2760.111 |
| Ordered batch including one invalid contract | 0.037 | 0.307 |

Common raw and public products agree within their two propagated absolute
budgets. The changed physical point produces different spectra. Exact repeats
report exact cache hits. Batch results retain order, isolate the invalid
middle request, and reproduce the accepted raw results exactly. Warm solves
retain fresh acceptance checks; these measurements do not imply that every
compatible request is cheaper than its cold counterpart.

Cold LCDM TT/TE/EE relative bounds are 0.003884, 0.003172 and 0.000288;
Planck bounds are 0.005426, 0.004131 and 0.000215. All individual and combined
checks pass the unchanged TT/EE 0.01 and TE 0.02 targets, including cross
normalization. LCDM evolves 93 state slots; Planck evolves 1,021 with 32
momentum points. Their cold refinements retain seven native and fourteen
independent-reference history reuses each. The 128 MiB raw-retention budget
causes 223/216 spills and 19/49 checked loads under actual global LRU pressure.
Incoming bundles can transiently exceed that retention budget before spill;
it is not a limit on the solver's full working memory.

The recorded machine is a four-core Intel i5-4690K at 3.50 GHz with 16 GiB,
macOS 26.2, Python 3.11.14, NumPy 1.26.4 and SciPy 1.12.0. The two workers
start together without simultaneous numerical tests or gates. Whole-worker
peak RSS across their seven operations is 1.92 GiB and 8.98 GiB. Compared
with the preceding physical-seed checkpoint, cold costs fall from 894.059
to 617.956 seconds for LCDM and 4697.143 to 3053.376 seconds for Planck.
Planck remains costly on this Intel machine; the measured baseline is explicit
rather than a promise of dense-window or sampler throughput. Historical time
targets remain optimization guides, not altered acceptance tolerances.

The evidence directory identified above retains
`lcdm-completion-runtime.json`, `planck-completion-runtime.json`, their logs
and separate `*-completion-readback.json` files.
`completion-numerical-environment.json` records machine, dependency/thread
environment, per-worker CPU cost and RSS. `numerical-completion-summary.json`
binds their payload digests and outcomes without rewriting earlier attempts.

The final focused handoff passes 215 runtime tests in 391.583 seconds, three
public scalar/batch/regular-metric regressions in 150.589 seconds, and nine
phase-space/workload-reader tests in 2.819 seconds. It includes independent
ungrouped scalar equations, kinetic single/pair/coarse references, pressure
and spill integrity, distinct physical-axis work, zero/localized/oscillatory
errors, joint bounds and weighted heartbeat accounting. This is the focused
verification requested for the slice; the human runs the full workflow.

These results establish that recorded sparse workload only. They did not
establish full-suite practicality, and their original closure claim is
withdrawn. Dense matched CAMB angular parity remains Slice Seven;
the universal matrix, sampler/application throughput and final bounded demo
remain Slices Eight through Eleven. Neither the scientific reference acceptance
nor these explicit numerical workload measurements join ordinary discovery.

### Full-workflow repair, 2026-10-29

The interrupted human workflow ran for 36,694 seconds and recorded 47
failure/error occurrences across 21 methods before its final report. Its
discoverable graph test had entered a real 99-multipole production solve:
442 modes, 16,367 source samples and about 689.6 million projection cells.
Five retained scalar kernel matrices alone represented about 25.7 GiB of
array payload. The global LRU budget did not bound local all-mode references.
The earlier three/four-multipole measurements could not expose this growth.

Projection now prepares, consumes and releases one Fourier-mode chunk for
every request, including adaptive requests and small maximum multipoles.
The same native validation, source-error propagation, geometry comparisons,
coarse projection and quadrature still execute. The Bessel recurrence also
partitions columns without changing any physical sample; its input/value
caches have byte budgets. Actual live kernel bytes and mode counts are
reported rather than claiming a constant as a measured peak.

The dense graph physics assertions are preserved in the explicitly invoked
`tests.project.lib.ccmbs_production_graph.ProductionGraphTestCase`. Run it
with the managed interpreter and `python -m unittest -v` when requesting
dense acceptance. Discovery instead tests manufactured wave rendering,
fresh-process artifact reload, digest corruption and the discovery boundary.
It does not earn a scientific acceptance certificate from manufactured data.
Ordinary real public-route and independent physics tests remain in discovery.

Bundled kinetic declarations preserve their existing temperature convention:
models without a declared heating parameter use their original temperature
ratio, while Planck retains its explicit heating factor. Compilation rejects
unbound physical-law symbols before numerical planning or evolution. The
early-prefix test observes actual scalar or batch integration stages, and
history error tests compare real reference estimates and inject a localized
history defect rather than assuming an improved solver must remain inaccurate.

Per-test profiling also exposed repeated scalar Jacobian work in stiff
vector-sector interval solves. A compiler proof of affine state dependence
now allows all Jacobian coefficient probes to share one graph evaluation;
the actual trajectory still uses the original scalar RHS. Collision products
retain the scalar column reduction order, including conservation-row
cancellation. Nonlinear systems and state-dependent collision coefficients
retain the general Jacobian path. Direct scalar-column comparisons and an
independent stiff analytic solution with a 1e-20 state test the optimization
without changing tolerances, equations, interval boundaries or projections.
Projected local intervals use one-step L-stable Radau integration rather
than rebuilding multistep BDF history at each output boundary. Whole-history
integration retains its BDF reference path. Both methods retain the same
component scaling and error tolerances and are checked against an independent
stiff analytic trajectory. Initial scale probes also share the validated
affine graph without dropping initially zero or extremely small states.
Alias-routing checks bind their bounded diagnostic request after preparation,
so public replanning honors the fixture's original hierarchy. They require
nonzero finite spectra. Independent sector physics/refinement comparisons
retain their original domains and thresholds. The direct Jacobian-column test
instead enters the public planner's expanded graph, stopping before numerical
integration.

The public planner expands the vector fixture to 113 states, compared with
38 in its direct source probe. Its initial Jacobian has only 360 nonzero
entries out of 12,769. Dense LU factorization made this actual public route
far more expensive than the earlier direct probe. Continuous integration now
selects sparse LU storage when the initial Jacobian is mostly zero, retaining
every nonzero coefficient and allowing new nonzeros at later times. Storage
format stays fixed for that solve, as required by the integrator. This changes
linear algebra storage, not equations, tolerances, hierarchy depth or samples.
An independent exact stiff trajectory checks BDF and Radau down to 1e-25
amplitudes, including a Jacobian that becomes dense after sparse startup.

Repair verification profiles individual test time and sampled process memory,
retaining progress and immediate tracebacks if interrupted. All 716 distinct
focused tests passed without skips across the repaired owner groups:

| Verification group | Passed | Measured seconds |
| --- | ---: | ---: |
| Complete CMB module, three completed partitions | 221 | 2831.318 |
| Three final alias cases and expanded public Jacobian case | 4 | 102.321 |
| Runtime owners, including sparse stiff analytic checks | 222 | 397.012 |
| Compiler, contracts and diagnostic/artifact readers | 187 | 306.047 |
| Bundled model declarations and physical outputs | 34 | 1737.267 |
| Frozen mapping, interfaces and adapter exports | 49 | 736.383 |

The Jacobian case overlaps the runtime group, giving 716 distinct tests and
complete 224-case CMB coverage. The bundled-model and interface profiles
precede the sparse-LU change; the final runtime/CMB groups check that numerical
change. Only the three alias fixtures and public Jacobian test changed during
the final partitioned reruns; their updated cases passed separately. Discovery
also imported all 1144 tests, included graph artifact checks and excluded the
explicit dense acceptance case. The retained profile manifests record these
producer boundaries under the output root's
`ccmbs-acceptance/completion-evidence/workflow-repair-20261029*` directories.
These partitioned measurements were followed by the successful full workflow
`devcovenant/logs/20261108T012634504227Z-run`: 1144 tests in 5978.307 seconds.
The human confirmed that run passed and the gate closed before authorizing
Slice Seven. That establishes the repaired development-loop result, not
ordinary angular acceptance or its cost; the remaining cost obligation is
recorded with the Slice Seven investigation below.

**Focused owners:** all direct runtime numerical/physics tests, public
`test_cmb` method selectors, cache/performance tests, and the existing
acceptance runner. Keep scalar CPU results as the comparison authority.

**Closure:** ordinary requests select and enforce adequate resolution with
measured/validated bounds, request-subset consistency, and identity-safe reuse.
Actual cold LCDM and Planck requests have reasonable measured costs and retain
usable products; errors or runtime blockers are repaired here. Heartbeats
and scoped estimates agree with completed work and phase attribution.
Finite output and a cheap diagnostic path alone cannot close this slice.

## Slice Seven - demonstrate actual matched CAMB parity

**Status:** in progress (2026-11-08); physical acceptance is not complete.
**Entry:** ordinary execution is accurate and practical. The current
low-ell ordinary-route investigation has not yet established this condition.

### Reference and durable-product prerequisites (2026-11-08)

The existing runner now exposes `--completion-preflight`, which performs no
CCMBS solve and retains every success/failure attempt with original checkout
provenance. Compact frozen requirements, historical full-case hashes and
current declaration bindings have separate roles. A corrupt retained
reference is rejected rather than silently regenerated. Numerical product
archives retain actual typed arrays, survive producer exit/temporary cleanup,
publish immutable binaries before atomic manifests and reject tampering,
missing arrays, wrong dimensions and escaped paths. Physical, numerical,
reference and assessment dependency identities are tested separately; pure
rendering changes do not invalidate the forward identity. These are
prerequisites, not accepted physical rows or a completed resume runner.

Preflight exposed an actual Planck normalization change made before this
slice: the current declaration resolves `omnuh2=0.000644866570625114` and
`omch2=0.11935498653217487`; the frozen independent point instead has
`omnuh2=0.0006441915396177796` and `omch2=0.1193556615631822`. Restoring a
missing default-temperature spelling recovers the massless reference hashes
without changing physics, but cannot reconcile these different densities.
The human authorized updating the affected independent reference after
checking the current normalization. Unmodified CAMB and an independent
Fermi-Dirac number-density calculation both reproduce the current density;
the latter agrees within 3.2e-10 fractionally. Only the Planck reference
changed, from SHA-256
`277731908a3ef8fe8c4a4ebfc8e759e94a53999e73b3e17153230b730027b206`
to `7f23ae11b641aa9f02aff6c6ead03fdeeedce76b1196ec6fe8ab297e5d40d769`.
The frozen requirements record the decision and preserve every physical
case, multipole, window, surface and tolerance. A lossless portable archive
retains all twelve canonical CAMB references; preflight validates those
arrays and their resolved physical inputs rather than expecting different
CAMB builds to reproduce an identical floating-point serialization.

A current LCDM native-history component comparison completed in 55.684
seconds. The main temperature/polarization/lensing source errors were below
0.4%; the small E quadrupole alone had a 1.24% relative difference. This is
component evidence, not angular parity. The ordinary unmodified TT/TE/EE
request at ell 2, 20 and 100 was interrupted after more than 36 minutes,
still in adaptive evolution. It produced no accepted angular spectrum.
Subsequent profiling found that the old initial-density proposal forced
unrelated background, native and source resolutions to grow together. That
proposal is removed. Native refinements now obey their phase criteria at
every level and share meshes within deterministic wave cohorts. Compatible
schedule batching preserves each mode's minimum subdivisions and limits
both total and per-interval extra work. The singleton executor uses the
measured faster equivalent scalar-row kernel. Independent validation remains
required. Already excessive observable-axis bounds now reject before the
separate doubled-k calculation; final acceptance still includes that
independent calculation when the preliminary bounds permit it.

A later ordinary diagnostic retained actual projected candidates at ell
2, 20 and 100. Its TT ratios to CAMB were approximately 0.980, 0.962 and
0.681; doubling the k grid left the last ratio effectively unchanged.
The actual upper wave number was 0.014965389806815388 inverse Mpc.
Independently truncating CAMB's transfer integral at that same endpoint
retained about 0.669 of its TT power near ell 100, while TE and EE retained
0.932 and 0.982. This isolates omitted high-k radial tails as a material
physical-domain defect. Increasing density inside the truncated domain
cannot repair missing support. This is diagnostic evidence from unaccepted
calculations, not a passing angular row.

Before expanding the matrix, repair physical-domain planning and validate
source sampling independently from the denser angular quadrature. Merely
checking the endpoint against the planner's own projection heuristic does
not measure omitted power. The older optional source-interpolation path
does not provide the independent interpolation evidence required here;
enabling it alone cannot establish fidelity. Ordinary low/full request
consistency, actual tail convergence and reasonable measured costs remain
open. Slice Seven therefore remains in progress.

### Source, angular and metric repairs (2026-11-08)

The ordinary planner now retains its full proposed k domain and separates
independently evolved native source meshes from angular-k quadrature.
Sources, rather than oscillatory transfers, are interpolated onto a common
angular grid. Separate nested angular and native-k calculations retain their
actual spectra and error bounds. Saved source times are independent of the
radial phase grid; freshly reevaluated native-time sources check that the
saved representation has not missed oscillations or a signal between knots.
Source promotion must strictly enlarge its actual representation while
preserving native evolution, background and radial resolution. The outer
k comparison must preserve the larger measured native-source bound, including
axes present only in its finer calculation.

An ordinary full-domain probe exposed overflow at k approximately 0.0371
inverse Mpc and eta approximately 10485 Mpc. The generated energy-only metric
equation introduces a k-squared/H expansion-time stiffness. Its owning
materializer now combines the equivalent Einstein energy and momentum
equations using dimensionally matched horizon weights. Both constraints
remain independently audited. Manufactured horizon limits, the scalar
correctness reference and gauge regressions check the formulation; these
checks do not certify its final angular spectra.

The pilot selector previously could keep splitting an already adjacent pair
of indices on an uneven wave grid. It now chooses only refinable gaps.
Scheduled physical stage contexts are stored as compact columns instead of
one dictionary per microstep. Evicting transient caches leaves future
precomputed values intact, and momentum lookups use their actual stage key.
The scalar RHS avoids singleton array allocation without changing equations,
collision stages or RK schedules. Focused storage, evolution and equation
checks passed (109 tests in 198.064 seconds), as did 81 angular, planning and
budget tests in 61.041 seconds. Source/axis repairs have their own regressions.
The ordinary diagnostic retained at
`ordinary-preflight/20261108T173531Z/report.json` was interrupted after
1383.238 seconds while still preparing independent native references. It
produced no accepted angular spectrum. There is no accepted full-window
physical case from these repairs yet.

On 2026-11-09, projection source-time interpolation was moved before the
k interpolation when its working representation fits the memory budget.
The streamed route remains available for larger representations. Independent
manufactured tests compare both routes, including signed sources, coarse radial
quadrature and propagated uncertainty. In one synthetic 2,000-node benchmark,
interpolation took 0.039 seconds rather than 0.449 seconds; this is not a
measurement of total scientific-acceptance runtime or evidence of parity.

The completion product chain now identifies the numerical spectrum actually
returned after refinement. Its named quadratures must match that projection's
wave grid, transfer components, primordial weighting and spectra on reload;
a later refinement cannot replace the returned result by event order. Tests
also retain the actual radial time, distance and weights, and reject rehashed
unrelated arrays and inconsistent operators. This repairs the
product linkage only: independent domain and axis evidence, consumer/graph
verification and the actual accepted cases remain open.
Measured-axis assessment now also requires retained base and refined spectra
on distinct effective grids and recomputes their discrepancy. This rejects
metadata-only or understated error claims; it does not supply the missing
physical calculations or validate an unmeasured domain bound. Declared
bounds cannot satisfy the assessment until an independent verifier for their
derivation and inputs is implemented.

The named-point runner screens sampled late electron-fraction histories
against CAMB before evolving perturbations. Its two background resolutions
and sampled histories are retained; a passing screen is not a physical
matching certificate. The initial massive-neutrino, wCDM and w0wa models
declare collapse-source reionization while their frozen CAMB references use
tanh. Representative background comparisons found maximum absolute electron
fraction differences of about 0.148. The physics and independent reference
must be matched before those cases can establish parity.
The wCDM initial point also declares photon density 0.000055 while its
2.7255 K CAMB reference resolves approximately 0.000043964; its expansion
rates differ by about 1.36% at redshift 1100. The named-point runner now
checks this physical input before the reionization screen or a full solve.
The mismatch must be resolved without changing the declared physics silently.
The existing declarations already allow independent photon density and
temperature, and the background resolver retains the declared density. A
focused model-adapter check varies that density at fixed temperature in all
three models and checks that the declared expansion responds. Matching the
independent reference does not require a model-convention change.

Before closure, finish actual physical k-domain/tail verification, complete
the ordinary full-window comparisons, and supply independently verified axis,
graph, consumer and reload evidence for all twelve comparable points. A
corrected proposal, a component pass or an interrupted ordinary probe must
not be recorded as that closure.

Durable investigation artifacts live below the configured output root's
`ccmbs-acceptance/completion-evidence/` directory, including
`native-components/current-lcdm-20261108T081437Z.json`, the preflight attempts
and the ordinary probe interruption record. Scientific acceptance remains
explicit and outside ordinary discovery and `devcovenant run`.

Focused prerequisite verification passed: 30 tests in 2.225 seconds. The
exact selection is:

```bash
source .venv/bin/activate
completion_test_owner=tests.copernican.lib.likelihoods.cmb
python -m unittest -q \
  "${completion_test_owner}.test_evidence_assets" \
  "${completion_test_owner}.test_diagnostics.CompletionContractTestCase" \
  "${completion_test_owner}.test_diagnostics.CompletionPersistenceTestCase" \
  "${completion_test_owner}.test_diagnostics.CompletionDependencyTestCase" \
  tests.project.lib.test_scientific_acceptance
```

**Owners:** extracted physics/numerical owners for residual defects, existing
CAMB fixtures, diagnostics, acceptance runner, graph tests, and readers.

**Physical prerequisites:** Before expensive case execution, reproduce and
resolve the applicable radiation and neutrino normalization, mass-limit,
curved-distance, and declared parameter-propagation findings above. Test
cross-route parameter responses and relevant physical domains with independent
expectations. Match each independent reference to the same actual physics
without silently replacing a declared law. CAMB in Copernican remains stock.
For nonstandard physical inputs, the human authorized an experimental CAMB
copy under `/tmp`; label its results separately from stock CAMB and record its
source identity, physical changes, build and inputs. Check its standard limit
against stock CAMB and its changed histories independently before trusting
its spectra. A cheap preflight can reject an inconsistent point, but cannot
certify physical parity.

The human approved an explicit transverse-geometry convention on 2026-11-10.
Models can declare FLRW curvature or a transverse-distance expression; wCDM
and w0wa declare FLRW geometry. Independent signed-curvature tests cover their
BAO and SNe distance helpers. This repair does not resolve the other physical
prerequisites or establish angular acceptance.
The q-resolved background now receives declared `model_parameters` as well
as `param_map`, preserving the latter's precedence as in hierarchy evolution.
A renamed phase-space temperature parameter is checked against the equivalent
literal declaration and a changed value. The human also approved an explicit
relativistic-density normalization. LCDM+Mnu, wCDM and w0wa now specify the
zero-mass q amplitude from their declared radiation and derive rest density
and cold-matter normalization from the same Fermi-Dirac moments. Independent
moment tests and the LCDM+Mnu zero-mass limit check this change. Matched CAMB
physical histories and affected reference identities required review.
The frozen initial LCDM+Mnu reference has `omnuh2=0.002684131415074082`;
the revised declaration resolves `0.0033249223708688874`. Both wCDM and w0wa
initial references have `omnuh2=0.0006441915396177796`, while their revised
declarations resolve `0.0007979813690085331`. Their complementary cold-matter
inputs also change. These are physical identity changes, not permission to
relabel the old CAMB arrays as matched. Standard CAMB's photon density and
reionization differences motivated the independent matching reference
described below; the affected cases still require actual parity comparisons.
The explicit completion preflight rejected the first stale LCDM+Mnu reference
in 133.004 seconds without running CCMBS. It now verifies artifact integrity
separately from physical-input identity and records the old and current
parameter values in the typed failure. This is an early diagnostic result,
not a matched reference or accepted scientific case.
With the revised wCDM initial point, a bounded screen found declared
`Omega_gamma=0.000055` versus standard CAMB's `0.00004396400582895195`
(25.1% relative difference). The sampled late ionization screen found a
12.7% maximum difference normalized to the reference peak, with 0.17%
between its two background resolutions. These are diagnostic screens, not
complete histories or permission to change the declared theory or reference.
The bounded `test_lcdm_mnu_declared_surfaces_are_finite` exercised the actual
revised hierarchy and projection in 151.228 seconds on this Intel host; it
used eight evolved modes and 19 multipoles. Its finite diagnostic output is
not an ordinary-route or independent-parity acceptance.

The prior distance and CMB expansion histories differed at redshift 1100 by
about 0.56% for LCDM+Mnu and 0.15% for wCDM and w0wa at their initial points.
The human approved extending the model convention with an explicit
`sound_horizon_redshift` expression. These three models now select their
declared CMB background for distance and ruler calculations, retain `z_rec`
as the recombination-ruler endpoint, and resolve the BAO drag epoch from the
same atomic background. This does not establish their full physical parity.

### Temporary experimental reference investigation (2026-11-10)

The authorized temporary CAMB 1.6.0 copy accepts an independent photon density
and externally integrated recombination and collapse-reionization histories.
Its neutrino radiation uses that photon normalization; an independent
Fermi-Dirac energy integral supplies the present total massive density. With
the experimental inputs absent, its H(z), electron fraction, and all raw
unlensed, lensed and lens-potential arrays through ell 2500 matched the stock
CAMB build exactly at one tested standard point. This is a standard-limit
check, not a proof of every possible experimental configuration.

For the revised wCDM initial point, sampled expansion agreed to 2.9e-8
relative, total neutrino density to 8.9e-6, and matter temperature to 1.6e-5.
The independent recombination and collapse integrations and temporary CAMB
histories differed from the native electron fraction by at most 0.00132
absolute on the sampled grid. These are component checks; they do not certify
the spectra. Successively refined temporary reference calculations still show
changes in low-ell EE and TE. Resolve and retain reference refinement evidence
before freezing or accepting any affected spectrum. Existing stock reference
arrays retain their original identities.
The temporary source's natural-spline interpolation of the first spherical
Bessel cell produced a spurious small-argument contribution when late
polarization remained nonzero near the observer. Evaluating the same spherical
Bessel function directly in that cell reduced the largest sampled raw-surface
change from 6001 to 12001 reionization nodes to about 2.1e-5 of its surface
scale for the checked wCDM point. Stock-mode behavior remains separately
checked; repeat the physical and refinement checks for every reference case.

The previous wCDM and w0wa declarations used a smooth dark-energy background
without perturbation states. A representative w=-0.9 diagnostic changed
low-ell TT by about 11% when CAMB's dark-energy stress was disabled. The
human authorized a conserved PPF closure, explicitly selected in these models
and absent from theories that do not declare it. Its Gamma equation and
Einstein density and momentum sources use the zero large-scale relative
momentum prescription of Fang, Hu and Lewis (2008). PPF is an approximation,
not an assertion about every possible dark-energy theory. Independent PPF
history, conservation, crossing, curvature and full-spectrum comparisons
remain necessary before this physical change is accepted.
An independent stock-CAMB PPF history fixture now checks the compiled Gamma,
density and momentum relations at 16 points, including either side of a
phantom-divide crossing. Gamma derivatives are independently estimated from
nearby CAMB times. The check exposed and corrected a factor-of-three error
in the momentum denominator. These local relations do not establish coupled
CCMBS history or spectrum parity.
For the w0wa initial point, the refreshed temporary PPF and previous
no-stress CAMB references agree exactly on every dense raw surface in their
shared w=-1, wa=0 limit. This checks that limit only.
The stiff PPF relaxation now uses exponential integration for eligible
batched modes and retains fully coupled BDF for the scalar fallback. In one
bounded wCDM solve, batched and BDF TT, TE and EE differed by at most
3.8e-5, 8.2e-6 and 2.7e-6 respectively when each difference was normalized
by its surface peak. The batched solve took 119 seconds and BDF 352 seconds
on this host. This comparison does not establish full-window accuracy or
independent spectral parity.

The nine affected physical points now have separately identified experimental
references. Their archived coarse and fine calculations retain all multipoles
from 2 through 2500, independently integrated input tables, background samples,
source/build hashes and generator scripts. Every original required multipole
remains in its comparison window; feature windows are also recalculated for
the changed physical inputs. The original stock arrays remain in the archive.
The reference comparator charges the measured coarse/fine discrepancy against
the unchanged scientific allowance. The human authorized a temporary
`handoff/` package containing the modified CAMB source, its generators and
reference inputs for transfer to another computer. The normal Copernican
dependency remains stock CAMB; see `handoff/HANDOFF.md` for recovery and
provenance checks. These calculations do not establish CCMBS parity. The five
wCDM and w0wa references now select
stock CAMB's PPF perturbation closure in the temporary input-matched build;
their superseded no-stress reference arrays remain archived separately.
Before the later PPF declaration change, the explicit 23-case completion
preflight passed in 118.497 seconds on this
host. It verified reference and physical identities without a CCMBS solve;
this is not scientific acceptance.
After the PPF declaration and reference changes, the same preflight passed
in 135.585 seconds before the subsequent integration changes. It likewise
did not run or accept a CCMBS spectrum.

**Work:**

1. Extend the existing runner to execute and persist named frozen cases and
   their complete windows. Resolve sparse-only handoff and completion-contract
   differences in that owner. Validate exact requests and references before
   expensive execution; persist each attempt so unaffected work can resume.
   Before relying on resume, implement and test physical/reference/numerical
   dependency identities, payload validation and durable readback after
   producer exit. Reject stale/missing/tampered cases and distinguish whole
   checkout provenance from reusable product dependencies. Do not defer
   these prerequisites to Slice Ten or relabel old payload identities.
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

## Slice Eight - reconcile all bundled theories and model convention

**Status:** pending. **Entry:** Slice Seven's matched physical acceptance
passes. Corrections discovered here that invalidate earlier evidence require
selective revalidation before this slice closes.

**Owners:** bundled model declarations and their theory sources; model
validator, compiler and adapter; background, species and perturbation owners;
SNe/BAO/CMB distance and sound-horizon paths; model templates and mirrors;
existing focused tests, diagnostics and retained evidence.

**Work:**

1. Inventory all ten bundled models individually: equations, assumptions,
   sampled and fixed parameters, units, domains, species and distributions,
   interactions, gauges, initial/boundary conditions, physical limits and
   observables. Map each equation and parameter to its implementation and
   applicable independent derivation or reference. Identify unsupported,
   missing or ambiguous mathematics instead of filling it with LCDM defaults.
2. Trace each shared quantity and parameter through background, SNe, BAO and
   CMB routes. Independently check expansion, curvature and distances, sound
   horizon and drag/recombination endpoints, densities, temperatures,
   neutrino mass limits, conservation, and response across representative
   physical and boundary points. Respect intentional theory-specific
   differences when explicitly justified; route agreement does not prove
   physics.
3. Resolve each investigation finding above, including QAUC radiation and
   recombination, QRSF normalization, TOG temporal and early limits, TORG
   radiation and sound speed, and USMF2 dynamics and constraints. Check all
   perturbation equations and their source, gauge and initial-condition wiring.
   Correct model declarations only where the model's own intended mathematics
   requires it; correct shared implementation at its owner. Make valid
   domains explicit without arbitrary clipping or a common thermal law.
4. Audit and repair name-dependent behavior, parameter propagation through
   phase-space backgrounds and evolution, dependencies and caches. Include
   equivalent consistently renamed declarations, fixed versus sampled model
   parameters, parameter changes and independent expected responses. Verify
   that all active declarations use their own equations and that failures
   distinguish invalid input from missing engine capability.
5. Review the complete model convention, including primordial spectra, and
   the canonical template and its package mirror against actual capability.
   Correct misleading template examples and document supported theory inputs
   and limitations. If required physics cannot be expressed without changing
   the model convention, stop and present the proposed contract amendment to
   the human for an explicit decision before implementing that change.
6. For each model retain an evidence and disposition record identifying the
   tested equations, domain and parameter responses, independent expectations,
   corrections, unsupported capabilities and unresolved questions. Require
   reproducible focused tests for each repair and relevant negative controls.
   Use bounded checks before expensive acceptance; share setup and exact
   products when valid without weakening independent or cold evidence.
7. Rerun affected physical and numerical acceptance after model or engine
   repairs, retaining unaffected products only under verified dependency
   identities. Keep scientific acceptance explicit and outside ordinary
   unittest discovery and `devcovenant run`; report actual measured cost.

**Closure:** all ten models have complete theory-specific equation and
parameter disposition, independent applicable checks, coherent shared routes,
resolved identified defects, and current affected acceptance evidence. No
model passes merely by loading, producing finite arrays, or agreeing with a
route sharing the same error. Convention limitations and decisions are
recorded honestly; unresolved required physics prevents closure.

## Slice Nine - demonstrate universal declarations and active sectors

**Status:** pending. **Entry:** Slice Eight model and convention review closes.

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
5. Consume the per-model equation and disposition evidence from Slice Eight;
   verify independent parameter responses and equivalent renamed declarations.
   The 23 cases do not excuse unresolved model or convention defects.
6. Preserve all applicable raw/public/sector/lensed surfaces and numerical
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

## Slice Ten - close application integration and final readiness

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
6. Extend Slice Seven's validated physical-product identity/readback to
   application comparisons, consumers and rendering invalidation. A pure
   rendering change must not trigger fresh unchanged physical evolution.
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

## Slice Eleven - prepare the ordinary LCDM/Planck 5/10 demonstration

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
* every bundled model has a complete equation, domain and parameter review,
  independent applicable evidence, and resolved cross-route discrepancies;
* every bundled/auxiliary declaration and nonzero required sector has
  complete theory-faithful accepted output, without hidden substitutions;
* solver and convention limitations are resolved or explicitly documented,
  with any necessary convention changes approved by the human;
* automatic engine planning meets the combined observable numerical budget
  with measured refinement or validated bounds over its declared domain;
* execution and the development suite have reasonable measured costs,
  with independent correctness decisions and informative work accounting;
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
