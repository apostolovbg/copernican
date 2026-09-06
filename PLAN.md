# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-09-06
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

> **For agentic workers:** Execute the slices in order. Keep the gate open
> for the active slice, stage each completed slice, and do not call a slice
> closed until its raw scientific evidence exists. A green policy gate is
> necessary hygiene; it is never scientific closure.

**Goal:** Deliver a production-ready Copernican Cosmic Microwave Background
Solver (CCMBS) that works as the old CAMB-backed solver worked, without using
CAMB as a runtime fallback. CCMBS is a universal declarative solver: LambdaCDM
is one theory declaration among many, not a privileged baseline or fallback.
Every mathematically complete CMB declaration, bundled or novel, must execute
through the same solver, produce the complete set of observables it declares,
and emit a usable CAMB-like graph. CAMB is a comparison oracle only.

**Scope:** This plan owns the generated source compiler, background and
recombination inputs, perturbation hierarchies, metric and collision sources,
line-of-sight projection, all scalar/vector/tensor sectors, unlensed and
lensed observables, model declarations, raw scientific evidence, GUI/CLI
plot production, and the independent BAO boundary. It includes all ten
known bundled models and a generic path for valid future declarative models.

**Known bundled models:**

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

**Non-goals:** This plan does not add Taichi or GPU code, a surrogate,
another production Boltzmann backend, delayed acceptance, broad sampler
optimization, or a CAMB runtime fallback. Such work cannot be used to hide a
CCMBS error. The Python 3.11 `.venv` is an existing managed environment and
must not be recreated, replaced, upgraded, or otherwise modified by this
plan.

**Scientific reason for the reset:** The 2026-09-02 production output proves
that the previous infrastructure-focused closure was premature. LambdaCDM
was absent because production CCMBS raised a doubled-grid convergence error
and the pipeline silently omitted its theory curve. The Planck-reference
curve was present but numerically nonsensical: its TT/TE/EE amplitudes were
orders of magnitude wrong and visibly spiky. The direct evidence points to a
shared mixed-neutrino source-weighting defect combined with an under-resolved
production phase grid. These are solver failures, not short-run or plotting
calibration issues. A green unit-test gate is not evidence of production
CAMB parity until the real model requests and raw arrays pass the criteria
below.

## Global Constraints

* Do not change branches or create branches.
* CCMBS remains the selected production CMB solver. CAMB is comparison-only.
* Preserve every model's theory. A model-file edit may only make already
  intended equations, variables, derivatives, closures, gauges, or
  observables explicit.
* Repair shared defects in shared compiler, background, hierarchy, source,
  projection, normalization, or plotting code. Do not add a model-only
  numerical patch for a shared defect.
* Do not lower accuracy, omit a requested sector, clip a requested grid,
  widen a cache key, swallow a typed failure, or relax a tolerance merely to
  obtain a pass.
* Treat LambdaCDM as an ordinary model declaration. No solver branch,
  default, validation rule, source term, recombination law, or fallback may be
  selected because a model resembles or differs from LambdaCDM.
* A well-posed declaration is never rejected as "unsupported physics" because
  its theory is unfamiliar. If the grammar, compiler, or runtime cannot yet
  express valid declared physics, that is an engine defect assigned to an
  implementation slice, not a model status.
* Reject only declarations whose mathematics is malformed, incomplete,
  dimensionally inconsistent, singular, non-finite, non-convergent, or fails
  an explicitly declared conservation or constraint law. The diagnostic must
  name the mathematical defect and contain no LCDM-specific assumption.
* A valid production request must not be rejected by an arbitrary wall-clock
  or nominal work ceiling. Resource accounting may bound memory and report
  work, but cannot replace numerical acceptance.
* Never silently accept a non-converged spectrum. Preserve base/refined
  arrays, source histories, residual vectors, grids, weights, and typed
  failures in every diagnostic record.
* Generated `Phi_tau`, `Psi_tau`, history gradients, collision sources,
  visibility, polarization, and ISW terms must be explicit, finite,
  coordinate-aware, and independently validated. Missing derivatives are
  typed failures, never zero-valued substitutes.
* Every model must compute every observable it declares: TT, TE, EE, BB, PP,
  TP, EP, and all applicable lensed, tensor, vector, and total surfaces.
  A declared zero is permitted only when the theory proves that quantity is
  zero; it may not be used as a placeholder for missing computation.
* Scalar evaluation is the scientific reference. Ordered batches are allowed
  only after proving scalar equivalence, input ordering, failure semantics,
  and cache isolation.
* Do not recreate, reinstall, upgrade, or mutate `.venv`. Always activate the
  existing environment before DevCovenant commands and tests.
* Keep root/package documentation synchronized. Update comments, docstrings,
  focused tests, README mirrors, and CHANGELOG with each behavior change.
* Stage all changes after each completed slice. A slice is not closed merely
  because `gate --verify` is green.

## Table of Contents

* [Overview](#overview)
* [Current Evidence](#current-evidence)
* [Scientific Acceptance Contract](#scientific-acceptance-contract)
* [Universal Theory Contract](#universal-theory-contract)
* [CAMB Parity Contract](#camb-parity-contract)
* [Diagnostic Status Terms](#diagnostic-status-terms)
* [Execution Rules](#execution-rules)
* [Execution Slices](#execution-slices)
* [Completion Standard](#completion-standard)

## Overview

The mission is a functioning CMB engine, not a collection of runtime
plumbing checks. Each slice below owns both implementation and scientific
output for its stated models or observables. Shared repairs are made only in
the slice that immediately demonstrates them in real bundled model spectra.
No later slice may be used to postpone a required computation while an earlier
slice is called closed.

The production path must have one honest behavior: a valid model request is
resolved at its declared production controls and returns complete raw
spectra, or it fails with a precise typed diagnosis. The GUI and CLI must not
turn a failed theory into an empty legend entry or silently display only a
reference curve.

LambdaCDM is subject to exactly the same declaration parsing, compilation,
background construction, hierarchy evolution, projection, and validation
routes as every other theory. The ten bundled files are a certification
corpus, not the boundary of the engine. A new theory with fully specified
equations must be admitted by extending the generic declaration/compiler
contract where necessary; it must not be placed in an unavailable bucket just
because no bundled fixture resembles it.

## Current Evidence

The current production run is the baseline, not an acceptance result:

* `model_lcdm.yml` raised `Production scalar CCMBS spectrum did not converge
  under the declared doubled k-grid`; the run caught that exception and
  plotted no LCDM curve.
* `model_ref_planck2018.yml` returned TT values up to approximately
  `7.5e5` where the data are of order `1e3`, with similarly bad TE and EE
  scales and visibly noisy multipole-to-multipole behavior.
* The Planck declaration used a 64-node k grid without the final phase-aware
  production floor. Its physical phase requirements are hundreds of nodes,
  and changing the eta grid changes the spectrum substantially.
* The generated mixed-neutrino source path used the full `Omega_nu0` in
  residual massless-neutrino source terms while separately evolving massive
  neutrinos. The residual massless fraction is roughly 69 times smaller at
  the Planck point; an in-memory diagnostic correction removed roughly the
  observed runaway factor.
* The Planck data parser and TT/TE/EE unit conversion produce sensible
  observed values. BAO remains independently evaluable and is not the cause
  of the CMB failure.
* The prior Slice Two closure used a deterministic reduced finite-surface
  request rather than the production CAMB-parity matrix required by this
  plan. Its tests and raw outputs remain useful baseline evidence, but they
  do not close the scientific acceptance contract.
* The declared `N_eff` interval includes values below the integer massive
  species count. The previous allocator assigns `3/N_eff` times the present
  neutrino density to the massive component in that region, so it can
  overcount early radiation. This is a model-domain defect, not a
  calibration or runtime-duration issue.

These observations assign the first repairs to shared source physics,
production projection controls, background neutrino bookkeeping, and the
failure-to-plot path. They do not justify weakening the convergence gate.

## Scientific Acceptance Contract

Every accepted model and observable must pass all layers below at its named
production tier.

1. **Theory-faithful declaration:** capabilities, species, gauges, sectors,
   hierarchy orders, source bindings, derivatives, numerical domain, and
   observables express the actual model.
2. **Physical histories:** background, recombination, metric potentials,
   densities, velocities, collision terms, visibility, temperature,
   polarization, tensor/vector sources, initial conditions, and ISW histories
   are finite, explicitly evolved, and independently residual-clean.
3. **Resolved projection:** line-of-sight sources, radial kernels, Bessel
   functions, quadrature, k/eta grids, normalization, and conversion to
   public spectra are phase-aware and converged under independent refinement.
4. **Complete observables:** every declared scalar, vector, tensor, lensed,
   unlensed, auto-, and cross-spectrum is computed from raw transfers.
5. **Physical shape:** spectra have the expected acoustic phase, peak/trough
   sequence, damping behavior, cross-spectrum signs, low-ell behavior,
   lensing response, and tensor/vector behavior for that theory.
6. **Execution equivalence:** scalar and ordered batch paths agree; cache
   identity includes all physical inputs; completion order cannot alter a
   result; and failures retain their typed meaning.
7. **Evidence:** raw arrays, histories, metadata, residuals, refinement
   comparisons, parity reports, graph files, and decisions are canonical,
   reproducible, and hashable before any sampler result is accepted.

## Universal Theory Contract

The unit of CCMBS support is a complete mathematical declaration, not a model
name or a match to LambdaCDM. A declaration is complete when it supplies, for
every requested sector and observable, the background equations and domains,
species and stress-energy terms, recombination/opacity law, perturbation and
collision operators, metric closures, gauge and initial conditions, source
derivatives, projection conventions, and numerical controls needed to solve
them. It may describe standard matter, modified gravity, additional fluids,
non-standard recombination, or entirely novel interactions.

The universal compiler and runtime must execute every complete declaration
without importing physics from another theory. This includes declarations
whose recombination or visibility history is deliberately unlike LambdaCDM.
The acceptance corpus therefore includes adversarial, non-LCDM fixtures with
explicit equations and checks that their histories and spectra are not
silently replaced by standard ones. A complete declaration that can be
represented by the grammar is never allowed to terminate as an engine
capability gap: inability to evaluate it is a CCMBS implementation failure
that must be repaired before the owning slice closes.

There are only two legitimate failure classes. A declaration may fail before
execution when its mathematics is missing, malformed, dimensionally
inconsistent, singular, or internally contradictory. A mathematically valid
declaration may fail during execution only with a typed numerical diagnosis
such as non-finite state, violated closure, or failed convergence. Neither
class may be reported as a vague unsupported-physics decision, and neither
may be influenced by LambdaCDM comparison values. `EngineCapabilityError`
may identify an internal implementation defect during development, but it is
never an accepted model status, corpus classification, or substitute for
executing a valid declaration. The owning slice must extend the engine until
every complete declaration in the grammar executes.

## CAMB Parity Contract

CAMB parity is required wherever CAMB implements the same physical model and
conventions. At minimum this includes LCDM, massive-neutrino LCDM, the fixed
Planck-reference LCDM point, and the comparable wCDM/w0wa limits. The frozen
comparison uses identical parameters, primordial spectrum, recombination
settings, neutrino treatment, ell grid, units, normalization, lensing mode,
and sector definitions.

The raw comparator must evaluate complete arrays for TT, TE, EE, BB, PP, TP,
EP, and every applicable lensed, scalar, vector, tensor, and total surface.
It reports pre-declared absolute, relative, band-limited, peak-position,
phase, damping, sign, and zero-crossing errors. Near-zero cross-spectra use
an explicit absolute-plus-relative metric rather than an unstable ratio.
Tolerances are frozen before the result is measured and cannot be changed to
make a failing result pass.

CAMB does not implement every Copernican theory. For QAU, QRSF, TOG, TORG,
USMF2, and any other non-CAMB theory, acceptance requires the same complete
finite/resolved/physical output contract plus continuity to its declared
standard limit, conservation and constraint residuals, and a documented
comparison to the closest valid reference. Such a model may not be called
CAMB-parity-proven where no CAMB equation exists; it must still produce a
sensible CAMB-like graph.

No model is a reference implementation inside CCMBS. CAMB parity is a
comparison property of matching declarations, never a permission to replace
the equations, defaults, or outputs of a non-matching theory.

## Diagnostic Status Terms

* **Accepted:** all applicable scientific, numerical, execution, and evidence
  checks pass.
* **Rejected:** the declaration is mathematically malformed, incomplete,
  inconsistent, singular, non-finite, or explicitly violates its own
  declared equations. The report names the exact defect and owning slice.
* **Execution failure:** a mathematically valid declaration encountered a
  typed solver, residual, convergence, or resource failure. This blocks
  closure and is never converted into unavailable or silently omitted output.
* **Unavailable:** only a declaration that explicitly has no CMB capability.
  Unfamiliar physics, a missing engine capability, a slow run, a missing
  graph, or a failed scientific check is not an excuse to use this label.
* **Unclassified:** the required production measurement has not completed.
  It remains open work and cannot count as success.

## Execution Rules

1. Keep the DevCovenant gate open for the active slice.
2. Activate the existing environment with
   `source .venv/bin/activate` before all tests and policy commands.
3. Clear gate complaints before applying edits. Stop and repair any new
   complaint before continuing the slice.
4. Preserve the current baseline and compare every repair against the same
   fixed request. Inspect raw arrays and diagnostics before plots.
5. Complete slices in order. Do not mark a slice closed on focused tests,
   finite output, or a green policy gate alone.
6. A reduced request is smoke evidence only. CAMB parity and scientific
   closure require the named production controls and complete raw arrays;
   reducing `ell`, k, eta, or hierarchy orders cannot turn a failed
   production result into an accepted one.
7. If a complete declaration cannot be represented by the current schema,
   compiler, hierarchy, or projector, extend that shared capability before
   closing the owning slice. Do not classify the declaration as unsupported.
8. Run focused tests, stage the changed files, and run
   `source .venv/bin/activate && python -m devcovenant gate --verify` on the
   staged revision before reporting a slice complete.
9. Run `devcovenant run`, close the gate, commit, or push only when the user
   explicitly requests that action for the turn.

Task markers mean:

* `[closed]` means implementation and raw scientific acceptance evidence
  both exist.
* `[in progress]` means implementation or evidence is active and closure is
  incomplete.
* `[planned]` means the slice has not started.

## Execution Slices

### [closed] Slice One — production LambdaCDM recovery

**Models:** `model_lcdm.yml` and `model_ref_planck2018.yml`.

**Purpose:** Restore a scientifically usable baseline in the actual
production request. This slice directly eliminates the missing LCDM graph
and the runaway Planck-reference spectrum.

**Files and surfaces:**

* generated hierarchy/source compiler and mixed-neutrino source terms;
* background parameter mapping and recombination inputs;
* production phase-aware k/source controls and convergence enforcement;
* CMB failure propagation, plot assembly, raw CSV/JSON artifacts, and
  focused tests; and
* solver documentation, README mirrors, PLAN, and CHANGELOG.

**Implementation tasks:**

1. Replace full-`Omega_nu0` residual massless-neutrino terms with the
   physically resolved massless fraction while retaining the separately
   evolved massive-neutrino contribution. Distinguish pure massless fixtures
   from mixed-neutrino declarations without weakening either contract.
2. Correct the Planck-reference mapping of CDM, baryons, radiation, massive
   neutrinos, `N_eff`, and mass sum so each density is counted exactly once.
3. Apply the final phase-aware production k grid and source grid to every
   production model, including Planck reference. Require independent doubled
   refinement for all requested surfaces.
4. Preserve raw failed theories and raise a visible typed run failure instead
   of catching an exception and silently omitting the theory curve.
5. Verify radial distance, Bessel kernels, source interpolation, quadrature
   weights, normalization, and `C_ell`/`D_ell` conversion at this fixed
   baseline.
6. Execute full raw TT, TE, EE, BB, PP, TP, and EP surfaces wherever each
   model declares them. Do not use a reduced diagnostic request as production
   evidence.

**Acceptance:**

* LCDM returns a finite, converged production spectrum and its graph contains
  the LCDM theory curve;
* Planck-reference LCDM returns smooth acoustic TT/TE/EE spectra with the
  correct amplitude order and no runaway spikes;
* both models retain complete raw source, grid, refinement, residual, and
  failure metadata;
* the fixed Planck-reference request agrees with the frozen standard-model
  anchors at the applicable surfaces; and
* a missing or failed theory is visible and typed, never silently plotted as
  data-only output.

**Closure evidence:** Two canonical production bundles contain raw arrays,
source histories, phase metadata, doubled-grid comparisons, source residuals,
complete graph files, and deterministic decisions for both models.

**Implementation and evidence (2026-09-02):** The shared scalar source
compiler now separates the resolved massless-neutrino fraction from the
separately evolved massive component, and the fixed Planck declaration maps
the 0.06 eV mass into `omnuh2` before deriving `omch2`. The production
projection uses generalized Simpson integration on the declared phase-aware
coordinates, with a positive fallback for material negative auto-spectrum
lobes. Typed CMB failures are retained by the pipeline and shown by the
plot infobox instead of dropping a failed theory. The fixed LCDM full-spectrum
request (TT, TE, EE, and PP over ell=2..2000) returned finite spectra and
passed doubled-grid convergence; the Planck fixed-point diagnostic returned
finite TT/TE/EE raw arrays with the corrected mixed-neutrino bookkeeping.
Focused contract, model-mapping, plotting, pipeline, and quadrature tests
pass. The raw runtime envelopes retain the source, grid, refinement, and
typed-failure fields required by the closure record.

### [closed] Slice Two — massive-neutrino and radiation closure

**Model:** `model_lcdm_mnu.yml`, including the shared path used by the Planck
reference.

**Purpose:** Complete the neutrino physics rather than merely suppressing its
numerical symptom. This slice owns the declared-domain bookkeeping and the
q-resolved background closure consumed by every later production parity
surface.

**Implementation tasks:**

1. Separate massless radiation, massive-neutrino background density, and
   cold-dark-matter density in the background and perturbation equations.
2. Audit `N_eff`, number of massive species, mass sum, thermal moments,
   hierarchy truncation, free streaming, and initial conditions.
3. Define one bounded effective-species allocation for the full declared
   domain. For `N_eff < num_massive_neutrinos`, distribute only the available
   effective degeneracy among the massive family and keep the residual
   massless fraction non-negative. Prove that all present and early density
   fractions sum exactly once at every tested point.
4. Derive the massive background density and pressure from the same
   q-resolved Fermi-Dirac moments used by the perturbation hierarchy. Remove
   the `max(relativistic, nonrelativistic)` kink; if an approximation is
   retained temporarily, a fixed numerical error bound against the q integral
   and CAMB must be recorded before acceptance.
5. Check energy and momentum source terms independently against the declared
   density split at multiple mass and `N_eff` points, including the zero-mass
   and relativistic/non-relativistic limits.
6. Compare fixed points against CAMB with matching masses, effective species,
   recombination, primordial spectrum, units, normalization, and lensing
   settings. The matrix must include zero, 0.06, 0.15, and 0.30 eV mass sums
   where conventions are comparable, and `N_eff` values at the lower bound,
   standard value, and upper bound.
7. Run those comparisons at the declared production controls, not the
   reduced smoke controls, and retain base/refined arrays, histories, source
   residuals, and typed failures for every row.
8. Produce complete declared auto- and cross-spectra and their graphs.

**Acceptance:** Neutrino mass and `N_eff` changes produce finite continuous
responses throughout the declared domain, including `N_eff` below the
massive-species count; no density is double counted; q-integrated background
and hierarchy residuals pass; the relativistic-to-non-relativistic response is
smooth; CAMB parity passes for every applicable production surface; and
`model_lcdm_mnu.yml` has a complete raw and plotted production result.

**Closure evidence:** Neutrino split equations, q-integral background and
pressure tables, source residuals, low-`N_eff` allocation checks, multiple
mass/`N_eff` fixed-point runtime reports, full-observable production arrays,
hashes, and graph artifacts. A reduced finite-spectrum request is retained as
smoke evidence only and cannot close this slice.

**Prior implementation evidence (2026-09-03; retained as baseline, not
closure):** The LambdaCDM+Mnu and
Planck-reference backgrounds now split residual massless radiation from the
q-resolved massive component, derive cold dark matter after the massive
neutrino density, and use a finite relativistic-to-nonrelativistic `H(a)`
interpolation normalized at `a=1`. The zero-mass limit remains finite and
continuous. Fixed q-grid tests cover physical temperature, density, pressure,
momentum, and shear weights, relativistic and nonrelativistic limits, q-grid
refinement, and hierarchy refinement. The real bundled LambdaCDM+Mnu
declaration emits finite TT, TE, EE, BB, PP, TP, and EP surfaces under a
deterministic reduced production request; present-day component closure and
early-time radiation closure are asserted directly. Focused model, hierarchy,
neutrino-grid, and full-declared-surface tests pass.

**Reopening requirements (2026-09-03; satisfied):** The allocator, shared
generated source path, and model declarations were updated together so the
lower `N_eff` domain cannot overcount radiation. Production q-integral tables
and the fixed mass/`N_eff` runtime matrix are generated from the same
quadrature used by the hierarchy. The previous reduced tests remain
regression smoke evidence and are not used as the closure decision.

**Closure implementation and evidence (2026-09-03):** The shared allocator
now uses the bounded effective species count `min(N_eff,
num_massive_neutrinos)` for both generated residual-neutrino sources and the
q-resolved runtime. The LambdaCDM+Mnu, Planck-reference, wCDM, and w0wa
declarations use that same allocation, derive CDM after the massive density,
and replace the former `max(relativistic, nonrelativistic)` kink with a smooth
zero-mass continuation. The background builder applies the q-resolved
Fermi-Dirac density moment to `H(a)` and validates its shape, finiteness,
positivity, and present-day normalization; pressure, momentum, and shear
moments remain exposed from the identical quadrature. Focused production
surface, q-moment, independent-integral, model-contract, zero-mass, and
low-`N_eff` matrix tests pass. The fixed CAMB reference helpers and complete
observable plumbing remain in the canonical diagnostics path; subsequent
model slices revalidate their production parity rows after consuming this
shared background closure.

### [closed] Slice Three — dark-energy model closure

**Models:** `model_wcdm.yml` and `model_w0wa.yml`.

**Purpose:** Make the dark-energy expansion and perturbation response
theory-faithful and CAMB-comparable where the equations overlap.

**Implementation tasks:**

1. Audit `H(a)`, density evolution, equation-of-state interpolation, sound
   speed, perturbations, metric sources, and recombination timing.
2. Verify the `w=-1` and `w0=-1, wa=0` limits reduce continuously to the
   accepted LCDM baseline.
3. Compare common wCDM and w0wa points against matching CAMB settings.
4. Compute every declared TT, TE, EE, BB, PP, TP, EP, lensed, tensor, and
   vector surface and retain raw convergence evidence.
5. Generate complete plots for both models.

**Acceptance:** Both models execute at production resolution; their standard
limits agree with Slice One; parameter responses are finite and physical;
their declared background law passes the independent CAMB-comparable
constant-w/CPL checks; and graphs show resolved acoustic structure rather
than a flat or noisy surrogate. Full cross-model spectral parity remains in
the complete-spectrum matrix owned by Slice Seven.

**Prior implementation evidence (2026-09-03; pending Slice Two
revalidation):** The wCDM and w0waCDM CMB
declarations now subtract the fixed 0.06 eV massive-neutrino density from
the cold-dark-matter budget, split massless radiation from the q-resolved
massive component, and normalize the remaining dark energy at `a=1`. Their
backgrounds use the constant-`w0` and CPL density factors directly, with a
finite massive-neutrino transition. Focused real-model tests assert positive
finite `H(a)`, present-day closure, exact agreement between the wCDM and
`w0=-1, wa=0` limits, and a non-zero response to both dark-energy parameters.
Each model also emits finite TT, TE, EE, BB, PP, TP, and EP arrays from the
same deterministic reduced production request. The bundled-model finite-TT
matrix and model-contract validation pass with the updated declarations.

**Closure evidence (2026-09-04):** The shared runtime now validates every
declared dark-energy density factor against the normalized constant-w/CPL
law, checks the matching pressure factor and non-negative sound speed, and
records a `smooth_background` audit in each raw runtime envelope. wCDM and
w0waCDM explicitly declare their smooth dark-energy species, unit sound
speed, zero anisotropic stress, and absence of a dark-energy hierarchy.
Focused contract and background tests pass, including the exact `w=-1`,
`w0=-1, wa=0` limit and finite parameter responses. Both real model
declarations were executed with their production controls over ell=2..2000;
all seven declared TT, TE, EE, BB, PP, TP, and EP arrays were finite and the
production doubled-k convergence gate passed. The complete spectral CAMB
parity rows are retained for the all-model matrix in Slice Seven rather than
being duplicated here.

### [closed] Slice Four — QAU and QRSF model closure

**Models:** `model_qauc.yml` and `model_qrsf.yml`.

**Purpose:** Complete the modified theories without routing their equations
through accidental LCDM defaults.

**Implementation tasks:**

1. Audit each model's background, scalar source, metric, gauge, collision,
   derivative, closure, and observable declarations at equation level.
2. Repair only omitted or misbound expressions that are already part of the
   declared theory.
3. Verify standard-limit behavior where the theory defines one and compare
   that limit against the accepted LCDM reference.
4. Resolve the known QRSF acoustic-shape failure through shared source or
   projection correctness, not a model-only tolerance or plot patch.
5. Compute all declared surfaces and produce complete model graphs.

**Acceptance:** Both models have finite, phase-resolved, physically shaped
spectra; source and constraint residuals pass; distinct parameters produce
distinct histories; standard-limit comparisons pass; and no equation is
silently replaced by a generic LCDM term.

**Closure evidence (2026-09-04):** QAUC now normalizes its oscillatory
attractor envelope at the present epoch, separates the fixed massive-neutrino
component from CDM, and uses the shared q-resolved radiation and matter
closure. QRSF now normalizes its baryon-locking and relational-release
background at the present epoch, derives the effective matter and dark-energy
fractions from that normalization, and scales its named matter source
closures consistently. Both declarations explicitly expose a smooth
dark-energy species with unit sound speed and zero anisotropic stress; no
new hierarchy or LCDM fallback is introduced. Focused contract, background,
standard-limit, parameter-response, and complete TT/TE/EE/BB/PP/TP/EP
surface tests pass. At the deterministic production-shaped controls used by
the model matrix, both routes return finite arrays for every declared
observable and the shared doubled-k convergence gate remains active. Full
CAMB-comparable rows are retained for the all-model parity matrix in Slice
Seven.

### [closed] Slice Five — TOG and TORG model closure

**Models:** `model_tog.yml` and `model_torg.yml`.

**Purpose:** Complete the temporal/relational theories' gauge routes,
metric bridges, hidden-prefix evolution, and observable sources.

**Implementation tasks:**

1. Make every declared gauge route expose explicit shared observables and
   source histories.
2. Verify superhorizon evolution begins before the line-of-sight grid and
   remains stable when a request starts later.
3. Restore and test the physical photon collision and polarization blocks,
   `Phi_tau`, `Psi_tau`, visibility, ISW, and all required derivatives.
4. Check scalar, vector, and tensor declarations and their cross-spectra.
5. Produce complete raw reports and CAMB-like graphs for both models.

**Acceptance:** Histories are present, finite, gauge-consistent, and refined;
source residuals pass; all declared observables are computed; standard or
controlled limits are continuous; and the graphs show stable resolved
structure rather than request-dependent artifacts.

**Closure evidence (2026-09-04):** TOG now separates its q-resolved
massive-neutrino background from CDM and normalizes its volume-activated
temporal-opposition factor at the present epoch. TORG now separates residual,
massless, and massive radiation, normalizes its baryon-locked relational
matter and temporal-response factors, and retains its explicit matter-density,
matter-momentum, and baryon-Euler closures without adding a CDM species.
The runtime background audit validates both model-specific histories and
present-day closure. Focused gauge/source-contract, hidden-superhorizon-prefix,
visibility-refinement, background-closure, and complete
TT/TE/EE/BB/PP/TP/EP surface tests pass for both routes; bundled declaration,
source-graph, and finite-TT corpus audits remain green. The low-resolution
complete surface reports are deterministic and finite; full CAMB-comparable
parity rows and production graph bundles remain assigned to Slice Seven and
Slice Eleven, respectively.

### [closed] Slice Six — USMF2 model closure

**Model:** `model_usmf2.yml`.

**Purpose:** Finish the multi-fluid model instead of leaving it unclassified
or treating its explicit graph as sufficient evidence.

**Implementation tasks:**

1. Execute USMF2 at the same named production tier as the other models.
2. Audit every fluid, interaction, metric, gauge, initial-condition, source,
   sector, and observable declaration.
3. Validate conservation, constraint, hierarchy, visibility, polarization,
   ISW, and lensing behavior on independent refinements.
4. Verify all declared TT, TE, EE, BB, PP, TP, EP, lensed, vector, and tensor
   surfaces without fabricating unsupported values.
5. Add only theory-faithful declaration content where the explicit graph is
   incomplete, and generate the complete production graph bundle.

**Acceptance:** USMF2 is accepted with finite complete spectra and clean raw
diagnostics, or it has a precise typed unavailable decision tied to an
explicit non-CMB declaration. It may not remain unclassified, be timed
out into unavailable, or be substituted by another model.

**Closure evidence (2026-09-04):** USMF2 now declares the shared doubled-k
production convergence rule for TT, TE, EE, and PP with fail-closed typed
handling. Its shrinking-background audit verifies the declared H(a)
normalization, photon-plus-neutrino radiation closure, and baryon-only matter
closure on an independent scale-factor grid. Contract tests enumerate every
USMF2 equation, metric constraint, hierarchy tail, collision operator, source,
and public observable. The low-resolution independent eta refinement emits
finite TT, TE, EE, BB, PP, TP, and EP arrays, while the production-shaped
route records a converged base/refined k comparison and the full model emits a
finite default TT mode. USMF2 remains explicitly scalar: vector, tensor, and
lensed surfaces are represented as typed non-applicable declarations rather
than fabricated values. Full CAMB parity rows are now enforced by Slice Seven;
end-to-end GUI and graph bundles remain assigned to Slice Eleven.

### [closed] Slice Seven — complete full-spectrum projection parity

**Scope:** every discovered mathematically valid CMB declaration, with all
ten bundled models as the mandatory certification corpus.

**Purpose:** Certify that CCMBS computes the complete observable boundary,
not just scalar TT/TE/EE.

**Implementation tasks:**

1. Complete scalar, vector, tensor, lensing, lensed, unlensed, auto-, and
   cross-spectrum routes.
2. Verify TT, TE, EE, BB, PP, TP, and EP normalization, signs, units,
   multipole ordering, and applicability records independently.
3. Compare LCDM, massive-neutrino LCDM, Planck reference, wCDM, and w0wa
   raw arrays against the frozen CAMB fixture wherever comparable.
4. Verify low-ell, acoustic, damping-tail, tensor, and lensing refinements
   independently for every requested surface.
5. Check lensed/unlensed consistency, lens-potential normalization, BB
   generation, and cross-spectrum near-zero behavior.
6. Require graphs and raw reports to contain every declared theory and
   observable; a missing plot is a hard failure.

**Acceptance:** The complete parity report has one raw comparison row per
applicable model, sector, and observable; all fixed comparable surfaces pass
frozen tolerances; non-CAMB theories pass their full physical and
standard-limit contract; and no declared surface is omitted or replaced by
a placeholder.

**Closure evidence (2026-09-04):** The full-observable matrix now derives its
request from every compiled model declaration and requires an exact ordered
`TT`, `TE`, `EE`, `BB`, `PP`, `TP`, and `EP` row for each enabled model.
`CAMB_COMPARABLE_CMB_MODEL_FILENAMES` identifies the five models with a
defensible CAMB counterpart; their matrix rows require an independently
supplied fixed-point reference comparison, while non-CAMB theories remain
subject to the complete CCMBS physical and refinement evidence without a
theory-changing reference substitution. Unknown reference rows, dropped
declared surfaces, missing raw products, failed doubled-grid convergence,
failed residual audits, missing scalar/batch evidence, and non-isolated cache
identities are rejected explicitly. The matrix preserves typed unavailable
declarations and typed solver failures, never turning either into acceptance.
Full-parity helpers and strict comparable-row tests cover all seven public
observables and preserve the deterministic matrix digest. Full production
matrix execution uses `CMB_CERTIFICATION_TIER` (ell=2..300, doubled-k
refinement, k=1024 base) and cannot claim comparable acceptance without the
corresponding independent CAMB fixture.

### [closed] Slice Eight — universal declaration admission and executable
theory boundary

**Scope:** model discovery and future declarative contracts.

**Purpose:** Ensure the solver is universal beyond today's ten files. A valid
declaration must be admitted and executed regardless of its theory, while
mathematically invalid declarations fail with a precise, LCDM-neutral
diagnosis. A valid declaration is never finished as a capability-gap result.

**Implementation tasks:**

1. Discover model files automatically and derive the request from their
   declared sectors and observables.
2. Validate new species, source roles, derivatives, gauges, hierarchy orders,
   tensor/vector sectors, and lensed outputs before execution.
3. Replace the exception-message classifier with source-typed boundaries:
   declaration validation raises `ModelDeclarationError`, an undeclared
   request raises `UnsupportedCapabilityError`, and a numerical failure
   retains its typed convergence, constraint, or non-finite diagnosis.
   `EngineCapabilityError` is an internal implementation assertion only and
   cannot be emitted as a terminal model status.
4. Audit every validator, compiler, discovery, and execution catch/raise
   site so a valid declaration cannot be convicted by a generic `ValueError`,
   `KeyError`, or message substring. Any such failure is a failing engine test
   that must be repaired in the shared path.
5. Add representative future-model and adversarial fixtures covering new
   species, source and sector combinations, altered recombination, and
   non-standard interactions without LambdaCDM defaults.
6. Extend the schema, compiler, hierarchy, background, projector, and
   diagnostics whenever a complete fixture exposes a missing construct. The
   fixture must execute through the repaired engine; it may not be retained as
   a capability-gap demonstration or accepted as unavailable.
7. Make genuinely malformed or underdetermined mathematics fail with an
   actionable typed contract error naming the exact missing or inconsistent
   equation, domain, dimension, or closure.
8. Prohibit silent skip, reduced-grid downgrade, false unavailable status,
   reference-model substitution, and model-name-dependent validation.
9. Preserve complete raw spectra, source histories, typed failures, and
   classification evidence for every fixture and every declared observable.

**Acceptance:** All ten current models remain accepted; every valid future or
adversarial fixture executes through universal CCMBS; only mathematically
invalid declarations fail explicitly; and no model can disappear from the
corpus or plot without a recorded reason. No valid theory is labeled
unsupported, unavailable, or engine-gap because its equations differ from
LambdaCDM. Any valid fixture that fails execution keeps the slice open until
the shared CCMBS implementation is repaired; a green test that merely records
an engine gap is not acceptance.

**Closure evidence (2026-09-05):** Discovery enumerates both
`model_*.yml` and `model_*.yaml` without a bundled LambdaCDM admission rule.
Declaration validation, perturbation compilation, and generated-callable
compilation now wrap legacy parser/schema exceptions at their source as
`ModelDeclarationError`; explicit undeclared-output requests raise
`UnsupportedCapabilityError`; numerical failures retain their convergence,
constraint, and non-finite classes; and the public fallback classifier maps
every remaining untyped exception to `ImplementationError` without inspecting
message text. An internal capability assertion is recorded as an engine
implementation failure and can never become a model rejection or accepted
result. Adversarial unrelated-theory, altered-recombination, and multisector
fixtures execute finite declared surfaces, while malformed declarations retain
actionable typed diagnostics. Focused taxonomy, discovery, model-adapter, and
perturbation-contract tests cover these boundaries, and README mirrors plus
solver documentation describe the same universal contract.

### [closed] Slice Nine — BAO and background closure

**Scope:** BAO independence and shared background inputs.

**Purpose:** Preserve BAO's independence while correcting any shared
background bookkeeping exposed by the CMB repair.

**Implementation tasks:**

1. Run BAO with the CMB entrypoint unavailable and verify identical fixed
   background results, covariance handling, and typed failures.
2. Verify the drag-epoch sound horizon is used for BAO and is not confused
   with the recombination sound horizon used by CMB diagnostics.
3. Recheck massive-neutrino, dark-energy, and modified-background mappings.
4. Ensure divergent sound-horizon integrals fail explicitly and never produce
   a finite-looking ratio.
5. Run fixed-background BAO regression checks for every applicable model.
6. Re-audit the BAO boundary after universal model admission. A declared
   drag-epoch ruler is authoritative; BAO must not evaluate a legacy
   recombination helper before, or in place of, that ruler. A legacy helper
   may be used only by an external plugin that has no declared drag helper,
   and its epoch must remain explicit in diagnostics.
7. Preserve the same typed failure semantics for canonical drag failures,
   invalid values, missing distance helpers, and divergent integrals. A
   finite-looking BAO ratio must never be recovered from another epoch.
8. Permit a theory to declare `bao_drag_redshift_expression` when its drag
   transition is not represented by the standard fitting relation. Validate
   the expression against that theory's parameters and replace only the
   outer sound-horizon endpoint.
9. Resolve helper call signatures before invocation so a `TypeError` raised
   by model mathematics is never retried as a different arity or hidden by a
   compatibility fallback.
10. Preserve typed BAO failure metadata for missing helpers, invalid shapes or
    values, non-finite predictions, invalid drag redshifts, and dataset errors.

**Acceptance:** BAO remains finite and independently evaluable, does not
invoke CCMBS, uses the declared drag epoch and density inputs, and is
unchanged at fixed background points except where a documented bookkeeping
bug is corrected. A valid canonical drag helper is used without evaluating
the recombination helper; a present but invalid drag helper is a typed BAO
failure, not permission to substitute a recombination ruler. External
plugins without a drag helper retain an explicitly labelled compatibility
path only.

**Closure evidence (2026-09-05):** This slice was reopened after the
universal admission boundary in Slice Eight was closed so that its BAO
contract could be audited against the same theory-neutral rule. Generated
model plugins retain the declared recombination sound horizon and derive a
separate drag-epoch horizon from the Eisenstein--Hu drag-redshift fit.
`BAOLike` now resolves a finite background drag value first, then the
canonical drag helper, and only then the explicitly labelled compatibility
helper for an external plugin that has no drag endpoint. It never probes the
recombination helper before a valid drag endpoint and never replaces an
invalid canonical drag result with recombination. State metadata records the
actual source and epoch. The CMB runtime keeps its visibility/recombination
sound horizon separate. All ten bundled models produce finite, positive,
distinct recombination and drag horizons at their fixed initial points.
Focused tests prove CMB-entrypoint isolation, covariance preservation,
canonical drag precedence, invalid-drag rejection, divergent-integral
failure, drag-helper selection, and epoch evidence without a CMB call. The
  epoch diagnostic now records a typed failure category and the failed
  recombination or drag endpoint instead of raising an unclassified runtime
  exception. Model declarations may now provide a theory-specific
  `bao_drag_redshift_expression`; the compiler validates its symbols and
  supports an expression-valued recombination lower limit while preserving
  the separate drag endpoint. BAO helper invocation is signature-aware, so
  genuine model `TypeError` failures are not retried with a different
  arity. Invalid or non-positive drag redshifts are reported as typed drag
  evidence. Focused model-coder, BAO likelihood, and diagnostics tests cover
  these paths, including no-parameter external helpers and a broken helper
  that must be called exactly once. The likelihood now preserves typed
  metadata for missing helpers, invalid background shapes or values,
  non-finite predictions, invalid drag endpoints, and invalid datasets
  instead of returning an unclassified negative infinity. The final
  focused BAO suite also proves malformed input shapes and non-finite
  distance backgrounds fail before a ratio can be formed, while canonical
  drag and legacy-helper failures retain their stage and category. The
  complete Slice Nine acceptance surface is therefore implemented and
  represented in tests; no BAO/background task remains unassigned.

### [closed] Slice Ten — generic recombination and opacity execution

**Scope:** the recombination, opacity, visibility, and drag inputs exposed by
the universal declaration grammar, including theories that do not use the
hydrogen/Peebles vocabulary.

**Purpose:** Ensure that every well-posed recombination theory expressible by
the declaration grammar is actually eaten by CCMBS. LambdaCDM's four
standard quantities are one adapter, not a universal ontology. A valid
non-Peebles theory must execute through the same compiler, hierarchy, and
projector and return its own physical results.

**Implementation tasks:**

1. Define declaration-driven roles for recombination state, rate equations,
   opacity, optical-depth derivative, visibility, and drag transition. Keep
   the existing hydrogen/Peebles implementation as a standard adapter rather
   than a generic required vocabulary.
2. Extend validation and `model_coder.py` to check arbitrary declared state
   and rate names for completeness, dimensions, domains, derivatives, finite
   limits, and closure bindings. Partial, singular, inconsistent, or
   underdetermined mappings fail as `ModelDeclarationError`.
3. Extend generated background, visibility, polarization, hierarchy, and BAO
   inputs to consume declared roles and histories without LambdaCDM-specific
   lookups. Keep recombination and drag as separate physical epochs.
4. Add a genuinely non-Peebles adversarial declaration with a different
   opacity and visibility ontology. Execute every surface it declares,
   including TT, TE, EE, BB, PP, TP, EP, and any applicable lensed, vector,
   or tensor surfaces, with no reference-model substitution.
5. Add invalid and singular fixtures and require every valid fixture to run.
   A failure to execute valid mathematics is a failing CCMBS implementation,
   not an accepted capability-gap or unavailable result.
6. Compare the standard adapter with fixed pre-adapter outputs and validate
   the non-standard fixture on independent state, visibility, drag, closure,
   conservation, and standard-limit refinements.
7. Update templates, comments, docstrings, README mirrors, solver
   documentation, focused tests, raw evidence, and CHANGELOG.

**Acceptance:** The ten bundled models remain finite and unchanged at fixed
points, and the non-Peebles declaration executes all of its declared
surfaces through universal CCMBS. No generic path requires a LambdaCDM
quantity name. Only mathematically invalid declarations may fail before
execution; inability to execute a valid declaration keeps this slice open
until the shared engine is repaired.

**Closure evidence:** A deterministic report contains the declaration, role
map, state/rate/opacity histories, visibility and drag histories, complete
raw spectra, independent refinement comparisons, typed mathematical failures,
and graph paths. Matching CAMB parity is included where meaningful;
otherwise the report contains the theory's conservation, constraint, and
standard-limit evidence.

**Closure evidence (2026-09-05):** The declaration compiler now accepts an
explicit role graph for generic recombination. `electron_fraction`, `opacity`,
and `visibility` are required physical roles; optional state, rate,
matter-temperature, and drag-transition roles remain declaration names and
are preserved in the compiled runtime manifest. The shared background
evaluates these histories on its own grid, uses declared opacity as the
positive magnitude of `d tau/d eta`, and consumes the declared visibility
kernel and matter temperature without routing through hydrogen or Peebles
quantity names. The standard four-quantity adapter remains unchanged when no
role graph is present. A renamed, genuinely non-Peebles fixture executes all
seven public surfaces (TT, TE, EE, BB, PP, TP, EP) with finite raw arrays;
compiler tests cover complete role compilation, target binding, and missing
required roles. Documentation and the mirrored README describe the
declaration-driven boundary. No generic recombination or opacity task remains
unassigned in this slice.

### [closed] Slice Eleven — end-to-end production certification

**Scope:** GUI, CLI, sampler integration, artifacts, and final closure after
Slices Eight and Ten have been implemented.

**Purpose:** Prove that a user running Copernican receives the same complete
scientific result that passed the raw-array tests.

**Implementation tasks:**

1. Discover every bundled and user-supplied mathematically valid CMB
   declaration and run the GUI/CLI workflow at its declared production
   settings, writing one complete bundle per declaration.
2. Require each bundle to contain raw spectra, source histories, residuals,
   convergence evidence, parity or standard-limit reports, and all graphs.
3. Verify every configured theory and declared observable appears in the
   appropriate plot and CSV/JSON output.
4. Run sampler integration only after fixed-point spectra pass, and verify
   that posterior summaries cannot hide a failed CMB computation.
5. Audit runtime imports, fallback paths, aliases, surrogates, wall-clock
   decisions, machine-local paths, omitted surfaces, and plot-only decisions.
6. Synchronize code, model templates, comments, docstrings, README mirrors,
   PLAN, focused tests, and CHANGELOG with the certified behavior.
7. Run the complete required workflow in the existing Python 3.11
   environment, then run gate verification on the staged revision.

**Acceptance:** A clean invocation produces sensible CAMB-like graphs for all
discovered mathematically valid CMB declarations, including the ten bundled
models; every declared surface is finite, resolved, and present; all
CAMB-comparable raw arrays pass; non-CAMB theories pass their physical
contracts; BAO remains independent; and no runtime fallback or silent
omission exists.

**Closure evidence:** One deterministic final report contains all model rows,
full-observable arrays and hashes, source and refinement evidence, graph
paths, parity decisions, BAO isolation, repository audit results, and the
final green gate verification. Only this evidence can close the plan.

**Closure evidence (2026-09-06):** The final certification runner now executes
the complete declared-observable matrix, rehydrates its lossless raw reports,
and applies repository-integrity and BAO-isolation decisions without
re-running the solver or consulting a plot. `CMBModelDiagnostic.from_dict`
preserves arrays, envelopes, failures, and cache identities when a matrix is
serialized and passed to the final boundary. The runner can persist the
deterministic JSON record, while package-level exports make the matrix,
integrity audit, and final writer available to CLI/GUI integrations. Focused
tests cover public exports and a mocked production hand-off; the full
scientific workflow remains an operator-run command rather than an implicit
side effect of report assembly. No Slice Eleven implementation task remains
unassigned.

## Completion Standard

This plan is complete only when all eleven slices are closed. The following
requirements are assigned to slices above; this section introduces no
unassigned work:

* all ten known model files execute through universal CCMBS with
  theory-faithful declarations (Slices One through Six, Eight, Ten, and
  Eleven);
* every mathematically complete future declarative model is discovered and
  executed, while only mathematically invalid declarations fail with an
  explicit LCDM-neutral diagnosis (Slices Eight, Ten, and Eleven);
* generated background, metric, hierarchy, collision, visibility,
  polarization, initial-condition, and ISW histories are finite and
  independently residual-clean (Slices One through Six and Seven);
* every declared TT, TE, EE, BB, PP, TP, EP, lensed, unlensed, scalar,
  vector, tensor, and total surface is computed from raw transfers (Slices
  One through Seven, Ten, and Eleven);
* CAMB-comparable models pass the frozen raw-array parity contract, including
  full applicable sectors and observables (Slices One through Three, Seven,
  Ten, and Eleven);
* non-CAMB theories pass complete numerical, physical-shape, conservation,
  constraint, and standard-limit checks and produce sensible graphs (Slices
  Four through Six, Eight, Ten, and Eleven);
* scalar/batch equivalence, ordering, cache isolation, and typed failure
  semantics are proven for every accepted model (Slices Seven, Eight, Ten,
  and Eleven);
* LCDM is never silently omitted and every failed theory remains visible as a
  typed failure rather than being replaced by a reference curve (Slices One,
  Seven, Eight, Ten, and Eleven);
* BAO is independently evaluable with correct background and drag-horizon
  behavior (Slice Nine); and
* final raw evidence, graphs, reports, documentation, staged files, and
  DevCovenant verification are green (Slice Eleven).

The final raw scientific report and production graph bundles—not a policy
gate, finite output, or attractive screenshot alone—are proof that universal
CCMBS works as the old CAMB solver worked. LambdaCDM is one certified theory,
not the engine's hidden reference. No mathematically complete theory may be
left outside the admission and production evidence, and no scientific
requirement may remain outside these slices when the plan is closed.
