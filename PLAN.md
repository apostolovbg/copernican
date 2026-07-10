# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-07-09
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

Use this plan to complete the native CMB solver.

The previous roadmap grouped several independent implementation sessions
inside broad slices. That structure allowed individual plumbing or proof
changes to be presented as closure while major physical work remained.

This replacement plan uses real sequential slices.

Each slice must be independently implementable, testable, documentable,
and closable in one work session. If a slice cannot be completed in one
session, the plan must be divided into additional slices before coding
continues. Do not create hidden sub-slices, work packages, phases, or
follow-up cleanup inside a slice.

The target condition is final:

* `standard: true` stays on the standard backend path.
* `standard: false` stays native, CAMB-free, and CLASS-free in production.
* The native path implements physical scalar, vector, and tensor
  Einstein-Boltzmann sectors.
* Massive neutrinos use one authoritative q-resolved hierarchy with
  physical momentum integration.
* Gauge support uses explicit transformations or gauge-invariant
  variables rather than aliases that force identical output.
* Collision handling is compiled from declared operator metadata.
* All standard CMB spectra are produced when the declared theory supplies
  the required physical sources.
* Completion is demonstrated through native absolute reference parity and
  spectrum convergence.
* TORG remains untouched during this roadmap.

## Table of Contents

* [Problem Preamble](#problem-preamble)
* [Current Baseline](#current-baseline)
* [Overview](#overview)
* [Execution Rules](#execution-rules)
* [Execution Slices](#execution-slices)
* [Completion Standard](#completion-standard)

## Problem Preamble

Copernican already has a substantial native declared-graph CMB engine.

The current branch includes:

* native upstream perturbation compilation;
* native background, evolution, projection, and caching;
* exact curved-sky lensing remapping;
* q-resolved massive-neutrino states;
* requested-spectrum filtering;
* full scalar spectrum-family plumbing;
* primordial `BB` preservation through lensing;
* multi-spectrum likelihood support;
* sector, parity, source-role, and conservation validation;
* generic execution of declared vector-like and tensor-like variables.

Those capabilities are the baseline. They are not proof that the native
solver is physically complete.

The remaining gaps are:

* scalar Einstein equations and metric-source normalization;
* physical photon, polarization, matter, and neutrino hierarchy closure;
* physical recombination, reionization, and tight coupling;
* physical q integration for massive neutrinos;
* generic collision-operator integration;
* regular scalar initial-condition modes;
* genuine synchronous and gauge-invariant support;
* physical vector Boltzmann hierarchies;
* physical tensor Boltzmann hierarchies;
* absolute native comparison with CAMB or CLASS;
* demonstrated numerical convergence.

This roadmap closes those gaps sequentially. No slice may be closed by
adding metadata, finite-array checks, source-string assertions, or
response-only tests while its physical target remains absent.

## Current Baseline

The following items are already closed and must not be added back to the
roadmap as unfinished work:

* The native solver route is selected for `standard: false`.
* The native route does not use CAMB or CLASS in production.
* The exact curved-sky lensing remapper exists.
* Gaussian lensing smoothing is removed.
* Artificial native spectrum scale constants are removed.
* Direct `PP` injection into lensed `BB` is removed.
* Declared primordial `BB` is included when calculating `lensed_BB`.
* Requested-spectrum dependencies are resolved before native projection.
* Requested spectra are included in native cache identity.
* Multi-spectrum likelihood results use returned spectrum positions
  rather than ell values as array indices.
* Photon-baryon Thomson momentum conservation is declared.
* The scalar materializer creates photon, baryon, CDM, polarization,
  massless-neutrino, and massive-neutrino states.
* Massive-neutrino q-bin states are generated and evolved.
* `TT`, `TE`, `EE`, `BB`, `PP`, `TP`, `EP`, and lensed outputs can be
  represented when declared.
* Generic vector-like and tensor-like graph variables can execute.
* Sector-incompatible cross spectra fail before execution.

The following remain open:

* physical scalar normalization;
* physical massive-neutrino quadrature;
* generic collision integration;
* physical gauge equivalence;
* complete initial modes;
* physical vector and tensor sectors;
* native absolute scientific parity;
* numerical convergence.

## Overview

The remaining work is divided into ten slices.

Slice One locks the physical convention.

Slice Two completes scalar Einstein equations.

Slice Three completes the photon, baryon, polarization, and thermodynamic
system.

Slice Four completes the scalar neutrino hierarchy and physical
massive-neutrino integration.

Slice Five replaces special-case collision handling with compiled generic
collision blocks.

Slice Six completes scalar initial modes and gauge support.

Slice Seven implements the physical vector sector.

Slice Eight implements the physical tensor sector.

Slice Nine establishes native absolute reference parity.

Slice Ten establishes convergence and closes the repository truth.

Each slice includes its own implementation, tests, documentation, and
changelog entry. There is no later cleanup slice.

## Execution Rules

* Execute slices strictly in order.
* Do not begin a later slice while an earlier slice is open.
* Each slice represents one complete work session.
* If a slice requires another work session, divide it into new numbered
  slices before implementation continues.
* Do not hide additional sessions under tasks, work packages, follow-ups,
  polish, or cleanup.
* Do not add optional or deferred physics.
* Do not modify TORG during this roadmap.
* Use a neutral native standard cosmology for acceptance testing.
* CAMB or CLASS may be used only as independent test references.
* Production native code must not import or call CAMB or CLASS.
* Do not alter `standard: true` behavior except where shared output
  consistency requires it.
* Do not add empirical output scales, direct spectrum injections, hidden
  damping, or test-only physical terms.
* Missing spectra must remain unavailable with a reason.
* Do not replace unavailable spectra with zeros.
* Physically zero and unavailable must remain distinct states.
* A test name must describe what its assertions actually prove.
* Grid-size activation tests do not count as convergence tests.
* Parameter-response tests do not count as absolute parity tests.
* Gauge labels do not count as independent gauge implementations.
* Tagged vector or tensor variables do not count as physical sectors.
* Every slice must update its touched documentation and `CHANGELOG.md`.
* Stage completed slice changes before beginning the next slice.
* Do not commit or push unless explicitly instructed.

Task markers mean:

* `[open]` active work for this roadmap;
* `[closed]` completed and validated for this roadmap.

## Execution Slices

### [closed] Slice One - Canonical CMB physical convention

Purpose:

Define one authoritative physical convention for every native CMB state,
equation, source, projection, and spectrum before further equations are
changed.

Depends on:

* Current native CMB baseline.

Probable affected files:

* `copernican/docs/cmb_solver.md`
* `copernican/docs/model_template.yml`
* `copernican/lib/perturbation_contract.py`
* `copernican/lib/cmb_projection_contract.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Use conformal time and comoving `k` in inverse Mpc.
* Select one named scalar convention as canonical.
* Define photon temperature multipoles.
* Define photon E- and B-polarization multipoles.
* Define baryon and CDM density and velocity variables.
* Define massless-neutrino variables and multipoles.
* Define massive-neutrino q-bin perturbations.
* Define scalar metric potentials and their signs.
* Define vector metric variables.
* Define tensor metric variables.
* Define velocity-divergence and anisotropic-stress signs.
* Define optical-depth and visibility conventions.
* Define line-of-sight temperature, E, B, and lensing sources.
* Define `C_ell` and `D_ell` output conventions.
* Define temperature, polarization, potential, and cross-spectrum units.
* Define the exact `PP`, `TP`, and `EP` normalization.
* Define the conventions passed into the lensing remapper.
* Define gauge transformations between the supported scalar gauges.
* Cite the physical equations used by the generated standard hierarchy.

Tasks:

* Add `copernican/docs/cmb_solver.md`.
* Record every native state with its mathematical definition and unit.
* Record every standard hierarchy equation intended for implementation.
* Record all source and spectrum normalization rules.
* Record the gauge transformations used in Slice Six.
* Add compile-time role and unit metadata where practical.
* Remove contradictory or undocumented convention claims.
* Add contract tests for required physical metadata.
* Update the readmes and changelog.

Done when:

* Every later slice can implement equations without inventing conventions.
* Every generated state has one documented meaning.
* Every standard spectrum has one documented normalization.
* No undocumented mixture of conventions remains.

### [closed] Slice Two - Scalar Einstein equations and metric sources

Purpose:

Replace the remaining proxy scalar metric system with the complete scalar
Einstein system in the convention fixed by Slice One.

Depends on:

* Slice One.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Replace present-day `Omega_i0` proxy sources with time-dependent
  background energy-density and pressure weights.
* Use the correct scale-factor dependence for matter and radiation.
* Implement the scalar energy constraint.
* Implement the scalar momentum constraint.
* Implement the anisotropic-stress relation.
* Implement the metric time-evolution relations required by the selected
  convention.
* Include photon, baryon, CDM, massless-neutrino, and massive-neutrino
  contributions through clearly defined source moments.
* Remove metric denominators and regularizers that are not part of the
  documented equations.
* Keep numerical low-k handling separate from the physical equation.
* Add runtime Einstein-residual diagnostics.
* Fail when declared physical constraints exceed their tolerance.

Tasks:

* Rewrite generated scalar metric derived expressions.
* Rewrite scalar constraints and closures.
* Add time-dependent background source scalars.
* Add dimensional and sign checks.
* Add Einstein energy, momentum, and shear residual tests.
* Add parameter-response tests for matter and radiation source changes.
* Update the solver documentation and changelog.

Done when:

* Scalar metric equations match the documented convention.
* Background source weights have the correct time dependence.
* Metric residuals remain below their declared tolerances.
* No proxy Poisson system remains in the generated standard hierarchy.

### [closed] Slice Three - Photon-baryon hierarchy and thermodynamics

Purpose:

Complete the physical photon, polarization, baryon, recombination,
reionization, tight-coupling, and scalar line-of-sight system.

Depends on:

* Slice Two.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Complete the scalar photon-temperature hierarchy.
* Complete the scalar E-polarization hierarchy.
* Use physical Thomson collision terms.
* Complete baryon continuity and Euler equations.
* Complete CDM continuity and Euler equations.
* Implement physical hierarchy truncation.
* Implement a governed tight-coupling approximation.
* Implement explicit tight-coupling entry and exit conditions.
* Ensure a stable transition to the full photon hierarchy.
* Complete native recombination.
* Complete native reionization.
* Produce physical optical-depth and visibility histories.
* Build the physical temperature monopole source.
* Build the physical Doppler source.
* Build the physical polarization source.
* Build the physical metric time-derivative ISW source.
* Remove non-derivative ISW placeholders.
* Preserve the exact native lensing remapper.

Tasks:

* Rewrite photon and polarization equations as needed.
* Replace terminal hierarchy damping with a physical closure.
* Implement and test the tight-coupling transition.
* Improve recombination and reionization accuracy.
* Rewrite scalar temperature and polarization source expressions.
* Add photon-baryon momentum-conservation tests.
* Add visibility-peak and visibility-width tests.
* Add tight-coupling/full-hierarchy agreement tests.
* Update the solver documentation and changelog.

Done when:

* The photon-baryon system is physically documented and implemented.
* Recombination and reionization meet the declared background thresholds.
* Tight coupling transitions without discontinuous physical output.
* Temperature and E sources match the documented line-of-sight equations.

### [open] Slice Four - Neutrino hierarchy and physical q integration

Purpose:

Complete massless- and massive-neutrino physics using one authoritative
q-resolved massive-neutrino hierarchy.

Depends on:

* Slice Three.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Complete the massless-neutrino hierarchy.
* Use a physical high-multipole closure.
* Keep one authoritative massive-neutrino q hierarchy.
* Remove independently evolved aggregate massive-neutrino states.
* Alternatively, convert aggregate names into strict algebraic aliases.
* Use the thermal background distribution.
* Include the correct q and epsilon factors.
* Use distinct physical weights for density, pressure, momentum, and
  anisotropic stress.
* Normalize perturbation moments against the matching background moments.
* Compute massive-neutrino background density and pressure from the same
  q grid.
* Preserve relativistic and nonrelativistic limits.
* Make q range and q count governed accuracy controls.
* Make the metric consume only the physical q-integrated moments.

Tasks:

* Replace normalized common q weights with physical quadrature.
* Add background-distribution factors.
* Add epsilon-dependent moment weights.
* Remove duplicate aggregate evolution.
* Add massless-limit tests.
* Add nonrelativistic-limit tests.
* Add q-integrated moment-consistency tests.
* Add evolved spectrum-response tests for different neutrino masses.
* Add q-grid convergence tests.
* Update the solver documentation and changelog.

Done when:

* Massive-neutrino density, pressure, momentum, and shear are physical q
  integrals.
* Aggregate quantities cannot drift from the q-bin hierarchy.
* Changing neutrino mass changes evolved and projected spectra.
* q-grid refinement produces convergent physical observables.

### [open] Slice Five - Generic compiled collision integration

Purpose:

Replace the remaining special-case Thomson integration path with generic
compiled collision blocks.

Depends on:

* Slice Four.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Compile every collision operator into resolved state slots.
* Compile the operator rate.
* Compile its coefficients or matrix.
* Compile its counterpart terms.
* Compile its conservation rule.
* Compile its integration strategy.
* Support explicit operators.
* Support exact operators with a declared exact form.
* Support implicit operators with a declared linear block.
* Preserve unhandled explicit collision terms in the ordinary RHS.
* Remove global suppression of shared collision symbols.
* Keep standard Thomson relaxation as one built-in compiled operator.
* Allow several collision operators in the same evolution interval.
* Fail before evolution for unsupported exact or implicit declarations.

Tasks:

* Extend collision-operator compiled data.
* Replace fixed Thomson coefficients in the integrator.
* Apply splitting only to selected compiled operators.
* Stop globally setting `collision_rate` to zero.
* Add multi-operator execution tests.
* Add renamed-state tests.
* Add changed-coefficient tests.
* Add custom explicit-operator tests.
* Add unsupported-strategy failure tests.
* Add conservation-residual tests.
* Update the solver documentation and changelog.

Done when:

* Collision evolution follows compiled theory metadata.
* Multiple interactions cannot disable one another.
* No generic runtime path assumes standard variable names.
* No declared interaction is silently removed.

### [open] Slice Six - Scalar initial modes and genuine gauge support

Purpose:

Implement complete regular scalar initial modes and explicit gauge
transformations.

Depends on:

* Slice Five.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Implement the regular adiabatic scalar mode.
* Implement baryon isocurvature.
* Implement CDM isocurvature.
* Implement neutrino-density isocurvature.
* Implement neutrino-velocity isocurvature.
* Include leading super-horizon series for all affected states.
* Satisfy the Einstein constraints at the starting time.
* Use the canonical gauge-invariant basis for observable construction.
* Implement conformal-Newtonian input mapping.
* Implement independent synchronous-gauge variables and equations.
* Implement explicit synchronous-to-invariant transformations.
* Implement a real gauge-invariant compilation route.
* Remove synchronous aliases that merely rescale Newtonian potentials.
* Remove gauge-invariant routing that simply executes the Newtonian branch.
* Permit declared custom gauge transformations where the standard
  transformation does not apply.

Tasks:

* Replace sparse mode seeds with complete series.
* Add starting-time constraint checks.
* Add synchronous scalar equations.
* Add compiled gauge transformations.
* Add gauge-invariant source construction.
* Add mode-leading-power tests.
* Add internal-history gauge tests.
* Add transformed-history agreement tests.
* Add observable-spectrum gauge-equivalence tests.
* Update the solver documentation and changelog.

Done when:

* Different gauges have genuinely different internal variables.
* Transformed gauge-invariant quantities agree.
* Final observables agree within the declared tolerance.
* Every supported scalar mode is regular and constraint-consistent.

### [open] Slice Seven - Physical vector Boltzmann sector

Purpose:

Replace synthetic vector-tagged execution as the only vector proof with a
physical vector Einstein-Boltzmann sector.

Depends on:

* Slice Six.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/cmb_projection_contract.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Add vector metric variables and Einstein relations.
* Add baryon and matter vorticity where physically supported.
* Add vector photon-temperature multipoles.
* Add vector photon E-polarization multipoles.
* Add vector photon B-polarization multipoles.
* Add vector massless-neutrino multipoles.
* Add vector massive-neutrino q multipoles where required.
* Add vector Thomson collision terms.
* Add regular vector initial conditions.
* Add vector temperature sources.
* Add vector E and B sources.
* Add vector `TT`, `TE`, `EE`, and `BB`.
* Preserve vector primordial `BB` through lensing.
* Keep sector-incompatible cross spectra rejected.

Tasks:

* Add vector hierarchy materialization.
* Add vector initial-mode materialization.
* Add vector source materialization.
* Add vector transfer and spectrum tests.
* Add analytic free-streaming-limit tests.
* Add vector collision-limit tests.
* Add vector sector-component output tests.
* Update the solver documentation and changelog.

Done when:

* Vector output comes from a physical hierarchy.
* The proof does not rely on a single custom vector variable.
* Vector temperature and polarization observables are finite and physical.
* Vector analytic limits pass their declared residual tolerances.

### [open] Slice Eight - Physical tensor Boltzmann sector

Purpose:

Replace synthetic tensor-tagged B-mode execution as the only tensor proof
with a physical tensor Einstein-Boltzmann sector.

Depends on:

* Slice Seven.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/cmb_projection_contract.py`
* `copernican/lib/model_coder.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Add tensor metric-wave evolution.
* Add tensor photon-temperature multipoles.
* Add tensor photon E-polarization multipoles.
* Add tensor photon B-polarization multipoles.
* Add tensor massless-neutrino anisotropic stress.
* Add tensor massive-neutrino q multipoles where required.
* Add tensor Thomson collision terms.
* Add regular primordial tensor initial conditions.
* Add tensor temperature sources.
* Add tensor E and B sources.
* Add tensor `TT`, `TE`, `EE`, and `BB`.
* Add tensor amplitude and tensor tilt.
* Preserve primordial tensor `BB` through lensing.
* Expose scalar, vector, tensor, and total spectrum components.

Tasks:

* Add tensor hierarchy materialization.
* Add tensor metric evolution.
* Add tensor initial-mode materialization.
* Add tensor source materialization.
* Add tensor amplitude-response tests.
* Add tensor tilt-shape tests.
* Add tensor neutrino-stress tests.
* Add tensor unlensed and lensed spectrum tests.
* Add sector-total consistency tests.
* Update the solver documentation and changelog.

Done when:

* Tensor output comes from tensor metric and Boltzmann hierarchies.
* The proof does not rely on a single synthetic `tensor_b` variable.
* Tensor `TT`, `TE`, `EE`, and `BB` respond physically to tensor inputs.
* Primordial tensor `BB` survives exact lensing remapping.

### [open] Slice Nine - Native absolute scientific parity

Purpose:

Demonstrate that the completed native solver reproduces an independent
Boltzmann reference in absolute physical output.

Depends on:

* Slice Eight.

Probable affected files:

* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/native_lensing.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Create one neutral `standard: false` native acceptance cosmology.
* Run it through the production native solver.
* Generate reference results through CAMB or CLASS only inside tests.
* Compare native background quantities.
* Compare recombination and visibility quantities.
* Compare native scalar `TT`, `TE`, and `EE`.
* Compare native `PP`, `TP`, and `EP`.
* Compare all four native lensed spectra.
* Compare native massive-neutrino responses.
* Compare native tensor outputs.
* Compare physical acoustic features.
* Do not use the standard adapter as the tested output.
* Do not accept amplitude response as absolute parity.

Required background thresholds:

* conformal age relative error at or below `0.2%`;
* sound horizon relative error at or below `0.2%`;
* visibility-peak redshift relative error at or below `0.5%`;
* visibility-width relative error at or below `3%`;
* recombination median relative error at or below `2%`;
* recombination 90th-percentile relative error at or below `5%`;
* reionization optical-depth relative error at or below `1%`.

Required scalar-spectrum thresholds:

* compare `TT`, `TE`, and `EE` over ell `2..2000`;
* compare `PP` over ell `10..1500`;
* compare lensed spectra over ell `2..2000`;
* `TT` median fractional error at or below `5%`;
* `TT` 90th-percentile fractional error at or below `10%`;
* `EE` median fractional error at or below `5%`;
* `EE` 90th-percentile fractional error at or below `10%`;
* normalized `TE` RMS error at or below `5%`;
* first three `TT` peaks within three ell;
* first three `EE` peaks within three ell;
* first three `TE` zero crossings within three ell;
* `PP` median fractional error at or below `10%`;
* `PP` 90th-percentile fractional error at or below `20%`;
* lensed `BB` median fractional error at or below `15%`.

Required non-scalar and neutrino thresholds:

* tensor `TT`, `EE`, and `BB` median fractional error at or below `10%`;
* massive-neutrino spectrum changes agree with the reference to `10%`;
* gauge-equivalent scalar spectra agree to `0.1%`;
* vector analytic-limit residuals meet their declared tolerances.

Tasks:

* Add native absolute background-reference tests.
* Add native absolute scalar-spectrum tests.
* Add native lensing-reference tests.
* Add native massive-neutrino response-reference tests.
* Add native tensor-reference tests.
* Fix physical source defects exposed by those comparisons.
* Record the accepted reference cosmology and tolerance table.
* Update the solver documentation and changelog.

Done when:

* Production native results meet all declared reference thresholds.
* The comparison is against an independent code path.
* No standard-backend result is presented as native validation.
* No response-only test is presented as absolute parity.

### [open] Slice Ten - Numerical convergence and final closure

Purpose:

Prove that the completed native solver converges numerically and close the
repository only after all implementation and scientific claims agree.

Depends on:

* Slice Nine.

Probable affected files:

* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/native_lensing.py`
* `copernican/lib/likelihoods/cmb/cmb.py`
* `copernican/lib/plotter.py`
* `copernican/lib/gui/plot_viewer.py`
* `copernican/docs/cmb_solver.md`
* `copernican/docs/model_template.yml`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `README.md`
* `copernican/README.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Demonstrate convergence of background sampling.
* Demonstrate convergence of eta sampling.
* Demonstrate convergence of k sampling.
* Demonstrate convergence of photon hierarchy depth.
* Demonstrate convergence of massless-neutrino hierarchy depth.
* Demonstrate convergence of massive-neutrino hierarchy depth.
* Demonstrate convergence of the q grid.
* Demonstrate convergence of source refinement.
* Demonstrate convergence of lensing quadrature.
* Fail when a requested accuracy tier is under-resolved.
* Record the active numerical envelope in validation output.
* Keep output availability explicit.
* Keep default plotting limited to `TT`, `TE`, and `EE`.
* Keep unlensed, lensed, lensing, sector, and diagnostic views separate.
* Replace weak tests whose names overstate their assertions.
* Remove obsolete acceptance-only tests.
* Update all user-facing claims.
* Run the complete local repository gate.
* Mark every slice closed only after the final gate succeeds.

Required convergence thresholds:

* final refinement changes `TT` by less than `1%`;
* final refinement changes `EE` by less than `1%`;
* final refinement changes normalized `TE` by less than `2%`;
* final refinement changes `PP` by less than `3%`;
* final refinement changes lensed `BB` by less than `5%`;
* final q-grid refinement changes accepted massive-neutrino spectra by
  less than `2%`;
* final hierarchy-depth refinement changes accepted spectra by less than
  `1%`;
* gauge-equivalent outputs remain inside their tolerance at every accepted
  accuracy tier.

Required regression checks:

* Changing `PP` changes the lensed spectra themselves.
* Primordial `BB` survives lensing.
* Missing spectra remain unavailable rather than fabricated.
* Multi-spectrum likelihood rows work for repeated and noncontiguous ell.
* Cache keys include structure, bound parameters, grids, requested
  spectra, and accuracy controls.
* Native production modules do not import or call CAMB or CLASS.
* Approximate lensing, direct spectrum injection, and empirical output
  scaling remain absent.
* Scalar, vector, tensor, and total components remain internally
  consistent.

Tasks:

* Add actual spectrum-convergence tests.
* Remove grid-size-only convergence claims.
* Add under-resolution failure tests.
* Add final cache-identity tests.
* Complete spectrum-availability metadata.
* Complete grouped plotting behavior.
* Update the model template.
* Update all CMB documentation.
* Update the final changelog entry.
* Run the complete local repository gate.
* Change every slice marker to `[closed]` only after success.

Done when:

* Every numerical control demonstrates convergence of physical output.
* Every source claim is supported by a scientific or structural test.
* Source, tests, documentation, public API, and changelog agree.
* The full local repository gate passes from a clean checkout.
* No item from this roadmap remains open or deferred.

## Completion Standard

This roadmap is complete only when all ten slices are `[closed]`.

The repository must then truthfully satisfy all of the following:

* Copernican ships a native, universal, theory-agnostic
  Boltzmann-hierarchy CMB solver.
* `standard: false` runs without a CAMB or CLASS production fallback.
* The standard native acceptance model contains physical scalar, vector,
  and tensor Einstein-Boltzmann sectors.
* Photon temperature, E polarization, B polarization, baryon, CDM,
  massless-neutrino, and massive-neutrino physics use one documented
  convention.
* Scalar metric sources use physical time-dependent background weights.
* Massive-neutrino metric moments are physical q integrals.
* No independent aggregate massive-neutrino state can drift from the
  q-resolved hierarchy.
* Collision integration is compiled from declared theory metadata.
* Multiple collision operators can run without silently disabling one
  another.
* Newtonian, synchronous, and gauge-invariant routes are connected by
  explicit transformations or invariant variables.
* Regular adiabatic, isocurvature, vector, and tensor initial modes are
  implemented.
* Every standard CMB spectrum is produced when the declared theory
  supplies the required physics.
* Unavailable, physically zero, and unrequested spectra remain distinct.
* Exact curved-sky lensing preserves primordial `BB`.
* Native scalar, tensor, lensing, and massive-neutrino outputs meet the
  independent-reference thresholds.
* Background, k, eta, hierarchy, q-grid, source, and lensing refinements
  demonstrate convergence.
* No empirical scales, source injections, hidden fallbacks, or
  acceptance-only physical equations remain.
* TORG remains unchanged.
* The complete repository gate passes from a clean checkout.
* Documentation and changelog statements match the measured code state.

No slice may be marked `[closed]` because a later slice is expected to fix
it. If any completion statement is false, the responsible slice remains
`[open]`.
