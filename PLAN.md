# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-08-08
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

Use this plan to complete the native CMB solver.

This roadmap uses real sequential implementation slices. Each slice has a
specific physical target and an explicit acceptance boundary.

Each slice must be independently implementable, testable, documentable,
and closable in one work session. If a slice cannot be completed in one
session, the plan must be divided into additional slices before coding
proceeds. Do not create hidden sub-slices, work packages, phases, or
follow-up cleanup inside a slice.

The target condition is final:

* Every production CMB model uses the native declared-graph path.
* CAMB and CLASS are independent test references only; production remains
  free of both backends.
* The production model contract has no solver-route boolean or backend
  fallback.
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

## Table of Contents

* [Problem Preamble](#problem-preamble)
* [Current Baseline](#current-baseline)
* [Overview](#overview)
* [Execution Rules](#execution-rules)
* [Execution Slices](#execution-slices)
* [Completion Standard](#completion-standard)

## Problem Preamble

Copernican has a substantial native declared-graph CMB engine.

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

The target capability gaps are:

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

The following baseline capabilities are established outside the open
acceptance work and must not be restored to the roadmap as unfinished work:

* A declared native route executes models carrying the transitional native
  marker.
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
* The scalar materializer creates photon, baryon, polarization, and each
  optional matter or neutrino state only when the contract declares it.
* Massive-neutrino q-bin states are generated and evolved.
* `TT`, `TE`, `EE`, `BB`, `PP`, `TP`, `EP`, and lensed outputs can be
  represented when declared.
* Generic vector-like and tensor-like graph variables can execute.
* Sector-incompatible cross spectra fail before execution.

The following acceptance areas are not established by the current baseline:

* physical scalar normalization;
* physical massive-neutrino quadrature;
* generic collision integration;
* physical gauge equivalence;
* complete initial modes;
* physical vector and tensor sectors;
* native absolute scientific parity;
* numerical convergence.

## Overview

The roadmap divides the target work into thirty-nine slices.

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

Slice Nine establishes the native reference foundation.

Slice Ten establishes shared control-model and test-model selection.

Slice Eleven creates the native LCDM model.

Slice Twelve migrates and audits every CMB model against a species-accurate
native contract without embedding LCDM assumptions in the shared compiler.

Slice Thirteen replaces per-mode native execution with a shared, batched
runtime architecture.

Slice Fourteen establishes adaptive transfer and projection convergence.

Slice Fifteen establishes native performance and architecture acceptance.

Slice Sixteen establishes the authoritative native scalar runtime and bans
shortcut implementations.

Slice Seventeen establishes scalar metric and species evolution correctness.

Slice Eighteen establishes regular scalar initial conditions and mode
families.

Slice Nineteen establishes declared tight coupling and hierarchy closure.

Slice Twenty establishes scalar evolution refinement convergence.

Slice Twenty-One establishes line-of-sight source conventions.

Slice Twenty-Two establishes independent projection kernels.

Slice Twenty-Three establishes scalar projection convergence.

Slice Twenty-Four establishes native scalar absolute parity.

Slice Twenty-Five establishes lensing remapping correctness.

Slice Twenty-Six establishes lensed scalar absolute parity.

Slice Twenty-Seven establishes massive-neutrino q-hierarchy correctness.

Slice Twenty-Eight establishes massive-neutrino absolute parity.

Slice Twenty-Nine establishes tensor hierarchy correctness.

Slice Thirty establishes tensor absolute parity.

Slice Thirty-One establishes gauge-equivalent scalar parity.

Slice Thirty-Two establishes physical vector hierarchy and parity.

Slice Thirty-Three removes native production route branching.

Slice Thirty-Four migrates the model corpus and native model assets.

Slice Thirty-Five completes user-facing native-only cutover.

Slice Thirty-Six isolates scientific references and package artifacts.

Slice Thirty-Seven establishes cross-sector numerical convergence.

Slice Thirty-Eight validates output, cache, and contract consistency.

Slice Thirty-Nine performs final scientific and repository closure.

Each slice includes its own implementation, tests, documentation, and
changelog entry. The roadmap contains no cleanup slice.

## Execution Rules

* Execute slices strictly in order.
* Do not begin a later slice while an earlier slice is open.
* Each slice represents one complete work session.
* If a slice requires another work session, divide it into numbered slices
  before implementation proceeds.
* Do not hide additional sessions under tasks, work packages, follow-ups,
  polish, or cleanup.
* Do not add optional or deferred physics.
* Use a neutral native standard cosmology for acceptance testing.
* CAMB or CLASS may be used only as independent test references.
* Production native code must not import or call CAMB or CLASS.
* During migration, preserve model physics and public output contracts;
  complete the native replacement before removing the standard route in
  Slice Thirty-Three.
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
* Every slice must have a targeted verification command with a three-minute
  wall-clock budget. A timeout is an implementation failure, not a reason
  to create a hidden chunk.
* A slice must satisfy every listed physical threshold before it is closed;
  passing the repository gate alone is not implementation acceptance.
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

* Each subsequent slice can implement equations without inventing
  conventions.
* Every generated state has one documented meaning.
* Every standard spectrum has one documented normalization.
* No undocumented mixture of conventions remains.

### [closed] Slice Two - Scalar Einstein equations and metric sources

Purpose:

Replace the proxy scalar metric system with the complete scalar
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

### [closed] Slice Four - Neutrino hierarchy and physical q integration

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

### [closed] Slice Five - Generic compiled collision integration

Purpose:

Replace the special-case Thomson integration path with generic
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

### [closed] Slice Six - Scalar initial modes and genuine gauge support

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

### [closed] Slice Seven - Physical vector Boltzmann sector

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

### [closed] Slice Eight - Physical tensor Boltzmann sector

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

### [closed] Slice Nine - Native reference foundation

Purpose:

Freeze the independent reference surface and establish the native background
and recombination baseline used by the parity slices that follow.

Depends on:

* Slice Eight.

Probable affected files:

* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/native_lensing.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `copernican/docs/model_template.yml`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Freeze one neutral native LCDM acceptance cosmology definition for the
  native model created in Slice Eleven.
* Run the existing native acceptance contract through the production native
  background and recombination path.
* Generate CAMB or CLASS reference fixtures only inside scientific tests.
* Compare native background quantities.
* Compare recombination, visibility, and reionization quantities.
* Validate projection-kernel analytic limits.
* Validate exact curved-sky remapper normalization and interpolation.
* Record reference provenance, numerical controls, and bounded work units.
* Keep full scalar, lensing, neutrino, tensor, gauge, and vector parity in
  the dedicated slices below.

Required background thresholds:

* conformal age relative error at or below `0.2%`;
* sound horizon relative error at or below `0.2%`;
* visibility-peak redshift relative error at or below `0.5%`;
* visibility-width relative error at or below `3%`;
* recombination median relative error at or below `2%`;
* recombination 90th-percentile relative error at or below `5%`;
* reionization optical-depth relative error at or below `1%`.

Tasks:

* Complete independent reference-fixture construction.
* Complete background and recombination acceptance metrics.
* Complete projection and remapper unit-level acceptance metrics.
* Record the accepted cosmology and tolerance table.
* Update the solver documentation and changelog.

Done when:

* The native acceptance contract executes through the native solver.
* All background and recombination thresholds pass.
* Reference fixtures are created outside production native execution.
* Projection and remapper normalization tests pass independently.
* No scalar or non-scalar absolute-parity claim is made by this slice.

### [closed] Slice Ten - Shared control and test model selection

Purpose:

Replace the hard-coded LCDM comparison control with an explicit control-model
and test-model pair shared by the CLI and GUI workflow layers.

Depends on:

* Slice Nine.

Probable affected files:

* `copernican/workflow.py`
* `copernican/lib/cli/**`
* `copernican/lib/gui/**`
* `copernican/lib/plotter.py`
* `copernican/lib/gui/plot_viewer.py`
* `copernican/lib/validation/**`
* `tests/copernican/lib/test_workflow.py`
* `tests/copernican/lib/test_plotter.py`
* `tests/copernican/lib/gui/**`
* `tests/project/lib/test_core.py`
* `README.md`
* `copernican/README.md`
* `copernican/docs/api_overview.md`
* `copernican/docs/gui_guide.md`
* `copernican/docs/gui_overview.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Add a control-model field with the current LCDM model as its default.
* Use `copernican/models/model_lcdm.yml` as the default.
* Keep the existing test-model page and selection as the test-model field.
* Add a control-model page immediately before the existing test-model page.
* Reuse the existing model-choice page behavior and validation in the new
  control page rather than creating a separate selection implementation.
* Add matching CLI control-model and test-model options.
* Represent both selections in one shared comparison request consumed by CLI
  and GUI execution paths.
* Permit comparisons between any two compatible declared models.
* Validate compatible observables, units, ell grids, and spectrum roles before
  comparison.
* Propagate resolved control and test identities into manifests, caches, run
  summaries, output names, plot titles, plot footers, and residual labels.
* Keep solver selection out of this feature; control and test are model roles,
  not backend choices.

Tasks:

* Generalize the hard-coded LCDM control input in the shared workflow layer.
* Add the control-model GUI page before the existing test-model page.
* Add CLI flags with equivalent shared request semantics.
* Replace hard-coded `LCDM vs. X` labels and filenames with resolved model
  identities.
* Add pair-selection, compatibility, manifest, plotting, CLI, and GUI tests.
* Update the user-facing documentation and changelog.

Done when:

* CLI and GUI construct the same control/test comparison request.
* LCDM remains the default control model.
* Any compatible model can be selected as control or test model.
* Outputs identify the actual control and test models without LCDM-specific
  assumptions.

### [closed] Slice Eleven - Native LCDM model

Purpose:

Create the first real production native LCDM model. The model must be a
declared physical graph rather than a documentation template or a route flag
that delegates to another backend.

Depends on:

* Slice Ten.

Probable affected files:

* `copernican/models/model_lcdm.yml`
* `copernican/docs/model_template.yml`
* `docs/model_template.yml`
* `copernican/lib/model_spec_validator.py`
* `copernican/lib/perturbation_contract.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Define LCDM background and recombination inputs in the native model shape.
* Define the scalar Einstein-Boltzmann graph, sources, projections, and
  numerical controls.
* Define regular adiabatic scalar initial conditions.
* Execute the model through the production native solver.
* Preserve the same cosmological parameters used by the independent
  reference model.
* Expose declared spectrum availability and provenance in the run manifest.

Tasks:

* Create a real native LCDM model file.
* Keep the native declaration at `copernican/models/model_lcdm.yml`.
* Validate its schema, units, source roles, and graph compilation.
* Add an end-to-end native LCDM smoke test.
* Remove any dependency on the documentation-only template for execution.
* Update native model documentation and the changelog.

Done when:

* The real native LCDM model validates and executes without CAMB or CLASS.
* Its scalar transfer and spectrum outputs are finite and declared.
* The model manifest records native execution and numerical provenance.
* No standard-backend result is used as the production output.

### [closed] Slice Twelve - Theory-accurate native model migration

Purpose:

Migrate every CMB theory model to the native declared-graph shape without
turning the native hierarchy compiler into a hidden LCDM solver. Production
must have one reusable evolution, projection, and lensing infrastructure, but
each model must declare its actual species, background sources, interactions,
closures, and available observables. The shared execution architecture and
its numerical acceptance budgets are specified in Slices Thirteen through
Fifteen.

Depends on:

* Slice Eleven.

Affected files:

* `copernican/models/*.yml`
* `copernican/lib/perturbation_contract.py`
* `copernican/lib/engine_adapter.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* `copernican/lib/model_coder.py`
* `copernican/docs/model_template.yml`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`
* `README.md`
* `tests/copernican/lib/test_engine_adapter.py`
* `tests/copernican/lib/test_model_coder.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/engines/test_engine_mcmc.py`

Scope:

* Inventory every model with CMB perturbations.
* Record each model's physical species, hierarchy families, background
  references, source roles, interactions, conservation rules, initial modes,
  projection roles, and unavailable observables.
* Translate every standard and custom CMB model into native declarations.
* Preserve model parameters, background quantities, priors, and observable
  contracts while replacing backend-specific execution assumptions.
* Declare equations, initial conditions, interactions, conservation rules,
  source roles, projections, units, and numerical controls for each model.
* Reuse the common native evolution, projection, and lensing infrastructure;
  do not add a solver branch or a solver implementation per theory.
* Compile each declared contract into a reusable representation exposing its
  graph structure, dependency closure, state layouts, collision plans,
  context layouts, and momentum-grid structure. The execution and cache
  architecture for that representation is owned by Slice Thirteen.
* Evolve massive-neutrino q bins through an efficient coupled numeric path
  without removing q-resolved physics or synthesizing aggregate states.
* Make hierarchy materialization conditional on the declared species and
  source graph. `cdm`, massive neutrinos, and any other species must not be
  synthesized to satisfy a compiler requirement.
* Represent theory-specific effects as declared background, source,
  interaction, constraint, or closure terms rather than as fake matter
  species. QRSF and TORG relational inertia use explicit native source
  closures; USMF declares CMB output unavailable until its perturbation
  closure exists.
* Add native validation and smoke execution for every model whose CMB output
  is declared available.
* Keep CAMB or CLASS references inside tests only.

Tasks:

* Reopen the model inventory and classify the physical ontology of every
  CMB manifest before changing compiler requirements.
* Replace the fixed LCDM species assumptions in the generated scalar graph
  with species-aware source assembly and optional hierarchy families.
* Generate density, velocity, momentum, shear, Einstein-source, and initial
  condition terms only when their declared species or source closure exists.
* Migrate model manifests after the compiler accepts species-accurate
  contracts. LCDM-family models may declare CDM; QRSF, TORG, and USMF do not
  declare it, and every other model is resolved from its own theory definition.
* Reject a model that claims CMB availability without a defensible native
  perturbation closure. Mark genuinely unsupported observables unavailable
  instead of producing a zero or substituting a standard-backend result.
* Add model-by-model tests for exact compiled species, source provenance,
  observable availability, and finite native spectra.
* Remove production assumptions that require a standard backend while keeping
  independent backend references in scientific tests.
* Update model and solver documentation and the changelog.

Done when:

* Every bundled CMB model has a reviewed ontology record and a native
  contract whose compiled species and sources match that record.
* The native scalar materializer has no unconditional LCDM species, density,
  velocity, or background assumptions.
* No model contains a synthetic CDM species merely to satisfy compilation.
* TORG, USMF, and every other non-LCDM theory either has an explicit native
  perturbation closure or declares CMB output unavailable.
* Every model with available CMB output has native validation and finite-
  spectrum smoke coverage.
* No production model requires CAMB or CLASS to produce CMB spectra.
* Model manifests distinguish unavailable, zero, and unrequested spectra.
* Generated generic numeric kernels preserve reference-interpreter results
  for scalar, interaction, gauge, tensor, vector, and q-resolved contracts.
* The model migration and species-accurate contract tests pass before the
  runtime architecture slices are attempted.

### [closed] Slice Thirteen - Native runtime architecture and batched execution

Purpose:

Replace repeated per-mode preparation and independent per-`k` adaptive
evolution with a shared runtime that preserves the complete declared physics.
This slice changes execution architecture, not model ontology or scientific
acceptance thresholds.

Depends on:

* Slice Twelve.

Probable affected files:

* `copernican/lib/model_coder.py`
* `copernican/lib/likelihoods/cmb/native_background.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_background.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_cache.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_evolution.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_projection.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Separate contract-static, cosmology-static, and request-specific work.
* Compile graph structure, dependency closure, state layouts, collision
  plans, context layouts, and momentum-grid structure once per runtime.
* Build background, recombination, visibility, and collision tables once per
  bound cosmology and reuse them across mode evolution and projections.
* Evolve required `k` modes in vectorized batches over a shared controlled
  conformal-time integration path rather than invoking one adaptive solver per
  mode.
* Execute declared expressions through generated numeric kernels without
  rebuilding dictionary contexts or resolving the graph at every stage.
* Preserve q-resolved massive-neutrino evolution and all declared scalar,
  vector, and tensor state variables in the batched path.
* Cache source histories and transfer tables for reuse across requested
  observables and repeated parameter proposals.
* Keep requested multipole work separate from unrelated high-multipole work
  unless the requested accuracy tier requires it.

Tasks:

* Define explicit runtime objects and cache identities for contract-static,
  cosmology-static, and request-specific data.
* Replace per-`k` adaptive evolution calls with a batched hierarchy RHS and a
  shared step/error-control strategy that remains stable for all modes in a
  batch.
* Move background, collision, momentum, and mode preparation out of
  Runge-Kutta stages and other hot loops.
* Add counters proving that compilation and static preparation do not scale
  with k modes, Runge-Kutta stages, or repeated parameter proposals.
* Add focused tests that reject a per-mode solver invocation and verify cache
  reuse, deterministic histories, finite states, and declared-sector parity.
* Keep the runtime generic: no theory-specific solver branch or hidden LCDM
  source may be introduced to obtain batching.

Done when:

* A fixed declared contract produces finite, deterministic batched histories.
* Contract compilation, background preparation, and collision preparation
  are reused at the documented cache boundaries.
* The execution path contains no independent adaptive ODE invocation for each
  requested `k` mode.
* q-resolved, scalar, vector, and tensor state layouts remain declared and
  physically sourced after batching.
* Repeated requests reuse compiled and cosmology-static work without
  changing the result.
* Runtime counters and focused architecture tests pass without claiming
  absolute reference-spectrum parity.

Implementation closure:

* `NativeRuntimeCacheIdentity` separates contract-static, cosmology-static,
  and request-specific cache keys for background and spectrum work.
* `_integrate_batched_rk4` provides one finite-checked shared RK4 schedule;
  generated scalar contracts use the vectorized hierarchy RHS, collision
  updates, tight-coupling masks, and source-history materialization for every
  compatible multi-mode grid.
* Adaptive source-mode requests reuse the same batch preparation, while
  declared modes that require a distinct hidden-prefix grid retain the
  existing deterministic declared evolution path.
* Runtime envelopes expose static-preparation, batch-size, stage, and
  substep counters. Focused tests cover shared scheduling, deterministic
  histories, cache-identity separation, repeated request reuse, finite output,
  and declared initial-mode execution.

### [closed] Slice Fourteen - Adaptive transfer and projection convergence

Purpose:

Make `k`, conformal-time, source-history, and line-of-sight refinement respond
to physical phase and quadrature error rather than fixed sparse-grid choices.

Depends on:

* Slice Thirteen.

Probable affected files:

* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/native_cache.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/lib/likelihoods/cmb/test_native_projection.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Select `k` nodes from requested multipoles, transfer phase, acoustic
  structure, visibility structure, and a declared refinement error.
* Refine conformal-time source histories around recombination, reionization,
  rapid background transitions, and oscillatory source regions.
* Project transfer histories with adaptive line-of-sight quadrature instead of
  relying on under-resolved sparse interpolation.
* Use stable Bessel recurrences or bounded interpolation with an explicit
  interpolation error estimate for every supported sector.
* Preserve low-multipole efficiency without paying for unrelated high-
  multipole accuracy.

Tasks:

* Implement independent refinement criteria for `k`, eta, source histories,
  and radial kernels.
* Add convergence ladders comparing successive physical refinements, not only
  node counts or parameter responses.
* Add under-resolution failures when a requested accuracy tier cannot be met
  within its declared work envelope.
* Verify that refined projections preserve source normalization, parity, and
  observable availability for scalar, vector, and tensor sectors.
* Keep reference spectra independent from production execution and reject any
  direct spectrum injection or empirical correction.

Done when:

* Successive physical refinements converge transfer histories and projected
  spectra within the declared internal tolerances.
* Visibility-era acoustic structure and low-multipole behavior remain stable
  under refinement.
* Under-resolved requests fail clearly before producing misleading spectra.
* Projection results remain finite, deterministic, and sector-consistent.
* Convergence tests exercise physical output rather than only grid activation.

Implementation closure:

* `native_adaptive.py` validates independent transfer, source-history, and
  line-of-sight projection accuracy sections with bounded node budgets,
  phase-point controls, absolute and relative tolerances, and named
  under-resolution failures.
* Transfer nodes respond to radial and acoustic phase and preserve declared
  multipole anchors. Source-time nodes respond to Fourier phase and visibility
  structure without reducing the generated scalar history grid.
* Native projection records independent coarse-source and lower-order
  line-of-sight estimates. Enabled surfaces expose measured absolute and
  relative errors and reject non-converged spectra before publication.
* Focused tests cover control validation, phase-aware k and eta grids,
  absolute-tolerance handling, under-resolution diagnostics, and an end-to-end
  adaptive native spectrum with finite transfer output.

### [closed] Slice Fifteen - Native performance and architecture acceptance

Purpose:

Prove that the shared runtime is fast enough for ordinary development while
retaining bounded work, cache reuse, and the declared numerical controls.

Depends on:

* Slices Thirteen and Fourteen.

Probable affected files:

* `copernican/lib/likelihoods/cmb/native_cache.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `tests/copernican/engines/test_engine_mcmc.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Benchmark representative native scalar, interaction, gauge, tensor,
  vector, and q-resolved massive-neutrino paths.
* Measure compilation, background, evolution, projection, and lensing work
  separately rather than treating total wall time as the only diagnostic.
* Prove repeated identical requests reuse compiled, background, and transfer
  work without changing numerical results.
* Enforce explicit runtime work limits and fail fast when an accuracy tier
  exceeds its budget.
* Run bounded fixed-cosmology scientific smoke coverage before the detailed
  scalar, lensing, neutrino, tensor, and gauge parity slices.

Required budgets:

* A representative full native LCDM spectrum completes within 180 seconds.
* A native joint MCMC smoke test completes within 60 seconds.

Done when:

* The required runtime budgets pass on the managed environment.
* Runtime counters identify each major phase and demonstrate cache reuse.
* A repeated request avoids recompilation and repeated cosmology-static work.
* Work envelopes reject unbounded requests before large runs begin.
* Focused performance and architecture tests pass before absolute parity work.

Implementation closure:

* `native_performance.py` owns the 180-second full-spectrum and 60-second
  joint-MCMC acceptance budgets, named overrun errors, and phase timers.
* Native runtime envelopes record compilation, background, preparation,
  evolution, projection, power-spectrum, and total wall times.
* Native cache telemetry records request counts, exact-result cache hits,
  aggregate phase time, graph-plan reuse, background reuse, and projection
  kernel hits without changing numerical results.
* Focused tests benchmark the full native LCDM surface, representative
  interaction, gauge, tensor, vector, and q-resolved paths, and the joint
  likelihood smoke workload.

### [closed] Slice Sixteen - Native scalar runtime authority

Purpose:

Establish one authoritative native scalar Einstein-Boltzmann runtime before
absolute spectrum parity work. This slice closes the architectural defect
that permits a hand-written approximation to diverge from the declared
graph.

Depends on:

* Slice Fifteen with its performance instrumentation retained.

Probable affected files:

* `copernican/lib/perturbation_contract.py`
* `copernican/lib/likelihoods/cmb/native_evolution.py`
* `copernican/lib/likelihoods/cmb/native_projection.py`
* `copernican/lib/likelihoods/cmb/copernican_cmb_solver.py`
* `tests/copernican/lib/test_perturbation_contract.py`
* `tests/copernican/lib/likelihoods/cmb/test_cmb.py`
* `copernican/docs/cmb_solver.md`
* `CHANGELOG.md`
* `PLAN.md`

Scope:

* Make the compiled declared scalar graph the sole production scalar
  evolution authority.
* Remove parallel hand-written scalar equations and alternate physics paths.
* Permit batching and caching only when they execute the same compiled graph
  or pass an explicit mathematical-equivalence test against it.
* Keep species, equations, constraints, closures, collision operators,
  initial conditions, and source roles declaration-driven.
* Preserve bounded runtime envelopes without reducing physical equations,
  hierarchy depth, source terms, or requested output surfaces.
* Keep CAMB and CLASS outside production execution.

Required tests:

* A generated scalar contract and its generic declared execution path produce
  the same state histories, source histories, transfers, and spectra.
* Any optimized or batched scalar path is proven equivalent to the generic
  path at fixed states, grids, and parameters.
* Production native modules contain no CAMB or CLASS import or call.
* Production code contains no empirical spectrum scale, direct spectrum
  injection, hidden damping, reference-output lookup, or shortcut branch.
* A declared species graph cannot acquire undeclared LCDM species or source
  terms during compilation or execution.
* The bounded runtime envelope rejects under-declared or unbounded work
  rather than silently lowering the requested physical calculation.

Done when:

* One compiled declared scalar runtime is authoritative in production.
* All alternate scalar physics paths are removed or mechanically proven
  equivalent by tests.
* The no-shortcut regression suite passes in under three minutes.
* Documentation and model declarations describe the authoritative runtime
  without historical migration wording.

### [closed] Slice Seventeen - Scalar metric and species evolution

Purpose:

Validate the physical scalar equations executed by the authoritative runtime
before adding initial-mode, closure, projection, or parity acceptance.

Depends on:

* Slice Sixteen.

Scope:

* Audit scalar metric, photon, baryon, matter, and massless-neutrino
  equations against the convention fixed by Slice One.
* Use time-dependent background density and pressure weights in every metric
  source.
* Enforce declared state, source, unit, and species ownership at compile and
  execution time.
* Evaluate Einstein energy, momentum, and shear constraints at named
  evolution anchors.
* Remove any remaining hard-coded scalar source or state branch.

Required acceptance:

* Constraint residuals stay within their declared tolerances throughout the
  accepted history.
* Generated and generic declared executions agree for fixed states, grids,
  parameters, sources, and transfers.
* A contract without a species or source does not acquire it implicitly.
* Finite-state and unit checks fail before projection when violated.

Done when:

* Scalar state histories are physically normalized, constraint-consistent,
  and entirely declaration-driven.
* Runtime envelopes record named scalar constraint anchors and full-history
  maxima, while generated scalar state and diagnostic units are checked
  before projection.

### [closed] Slice Eighteen - Scalar initial conditions and mode families

Purpose:

Complete regular scalar initial conditions independently of tight coupling and
full-spectrum parity.

Depends on:

* Slice Seventeen.

Scope:

* Implement the regular adiabatic initial series for every declared scalar
  species and metric state.
* Implement each supported declared scalar isocurvature family.
* Seed all gauge routes through explicit transformations or invariant
  variables rather than forced aliases.
* Validate early-start and hidden-prefix evolution at fixed modes.
* Reject an initial mode that violates declared constraints or finiteness.

Required acceptance:

* Initial states satisfy the declared Einstein and collision constraints.
* Adiabatic and each supported isocurvature mode produce distinct,
  reproducible source histories where physics requires it.
* Early-start and hidden-prefix histories agree within `5%` on the accepted
  comparison surface.
* Initial-mode tests use absolute states and sources, not response ratios.

Done when:

* All supported scalar initial modes are regular, finite, constrained, and
  connected to the declared graph.
* Automatic mode selection rejects multiple families instead of choosing a
  hidden priority order.
* Explicit start expressions remain authoritative when layered onto a
  metadata-only hierarchy declaration.
* Massive-neutrino q-bin seeds follow the selected scalar family rather than
  reusing adiabatic expressions for isocurvature modes.
* Absolute source-history tests cover adiabatic and every supported
  isocurvature family, with the hidden-prefix comparison below the declared
  five-percent threshold.

### [closed] Slice Nineteen - Declared tight coupling and hierarchy closure

Purpose:

Complete the declared fast-collision regime and terminal hierarchy behavior
without a scalar-specific physics path.

Depends on:

* Slice Eighteen.

Scope:

* Implement declared tight-coupling entry and exit conditions.
* Apply declared exact and implicit collision operators with momentum
  conservation and their declared fast-manifold targets.
* Implement the declared first-order temperature and polarization closures.
* Implement terminal closures for photon and massless-neutrino hierarchies.
* Preserve collision invariants through every split or implicit step.

Required acceptance:

* Entry and exit use declared hysteresis thresholds and collision metadata.
* Exact collision matrix limits match independent linear-algebra fixtures.
* Increasing hierarchy depth changes the terminal surface only within the
  later convergence tolerance, without changing declared equations.
* Collision conservation failures stop before evolution proceeds.

Done when:

* Tight coupling and terminal closure are declaration-driven, conserved,
  finite, and directly tested at their physical limits.

### [closed] Slice Twenty - Scalar evolution refinement convergence

Purpose:

Prove convergence of scalar histories and sources after equations, initial
conditions, collisions, and closures are individually complete.

Depends on:

* Slice Nineteen.

Scope:

* Compare scalar histories under doubled evolution sampling.
* Compare scalar histories under increased photon and massless-neutrino
  hierarchy depth.
* Compare low, recombination, and late-time source anchor regions.
* Keep runtime-envelope controls explicit and reject under-resolved requests.
* Record independent state-history and source-history errors for every enabled
  evolution refinement.

Required thresholds:

* Accepted state and source histories change by less than `1%` at each anchor
  region under the declared refinements.
* Increasing hierarchy depth changes accepted scalar histories by less than
  `1%`.
* The targeted convergence command completes within three minutes.

Done when:

* Scalar evolution convergence is demonstrated with physical output rather
  than grid-size activation or response-only evidence.
* Declared evolution controls expose fine/coarse sample counts, anchor errors,
  and bounded work accounting; strict requests reject under-resolved history
  comparisons before publishing an accepted spectrum.

### [closed] Slice Twenty-One - Line-of-sight source conventions

Purpose:

Implement and independently validate every scalar line-of-sight source before
testing radial integration.

Depends on:

* Slice Twenty.

Scope:

* Implement declared temperature, Doppler, ISW, polarization, and
  lensing-potential sources.
* Preserve visibility-era source refinement and direct declared histories.
* Validate temperature and E-mode transfer normalization independently of
  evolution normalization.
* Keep source availability tied to declared sectors and source roles.
* Resolve each declared source role through the shared projection contract,
  including the second-derivative temperature kernel.
* Record finite source-role coverage, sample counts, and history convergence
  diagnostics in the runtime envelope.

Required acceptance:

* Analytic source fixtures reproduce each source term and sign.
* Source histories remain stable under a source-grid refinement.
* A missing source makes its spectrum unavailable with a reason and does
  not fabricate zeros or evaluate an unrelated sector.
* The targeted source-contract and projection tests complete within three
  minutes.

Done when:

* Scalar line-of-sight sources are canonical, normalized, and declaration-
  driven before radial projection is assessed. Temperature source roles use
  the shared kernel contract, finite source histories are reported, and
  missing histories fail with an explicit availability error.

### [closed] Slice Twenty-Two - Independent projection kernels

Purpose:

Validate radial kernels independently of scalar source evolution and
line-of-sight integration.

Depends on:

* Slice Twenty-One.

Scope:

* Validate spherical-Bessel, derivative, spin-2, tensor, vector, and lensing
  kernels against analytic or SciPy references.
* Validate kernel batching at fixed inputs against scalar evaluation.
* Validate interpolation, endpoint, parity, and zero-crossing behavior.
* Reject sector-incompatible kernels before projection.

Required thresholds:

* Nonzero kernel reference values agree within `1e-10` relative tolerance.
* Batched and scalar kernel values agree within `1e-12` at fixed inputs.
* Kernel tests complete within three minutes without reducing requested
  multipole or radial ranges.

Done when:

* Every production projection kernel has an independent numerical or
  analytic acceptance test.
* Spherical-Bessel values and derivatives agree with SciPy at positive,
  negative, and zero arguments, including the signed parity limits.
* Batched scalar, spin-2, vector, and tensor kernels remain finite at the
  zero-argument endpoint and agree with their analytic limits.
* Projection contracts reject incompatible sector and kernel combinations
  before line-of-sight integration.

### [closed] Slice Twenty-Three - Scalar projection convergence

Purpose:

Prove convergence of scalar transfer functions and unlensed spectra after
source and kernel correctness are established.

Depends on:

* Slice Twenty-Two.

Scope:

* Refine the k grid, eta grid, source grid, and radial integration together
  only through explicit accuracy controls.
* Preserve low-ell, visibility-era, and high-ell projection regions.
* Compare `TT`, `TE`, `EE`, and `PP` on fixed accepted anchor surfaces.
* Validate requested-spectrum filtering and output availability.

Required thresholds:

* Doubling k and eta projection resolution changes `TT`, `TE`, and `EE` by
  less than `1%` on the accepted surface.
* Source and kernel refinements preserve transfer normalization.
* A requested projection never evaluates an unrelated sector or fabricates
  a missing spectrum.

Done when:

* Scalar transfer functions and unlensed spectra converge independently of
  any external reference spectrum on fixed low-ell, visibility-era, and
  high-ell anchor surfaces.
* Nested k and eta projection surfaces meet the one-percent `TT`, `TE`, and
  `EE` threshold, with finite `PP` output and explicit source-history
  convergence diagnostics.
* Requested-spectrum filtering evaluates only required source terms and
  rejects unavailable spectra before runtime work begins.

### [closed] Slice Twenty-Four - Native scalar absolute parity

Purpose:

Establish absolute scalar parity for the native LCDM acceptance model after
scalar evolution and projection have separately converged.

Depends on:

* Slice Twenty-Three.

Scope:

* Compare native `TT`, `TE`, and `EE` over ell `2..2000`.
* Compare native `PP` over ell `10..1500` and `TP` and `EP` over their
  declared supported ranges.
* Compare acoustic peak locations and TE zero crossings.
* Generate CAMB or CLASS results only in independent scientific tests.
* Prohibit response ratios, empirical scales, injections, and fallback output
  as parity evidence.

Required thresholds:

* `TT` and `EE` median and 90th-percentile errors are at or below `5%` and
  `10%`.
* Normalized `TE` RMS error is at or below `5%`.
* The first three TT and EE peaks and TE zero crossings are within three
  ell.
* `PP` median and 90th-percentile errors are at or below `10%` and `20%`.

Done when:

* Native scalar output meets every absolute fixed-cosmology threshold and
  the fixture proves that the production graph has no undeclared LCDM term.

Implementation record:

* The fixed scalar acceptance surface uses the native declared graph for
  `TT`, `TE`, `EE`, `PP`, `TP`, and `EP`; CAMB remains an independent test
  reference.
* Scalar k quadrature retains declared multipole, conformal-distance, and
  sound-horizon phase anchors inside the bounded node budget.
* Phase-aware k quadrature is explicitly enabled by the acceptance contract;
  other contracts retain the bounded anchor-and-gap grid unless they opt in.
* Continuous exact collision blocks apply their compiled matrix and damping
  terms once, while split collision outputs remain suppressed from the
  generated equation graph.

### [closed] Slice Twenty-Five - Lensing remapping correctness

Purpose:

Validate lensing normalization and interpolation independently before
feeding native scalar spectra through the complete lensed pipeline.

Depends on:

* Slice Twenty-Four.

Scope:

* Validate remapping normalization with independent unlensed inputs.
* Validate remapping interpolation, quadrature, endpoints, and parity.
* Validate the declared `PP` convention and lensing-potential dependence.
* Prohibit Gaussian smoothing, direct spectrum injection, and empirical
  output scaling.

Required acceptance:

* Analytic remapping fixtures meet their declared normalization tolerance.
* Independent interpolation fixtures remain finite and converge under radial
  refinement.
* Changing `PP` changes the remapped spectrum itself.

Done when:

* The remapper is physically normalized, interpolation-stable, and isolated
  from production reference output.

Implementation record:

* Curved-sky remapping validates finite compatible `cls` and `clpp` surfaces,
  ordered interior quadrature nodes, and the declared sampling envelope before
  numerical work begins.
* Zero-deflection identity and sampling-refinement fixtures cover analytic
  normalization and interpolation stability without reference-spectrum
  participation.

### [closed] Slice Twenty-Six - Lensed scalar absolute parity

Purpose:

Compare the complete native scalar-to-lensed pipeline with an independent
reference using native unlensed spectra and native `PP`.

Depends on:

* Slice Twenty-Five.

Scope:

* Compare lensed `TT`, `TE`, `EE`, and `BB` over the declared ell range.
* Verify primordial and generated B-mode sources survive lensing.
* Keep lensed and unlensed availability states distinct.

Required thresholds:

* Lensed `TT`, `TE`, and `EE` meet the scalar absolute parity thresholds.
* Lensed `BB` median fractional error is at or below `15%`.
* Lensed output changes when native `PP` changes.

Done when:

* Full lensed scalar parity passes without reference-spectrum participation
  in the production path.

Implementation record:

* Exact lensed assembly requires a contiguous zero-based analysis surface and
  derives every lensed `TT`, `TE`, `EE`, and `BB` result from native unlensed
  transfers plus native `PP`.
* Orchestration tests verify that declared odd-parity input survives the
  remapper and that sparse public requests cannot bypass contiguous analysis.

### [closed] Slice Twenty-Seven - Massive-neutrino q-hierarchy correctness

Purpose:

Validate the physical q-resolved massive-neutrino hierarchy before absolute
massive-neutrino spectrum comparison.

Depends on:

* Slice Twenty-Six.

Scope:

* Validate q nodes, weights, thermal momentum factors, and quadrature order.
* Validate density, pressure, momentum, and shear source moments.
* Validate relativistic-to-nonrelativistic background transitions.
* Ensure only contracts that declare a massive-neutrino species receive q
  states and source terms.
* Remove any aggregate massive-neutrino state that can drift from q bins.

Required acceptance:

* Fixed-distribution moment fixtures meet independent quadrature tolerances.
* q refinement changes accepted moments within the final convergence
  tolerance.
* Models without a declared massive-neutrino species remain free of q
  states and massive-neutrino sources.

Done when:

* The resolved q hierarchy is the sole massive-neutrino physical authority.

Implementation record:

* Massive-neutrino grids now validate their count, positive q bounds,
  second-order logarithmic trapezoid rule, finite nodes, and positive
  quadrature weights before entering the runtime cache.
* Density, pressure, momentum, and shear moments retain their distinct
  distribution, epsilon, and velocity factors; strict aliases remain derived
  from the q-bin state surface.
* Independent log-q fixtures cover the physical moments, invalid grid
  definitions, q refinement, and the absence of q runtime state when no
  massive-neutrino species is declared.

### [closed] Slice Twenty-Eight - Massive-neutrino absolute parity

Purpose:

Compare the q-resolved massive-neutrino native physical source spectra at
fixed cosmologies with independent quadrature references.

Depends on:

* Slice Twenty-Seven.

Scope:

* Compare absolute density, pressure, momentum, and shear source spectra for
  models that explicitly declare massive neutrinos and q hierarchies.
* Use fixed cosmologies rather than response ratios or `sum_mnu` responses.
* Cover the relativistic and nonrelativistic transition regimes.

Required threshold:

* Accepted massive-neutrino source-spectrum errors are at or below `10%`.

Done when:

* Absolute massive-neutrino source parity passes across both transition
  regimes and remains tied to the resolved q hierarchy.

Implementation record:

* Massive-neutrino dipoles evolve as the regularized physical variable
  `v(q,a) Psi_1`, removing the nonrelativistic `1/v` singularity from the
  declared q equations.
* The q-integrated momentum source uses the matching `q^4 f_0 / v` weight,
  while density, pressure, and shear retain their independent physical
  moments.
* Scientific tests compare all four absolute source spectra against direct
  log-q quadrature at fixed relativistic and nonrelativistic cosmologies,
  with a ten-percent acceptance boundary and no response-ratio evidence.

### [closed] Slice Twenty-Nine - Tensor hierarchy correctness

Purpose:

Complete the physical tensor hierarchy and tensor source normalization before
absolute tensor parity.

Depends on:

* Slice Twenty-Eight.

Scope:

* Implement and validate tensor metric, photon, polarization, and neutrino
  equations from declared tensor roles.
* Validate tensor amplitude, tilt, initial conditions, and terminal closure.
* Validate tensor source normalization and tensor radial-kernel selection.
* Keep tensor physics separate from scalar and vector state names.

Required acceptance:

* Tensor state histories are finite and satisfy declared constraints.
* Tensor source and radial-kernel fixtures match independent analytic limits.
* Increasing tensor hierarchy depth changes accepted histories within `1%`.

Done when:

* The tensor sector is physical, converged at its working depth, and
  declaration-driven.

Implementation record:

* The generated tensor graph evolves the metric wave, photon and neutrino
  spin-2 temperature moments, and photon E/B polarization moments without
  scalar-style monopole, dipole, or low-order polarization placeholders.
* Regular superhorizon metric, neutrino-stress, and photon-collision
  constraints use the radiation free-streaming fraction and are validated
  before each tensor mode evolves.
* Independent fixtures validate tensor temperature and polarization source
  normalization, spin-2 radial-kernel selection, and every terminal closure.
* Fixed source-history comparisons show that increasing the photon,
  polarization, and neutrino hierarchy depths changes accepted tensor
  histories by less than one percent.

### [closed] Slice Thirty - Tensor absolute parity

Purpose:

Establish absolute tensor spectrum parity using fixed-cosmology independent
references.

Depends on:

* Slice Twenty-Nine.

Scope:

* Compare native tensor `TT`, `EE`, and `BB` absolutely.
* Compare tensor unlensed and lensed outputs.
* Keep the proof independent of synthetic tensor probes and response ratios.

Required threshold:

* Tensor `TT`, `EE`, and `BB` median fractional errors are at or below `10%`.

Done when:

* Tensor absolute parity passes and tensor primordial `BB` survives native
  lensing.

Implementation record:

* The fixed tensor acceptance cosmology uses `r = 0.1`, `nt = 0`, and the
  native declared tensor hierarchy on a contiguous projection surface.
* Independent CAMB calls provide absolute tensor `TT`, `EE`, and `BB`; the
  tensor contribution to CAMB's lensed total is isolated from its lensed
  scalar surface without calling production native code.
* Native unlensed and remapped tensor `TT`, `EE`, and `BB` pass the
  ten-percent median fractional-error boundary at the declared reference
  multipoles.
* The native remapper retains finite positive primordial tensor `BB` while
  using the independently evolved native scalar lensing potential.

### [closed] Slice Thirty-One - Gauge-equivalent scalar parity

Purpose:

Validate Newtonian, synchronous, and gauge-invariant scalar routes against
the same declared theory graph.

Depends on:

* Slice Thirty.

Scope:

* Validate explicit gauge transformations and invariant variables.
* Compare gauge routes at fixed cosmology, grids, initial mode, and source
  roles.
* Prevent LCDM matter aliases or gauge labels from forcing identity.
* Compare both histories and final scalar spectra.

Required threshold:

* Gauge-equivalent scalar spectra agree to `0.1%` on the accepted surface.

Done when:

* Gauge routes agree through explicit transformations without alias-forced
  equality.

Implementation record:

* Compiled perturbation manifests expose the scalar gauge-equivalence route,
  Newtonian observable basis, metric state names, and derived bridge nodes.
* Conformal-Newtonian, synchronous, and Bardeen-invariant contracts retain
  distinct compiled metric state graphs rather than passing through labels or
  LCDM aliases.
* A fixed-cosmology acceptance test compares visible source histories and
  scalar `TT`, `TE`, `EE`, `BB`, `PP`, `TP`, `EP`, and lensed surfaces across
  all three routes at the `0.1%` boundary.
* Synchronous transformations are checked through `Phi_from_synchronous`
  and `Psi_from_synchronous`; invariant spectra are checked through the
  declared `Phi_gi` and `Psi_gi` states.

### [closed] Slice Thirty-Two - Physical vector hierarchy and parity

Purpose:

Complete and validate the physical vector sector independently of scalar and
tensor acceptance.

Depends on:

* Slice Thirty-One.

Scope:

* Implement vector metric, matter, photon, polarization, and source roles
  from the declared vector graph.
* Validate vector parity, normalization, initial conditions, and terminal
  closure.
* Compare vector radial kernels and generated vector spectra with analytic
  flat-space limits.
* Keep sector totals consistent with their declared inputs.

Required acceptance:

* Vector analytic-limit residuals meet their declared tolerances.
* Vector source normalization and parity remain stable under refinement.
* A scalar-only contract does not acquire vector state or source terms.

Done when:

* Generated vector output passes physical analytic and absolute acceptance
  tests.

Implementation record:

* The generated vector graph evolves the transverse metric shear
  `sigma_vector`, baryon and optional CDM vorticity, photon heat flux and
  anisotropic stress, massless-neutrino vector moments, and photon E/B
  polarization hierarchies. Thomson vector drag and free-streaming terminal
  closure remain declaration-driven.
* The compiled manifest records the physical vector metric state, hierarchy
  role groups, even/odd parity, free-streaming closure, and all four vector
  radial kernels. Scalar-only manifests report no vector states, sources, or
  kernels.
* Independent flat-space fixtures validate vector temperature, E, and B
  radial kernels and their signed parity limits. Finite transfer and
  spectrum checks cover vector `TT`, `TE`, `EE`, and `BB`, and exact lensing
  retains declared vector primordial `BB`.
* The vector acceptance tests verify nonzero odd-parity output, physical
  source normalization, terminal hierarchy materialization, and isolation
  from scalar-sector contracts.

### [closed] Slice Thirty-Three - Native production route cutover

Purpose:

Remove production solver-route branching from the core model and execution
contracts after native scalar, lensing, neutrino, tensor, gauge, and vector
acceptance is complete.

Depends on:

* Slice Thirty-Two.

Scope:

* Remove the production `standard` solver-route boolean and backend fallback
  from the schema, validator, coder, adapters, cache, and CMB facade.
* Keep CAMB and CLASS imports confined to independent scientific tests.
* Remove obsolete compatibility readers, aliases, and bridge paths.
* Preserve explicit unavailable-spectrum behavior and native cache identity.

Required acceptance:

* Production execution has one native declared-graph route.
* Negative tests reject removed route flags and backend fallback requests.
* Production modules contain no CAMB or CLASS import or call.

Done when:

* Core production execution cannot select or fall back to a second CMB
  solver.

Implementation record:

* The schema, compiler, model adapter, cache identity, likelihood facade,
  run pipeline, and run manifest expose one native declared-graph route.
  Removed `backend`, `standard`, and `backend_mapping` selectors fail before
  graph compilation.
* Production CMB, BAO, statistics, and utility modules contain no CAMB or
  CLASS imports or calls. The independent CAMB reference helper resides in
  the scientific test package and cannot be imported through the production
  package.
* Model declarations, documentation templates, and synthetic integration
  fixtures use the route-neutral contract. Native runtime preparation owns
  parameter binding, unavailable spectra remain explicit, and cache
  signatures retain compiled native graph identity.
* Targeted contract, adapter, likelihood, manifest, model-template,
  integration, packaging, and production-import-isolation tests pass within
  their three-minute command budgets.

### [closed] Slice Thirty-Four - Model contract migration and asset cutover

Purpose:

Migrate every shipped CMB model to the native theory-neutral contract after
the production route has one authority.

Depends on:

* Slice Thirty-Three.

Scope:

* Remove transitional route metadata and LCDM-only descriptions from every
  model declaration.
* Make each model's species, equations, sources, initial modes, and sectors
  explicit and theory-accurate.
* Keep one canonical native LambdaCDM declaration at `model_lcdm.yml` and
  remove the duplicate model asset.
* Regenerate model caches, manifests, validation fixtures, and references.

Required acceptance:

* Every shipped model validates and executes through the native contract.
* No model contains undeclared LCDM species, source terms, or historical
  migration wording.
* Renamed model assets resolve consistently in tests, manifests, and docs.

Done when:

* The model corpus is native, theory-accurate, and free of transitional
  route artifacts.

Implementation record:

* The model library contains one canonical `model_lcdm.yml` native asset.
  Model discovery, tests, manifests, and documentation resolve that filename.
* Every CMB-capable model compiles non-empty equations, sources, observables,
  and regular initial data from its declared scalar sector, species, hierarchy
  families, collision rules, and model-specific source closures.
* QRSF, TORG, and USMF contain no CDM parameter or background alias. QRSF and
  TORG compile their baryon-locked relational source closures; USMF keeps CMB
  output explicitly unavailable.
* Corpus validation records exact species and source inventories, rejects
  route metadata and developmental wording, and executes finite native smoke
  spectra for every model that declares CMB availability.

### [closed] Slice Thirty-Five - User-facing native-only cutover

Purpose:

Align CLI, GUI, run builders, plot labels, and user documentation with the
single native production route.

Depends on:

* Slice Thirty-Four.

Scope:

* Remove public CAMB/native solver selection from CLI and GUI.
* Preserve control-model and test-model selection as the comparison API.
* Update run summaries, plot footers, cache labels, and error messages to
  describe native execution and arbitrary control/test model pairs.
* Update user-facing model and solver documentation without historical
  migration language.

Required acceptance:

* CLI and GUI expose no second CMB solver choice or backend flag.
* Control and test model choices remain identical through shared execution
  code.
* User-facing labels report the selected model pair rather than LCDM versus
  an implicitly selected theory.

Done when:

* User-facing workflows truthfully expose one native CMB engine and flexible
  model comparison.

Implementation record:

* CLI and GUI workflows select exactly one control model, one test model, and
  one sampler engine. They expose the fixed Copernican native declared-graph
  CMB identity for provenance without a CMB solver or backend selector.
* Manifest loading, execution, result writing, CSV export, plotting, and
  analysis consume the same ordered control/test comparison. Missing role
  records fail, while same-model comparisons retain distinct control and test
  outputs.
* Native CMB manifest, cache, runtime, footer, and error labels use one
  production identity. Transitional custom-route labels and implicit LCDM
  comparison assumptions are absent from user-facing surfaces.
* Repository and package documentation distinguish sampler engines from the
  native CMB engine, describe arbitrary model pairs, and direct physical solver
  details to the canonical CMB convention document.
* Targeted CLI, GUI, manifest, executor, plotting, analysis, result-writing,
  cache, model-selection, and synthetic-integration tests cover the cutover.

### [closed] Slice Thirty-Six - Scientific reference and package isolation

Purpose:

Separate scientific reference tooling from production packaging and assets
after the native-only route and model corpus are complete.

Depends on:

* Slice Thirty-Five.

Scope:

* Move CAMB or CLASS to test/development dependency surfaces where packaging
  permits.
* Keep independent reference builders and fixtures outside production
  modules and installed package entry points.
* Update lockfiles, license reports, manifests, cache identities, and package
  discovery coherently.
* Remove obsolete CAMB-style assets without deleting independent references.

Required acceptance:

* A production installation imports and runs native CMB execution without
  CAMB or CLASS installed.
* Scientific reference tests still build independently when their dependency
  is present.
* Dependency, license, manifest, and cache artifacts are synchronized.

Done when:

* Production and scientific-reference dependency boundaries are explicit and
  mechanically enforced.

Implementation record:

* The default package dependency manifest and packaged runtime lock exclude
  CAMB and CLASS. The repository workspace manifest retains CAMB as an exact
  independent scientific-reference dependency for tests.
* Package and workspace license inventories follow their owning dependency
  surfaces. Wheels and source manifests exclude workspace licenses, test
  reference modules, obsolete CAMB adapters, and bytecode artifacts.
* Native runtime cache identities include the canonical native execution
  engine. The independent CAMB helper records its own provider and version
  identity entirely under the test tree.
* Installed-package validation inspects wheel metadata and assets, blocks
  CAMB and CLASS imports, and executes a finite native declared-graph spectrum.
  Focused dependency, license, manifest, package-discovery, cache, and
  independent-reference tests enforce the boundary.

### [open] Slice Thirty-Seven - Cross-sector numerical convergence

Purpose:

Prove final physical-output convergence across sectors and numerical controls
after native-only production is established.

Depends on:

* Slice Thirty-Six.

Scope:

* Refine background, eta, k, photon, massless-neutrino, massive-neutrino,
  tensor, and vector hierarchy controls.
* Refine q grids, source grids, and lensing quadrature.
* Record the active numerical envelope in validation output.
* Fail when an accuracy tier is under-resolved instead of silently lowering
  requested physics.

Required thresholds:

* Final refinement changes `TT` and `EE` by less than `1%`.
* Final refinement changes normalized `TE` by less than `2%` and `PP` by less
  than `3%`.
* Final refinement changes lensed `BB` by less than `5%`.
* q-grid refinement changes accepted massive-neutrino spectra by less than
  `2%`.
* Hierarchy refinement changes accepted spectra by less than `1%`.

Done when:

* Every physical numerical control demonstrates converged output and every
  requested accuracy tier has an explicit bounded envelope.

### [open] Slice Thirty-Eight - Output, cache, and contract consistency

Purpose:

Validate public output availability, cache identity, plotting surfaces, and
cross-sector consistency before final repository closure.

Depends on:

* Slice Thirty-Seven.

Scope:

* Keep unavailable, physically zero, and unrequested spectra distinct.
* Include structure, bound parameters, grids, requested spectra, and
  accuracy controls in cache identity.
* Validate repeated and noncontiguous ell in multi-spectrum likelihoods.
* Keep scalar, vector, tensor, lensed, unlensed, lensing, sector, and
  diagnostic plotting surfaces separate.
* Remove acceptance-only tests and replace tests whose names overstate their
  assertions.

Required acceptance:

* Changing `PP` changes lensed spectra, and primordial `BB` survives
  lensing.
* Missing spectra remain unavailable rather than fabricated.
* Cache reuse never returns a result for a changed contract or accuracy
  control.
* Scalar, vector, tensor, and total components remain internally consistent.

Done when:

* Public APIs, cache behavior, plotting surfaces, and spectrum metadata
  agree with the native physical graph.

### [open] Slice Thirty-Nine - Final scientific and repository closure

Purpose:

Perform the final end-to-end acceptance and close the roadmap only when every
implementation, scientific, architectural, and documentation claim agrees.

Depends on:

* Slice Thirty-Eight.

Scope:

* Re-run the complete local repository gate from a clean staged state.
* Run the complete required workflow tests with bounded diagnostics and
  inspect their artifacts rather than accepting silent timeouts.
* Recheck that source, tests, model declarations, public APIs, docs, caches,
  manifests, dependencies, and changelog entries agree.
* Recheck that production code has no empirical scales, source injections,
  hidden fallbacks, reference lookups, or acceptance-only equations.
* Change every remaining slice marker to `[closed]` only after its own
  acceptance evidence is present.

Required acceptance:

* All final scientific thresholds from Slices Seventeen through Thirty-Eight
  remain green together.
* The complete repository workflow passes from a clean checkout.
* No slice remains open, deferred, or supported by an unmeasured claim.

Done when:

* Copernican ships a fast, theory-agnostic, declarative native CMB solver
  with absolute reference parity, demonstrated convergence, native-only
  production execution, and complete repository acceptance.
* The full local repository gate passes from a clean checkout.
* No item from this roadmap remains open or deferred.

## Completion Standard

This roadmap is complete only when all thirty-nine slices are `[closed]`.

The repository must then truthfully satisfy all of the following:

* Copernican ships a native, universal, theory-agnostic
  Boltzmann-hierarchy CMB solver.
* Every production CMB model uses the native declared contract without a
  CAMB or CLASS fallback.
* The native LCDM acceptance model contains physical scalar, vector,
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
* Contract compilation, background preparation, and hierarchy evolution use
  the shared batched runtime architecture.
* Transfer and line-of-sight projection use adaptive physical refinement
  rather than under-resolved fixed sparse grids.
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
* The complete repository gate passes from a clean checkout.
* Documentation and changelog statements match the measured code state.

No slice may be marked `[closed]` because another slice is expected to fix
it. If any completion statement is false, the responsible slice remains
`[open]`.
