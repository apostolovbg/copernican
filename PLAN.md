# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-09-09
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

> **For agentic workers:** Execute the slices in order. Keep the gate open
> for the active slice, stage each completed slice, and do not call a slice
> closed until its implementation and raw scientific evidence both exist. A
> green policy gate is necessary hygiene; it is never scientific closure.

**Goal:** Deliver a working Copernican Cosmic Microwave Background Solver
(CCMBS) that replaces the old CAMB-backed production path. CCMBS is a
universal declarative solver. LambdaCDM is one ordinary theory declaration,
not a baseline, fallback, privileged branch, or definition of valid physics.
Every mathematically well-posed declaration that the grammar can describe,
including unfamiliar future theories, must be consumed by the same engine,
solved with automatically selected numerical resolution, and rendered as a
usable CAMB-like graph. CAMB is a comparison oracle only.

**Scope:** This plan owns the declarative model contract, expression compiler,
automatic numerical planner, background and recombination evolution, metric
and species hierarchies, collision and opacity operators, scalar/vector/tensor
line-of-sight projection, unlensed and lensed TT/TE/EE/BB/PP/TP/EP surfaces,
normalization and units, diagnostics, GUI/CLI graph production, all bundled
models, novel declarative models, CAMB comparison, sampler integration, and
the independent BAO background boundary.

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
optimization, or a CAMB runtime fallback. Such work cannot conceal a CCMBS
defect. The existing Python 3.11 `.venv` is managed outside this plan and
must not be recreated, replaced, upgraded, or otherwise modified.

**Reason for the reset:** The production output from 2026-09-02 showed the
actual failure mode. LambdaCDM produced no CMB curve because CCMBS rejected
its production request as non-converged. The Planck-reference curve that did
appear was spiky and orders of magnitude away from the observations. The
repository accumulated model-specific numerical recipes while attempting to
repair shared projection and evolution defects. Those recipes are not theory
and cannot be the scientific contract. This reset removes that approach. The
engine, not each YAML file, must determine resolution and prove convergence.

## Global Constraints

* Do not change branches or create branches.
* CCMBS is the selected production CMB solver. CAMB is comparison-only.
* Model files contain theory, not solver implementation instructions.
  `numerical`, `accuracy_controls`, hierarchy floors, ODE tolerances,
  quadrature counts, coupling thresholds, fixed integration starts, and
  production work limits are forbidden in model declarations.
* Requested ell ranges and observable subsets are request/API inputs. The
  engine derives all internal k, eta, q, hierarchy, ODE, source, and
  projection resolution from those requests and from the declared physics.
* No model-specific numerical patch may repair a shared engine defect.
* Preserve every theory. Model-file changes may only remove implementation
  controls or make the theory's intended equations, domains, derivatives,
  closures, gauges, interactions, and observables explicit.
* Never lower accuracy, omit a declared surface, clip a requested range,
  widen a cache key, swallow a typed failure, or relax a tolerance merely to
  obtain a pass.
* LambdaCDM must follow exactly the same parser, compiler, planner,
  background, hierarchy, projection, and validation routes as every other
  declaration. No model-name or LambdaCDM-family inference is permitted.
* A mathematically well-posed declaration is never rejected as unsupported
  physics. If the grammar or runtime cannot express it, extend CCMBS in the
  owning slice. The only accepted declaration failures are malformed,
  incomplete, dimensionally inconsistent, singular, non-finite, internally
  contradictory, or explicitly constraint-violating mathematics.
* A valid request may fail only with a precise typed numerical diagnosis, such
  as non-finite evolution or failed convergence. It may not become an empty
  graph, an unavailable-model status, or a silently substituted spectrum.
* The engine may not use an arbitrary wall-clock or nominal work ceiling as
  numerical acceptance. It must adapt resolution until convergence or return
  the raw evidence and typed reason it cannot do so.
* Never silently accept a non-converged spectrum. Preserve grids, weights,
  source histories, transfer arrays, residuals, and both refinement products.
* Generated metric derivatives, visibility, collision sources, polarization,
  ISW, tensor/vector sources, and initial conditions must be explicit,
  finite, coordinate-aware, and independently validated. Missing derivatives
  are failures, never zero-valued substitutes.
* Every declared observable must be computed. This includes TT, TE, EE, BB,
  PP, TP, EP and all declared scalar, vector, tensor, lensed, unlensed, auto,
  cross, and total surfaces. A zero surface is valid only when the declaration
  proves it is physically zero.
* Scalar execution is the reference. Ordered batching and caching are valid
  only after scalar equivalence, input ordering, failure semantics, and cache
  isolation are proven.
* BAO consumes the generated background's drag ruler independently of the CMB
  likelihood. Recombination and drag sound horizons are distinct quantities.
* Root and package documentation remain synchronized. Every behavior change
  updates focused tests, comments/docstrings, README mirrors, and CHANGELOG.
* Always activate the existing `.venv` before DevCovenant commands and tests.

## Table of Contents

* [Overview](#overview)
* [Model Contract](#model-contract)
* [Automatic Numerical Contract](#automatic-numerical-contract)
* [Scientific Acceptance Contract](#scientific-acceptance-contract)
* [Universal Theory Contract](#universal-theory-contract)
* [CAMB Parity Contract](#camb-parity-contract)
* [Diagnostic Status Terms](#diagnostic-status-terms)
* [Execution Rules](#execution-rules)
* [Execution Slices](#execution-slices)
* [Completion Standard](#completion-standard)

## Overview

The mission is a CMB engine, not a collection of infrastructure checks or
hand-tuned model recipes. Each slice below owns implementation and raw
scientific evidence for its stated behavior. No slice is a verification-only
placeholder: its acceptance tests exercise real CCMBS requests and produce
raw arrays or artifacts that the next slice can consume.

The production path has one honest behavior. A valid declaration is planned,
evolved, projected, validated, and returned with complete requested spectra;
or it fails with a precise diagnostic containing the failing mathematical or
numerical evidence. The GUI and CLI must never turn a failed calculation into
an absent curve or display only a reference curve.

The numerical planner is a first-class engine component. It inspects the
requested ell surface, source visibility width, acoustic and radial phase,
momentum distributions, hierarchy stiffness, and declared equations. It
chooses and refines resolution, verifies independent convergence, and records
the decision. No YAML model is allowed to carry a second, competing planner.

## Model Contract

The accepted model schema describes only physics:

* parameters, dimensions, priors, and physical domains;
* background species, stress-energy terms, and equations of state;
* recombination, opacity, visibility, and drag equations;
* perturbation variables, derivatives, gauges, sectors, and equations;
* collision and interaction operators, closures, and conservation laws;
* initial and boundary conditions;
* source bindings, projection conventions, and declared observables.

The validator rejects solver controls wherever they occur in a model file. A
request may provide an observable range or an explicit diagnostic purpose,
but not an internal node count or tolerance. The compiled contract exposes
the physics and its dimensions to CCMBS; the planner owns every numerical
choice and reports the resolved choices in runtime telemetry.

Migration of the ten bundled files is mandatory. Removing a numerical field
must not change a theory's equations or parameter semantics. If a former
field encoded an actual physical domain, it is rewritten as a mathematical
domain or validity condition and is handled by the validator, not by a grid
builder.

## Automatic Numerical Contract

CCMBS must derive resolution rather than guess it. The planner performs the
following closed loop for every request:

1. Analyze equation scales, stiffness, interaction rates, visibility width,
   acoustic phase, radial phase, q-distribution support, and requested ell
   coverage.
2. Choose initial grids and hierarchy truncations from error estimators and
   physical support, with no model-provided counts.
3. Evolve scalar reference modes with adaptive ODE error control and explicit
   tight-coupling entry/exit detection.
4. Construct source and line-of-sight grids from source features and kernel
   phase, not from a fixed eta or k ladder.
5. Refine each independent surface—background, q, hierarchy, source, k, eta,
   and projection—until its declared numerical error bound is met.
6. Recompute selected modes with a scalar reference path and compare all
   batch/cache paths before accepting the result.
7. Return the resolved plan, error estimates, refinement ratios, and raw
   products alongside the spectra.

The planner must be deterministic for identical physical inputs and request
shapes, but it may choose different grids for different theories or ell
surfaces when their equations require it. A failure must identify the
surface, resolution levels, residuals, and remaining error; it must never be
fixed by silently accepting an under-resolved spectrum.

## Scientific Acceptance Contract

Every model and observable must pass all layers below at its automatically
resolved production tier.

1. **Theory fidelity:** the compiled contract contains only the supplied
   equations and declarations, with no imported LambdaCDM assumptions.
2. **Physical histories:** background, recombination, metric potentials,
   densities, velocities, collisions, visibility, polarization, tensor/vector
   sources, initial conditions, and ISW histories are finite and
   residual-clean.
3. **Automatic resolution:** all independent numerical surfaces converge under
   planner-selected refinement and retain their raw evidence.
4. **Complete observables:** every declared TT/TE/EE/BB/PP/TP/EP and applicable
   lensed, unlensed, scalar, vector, tensor, and total surface is present.
5. **Physical shape:** acoustic phase, peak/trough sequence, damping,
   low-ell behavior, cross-spectrum signs, lensing response, and tensor/vector
   behavior are sensible for the declared theory.
6. **Execution equivalence:** scalar, batch, cache-warm, and cache-cold paths
   agree bitwise within the declared floating-point envelope.
7. **Evidence:** raw arrays, histories, grids, weights, residuals, planner
   decisions, parity rows, graph files, and failure decisions are canonical,
   reproducible, and hashable.

## Universal Theory Contract

The unit of support is a complete mathematical declaration, not a model name
or a resemblance to LambdaCDM. A complete declaration supplies equations and
domains for every requested sector, species stress-energy, recombination and
opacity law, perturbation and collision operators, metric closures, gauge,
initial conditions, source derivatives, projection conventions, and physical
observables. It may describe ordinary matter, modified gravity, extra fluids,
non-standard recombination, or an entirely novel interaction.

The compiler and runtime must consume every complete declaration expressible
by the schema. A theory with unfamiliar names or different equations is not
an unavailable model. A genuine grammar/compiler limitation is an engine bug
and is repaired by extending the generic path. The engine must not contain
model-name branches that decide whether a declaration is edible.

A declaration can be rejected before execution only for malformed,
incomplete, dimensionally inconsistent, singular, non-finite, or internally
contradictory mathematics. A valid declaration can fail during execution only
with typed numerical evidence such as a non-finite state, violated declared
constraint, or failed adaptive convergence. `EngineCapabilityError` may help
locate a development defect, but it is never an accepted final status for a
valid theory and never a substitute for implementing the missing path.

## CAMB Parity Contract

CAMB parity is required wherever CAMB implements the same physics and
conventions: LambdaCDM, massive-neutrino LambdaCDM, the fixed Planck-reference
point, and matched wCDM/w0wa limits. The comparator uses identical physical
parameters, primordial spectrum, recombination and neutrino conventions,
ell grid, units, normalization, lensing mode, and sector definitions.

The comparator evaluates complete raw arrays for TT, TE, EE, BB, PP, TP, EP
and every applicable lensed, scalar, vector, tensor, and total surface. It
reports absolute, relative, band-limited, peak-position, phase, damping,
sign, and zero-crossing errors. Near-zero cross-spectra use an explicit
absolute-plus-relative metric. Several fixed points, mass points, and ell
bands are mandatory; one synthetic fixture or one low-ell smoke request is
not parity evidence.

For theories CAMB does not implement, CCMBS still computes every declared
surface and passes internal physical invariants, adaptive convergence, and
shape checks. Such theories are not falsely labelled CAMB-equivalent.

## Diagnostic Status Terms

* **valid:** the declaration is mathematically complete and its requested
  surfaces execute with finite converged evidence.
* **malformed:** validation found a named mathematical, dimensional, domain,
  or consistency defect in the declaration.
* **non-finite:** execution produced a non-finite physical state or source.
* **non-converged:** independent planner refinement remains outside its bound;
  both products and the measured error are retained.
* **engine defect:** a valid declaration exposed a missing compiler/runtime
  path; this is actionable development evidence, never model rejection.
* **physical zero:** a declared surface is identically zero only when its
  equations and symmetries prove that result.

## Execution Rules

1. Confirm the existing managed environment and activate `.venv`; never
   recreate or mutate it.
2. Open the DevCovenant gate before edits and clear blocking complaints.
3. Work only on the active slice and stage all files when its implementation
   and evidence are complete.
4. Run focused acceptance tests for the slice. Do not call a slice closed on
   a policy gate alone.
5. Run `devcovenant gate --verify` before reporting. A user-run
   `devcovenant run` remains a required full-suite confirmation.
6. Inspect run artifacts before interpreting failures. Fix source defects,
   not symptoms, and rerun the affected acceptance criteria.

## Execution Slices

### [planned] Slice One — pure theory declarations and automatic planner

Remove all solver implementation controls from the ten model files and from
the accepted model schema. Introduce the engine-owned numerical-planning
contract and request-level observable/ell inputs. Migrate model validation,
compiled dataclasses, cache identity, manifests, docs, and fixtures so no
runtime path reads a model `numerical` or `accuracy_controls` block.

Implement the planner's physical scale analysis and deterministic baseline
resolution selection. It must derive initial k/eta/q grids, hierarchy orders,
ODE tolerances, tight-coupling transitions, and integration starts from
equations and request shape, then expose its decisions in telemetry. Preserve
the scalar reference path and make under-resolution an explicit diagnostic.

Acceptance requires every bundled declaration to validate without numerical
knobs, a novel renamed declaration to compile identically, planner decisions
to be finite and deterministic, cache keys to include all physical inputs,
and no model-name branch or hidden standard-cosmology default to be found by
repository audit. Raw planner manifests are required for at least LCDM,
USMF2, QAU, QRSF, TOG, TORG, wCDM, and w0wa.

### [planned] Slice Two — automatic background, recombination, and drag

Move background resolution, recombination, opacity, visibility, and drag
transition selection entirely into the planner/runtime. Derive adaptive
scale-factor and conformal-time grids from equation curvature, interaction
rates, visibility width, and requested source accuracy. Use one physical
background contract for CMB recombination quantities and an independent drag
ruler for BAO.

Implement q-resolved massive-neutrino density and pressure from the same
thermal distribution used by perturbations. Close `N_eff`, mass, radiation,
CDM, baryon, and dark-energy bookkeeping over the full declared domains,
including `N_eff` below the integer massive-species count, without `max()`
interpolation or double counting. Validate `H(a=1)=H_0`, positivity, smooth
derivatives, visibility normalization, and recombination/drag independence.

Acceptance requires automatic background refinement below the declared bound
at physical anchors, several neutrino masses and `N_eff` values, continuous
zero-mass behavior, independent BAO evaluation with the CMB entry point
absent, and raw background/drag evidence for every bundled model that uses
those quantities.

### [planned] Slice Three — automatic hierarchy, collisions, and initial data

Make hierarchy depth and ODE resolution adaptive for every declared scalar,
vector, and tensor family, including q-resolved massive neutrinos. Derive
tight-coupling entry and exit from the declared collision rate and phase
scales. Compile every collision matrix, damping term, counterpart, closure,
and conservation rule from the declaration; remove heuristic zero or fixed
coefficient substitutions.

Compile and validate explicit `Phi_tau`, `Psi_tau`, history derivatives,
metric closures, visibility sources, polarization hierarchy, ISW terms, and
all initial/boundary conditions. The engine must evolve a hidden early prefix
when the requested line-of-sight grid starts later, while preserving the same
physical history for late-start requests.

Acceptance requires scalar-vs-batch equivalence, finite residual-clean
histories, adaptive hierarchy and q refinements, collision conservation
evidence, stable low-ell results when request ranges change, and distinct
histories for distinct declared modes and gauges. Missing derivatives or
collision terms must fail explicitly with their source name.

### [planned] Slice Four — universal source graph and all-sector projection

Implement the generic source compiler and line-of-sight projection for every
declared source and kernel. Derive k sampling from radial and acoustic phase,
visibility features, and the requested ell ceiling; derive eta sampling from
source curvature and kernel support. Use positive, order-aware quadrature that
does not assume uniform log-k nodes, and perform independent k, eta, source,
and projection refinements until convergence.

Complete scalar, vector, and tensor temperature/E/B kernels, cross-sector
routing, primordial normalization, units, and sign conventions. Keep all
projection products and refinement arrays in the diagnostic result. A valid
request may not be dropped because its grid is difficult; the planner must
refine it or report the raw numerical obstruction.

Acceptance requires smooth low- and intermediate-ell acoustic structure for
LCDM and the Planck reference, alternating-sign TE, structured EE, finite
tensor/vector surfaces, and convergence of raw TT/TE/EE/BB/PP/TP/EP arrays at
the planner-selected production range. The tests must inspect arrays before
plotting.

### [planned] Slice Five — complete observable post-processing

Implement and validate all declared unlensed and lensed surfaces, including
the lensing potential, remapping, BB generation, TP/EP cross spectra, and
scalar/vector/tensor totals. Enforce declared physical zeros only from proved
symmetries. Remove any path that returns only TT/TE/EE or silently substitutes
an absent BB/PP component.

Acceptance requires complete surface sets for every model that declares them,
finite covariance-compatible units, positive auto spectra, correct cross
signs, stable lensing response, and scalar-vs-batch/cache identity for every
surface. Raw transfer components and post-processing intermediates must be
stored in the canonical evidence artifact.

### [planned] Slice Six — universal model corpus and grammar extension

Run the same engine against all ten bundled theories after their numerical
blocks are removed. Repair shared compiler/runtime paths for QAU, QRSF, TOG,
TORG, USMF2, wCDM, w0wa, massive-neutrino LCDM, and the Planck reference.
Add adversarial declarations with unrelated names, altered recombination and
opacity equations, extra sectors, non-standard interactions, and every
declared observable. If a valid declaration exposes a grammar or execution
gap, extend the generic schema/compiler/runtime here; do not add a capability
fixture or classify the theory as unavailable.

Acceptance requires every complete bundled and adversarial declaration to
execute finite converged requested surfaces, with no LambdaCDM-name dependence,
no engine-capability status left unresolved, and raw model manifests showing
the same universal route. Malformed mathematics must still fail with a named
model-independent validation error.

### [planned] Slice Seven — CAMB parity and production graph recovery

Build an independent CAMB comparison harness using matched physical
conventions, not a CAMB runtime fallback. Compare multiple fixed points for
LCDM, massive-neutrino LCDM, the Planck reference, wCDM, and w0wa over the
complete applicable surface and production ell range. Record raw CCMBS/CAMB
arrays, all error metrics, peak positions, phases, damping tails, signs, and
artifact hashes.

Run the real GUI/CLI graph path with normal production requests. LCDM and the
Planck reference must both produce sensible CAMB-like TT/TE/EE/BB/PP/cross
graphs, with no missing curve caused by a convergence exception. The graph
path must display an explicit typed failure instead of silently omitting a
theory when any genuinely invalid request fails.

Acceptance is actual parity evidence, not synthetic arrays: complete raw
reports and graph artifacts pass the declared numerical and physical-shape
bounds at several fixed points and mass values. The production Planck
likelihood receives finite, non-catastrophic spectra and a sane CMB chi-square.

### [planned] Slice Eight — end-to-end solver closure

Exercise the normal runner, model adapter, likelihood assembly, sampler, CSV
and plot exporters, GUI/CLI, cache reuse, and failure reporting against all
bundled models and representative novel declarations. Confirm that BAO uses
only the independent drag background boundary and that CMB failures never
corrupt SNe/BAO results.

Run repeated fixed-seed short posterior regressions only after the production
graphs and raw parity reports pass. Verify that the posterior does not repair
bad CMB spectra by driving `H_0`, matter, baryon, or `N_eff` to absurd values.
Publish the final corpus matrix, complete observable artifacts, parity rows,
planner evidence, graph hashes, and a reproducible closure manifest.

Acceptance requires the full declared test suite, all slice acceptance tests,
green DevCovenant verification, finite and physically sensible graphs for all
ten bundled models, complete declared observables, CAMB parity wherever
applicable, and explicit mathematical diagnostics for every intentionally
invalid declaration.

## Completion Standard

This plan is complete only when all of the following are true:

* no bundled model contains solver numerical controls or hidden numerical
  overrides;
* CCMBS derives and records its own resolution for every request and never
  silently accepts an under-resolved result;
* all ten bundled models and representative mathematically complete novel
  declarations execute through one universal theory-to-engine route;
* all declared scalar, vector, tensor, unlensed, lensed, auto, and cross
  surfaces—including TT, TE, EE, BB, PP, TP, and EP—are computed;
* LCDM and the Planck reference produce sensible CAMB-like graphs through the
  normal GUI/CLI production path, with no missing curves;
* CAMB parity passes for every physically comparable model and surface at
  multiple fixed points and the production ell range;
* non-CAMB theories pass finite, converged, theory-faithful internal checks;
* background, recombination, drag, BAO, likelihood, sampler, cache, export,
  and failure boundaries are independently evidenced;
* raw arrays, histories, grids, residuals, planner decisions, parity reports,
  graph files, and hashes are reproducible and attached to the closure
  manifest;
* the complete tests and DevCovenant gate are green without suppressions,
  skipped physics, relaxed acceptance, or CAMB fallback; and
* `PLAN.md` has no outstanding requirement outside an implementation slice.

Until every item above is demonstrated, CCMBS is not scientifically closed.
