# Development Plan
**Doc ID:** PLAN
**Doc Type:** plan
**Project Version:** 12.0.26
**Project Stage:** stable
**Maintenance Stance:** active
**Compatibility Policy:** forward-only
**Versioning Mode:** versioned
**Last Updated:** 2026-08-18
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->
This opening section is managed by DevCovenant.
Use `PLAN.md` to track active implementation work below this block.
<!-- DEVCOV:END -->

> **For agentic workers:** Execute the slices in order and use the
> repository gate workflow. Each slice has independent implementation,
> correctness, scientific, and performance acceptance evidence.

**Goal:** Establish a clean exact solver and sampler architecture with a
numerically converged CCMBS reference path.

**Architecture:** Copernican exposes independent sampler and CMB-solver
contracts. The current MCMC and nested implementations become sampler
backends, and the current declared-graph numerical path becomes CCMBS
(Copernican Cosmic Microwave Background Solver). Both samplers consume the
same injected CMB solver and likelihood contracts; a later plan may add
CCMBS-Taichi and a Taichi sampler without changing those public boundaries.

**Tech Stack:** Python 3.11, NumPy, SciPy, emcee, multiprocessing, Tk,
declared-graph CMB execution, focused unittest suites, and DevCovenant. This
plan does not add Taichi or require a GPU; it makes the interfaces and
provenance Taichi-ready.

## Global Constraints

* Do not change branches, create branches, or alter repository workflow.
* Remove surrogate and delayed-acceptance production behavior completely.
* Keep exact scalar evaluation as the scientific reference and default.
* Keep the current NumPy/SciPy declared-graph implementation numerically
  unchanged except where dependency injection, naming, or an explicitly
  recorded production convergence floor requires movement.
* Do not introduce a production CAMB or CLASS fallback.
* Both MCMC and nested sampling must evaluate CMB likelihoods through the
  selected solver contract.
* CCMBS is the default solver identity for the current reference backend.
* Solver selection and sampler selection are independent manifest choices.
* Batch evaluation must preserve input ordering, per-item diagnostics, typed
  failures, and parameter-dependent cache isolation.
* Do not broaden parameter-dependent cache keys across unequal parameter
  points or lower numerical accuracy to manufacture performance evidence.
* Do not add Taichi as a dependency in this plan. A later plan owns the
  Taichi implementation, device matrix, and GPU acceptance evidence.
* Treat Vulkan as the first AMD-capable Taichi target; never assume CUDA is
  available on an AMD device.
* Internal identifiers, module stems, manifest keys, and repository paths must
  not use the legacy `engine` or `cosmo` prefixes after migration.
* Scientific prose may use “cosmology” and “cosmological”.
  Filenames may retain source names only with explicit provenance.
* Forward-only compatibility applies: do not leave hidden aliases, legacy
  readers, or silent translation paths behind after migration.
* Full `copernican.validation` workloads are not acceptance dependencies.
  Use bounded focused tests and fixed scientific reference fixtures.
* Preserve root/package documentation synchronization and update changelog,
  comments, docstrings, tests, and generated mirrors in the same slice.
* Stage all changes after each completed slice. Do not commit or push unless
  explicitly instructed.

## Table of Contents

* [Overview](#overview)
* [Current State and Decisions](#current-state-and-decisions)
* [Target Architecture](#target-architecture)
* [Naming and Migration Contract](#naming-and-migration-contract)
* [Scientific and Performance Contract](#scientific-and-performance-contract)
* [Execution Rules](#execution-rules)
* [Execution Slices](#execution-slices)
* [Completion Standard](#completion-standard)

## Overview

This plan deliberately has two slices, the minimum useful decomposition for a
cross-cutting migration of terminology and runtime boundaries:

* Slice One removes the discarded approximation path and migrates all sampler
  terminology, public configuration, manifests, discovery, tests, and docs.
* Slice Two introduces the selectable CCMBS solver contract and routes both
  current samplers through it while preserving the exact reference behavior.

The later Taichi plan starts only after these two slices are closed. It will
add a Taichi sampler and CCMBS-Taichi implementation behind the contracts
defined here. It must not reopen the naming migration or redesign manifest
selection while implementing GPU kernels.

## Current State and Decisions

The current runtime has two sampling modules under
`copernican/samplers/`: an ensemble MCMC implementation and a nested-sampling
implementation. Both construct joint SNe, BAO, and CMB likelihoods with
canonical sampler names and metadata. The shared model adapter owns model
contracts independently of either sampler.

The current CMB path is a declared-graph NumPy/SciPy implementation reached
through `copernican/lib/likelihoods/cmb/cmb.py` and the CCMBS orchestrator in
`copernican/lib/likelihoods/cmb/orchestrators/ccmbs.py`. Its identity is now
the stable CCMBS solver identity in manifests and GUI text. Solver selection
remains the explicit registry work assigned to Slice Two.

The CMB package keeps its public facade and value types at the package root.
Selectable backends and their registry live under `solvers/`; CCMBS execution
coordination lives under `orchestrators/`; and numerical helpers, cache
ownership, convergence, performance, and lensing live under `runtime/`. The
old ambiguous module stems and the former backend-prefixed vocabulary are
removed from repository-owned CMB code, tests, documentation, manifests, and
diagnostics.

The surrogate and delayed-acceptance path is not part of the target
architecture. It approximates the full joint posterior, changes the sampler
algorithm, and introduces a separate scientific validation burden. Remove its
configuration, code, provenance, tests, documentation, and plan references;
do not replace it with a no-op alias.

The exact scalar sampler, exact CMB spectra, likelihood ordering, declared
failure taxonomy, cache identities, and declared numerical envelope remain
the comparison authority. A short run can prove workflow plumbing, but not
posterior convergence or scientific validity.

The bundled production scalar graphs use at least 64 transfer-wave-number
nodes. The former 18-node setting is retained only by bounded low-resolution
fixtures that explicitly exercise contract behavior; it is not an acceptable
production reference because it changes the spectrum materially as the
requested multipole range is refined.

This floor closes transfer-grid instability, not absolute scalar calibration.
Fixed-point reference anchors must still establish source normalization,
temperature-polarization sign, and cross-spectrum parity before CCMBS is
declared scientifically equivalent to the independent reference fixture.

## Target Architecture

### Sampler package

The canonical package is `copernican/samplers/`. The current modules migrate
as follows:

* The MCMC and nested modules live at `sampler_mcmc.py` and
  `sampler_nested.py`.
* `SAMPLER_KIND`, `SAMPLER_LABEL`, `SAMPLER_VERSION`, `SAMPLER_SETTINGS`, and
  `SAMPLER_PROGRESS_CHUNKS` are the only sampler metadata constants.
* `sample_parameters` is the canonical callable in both sampler modules.
* `fit_sne_parameters`, `resolve_fit_function`, and related compatibility
  names are removed rather than retained as legacy aliases.

The sampler resolver accepts one canonical callable and one capability
descriptor. A sampler receives model plugins, datasets, a selected CMB
solver, sampling settings, a seed, and progress callbacks. It returns the
existing result shape with sampler-neutral parameter and diagnostic names.
The two samplers may differ internally, but neither may import a concrete
CCMBS implementation directly.

### Model adapter package

`copernican/lib/model_adapter.py` is renamed to a model-oriented adapter,
with `ModelPlugin` and `model_plugin_validation` renamed accordingly. This
adapter owns model metadata, priors, distance functions, and the immutable
declared CMB contract; it is not a sampler and must not retain sampler
terminology.

`copernican/lib/sampler_capabilities.py` becomes sampler capabilities. Its
public types and resolver names use `SamplerSetting`,
`SamplerProgressChunk`, `SamplerCapabilities`, and
`get_sampler_capabilities`.

### CCMBS solver contract

Introduce a solver protocol and registry under the CMB likelihood package.
The protocol must expose:

* a stable solver identifier and human label;
* scalar spectrum evaluation from a prepared model contract;
* ordered batch evaluation with one typed result per input;
* capability metadata for supported spectra, grids, accuracy tiers, and
  execution backends;
* cache and phase-timing provenance;
* typed domain, convergence, non-finite, and performance failures;
* a preparation hook for immutable structural assets and a cleanup hook for
  device or worker resources.

The current implementation becomes the reference CCMBS backend. Use a stable
identity such as `ccmbs_numpy` internally and the user-facing label
`CCMBS — Copernican Cosmic Microwave Background Solver`. The public solver
name is CCMBS; “NumPy/SciPy reference backend” describes its
implementation, not a second scientific model.

`cmb.py`, likelihood classes, posterior construction, and both samplers must
receive a solver object or registry-resolved solver. The default resolver
returns CCMBS, so existing callers retain exact behavior when no solver is
declared. A manifest may select a registered solver explicitly, and the
resolved solver identity must be written to run and result provenance.

The later Taichi implementation will register beside the reference backend,
for example as `ccmbs_taichi`. It must consume the same prepared contract and
return the same public result and failure shapes. No Taichi code belongs in
this plan.

### Selection and execution flow

The manifest uses independent selections:

```yaml
selection:
  sampler:
    name: copernican.samplers.sampler_mcmc
  cmb_solver:
    id: ccmbs_numpy
```

The run configuration, GUI builder, CLI confirmation, executor, pipeline,
result writer, and run manifest all use `sampler` and `cmb_solver`. The
pipeline resolves both before constructing the control and test posterior,
then passes the same solver contract to MCMC or nested sampling. A solver is
not selected implicitly from the sampler, and a sampler is not selected from
the CMB model.

For a future GPU backend, the solver owns one device context per process and
offers batched exact evaluation. The MCMC ensemble wave is the natural batch
boundary. Nested sampling remains sequential in its evidence update, but its
candidate likelihood calls use the same scalar/batch contract when batching is
safe. The plan must not create one GPU context per multiprocessing worker.

## Naming and Migration Contract

### Legacy sampler vocabulary

Replace all repository-owned uses of the following names in code, tests,
manifests, generated output, docs, comments, and GUI text:

* `engines/`, legacy engine module stems, and engine-based discovery;
* `ModelPlugin`, `SamplerSetting`, `SamplerCapabilities`, and
  `get_sampler_capabilities`;
* `engine`, `engine_kind`, `engine_module`, and `ENGINE_*` configuration keys;
* “sampler engine” and “CMB engine” labels.

The canonical user-facing terms are “sampler”, “CMB solver”, and
“CCMBS”.
Tests must assert the new manifest and provenance keys rather than accepting
both old and new shapes.

### CMB solver vocabulary

Use the stable `CCMBS_ID` and `CCMBS_LABEL` identifiers for the current
reference solver. The solver registry, module rename, and selectable solver
entrypoint are Slice Two work; Slice One only removes the old engine identity
and labels.

### `cosmo` prefix cleanup

Inventory every `cosmo`-prefixed Python identifier, module stem, internal
manifest key, and repository-owned path. Rename parser modules to a neutral
dataset parser convention, rename parameter variables and result fields to
`model` or `parameter` terminology, and update discovery and all references.

Do not mechanically rename scientific prose containing “cosmology” or
“cosmological”. Do not silently rename immutable upstream data artifacts:
either preserve their source filenames in a documented allowlist or create a
repository-owned neutral alias while retaining the original filename and
hash in provenance. The Slice One allowlist is limited to the upstream Union3
Stan sources `copernican/datasets/sne/union3/stan_code_fixed.txt` and
`stan_code_simple.txt`, whose `cosmo_model` variable is part of the imported
source text. The final stale-name scan must report every exception.

### Approximation removal

Remove `copernican/engines/surrogate.py`, the delayed-acceptance branches and
arguments in sampler and pipeline APIs, surrogate configuration validation,
manifest/result provenance fields, GUI controls, and all surrogate tests and
docs. Remove the corresponding plan slice and acceptance claims. A manifest
that still requests delayed acceptance must fail as an unsupported setting;
the runtime must not silently ignore it or reinterpret it as exact sampling.

## Scientific and Performance Contract

The exact CCMBS scalar result is authoritative for every comparison. Fixed
point spectra must preserve requested names, multipole ordering, sectors,
lensed/unlensed distinctions, diagnostics, and typed failures. Likelihoods,
priors, constraints, cache identities, and result serialization must remain
unchanged after sampler and solver injection.

Every bounded benchmark records model and dataset identity, seed, sampler,
solver, requested spectra, numerical grids, accuracy tier, cache state,
process/thread settings, phase timings, scalar/batch item counts, and
GUI/headless mode. A faster elapsed number without this context is not
accepted. Full `copernican.validation` workloads are not an acceptance
dependency.

The later Taichi plan must independently demonstrate Vulkan/AMD device
availability, kernel correctness, CPU/reference parity, precision, and
throughput. This plan only defines the array, preparation, capability, and
diagnostic seams needed for that implementation.

## Execution Rules

1. Open a DevCovenant gate before edits and clear all gate complaints.
2. Work from the active `.venv` for policy commands and focused tests.
3. Complete the slices in order; do not start Slice Two if Slice One's
   naming and approximation-removal acceptance is incomplete.
4. Update implementation, tests, docs, comments, docstrings, manifests,
   mirrors, and changelog together whenever a contract changes.
5. Use bounded focused tests and exact scientific fixtures; do not run the
   full validation workload as a substitute for acceptance.
6. Stage all changes at the end of each slice.
7. Run `source .venv/bin/activate && python -m devcovenant gate --verify`
   on the staged revision before reporting the slice complete.
8. Do not run `devcovenant run`, commit, or push unless explicitly requested
   for that turn.

Task markers mean:

* `[planned]` identifies work to be executed in a future gate.
* `[closed]` identifies work completed in substance and acceptance evidence.

## Execution Slices

### [closed] Slice One — Sampler vocabulary and exact-path cleanup

**Purpose:** Remove the discarded surrogate path and migrate all public and
internal sampler terminology without changing exact numerical behavior.

**Files and surfaces:**

* rename `copernican/engines/` and its MCMC/nested modules;
* rename sampler capability, model-adapter, configuration, manifest,
  executor, pipeline, workflow, GUI, CLI, and result-writer symbols;
* migrate sampler tests and discovery fixtures;
* remove `surrogate.py`, delayed-acceptance branches, settings, aliases,
  provenance, tests, docs, and generated references;
* inventory and rename internal `cosmo` prefixes with documented source-file
  exceptions;
* update `README.md`, `copernican/README.md`, package/repository docs,
  `SPEC.md`, manifests, mirrors, and `CHANGELOG.md`.

**Implementation tasks:**

1. Build a complete old-to-new symbol, path, manifest-key, and documentation
   inventory before moving files; record immutable source-name exceptions.
2. Move sampler modules and tests, replace discovery and imports, and expose
   only `SAMPLER_*` metadata plus `sample_parameters`.
3. Rename model adapter and sampler capability types so no sampler contract
   depends on an `Engine*` class or `engine_*` module.
4. Remove delayed-acceptance and surrogate arguments from every public call
   path, reject stale requests explicitly, and delete the approximation
   implementation and its tests.
5. Rename run configuration and manifest selection fields from `engine` to
   `sampler`, update GUI/CLI labels, and rewrite result provenance.
6. Rename internal `cosmo` prefixes, update parser discovery and dataset
   references, and preserve source-artifact hashes for allowlisted names.
7. Run focused sampler, manifest, GUI, dataset discovery, and exact seeded
   chain tests; run a stale-name scan with only the documented exceptions.

**Acceptance:**

* No production import, setting, manifest field, GUI control, or provenance
  record supports surrogate or delayed acceptance.
* MCMC and nested modules are discoverable as samplers and expose the same
  canonical callable shape.
* Existing exact scalar fixtures produce the same samples, likelihoods,
  spectra, failures, and serialization values under the renamed API.
* No legacy engine alias or silent `cosmo` compatibility path remains.
* The stale-name scan is empty except for explicitly documented upstream
  source filenames and ordinary scientific prose.

### [closed] Slice Two — CCMBS registry and solver-injected samplers

**Purpose:** Make the current exact CMB implementation a selectable CCMBS
solver and route both samplers through a Taichi-ready solver contract.

**Files and surfaces:**

* create the CMB solver protocol, capability descriptor, registry, and
  selection validation;
* rename and adapt the current declared-graph implementation as the CCMBS
  NumPy/SciPy reference backend;
* update `cmb.py`, CMB likelihoods, posterior construction, sampler modules,
  run configuration, manifest, executor, pipeline, GUI, and result writer;
* add scalar/batch solver contract tests, solver-selection tests, and paired
  MCMC/nested CMB fixtures;
* update solver documentation, package/repository README mirrors, and the
  changelog.

**Required interfaces:**

```python
class CMBSolverProtocol(Protocol):
    solver_id: str
    solver_label: str

    def capabilities(self) -> Mapping[str, object]: ...
    def prepare(self, contract: Mapping[str, object]) -> object: ...
    def evaluate(
        self, prepared: object, ells: Sequence[int], *,
        spectra: Sequence[str], workload: str,
    ) -> CMBResult: ...
    def evaluate_batch(
        self, prepared: Sequence[object], ells: Sequence[int], *,
        spectra: Sequence[str], workload: str,
    ) -> tuple[CMBResult, ...]: ...
```

The exact result type carries spectra, requested ordering, diagnostics, cache
provenance, phase timings, and a typed failure when evaluation is unsuccessful.
The reference solver may implement `evaluate_batch` by ordered scalar
adaptation, but it must satisfy the same isolation contract a future Taichi
backend will use.

**Implementation tasks:**

1. Define the protocol, result type, capability schema, registry, and default
   CCMBS resolver without changing numerical kernels.
2. Move the current declared-graph executor behind the CCMBS reference
   adapter and preserve all existing cache identities and error classifiers.
3. Inject the resolved solver into CMB likelihood and posterior factories;
   remove direct concrete-solver imports from both sampler modules.
4. Add independent `sampler` and `cmb_solver` manifest configuration, resolve
   both before control/test sampling, and persist their identities and
   capabilities in run provenance.
5. Route MCMC and nested sampling through the same solver-aware likelihood
   construction and verify both with CMB-enabled bounded fixtures.
6. Verify scalar/batch ordering, fixed-point numerical parity, typed failures,
   cache isolation, and result serialization for the CCMBS reference.
7. Add a backend capability/probe seam that can later report Taichi Vulkan or
   AMDGPU devices without importing Taichi in this plan.

8. Preserve the production CCMBS transfer-grid floor: bundled scalar model
   contracts declare `minimum_k_sample_count: 64` and set their numerical
   `k_sample_count` to at least that value. Add a focused regression test so
   future solver injection cannot reintroduce the unstable 18-node path.

**Acceptance:**

* CCMBS is the documented default solver and the current exact implementation
  remains the numerical reference.
* MCMC and nested samplers both run CMB likelihoods through the selected
  solver, with no hard-coded solver identity in either sampler.
* A manifest can select CCMBS explicitly, and an unknown solver fails before
  expensive sampling begins.
* Ordered scalar and batch results, diagnostics, typed failures, caches, and
  spectra remain equivalent to the pre-migration reference.
* Solver capabilities and provenance are complete enough for a later Taichi
  Vulkan implementation to register without changing sampler contracts.

**Closure evidence:** The CCMBS protocol, ordered result contract, default
NumPy/SciPy registry adapter, independent manifest selection, sampler
injection, result and chain provenance, and focused scalar/batch/selection
tests are implemented. A staged `gate --verify` is the closure checkpoint;
the required full DevCovenant run remains operator-managed.
## Completion Standard

Slice One is closed when its staged revision has a green `gate --verify`.
This plan is complete only when both slices are closed in order and the
staged revision has a green `gate --verify`.

Completion requires:

* no surrogate or delayed-acceptance production path;
* sampler terminology and paths used consistently across code, manifests,
  GUI, CLI, tests, docs, and generated output;
* no unapproved internal `engine` or `cosmo` prefixes;
* exact scalar CCMBS reference behavior preserved;
* bundled production scalar CCMBS contracts use the declared 64-node transfer
  floor and reject under-resolved production settings;
* independent sampler and CMB-solver selection in manifests;
* both MCMC and nested sampling using the selected solver contract;
* ordered batch, cache, failure, diagnostics, and provenance contracts
  preserved;
* Taichi-ready array, capability, preparation, and device seams without a
  Taichi implementation or dependency;
* root/package documentation, comments, tests, mirrors, and changelog aligned.

A green policy gate or a finite spectrum is not proof of scientific validity.
The next Taichi plan must independently demonstrate Vulkan/AMD device
availability, kernel correctness, CPU/reference parity, numerical precision,
and measured throughput before enabling a GPU backend as a production option.
