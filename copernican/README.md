# copernican
**Doc ID:** README
**Doc Type:** repo-readme
**Project Version:** 12.0.26
**Last Updated:** 2026-11-09
**DevCovenant Version:** 1.0.1b6

<!-- DEVCOV:BEGIN -->

<!-- DEVCOV:END -->

**Version:** 12.0.26

![Copernican banner](https://raw.githubusercontent.com/apostolovbg/copernican/main/copernican/docs/banner_github.png)

## Overview
Copernican is a Python toolkit for evaluating cosmological models against
SNe Ia, BAO, and CMB observations. It gives researchers one manifest-driven
workflow for selecting data, choosing a model, running the sampler, and
keeping the results tied to the exact inputs that produced them.

The same manifest can drive the command-line interface or the GUI. That keeps
interactive runs and scripted runs on one configuration surface, with the same
seed handling, control-model and test-model selection, dataset selection,
sampler choice, CMB solver choice, and output layout.

Copernican is built for reproducibility. Every run writes a manifest, one
canonical run log, summary artifacts, plots, and chain outputs into a per-run
directory under `~/copernican_output/`, so a result can be replayed or audited
later without guessing which options were used.

The package includes the model library, trusted dataset parsers, sampler
samplers, validation manifests, and supporting analysis tools needed for the
full workflow. Every bundled CMB model declares the declared graph contract;
models with available CMB output use the same declared solver, while a model
without a defensible perturbation closure reports CMB output as unavailable.
USMF2 now executes through that same declared route. Its theory-specific
shrink-field closure, sourced initial conditions, metric constraints,
projection sources, and public observables are evaluated directly from the
declared graph without a CAMB, CLASS, or LCDM substitution. Its production
route records a fail-closed doubled-k comparison and audits the shrinking
background's radiation and baryon-only matter closure; unsupported vector,
tensor, and lensed surfaces remain typed non-applicable declarations.

The CMB subsystem's physical state convention, hierarchy equations, collision
operators, gauge routes, line-of-sight sources, spectrum units, lensing
inputs, engine-owned numerical planning, and independent-reference
boundaries are
documented in
[`copernican/docs/cmb_solver.md`](copernican/docs/cmb_solver.md).

CCMBS compiles one model-independent source graph before numerical evolution.
The graph records every declared source, transfer component, radial kernel,
sector route, and spectrum edge in a digest-addressed runtime envelope.  The
planner selects phase-aware k and eta resolution from the requested surface;
source, transfer, and line-of-sight refinements retain their raw arrays and
reuse evidence.  Scalar, vector, and tensor routes therefore share one
validated projection contract without model-name dispatch or fabricated
zero-valued sources.

Explicit stage diagnostics retain native state and source eta grids plus raw
transfer components. They compare bounded waves with an independent
test-owned reference and bypass spectrum, transfer, source-history, and
initial-state result caches, so diagnostic evidence cannot alter production
results.

Normal public requests, including explicit ell arrays, use the engine's final
production accuracy envelope and request-shaped grids. Reduced grids are
available only through an explicit diagnostic boundary. YAML model validation
rejects solver-owned numerical and accuracy blocks; compatibility fixtures are
translated after clean source validation and never become production metadata.

The request-shaped production path is coordinated by the CMB runtime planner,
phase-aware adaptive grid builder, and projection convergence layer in
`copernican.lib.likelihoods.cmb.runtime`. They retain distinct base and
refinement grids, fill duplicate phase anchors deterministically, and preserve
the final grid and convergence evidence in the runtime envelope. The
diagnostics API replans prepared contracts from its requested ell range before
applying diagnostic overrides, so reduced diagnostic requests cannot inherit a
stale production-sized grid.

The adaptive runtime closes phase resolution against measured spacing, not
only against a theoretical node-count floor. Production k-grid convergence
retains distinct nested grids, measured new-mode work, cache state, and
shape-aware digests. Cold refinement evolves every new k mode; warm reuse is
accepted only for the matching finer calculation. An identical or non-nested
refinement cannot be reported as evidence. Bounded graph tests persist raw
arrays, refinement records, and a manifest, then reload and hash them from a
fresh process after plot cleanup.

Corpus admission and execution keep theory labels separate from physical
identity. Renaming a complete declaration preserves its route, compiled
runtime signature, background identity, hierarchy schedule, and projection
cache key while manifests still report the requested label. Request-local
diagnostic grids update the engine plan consistently, including prepared
runtimes, so a corpus matrix cannot silently run a stale numerical plan.

The compiled contract audit in `copernican.lib.cmb_contract` produces a
machine-testable model-by-capability matrix for `TT`, `TE`, `EE`, `BB`, `PP`,
`TP`, and `EP`. It derives species, sectors, hierarchy families, projection
roles, and early unsupported-combination diagnostics from declarations rather
than theory names or model filenames.

Declared spectrum requests run that capability preflight before background
construction, so unsupported observables fail without beginning evolution.

Before declared scalar evolution begins, Copernican audits the requested k grid
against the model's declared numerical limits and preflights every mode on the
coupled Einstein constraint surface. The runtime records the ordered mode set,
residual terms, normalization scales, and constraint provenance with the
spectrum data so scientific failures can be traced without silently skipping
high-k modes.
Generated scalar contracts keep their declared regular superhorizon metric
seed during this preflight. The nearly singular early-time energy constraint
is validated rather than solved for a replacement potential, preventing
low-k round-off from collapsing the initial metric and corrupting every
projected spectrum.

Final generated scalar spectra resolve the measured radial/acoustic Fourier
phase and validate a distinct nested finer ladder. An arbitrary fixed mode
floor does not substitute for that comparison. The hierarchy uses
quarter-cycle Runge-Kutta stages through recombination, then the declared
phase target for the late integrated Sachs-Wolfe tail. Production
line-of-sight grids resolve the actual visibility profile and Fourier phase;
the search allowance is not a required output size. Their independently
finer grids retain every coarse node. Runtime evidence records the actual
node counts and spacing.

Transfer phase refinement is nested. The engine evolves a bounded base k
ladder once, preserves those modes, and evolves the declared hierarchy at
every new phase-quadrature node before projection. Background products and
matching cached source histories remain reusable. Runtime evidence records
the physical phase status, base and refined node counts, cold and warm mode
counts, refinement timing, measured work, and reuse flags.

Evolution acceptance compares independently evolved state and source
histories over every shared node and interval midpoint, with additional
anchors taken from the actual visibility and interaction histories. Source
representation is then measured separately by adaptive interpolation on a
materially distinct nested eta grid while evolution stays fixed. Per-product
global errors drive acceptance, while local near-zero errors remain retained
diagnostics. Localized spikes, sign changes, and near-zero crossings therefore
remain visible outside the named anchors. Projection quadrature uses nested
phase-aware eta grids, positive trapezoid weights, and the effective projected
k endpoint. It compares the resulting spectra per product and retains both
grid identities and measured differences. The runtime separately reports
evolution-history, source-representation, k-refinement, and line-of-sight
errors.

Evolved scalar Einstein residuals use the sum of absolute declared equation
terms as their dimensionless normalization. Copernican records the maximum
location and physical regime, tolerance and normalization provenance, and
coarse-to-intermediate-to-reference convergence evidence; a grid below its
declared reference eta resolution is reported as under-resolved rather than
accepted as a physical result.
The generated shear residual evaluates its declared correction directly so
nearly equal metric potentials do not create a floating-point-only breach.

All bundled CMB-capable models execute through the Copernican declared
declared-graph CMB solver. The manifest selects the solver independently from
the sampler; `ccmbs` is the single public CCMBS solver identity, while its
selected execution backend and capabilities are persisted in provenance. CAMB
and CLASS are
independent scientific reference tools used by tests, not production spectrum
solvers.

CCMBS owns hardware selection internally. Its portable CPU path is the
correctness baseline, while an optional accelerator may run only validated
fixed-shape kernels and must match the CPU result before use. Ordinary
installation and tests do not require Taichi or any other accelerator runtime;
unavailable hardware falls back inside the same CCMBS engine rather than
creating a second solver identity. The public execution orchestrator in
`runtime/execution.py` composes focused owners for hierarchy evolution,
collisions, line-of-sight grids, spectrum projection, numerical controls,
diagnostics, and evidence identities. Those owners retain the same request and
cache semantics while keeping solver physics separate from the measurements
and persisted evidence that observe it.

A default package installation has no CAMB or CLASS dependency. The repository
workspace lock includes CAMB only for independent scientific-reference tests;
the packaged runtime lock and installed license inventory contain declared
production dependencies only.

Requested spectra resolve only declared model dependencies. A model without
a defensible perturbation closure reports the affected CMB outputs as
unavailable rather than substituting another solver or a zero spectrum. See
[`copernican/docs/cmb_solver.md`](copernican/docs/cmb_solver.md) for the graph,
physical, numerical, lensing, caching, and reference conventions.

CMB likelihoods and result tools preserve exact spectrum names and row order,
including repeated or noncontiguous multipoles. Scalar, vector, tensor,
total, lensed, unlensed, lensing-potential, and diagnostic surfaces retain
separate metadata in plots, diagnostics, and long-form CSV output. Runtime
payloads distinguish computed, unrequested, physically zero, and unavailable
spectra, while cache identities bind the graph, parameters, numerical grids,
accuracy controls, requested spectra, and multipole sequence.
Generated vector hierarchies evolve Thomson drag and polarization with exact
collision blocks at every epoch. Higher photon moments receive exponential
damping in those blocks, keeping stiff collisions out of explicit streaming
steps and preventing spurious transfer growth.
The shared CMB χ² helper accepts an explicit solver selection for
deterministic likelihood tests; normal application calls retain the CCMBS
default.
The post-processing boundary records the complete dependency graph for every
requested surface. It hashes raw transfer components, unlensed products, and
public outputs, validates finite non-negative auto spectra, preserves cross
signs, and records proved physical-zero states. Exact lensed remapping uses a
bounded cache keyed by its scaled inputs, output convention, sampling factor,
and ell grid, so an exact repeat reuses the lensing intermediate rather than
rerunning the remapper.

CCMBS `runtime/mode_evolution.py` retains the declared integration choice
when collecting diagnostic evidence. The accuracy control
`continuous_collision_solver: true` explicitly selects coupled BDF collision
integration for supported declarations. Its error control normalizes each
state by its initial amplitude or propagated initial physical drive, so small
velocity and shear moments retain their gravitational effect. Compatible
small batches keep their computed histories rather than repeating every mode
through a different scalar solver. Focused early-time regressions compare
independently refined RK and BDF histories before late evolution or spectrum
projection.

Declared CMB validation records the resolved numerical envelope in each run
manifest. The named final tier requires bounded background, transfer,
source, hierarchy, momentum-grid, and lensing controls and rejects
under-resolved requests before expensive evolution. Momentum acceptance
compares doubled-count quadrature with independently extended low- and high-q
tails across the active thermal mass domain. Hierarchy acceptance uses a
free-streaming spherical-tail bound tied to its closure assumptions, terminal
multipole, and visibility-phase domain; configured floors alone do not earn a
validated status. Cross-sector refinement tests enforce the documented `TT`,
`TE`, `EE`, `PP`, lensed `BB`, q-grid, and hierarchy thresholds. Explicit
graphs without a sector registry retain their active sector identity from
compiled observable and tensor-character metadata, so runtime envelopes
cannot silently omit executed sectors.

The fixed-parameter CCMBS diagnostics API is independent of samplers and
plots. `discover_cmb_model_records()` enumerates `.yml` and `.yaml` model
declarations and retains typed ready, unavailable, rejected, or engine-error
rows; `discover_cmb_plugins()` returns only ready CMB models;
`run_cmb_model_diagnostic()` captures raw transfer components, raw and public
TT/TE/EE spectra, runtime-envelope metadata, and doubled-k-grid refinement
errors before plotting. `run_bundled_cmb_diagnostics()` applies the same
report shape to the full corpus, preserving explicit non-convergence instead
of hiding it. Unavailable means that the theory explicitly declares no CMB
output. Invalid or incomplete mathematics is rejected, while a valid
declaration that exposes missing compiler or runtime machinery is an
implementation failure and therefore CCMBS work, never unsupported theory.
LambdaCDM and novel declarations use the same admission rules. Deliberately
non-standard recombination is accepted when its declaration is complete; the
standard hydrogen quantities are an adapter, not a universal vocabulary.
Declarations with a different recombination ontology may instead provide
named `background.recombination.quantities` and map them through
`background.recombination.roles` for `electron_fraction`, `opacity`, and
`visibility` (with optional state, rate, matter-temperature, and drag roles).
CCMBS evaluates those role histories directly on its shared background grid;
it does not substitute a hydrogen or Peebles quantity.
`compare_cmb_spectra_to_reference()` adds backend-neutral
fractional auto-spectrum and normalized cross-spectrum checks for independent
fixed-point fixtures. Generated scalar contracts validate explicit metric
derivatives and compiler-backed source and closure expressions before runtime;
`Psi_tau` and `Phi_history_tau` use explicit runtime history-gradient
bindings, so missing derivatives cannot become implicit zero histories. BAO
has a fixed-background regression that deliberately breaks the CMB entrypoint
and verifies that BAO evaluation remains usable.
Fixed-point diagnostics retain the base raw spectra when their explicit
doubled-k comparison rejects a model; the nested production comparison is
deferred on that evidence path so convergence failures remain inspectable.
Full observable parity is available through
`compare_full_cmb_observable_parity()` and `build_cmb_parity_report()` in
`copernican.lib.likelihoods.cmb.diagnostics`. These helpers compare raw
`C_ell` or `D_ell` arrays for TT, TE, EE, BB, PP, TP, EP, lensed surfaces,
and explicit scalar/vector/tensor/total sectors. They retain finite and
physical-shape checks, cross-spectrum sign checks, C-to-D convention checks,
independent refinement evidence, fixture digests, and optional parameter
response points; missing or mismatched rows are rejected rather than
interpolated or silently substituted.
Fixed-point reports also carry compact raw source-history samples and
independently recomputed metric, visibility, polarization, and ISW residuals;
an unavailable or failed audit remains explicit rather than being treated as
parity evidence. Raw projection certification records those residuals for the
generated-hierarchy slice, while final certification remains the acceptance
boundary.
Generated histories also retain fine and coarse eta arrays in
`source_history_refinement`. The `source_history_bundle_digest` binds those
arrays, raw source terms, hierarchy and initial-state residuals, derivative
provenance, audit controls, and the independent closure decision to one
deterministic SHA-256 value. Explicit model graphs report a typed
`not_applicable` digest instead of fabricating generated-history evidence.
Production contracts may additionally declare a doubled scalar
`k_sample_count` convergence rule. CCMBS evaluates the declared TT, TE, and
EE surfaces at both resolutions, records per-spectrum errors in the runtime
envelope, and raises a typed convergence failure when the declared tolerance
is not met. `audit_bundled_cmb_contracts()` separately inventories every
bundled declaration for contract version, gauge, sectors, spectra, hierarchy
families, numerical bounds, and runtime-envelope consistency; this structural
audit does not substitute for hierarchy or CAMB scientific validation.
`audit_bundled_cmb_declarations()` adds the model-facing classification used
by the declaration slice: `ready` means the model-owned contract and source
graph are complete, `rejected` means an enabled declaration is structurally
incomplete, and `unavailable` is reserved for a model that explicitly disables
CMB output. The report also records whether execution uses the shared
generated scalar hierarchy or an explicit model-authored scalar graph, along
with any theory-specific source equations. QRSF and TORG retain their named
baryon-locked source closures; USMF2 is classified as an explicit scalar
graph, never as a generated-hierarchy or LCDM fallback.
The final certification helpers build a deterministic matrix from those raw
reports, reject missing residual or reference evidence, retain every raw
array, and write a hash-addressed JSON record. The independent test-owned
CAMB fixture freezes the LCDM parameter point, multipole ordering, `D_ell`
normalization, TT/TE/EE tolerances, and CAMB provenance. BAO isolation tests
deliberately make the CMB entrypoint unavailable while requiring identical
fixed-background values and covariance handling.
Stored CAMB fixtures retain exact content-digest validation. Rebuilding the
same fixture on another compiler or platform checks identical metadata and
numerical spectra within `rtol=1e-6` plus `1e-8` times each reference array's
peak magnitude. This rebuild envelope preserves exact zero spectra and leaves
the separate CCMBS scientific parity tolerances unchanged.
`build_final_cmb_certification_report()` extends that boundary with solver
and dataset identities, fixture hashes, per-model raw evidence digests, and
explicit integrity decisions. `assess_bao_cmb_isolation()` compares captured
BAO values, covariance metadata, and typed failures without invoking CCMBS.
Bounded parity reports also retain the ordinary-route solver identity and
selected backend diagnostics beside the raw comparison rows, so hardware
provenance cannot be confused with a separate physical solver.
`run_final_cmb_certification()` is the production hand-off: it executes the
full declared-observable matrix, rehydrates every raw report, applies the
repository and BAO-isolation gates, and optionally writes one deterministic
JSON artifact. It never invokes sampling or accepts a plot in place of raw
scientific evidence.
`read_final_cmb_certification_report()` reloads that artifact and rejects any
post-run mutation by verifying its recorded digest. A certified final report
also revalidates its source identity and retained completion artifacts. The
completion contract requires all 23 physical and auxiliary cases, complete
surfaces, independent reference windows, and resolved numerical axes;
synthetic wiring reports cannot certify CCMBS. `diagnostics.py` applies both
raw and public spectrum bounds without fitting amplitude or phase.

Before an explicit completion-matrix attempt, run
`python -m tests.project.lib.scientific_acceptance --completion-preflight`
in the activated repository environment. This validates the frozen physical
points and independent reference identities without starting CCMBS evolution.
Every attempt persists its original checkout provenance and any mismatch;
preflight readiness never means scientific acceptance. Current declaration
bindings remain separate from frozen requirements and historical full-case
hashes. The portable `ccmbs_comparable_camb.json.gz` fixture preserves all
twelve canonical reference arrays and the approved Planck density update;
no acceptance window or tolerance changes with that update.
The explicit `--completion-point CASE_ID` option retains named ordinary
attempts and completed numerical arrays, including failed or interrupted
attempts. Its quantitative parity flags remain separate from completion;
the full physical-domain, independent-axis and consumer evidence is still
required. `PLAN.md` records the unresolved low-ell high-k-tail defect.
The `evidence_assets` owner retains immutable compressed arrays with
atomic manifests, verifies every array on readback, and rejects missing,
tampered or escaped artifacts. Completion diagnostics require actual grid,
weight, source and transfer payloads with compatible dimensions. Separate
physical, numerical, reference and assessment identities distinguish reusable
forward products from rendering changes. The acceptance matrix remains
explicit and outside ordinary test discovery and `devcovenant run`.

Generated model plugins expose separate recombination and drag-epoch sound
horizons. BAO selects `get_sound_horizon_rs_drag_Mpc()` and records its
epoch, source, and fitted drag redshift; the CMB background continues to use
its recombination-era visibility sound horizon. Models without the optional
drag helper retain a typed legacy fallback for compatibility. When a drag
helper is present, BAO never evaluates the recombination helper first, and an
invalid drag value is reported as a BAO failure rather than silently replaced.
The independent epoch diagnostic also preserves whether a recombination or
drag integral failed, including the typed failure category. A declaration may
provide `bao_drag_redshift_expression` when its drag transition is not the
standard Eisenstein--Hu fit; CCMBS compiles that expression and uses it only
for the drag-era BAO ruler, leaving the recombination endpoint independent.
Missing helpers, invalid background shapes or values, non-finite predictions,
and invalid drag endpoints are retained as typed BAO metadata rather than
being collapsed into an unexplained likelihood value.
The corpus baseline records pre-repair behavior before shared physics
changes. `CMB_CORPUS_BASELINE_REQUEST` freezes the initial-guess parameters,
ordered TT/TE/EE multipoles, k/eta node counts, source-anchor rule, and
doubled-k refinement. `run_bundled_cmb_corpus_baseline()` evaluates each
frozen `BUNDLED_CMB_MODEL_FILENAMES` row once, attaches contract and
source-graph audits, and preserves raw source histories, transfer components,
public and raw spectra, residual vectors, numerical metadata, cache
identities, and typed decisions before a plot is made.

USMF2 follows named finite node-count tiers rather than a wall-clock limit.
An incomplete progression is recorded as `unclassified` with its remaining
work and never relabelled `unavailable`. `build_cmb_corpus_baseline_report()`
and `write_cmb_corpus_baseline_report()` serialize the ten exact rows in a
canonical order with per-row and report SHA-256 digests. Rejected evidence is
therefore preserved as a scientific baseline rather than hidden as a pass.

The later strict bundled-model matrix uses `CMB_CERTIFICATION_TIER` for final
scientific acceptance. `run_bundled_cmb_matrix()` evaluates each filename
through CCMBS, while `build_bundled_cmb_matrix_report()` and
`write_bundled_cmb_matrix_report()` require explicit contract identity,
source residuals, doubled-grid evidence, and scalar/batch cache-isolation
evidence before counting a model as accepted. The exact ordered batch audit
is exposed by `assess_scalar_batch_cache_evidence()`; an unmeasured check
remains an explicit non-passing decision. A complete matrix exposes
`decision_complete: true` even when a model is explicitly rejected, while
`success` remains false until every row is scientifically accepted. The
opt-in matrix execution can also evaluate two bounded parameter points through
both public scalar and ordered batch paths, recording per-point equality and
cache identities; a model with no independently variable parameter remains
explicitly unavailable for that comparison.
The full-observable matrix enforces the universal declaration boundary.
`declared_cmb_spectrum_names()`
reads every angular-power observable from a compiled model, so a matrix
request cannot silently collapse to TT/TE/EE. `run_bundled_cmb_full_matrix()`
executes those per-model requests, including BB, PP, TP, EP, and any lensed
names a future contract declares. `build_bundled_cmb_full_matrix_report()`
requires an exact declared/requested match and records one classification per
model: accepted, rejected with typed evidence, or unavailable only when the
declaration explicitly disables CMB output. Raw spectra, transfer histories,
source residuals, doubled-grid evidence, scalar/batch comparisons, and cache
identities remain nested in each row; missing or unclassified models are never
promoted to a passing corpus result. `write_bundled_cmb_full_matrix_report()`
serializes the same hashable matrix without dropping any raw fields.
`CAMB_COMPARABLE_CMB_MODEL_FILENAMES` names the five bundled declarations with
a defensible CAMB counterpart. A reference-required row is rejected unless an
independent fixed-point comparison is present and converged; fixtures are
never manufactured or borrowed from another theory. Unknown reference rows
and dropped declared observables fail validation.
Tier-controlled parity uses the same declared numerical overrides and workload
for both paths. The cached scalar wrapper retains its typed result, raw
unscaled spectra, request metadata, solver identity, and phase provenance so
the audit compares solver products rather than plot output.
The production parity and graph boundary uses the shared
`canonical_cmb_theory_spectra()` payload is consumed by likelihoods,
diagnostics, and graph rendering. Independent CAMB fixtures retain complete
raw `C_ell`/`D_ell` surfaces, fixed-point metadata, physical-shape evidence,
and artifact hashes; cache reports retain request-state and cross-request
reuse evidence. CMB graphs display typed execution or surface failures when a
theory cannot produce a valid curve.
The bounded external CAMB evidence boundary uses the
test-owned reference builder executes CAMB directly on one exact ell grid for
the five comparable bundled cosmologies and retains raw `C_ell`/`D_ell`
surfaces, contract and numerical-plan digests, and artifact hashes. The
fail-closed comparator rejects a changed raw surface. Canonical
CCMBS-versus-CAMB acceptance at the bounded scientific handoff remains
assigned to Slice Nineteen; CAMB-only rows are never reported as canonical
parity.
Slice Fourteen fixes the bounded parity request to the sparse ell grid
`(2, 20, 100, 200, 500, 800, 1200, 1500, 2000, 2500)` and records the
per-surface tolerance matrix. `build_cmb_parity_matrix_report()` requires an
exact model and declared-surface set, fixture digest, and independent
refinement decision for every row; missing, extra, or rejected evidence is
retained as a non-passing decision.
The explicit `run_cmb_parity_matrix()` command executes one ordinary public
CCMBS request per row, retains production convergence evidence, and records
typed execution failures. It is excluded from ordinary test discovery and
cannot report parity until the retained comparisons pass.
`write_cmb_parity_matrix_report()` persists the complete report, including raw
arrays and its integrity digest, for reload verification in another process.
Named fixed-point contracts and CAMB rows are supported for response points;
each point receives its own ordinary-route execution and parity decision.
`run_fixed_lcdm_cmb_parity()` defines the explicit scientific handoff:
fixed massless-neutrino LCDM with TT/TE/EE on the canonical sparse grid. It
rejects mismatched CCMBS/CAMB physical inputs, requires CAMB's resolved
defaults, complete raw surfaces, source/evolution convergence, exact cache
identity, and solver/backend provenance. The test-owned
`build_fixed_lcdm_camb_parity_row()` supplies the independent CAMB row.
`tests/project/lib/scientific_acceptance.py --output PATH` performs the
explicit handoff and `--verify PATH` reloads the durable report in a separate
process through `read_cmb_parity_matrix_report()`. The handoff returns a
quantitative accepted/rejected decision and remains outside ordinary test
discovery and full-resolution production work.
The ordinary production planner owns transfer, source, projection, and
evolution refinement. Independently evolved dense feature histories and
compact evidence for background, q, hierarchy, evolution, source, projection,
and physical limits are retained in the public performance record. Physical
limit evidence checks both k and eta endpoints, integration start, omitted
visibility tails, thermal background support, and lensing ell support. The
bounded production graph test uses that ordinary route, confirms wave-bearing
TT/TE/EE output, and proves exact-repeat reuse without a diagnostic bypass or
a second cold solve. A compact ordinary final-tier graph also runs both a
multi-spectrum request and its TT subset, requires every active numerical
axis to meet its measured budget, and verifies that the shared TT result is
unchanged.
The production planner separates native wave modes, saved source times and
angular quadrature. `spectrum_projection` interpolates independently evolved
source meshes onto a common angular mesh and evaluates the radial kernels
there. It measures source-k and angular-k errors separately. Source-time
interpolation is checked against sources reevaluated from native histories;
radial refinement holds those physical histories fixed. The full planned k
support is retained independently of the requested multipole subset. These
checks do not by themselves establish convergence of the physical k domain.
Joint MCMC
evaluation may defer the doubled-grid comparison, but it cannot bypass the
declared final phase-grid floor; a low-node smoke path is restricted to the
explicit diagnostic matrix.
The hierarchy runtime now derives a structural schedule from the compiled
families, collision operators, and initial conditions. Compatible modes share
one schedule; incompatible scalar, vector, tensor, and q-resolved modes remain
partitioned. Initial states and phase schedules use bounded, full-identity
caches, and runtime envelopes report schedule signatures, cache reuse, and
the hidden early-prefix decision alongside the raw spectra.
The `mode_evolution` stage cache keeps precomputed background and momentum
contexts in compact columns. Bounded working dictionaries can be evicted
without dropping future sampled values or changing RK stages. The adaptive
pilot selector excludes adjacent indices when splitting gaps, so irregular
wave grids cannot stall preparation.
The fixed Planck-reference declaration maps its 0.06 eV neutrino mass to
`omnuh2` and subtracts that contribution from `omch2`; the generated mixed
hierarchy weights the residual massless-neutrino source by its own density.
This keeps massive and massless radiation from being counted twice while
leaving pure-massless LCDM contracts on their original `Omega_nu0` path.
The LambdaCDM+Mnu and Planck reference backgrounds now apply the same split
to `H(a)`: massless radiation stays proportional to `a^-4`, while the
q-resolved Fermi-Dirac density and pressure moments supply the massive
component at every scale factor. The declared transition is a smooth
numerical baseline and is corrected to the q integral before recombination.
The effective massive-species allocation is bounded by `N_eff`, so points
below three massive species cannot overcount the early radiation.
The engine-owned background planner allocates its physical scale-factor
budget across the radiation scaffold, recombination visibility feature, and
late reionization interval, then performs an independent doubled-grid
refinement. If that measured comparison remains outside its bound, the
engine continues through a bounded refinement ladder instead of treating a
single theoretical node count as convergence. The accepted finer background
becomes the production background. Each attempt measures endpoints, opacity,
visibility normalization, electron fraction, sound speed, drag quantities,
and massive-neutrino density and pressure when active. The background owner
`runtime/background.py` integrates baryon drag as opacity divided by
`R = 3 rho_b / (4 rho_gamma)` through the complete remaining recombination
tail. The independently evolved late reionization component is excluded
from that ruler. The drag integral continues beyond the visibility peak;
the recombination and drag epochs retain distinct sound horizons.
Declared reionization rates evaluate `z = 1/a - 1` at their actual endpoint
and midpoint stages, so equivalent scale-factor and redshift laws agree.

The standard `model_lcdm.yml` and fixed `model_ref_planck2018.yml`
declarations explicitly use tanh reionization, with hydrogen width 0.5 in
redshift and a second helium transition at redshift 3.5 with width 0.4.
The declared midpoint is calibrated to optical depth 0.054. These are
reference-model assumptions, not a model-name rule in CCMBS. Alternative
models retain their declared ionization-rate laws. The background also
accepts a declared total `electron_fraction` expression; contradictory rate
and fraction inputs or fractions outside physical bounds are rejected.
The resolved calibration is retained in background evidence.

Both reference models fix the measured blackbody temperature at 2.7255 K.
Photon density follows the blackbody law, proportional to temperature to
the fourth power divided by the square of H0, in distance, sound-horizon
and CMB declarations. Photon density is no longer an independent fitted
parameter in these two models. Other theories retain their own explicit
radiation and reionization declarations.

The reference declarations select `geometry: cmb.background`. The compiler
in `model_coder.py` derives distances and both recombination/drag sound
horizons from the same native expansion and atomic histories as CMB.
A separately fitted recombination-redshift parameter and duplicate symbolic
H(z)/sound-horizon formulas are removed from these two models. Other models
can retain their symbolic geometry or explicitly select the shared route.

Their `effective_three_level` recombination prescription uses coupled
hydrogen, helium and gas-temperature evolution in `runtime/recombination.py`,
including helium escape/continuum/triplet channels and Compton scattering.
Custom atomic quantities and generic recombination roles remain separate
explicit prescriptions. No production background calls CAMB.

The Planck declaration specifies thermal neutrino heating with
`standard_neutrino_neff: 3.044`, three degenerate masses totaling 0.06 eV,
and the corresponding Fermi-Dirac density. Expansion and kinetic perturbations
share that temperature and density, leaving no additional massless 0.044
component. Automatically planned momentum grids generate q-resolved scalar
states with physical energy, momentum and shear normalization.

The scalar hierarchy in `perturbation_contract.py` evolves the curvature
potential with a conditioning-weighted combination of the Einstein energy
and momentum equations. This avoids both a singular velocity division near
the superhorizon limit and the energy equation's stiff k-squared/H damping
at large wave number. Both constraints remain independent checks; initial
radiation-density
projection closes both constraints while preserving the primordial metric.
`runtime/mode_evolution.py` retains the evolved metric and source histories.
Its scalar stage contexts reuse compiler-ordered slot names. The graph
compiler reuses each condition's common symbol domain, and `runtime/cache.py`
measures array payloads only for stores with a byte budget, retaining those
sizes through replacement and eviction. These changes preserve equations,
scalar conversion and numerical controls.
`runtime/history_store.py` retains raw mode sources across one request's
nested refinements even when the global LRU evicts them. Its memory budget
spills exact arrays into request-local files with checked digests; those
files are released on success or failure. It retains no convergence
certificates. Sampling node counts are separate from each mode's physical,
graph, grid and numerical identity. Direct runtime ownership tests are
included in ordinary unittest discovery.

The parameter-scoped `resolution_hint` cache stores starting numerical controls
for matching physical points and Fourier support. It excludes observation
sampling labels, carries no certificate and preserves every accuracy target.
Joint-budget refinement uses measured axis improvement to choose its next
trial. It avoids repeating an unproductive grid increase while another
material error can be reduced. The unchanged total-error test still decides
acceptance. Source/projection refinement changes LOS sampling independently
of the background seed. Compatible pilot/reference states survive these
sampling changes, keyed by their exact physical and numerical inputs. Sources
and acceptance checks are recomputed from the retained raw states.
Background refinement starts from its accepted internal seed through a
separate control, so an obsolete input cannot cause a no-op refinement or
silently increase the unrelated LOS sampling seed.
Native and LOS densities remain independent of the background node count.
Native wave cohorts share phase-resolved meshes, and compatible schedule
groups preserve every mode's required subdivisions while bounding total
extra work. Promoted coarse/intermediate/fine comparisons apply the same
resolution criteria at every level. Every independent native and observable
check still runs; no starting proposal is an acceptance certificate.
Fresh requests still perform their numerical checks; exact accepted results
retain their usual cache identity. Clearing result products preserves these
starting hints, while clearing parameter caches removes them.

Evolution retains its full independently planned native history. Large saved
batch trajectories use temporary file-backed arrays; live RK working arrays
determine mode chunking. Resident ndarray cells and mapped history cells are
reported separately; the operating system can also cache mapped pages.
Reordering an observation request preserves compatible physical histories.
Changed physical parameters, equations, distributions or effective controls
require fresh work. `runtime/equation_kernel.py` groups validated equations by
their arithmetic structure, preserving operand bindings and operation order.
Scalar execution remains an independent comparison. A single pending compatible
mode keeps that executor after cache reuse; new independent transfer nodes use
chunked batches with separately recorded reference work. Split collision
schedules resolve the spectral scale of the actual declared interaction
operator, including its coefficients, rather than assuming every interaction is
Thomson opacity. Stiff declared blocks without a fast manifold use error-
controlled local implicit integration of their full equations; interval
boundaries and conservation checks remain active. A fixed RK substep ceiling
cannot reject an otherwise resolvable block.
`runtime/stage_background.py` evaluates the actual declared background at
scheduled RK coordinates in bounded vector blocks. It preserves every stage
coordinate and physical interpolant, and compares all returned fields with
the scalar owner. Momentum stage bindings separate constant fields from
varying columns instead of rebuilding their array metadata for every stage.
Interior equation operands gather native state rows directly, including when
their terminal rows need declared closure values. Only those replacements
and actual graph dependencies enter the RHS context. Derived state rebinding,
numeric promotion, diagnostic contexts and independent scalar comparisons
retain their original semantics.
`runtime/performance.py` charges nested phases exclusively and merges outer
refinement timings into the request. Heartbeats distinguish phase, scope and
request elapsed time. Known RK schedules use mode-row substeps as weights;
projection uses ell/eta cells. Each estimate names its work scope, while
unresolved future adaptive work remains unknown. Scalar adaptive integration
reports RHS activity without inventing a known remaining-work total.
Grid messages label conformal-time node counts as `eta_samples` or
`source_eta_samples`; these counts are not wall-clock estimates.

Adaptive evolution keeps the hidden early-time prefix in addition to its
controlled late-time mesh. `runtime/adaptive.py` resolves measured opacity,
expansion and visibility features, actual Fourier phase and physical rate
transitions. Independent coarse grids cannot overwrite the shared fine grid.
Selected actual RK histories and their compiled source products are checked
before scheduling the remaining modes; successful pilot work is retained.
An unresolved engine-owned evolution mesh is refined without a fixed node
ceiling; measured stagnation remains a failure. Each state's own amplitude
and absolute tolerance govern its history error, while pointwise errors at
zero crossings remain diagnostic. `runtime/evolution.py` includes collision
phase after leaving the declared fast manifold: exact damping alone does not
make a split, driven collision solution accurate.
Native states use a shape-preserving cubic Hermite interpolant before source
evaluation. Its stable harmonic slopes retain tiny moments and avoid
artificial ISW corners caused by differentiating linearly resampled metric
histories. Independent evolution checks compare the actual compiled sources
as well as native states. Exact matching independent reference histories
also survive nested refinements; each comparison is evaluated again.
Velocity-isocurvature density seeds use the independent momentum source,
so they resolve before any provisional state exists. `contracts_audit.py`
uses the runtime's generated source-graph validator, including every energy
equation input. Public massive-neutrino velocity and shear use the physical
enthalpy density (rho plus pressure) at each epoch; normalized q moments
remain separate. BAO data comparisons use the baryon drag sound horizon.

Hierarchy families declare `physical_role` and kinetic `phase_space` metadata.
`phase_space.py` resolves the declared occupation law, logarithmic derivative,
temperature, mass and quadrature support independently of species labels.
Numerical support and density are measured separately, including nonthermal
features outside the initial thermal search interval.

Ordinary production checks independently finer background, hierarchy depth,
and applicable momentum quadrature on informative actual modes before the
remaining surface. Evolution and source comparisons remain independent.
`runtime/error_budget.py` propagates measured source uncertainties through the
absolute line-of-sight operators and signed-spline quadrature weights, then
sums all applicable axis bounds with projection and Fourier refinement errors.
Cross-spectrum checks include uncertainty in their normalization. Joint
transfer products include interactions between different source-error axes,
and the final budget retains the larger coarse/finer-grid axis bounds.
Background checks also propagate changes in the radial/geometry operator. These
bounds
condition on measured source discrepancies; their sampled physical domain and
between-wave maximum-envelope assumption remain explicit. They do not certify
agreement with an external scientific reference.

The explicit `tests.project.lib.numerical_planning_acceptance` command records
ordinary LCDM/Planck angular costs, numerical budgets, exact and compatible
reuse, changed requests/parameters, subsets, cache pressure and batch failure
isolation. It preserves each operation and its producer identity. Adaptive
subsets
compare raw and public products against the sum of their recorded numerical
uncertainties; different accepted meshes need not agree to roundoff.
`--assess-only` reads retained measurements into a separate digest-bound
readback without changing the producer report or rerunning its solver.
These measurements and scientific acceptance remain outside unittest
discovery and `devcovenant run`.
Historical sparse numerical measurements cover seven actual operations
for each model. On the recorded Intel i5-4690K, cold LCDM and Planck requests
take 617.956 and 3053.376 seconds; exact repeats take 0.015 and 0.172 seconds.
These are final-control numerical measurements at ell=(2,10,20), including
the full time domain and independent physical-axis comparisons. `PLAN.md`
records the workload variants, memory costs and source-bound evidence. They
did not establish full-suite practicality; that closure claim is withdrawn.
Matched external angular parity remains a separate acceptance requirement.

Projection now consumes one bounded mode chunk before releasing its radial
kernels, including when adaptive comparisons are enabled and the maximum
multipole is small. The same source, geometry and coarse/fine error checks
remain active. Bessel recurrence columns and array caches have memory budgets;
these partition work without discarding physical samples or relaxing accuracy.
The runtime reports actual retained kernel bytes and live mode counts.
Physical phase-space expressions must bind declared parameters before
numerical planning; missing thermal constants are compilation errors.
For compiler-proven affine state equations, continuous interval Jacobians
evaluate their coefficient probes together. The scalar trajectory, original
error tolerances, collision reduction order and boundary handling remain
unchanged. Nonlinear equations retain the general Jacobian calculation.
Local intervals use one-step stiff integration to avoid restarting multistep
history at every projected boundary; whole-history BDF remains independently
tested. Initial scale probes preserve zero and tiny driven states.
Continuous solvers select sparse LU for mostly zero Jacobians, retaining
every coefficient and any new nonzeros as the equations evolve. This avoids
dense factorization of automatically expanded hierarchies without reducing
their depth or changing their error tolerances.

Dense graph physics acceptance is explicitly invoked with
`python -m unittest -v
tests.project.lib.ccmbs_production_graph.ProductionGraphTestCase` in the
managed environment. Its original physics assertions remain intact. Ordinary
`test_ccmbs_production_graph` discovery checks manufactured wave rendering,
artifact hashes and fresh-process reload without initiating production work.
It does not certify the manufactured data as scientific solver acceptance.
Alias-routing fixtures explicitly bind their diagnostic request and original
role-based hierarchy after preparation, so public replanning cannot turn a
label-equality check into numerical production. They require finite nonzero
spectra. Independent sector physics checks retain their domains and accuracy
thresholds; a separate check validates every Jacobian column of the expanded
public graph before integration starts.

The explicit `tests.project.lib.native_history_acceptance` command compares
both reference models' native modes and source terms before angular work.
Its `--refine` option additionally checks measured changes under finer
resolution. Reports retain the physical inputs, source identity and verdict.
This component check and the full scientific acceptance remain outside
ordinary discovery and `devcovenant run`.

Fast collision equilibrium must hold relative to both streaming and
expansion: the tight-coupling criterion in `runtime/evolution.py` uses
`max(abs(k), abs(Hconf))`. This prevents very long wavelength modes from
remaining on an instantaneous polarization equilibrium after decoupling.

Massive-neutrino density and pressure histories are retained
from the same q quadrature used by perturbation evolution, including the
continuous zero-mass limit.
The wCDM and w0waCDM CMB declarations use that same density closure while
evolving constant-`w0` and CPL dark-energy factors, respectively; their
`w=-1`, `wa=0` limit is continuous with the shared background path.
Both contracts declare dark energy as a smooth background component with
unit rest-frame sound speed and zero anisotropic stress. CCMBS validates the
analytic density and pressure factors on the full scale-factor grid before
recombination and line-of-sight evolution.
QAUC and QRSF likewise expose their theory-specific background terms
explicitly: QAUC normalizes its oscillatory attractor at the present epoch,
while QRSF normalizes its baryon-locking and relational-release sectors.
QRSF retains its named matter-density, matter-momentum, and baryon-Euler
closures, and neither model is routed through a generic LCDM surrogate.
TOG normalizes its volume-activated temporal-opposition term and separates
the q-resolved massive-neutrino background from CDM. TORG keeps its
baryon-locked relational inertia and explicitly splits residual, massless,
and massive radiation without introducing an independent CDM species.
Pipeline summaries now retain typed CMB failures, and CMB plot infoboxes show
the failure category and message when a theory cannot be computed; a failed
theory is never silently reduced to an unlabeled data-only panel.
Source histories are cached independently from complete spectra only when
their structural contract, parameter identity, solver, source grid, and
requested source roles match exactly. Runtime envelopes report source-cache
hits and misses, while the phase-aware k status records the physical node
requirement and whether the bounded grid resolves it. A contract may require
that phase check explicitly; an under-resolved grid is then rejected with its
metrics instead of being presented as converged.
Phase-aware k ladders use generalized Simpson integration on their actual
log-k coordinates, retaining physical anchors while removing the first-order
bias of a trapezoid rule. Material negative auto-spectrum lobes still fall
back to the positive trapezoid rule. Fixed-point
diagnostics also record ordered TT peaks and troughs, damping, TE sign
changes, and EE peaks directly from raw arrays before plotting.
The generated scalar hierarchy uses the temperature-normalized moment
`Pi = Theta_gamma,2 + 6 E_gamma,2`. CAMB's native quadrupoles equal four
times these temperature multipoles, giving `polter = 0.4 Pi` and the `3/4`
E-source coefficient. Temperature's integrated quadrupole source
uses the matching `1/4` ordinary and `3/4` second radial derivative
coefficients. CCMBS applies both to the undifferentiated visibility-weighted
moment, avoiding noisy second time derivatives of sampled histories.
The photon continuity equation uses `+Phi'`, preserving adiabatic
photon-baryon entropy for equal velocities. At integration stages the
generated conformal Hubble rate is evaluated from the same declared `H(a)`
as the coordinate conversion. Independent entropy, angular Thomson,
compact CAMB history, and actual application composition regressions cover
these conventions; see the
[CCMBS development checks](copernican/docs/cmb_solver.md#development-checks).
The explicit `scientific_acceptance --development-checks` commands report
test correctness and measured cost separately. Exceeding a historical cost
target remains visible in telemetry and does not fail passing tests. Failed,
skipped or missing tests still fail the command; the separate scientific
acceptance calculation remains outside ordinary discovery and DevCovenant
workflow runs.
The generated momentum combination eliminates its declared metric derivative
before floating-point evaluation to preserve small source terms under
cancellation. Exact scalar collision splitting uses a 0.1-radian phase
ceiling; exact collision exponentials alone do not control splitting error.
Coarsened projection batches preserve zero-width optional vector and tensor
sectors without indexing absent kernels, so scalar refinement remains warning
free under NumPy's strict empty-axis rules.

Generated scalar manifests publish a source-closure summary with derivative
kind, conformal-time coordinate, binding, dependencies, source-role ownership,
closure names, and Einstein residual names. Runtime envelopes repeat that
provenance and require finite, eta-aligned raw source-history fields before
independent residual recomputation. Generated constraints use strict term
lookup, so an omitted term raises a typed failure instead of being normalized
against its own residual. Explicit model graphs are recorded as
`not_applicable` for this generated summary and are never silently rewritten.

Declared MCMC workers prepare immutable graph structure once per model and
reuse it across parameter proposals. Bounded caches distinguish structural,
parameter-dependent, and complete-result data, while request diagnostics
record cold, warm, and exact-hit states with phase timings and work units.
The public `compute_cmb_spectrum_batch` contract evaluates an ordered
sequence of declared contracts and returns one serializable result per input,
including typed failures and cache provenance. It starts as an exact
scalar-to-batch adapter, so parameter-dependent state remains isolated while
shared-structure and vectorized kernels are validated independently.
Each successful result also records the requested multipole and spectrum
order, solver identity, and optional raw-spectrum payload. The diagnostics
parity audit compares public arrays exactly and compares raw arrays and
runtime metadata whenever those fields are present; missing cache identities
or required request metadata cannot pass as evidence.
The MCMC sampler exposes `cmb_batch_size` as an explicit opt-in setting;
the default `0` keeps the exact scalar sampler and fallback path unchanged.
Sampler progress also reports worker-pool launch and walker initialization
as explicit phases, including elapsed time, measured rate, remaining work, and
ETA in both CLI output and GUI progress snapshots. Burn-in and production
count iterations separately from cumulative walker evaluations. Worker logs
record runtime preparation duration so expensive startup is distinguishable
from posterior evaluation. Each CMB proposal logs its parameter vector,
elapsed time, cache state, effective ell/k/eta grids, phase timings, and work
units. Worker processes emit a heartbeat every 30 seconds and enforce the
configurable `cmb_evaluation_timeout_seconds` watchdog (default 300 seconds),
returning typed incomplete execution evidence instead of blocking an ensemble
step or converting a timeout into a rejected posterior proposal.
Primordial-only parameter rebounds reuse bounded transfer products and rerun
only primordial power integration; changed cosmological parameters retain
separate transfer identities, and adaptive refinement keeps its full path.
Exact split collision half-steps absorb collision stiffness, so their
magnitude does not create redundant Runge-Kutta microsteps after a declared
tight-coupling transition. Only valid parameter-domain exclusions become
rejected proposals; contract, convergence, non-finite, constraint, and
capability failures stop execution with typed diagnostics. Runtime telemetry
records phase timings, cache states, and work units without imposing a
wall-clock limit on valid solver evaluations. Large fixed-point requests use
deterministic ordered mode and projection chunks, and their engine-planned
numerical controls remain visible in the runtime envelope.

Batched exact collision updates group repeated dense operator matrices and
reuse one well-conditioned eigensystem while preserving analytic small-block
updates and the scalar exact fallback. The runtime envelope records the
kernel elapsed time, mode and fallback counts, result-array allocation
events, cache evidence, and shape-aware raw input/output digests. This CPU
baseline is deterministic and remains the correctness authority for any
future accelerator backend.

CCMBS may select an internal Taichi execution backend when its validated
fixed-shape kernels and an available device support the work. Unsupported
work, unavailable packages, and unsuitable devices fall back inside CCMBS to
the NumPy CPU correctness path. Backend and fallback evidence is deterministic
and path-free; callers cannot select CPU or Taichi as separate solvers, and no
Taichi dependency is required for ordinary installation or test discovery.

Ensemble fit results also retain an `ensemble_performance` record with total
and per-stage timings, requested and effective worker counts, the CPU-derived
worker limit, nominal proposal evaluations, and failed-request counts. The
record includes the configured CMB proposal timeout. Spawned workers request
one numerical thread, and the record marks any process or numerical
oversubscription. The reference validation manifest fixes seed 0,
five burn-in steps, ten production steps, 32 walkers, and a three-worker pool;
its copied run manifest and parameter summary preserve this provenance.

Copernican ships as a managed Python application. The repository keeps the
bootstrap interpreter, virtual environment, and locked dependencies in view so
source checkouts and installed copies follow the same launch path.

## Launch Copernican

Start in the repository root. The commands below bootstrap the managed
environment, install the locked dependencies, and launch the CLI or GUI.

### Bootstrap the private interpreter

macOS and Linux:

Download the Python 3.11 build.

```
mkdir -p .python
arch="$(uname -m)"
case "$(uname -s)" in
    Darwin)
        plat="apple-darwin"
        ;;
    Linux)
        plat="unknown-linux-gnu"
        ;;
    *)
        echo "Unsupported platform." >&2
        exit 1
        ;;
esac
base="https://github.com/astral-sh/python-build-standalone/releases"
file="download/20251028/cpython-3.11.14+20251028-${arch}-${plat}"
file="${file}-install_only.tar.gz"
url="$base/$file"
curl -fL "$url" | tar -xz -C .python --strip-components=1
```

Windows PowerShell:

Download the Python 3.11 build.

```
New-Item -ItemType Directory -Force .python | Out-Null
$base = "https://github.com/astral-sh/python-build-standalone/releases"
$file = "download/20251028/cpython-3.11.14+20251028-amd64-pc-windows-msvc"
$file = "${file}-install_only.tar.gz"
$url = "$base/$file"
Invoke-WebRequest -Uri $url -OutFile python.tar.gz
tar -xzf python.tar.gz -C .python --strip-components=1
Remove-Item python.tar.gz
```

Windows cmd:

Download the Python 3.11 build.

```
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
    "$base = 'https://github.com/astral-sh/python-build-standalone/releases'; ^
     $file = 'download/20251028/'; ^
     $file = $file + 'cpython-3.11.14+20251028-'; ^
     $file = $file + 'amd64-pc-windows-msvc'; ^
     $file = $file + '-install_only.tar.gz'; ^
     $url = $base + '/' + $file; ^
     New-Item -ItemType Directory -Force .python | Out-Null; ^
     Invoke-WebRequest -Uri $url -OutFile python.tar.gz; ^
     tar -xzf python.tar.gz -C .python --strip-components=1; ^
     Remove-Item python.tar.gz"
```

### Create the managed virtual environment

macOS and Linux:

```
./.python/bin/python3 -m venv .venv
```

Windows PowerShell:

```
.\.python\python.exe -m venv .venv
```

Windows cmd:

```
.\.python\python.exe -m venv .venv
```

### Activate the environment

macOS and Linux:

```
source .venv/bin/activate
```

Windows PowerShell:

```
.venv\Scripts\Activate.ps1
```

Windows cmd:

```
.venv\Scripts\activate.bat
```

### Install the locked dependencies

This installs the exact package versions Copernican expects into the active
environment.

```
python -m pip install -r requirements.lock
```

### Run Copernican

Start the command-line interface.

```
python -m copernican --cli
```

Start the graphical interface.

```
python -m copernican --gui
```

The GUI opens directly in the active `.venv` on every supported platform.

If Copernican is installed in the same `.venv`, use these commands instead.

```
copernican --cli
```

```
copernican --gui
```

See [docs/packaging.md](docs/packaging.md#launch-copernican) for the
packaging notes that sit alongside these commands.

Each run keeps its own run logs inside the generated
`~/copernican_output/copernican-run_*` folder.

## Repository Layout
- `copernican/lib/` contains shared runtime helpers, GUI scaffolding,
  analysis tools, plotting helpers, and the declared CMB internals.
- `copernican/models/` houses the YAML model definitions and their metadata.
- `copernican/samplers/` collects the sampler back ends.
- `copernican/datasets/` bundles the trusted datasets and parser metadata.
- `copernican/validation/` holds the validation runner and reference
  manifests.
- `docs/` contains the long-form manual set.
- `ABOUT.md`, `AGENTS.md`, `CHANGELOG.md`, `CITATION.cff`, `PLAN.md`,
  `SECURITY.md`, and `SUPPORT.md` describe the front-door package contract.

## Run Builder and GUI
The GUI keeps the same manifest model as the CLI. The Run Builder walks
through seed, control model, test model, dataset, sampler, and plan
panels. The control model defaults to `model_lcdm.yml`, while the test model
is selected independently. The Save Manifest page stays locked until each
step has a selection; and the Start Run action renames the workspace to
`copernican-run_<timestamp>` before launching the worker. The Run Settings
panel mirrors the CLI prompts for walkers, burn-in, production steps, and pool
size so GUI runs and CLI runs use the same run metadata.

Sampler convergence diagnostics are evaluated over sampled coordinates.
Parameters locked by equal lower and upper bounds remain in posterior outputs
with deterministic diagnostics: R-hat is `1.0`, and bulk and tail effective
sample sizes equal the retained draw count. If ArviZ cannot produce finite
diagnostics for a sampled coordinate, the sampler uses its conservative
internal finite estimator.

The manifest stores one comparison request containing both model identities.
Compatibility checks cover declared observables, units, multipole grids, and
spectrum roles before execution. Summaries, CSV files, posterior artifacts,
plot footers, and residual labels use the resolved control/test pair rather
than assuming an LCDM control.
Posterior plotting reads that pair from the saved manifest, and direct
plotting calls provide the same comparison object.

The run worker is the sole owner of the canonical log file. It sends
structured severity-preserving events to the Run Monitor's in-memory log box,
while progress snapshots use a separate channel. GUI forwarding never writes
back to the worker file, so each event and selected dataset-ingestion record
appears once. Metadata dialogs open with the system default application and
use the same launch behavior as the rest of the GUI.
Run and validation workers use the active interpreter from the directory that
contains the importable `copernican` package, so source checkouts do not
require an editable package installation.
Immediately before launch, the GUI writes the displayed confirmation snapshot
to the run workspace. The worker therefore receives the same models, datasets,
seed, and sampler settings shown in the Run Builder.

## Analysis Workspace
The Analysis tab provides Run Summary, Posteriors, and Comparisons tools.
Run Summary ingests a saved run folder and renders the manifest, parameter
summary, and log in a scrollable panel. Posteriors lists `posterior-*.nc`
snapshots and renders trace and histogram views in the shared plot viewer.
Comparisons loads two run folders and reports parameter shifts, dataset count
deltas, and χ² differences in a structured view.

## Validation
The Validation tab runs the reference manifest against the shipped datasets,
streams the CLI output into the GUI, and stores the resulting summary in
`~/VALIDATION.md` alongside the per-run output directory. The manifest keeps
the regression baseline deterministic so validation reports stay repeatable.

## Documentation and Policy
The package docs mirror the root docs so installed copies and repository
copies stay aligned. `README.md` is the canonical source for
`copernican/README.md`. `docs/gui_guide.md` explains the GUI,
`docs/cli_guide.md` explains the CLI, `docs/run_manifest.md` covers manifest
structure, and `docs/packaging.md` covers setup and distribution tasks.

## Maintenance Helpers
Command-line users can work without the GUI:

- `python -m copernican --catalogue-summary`
- `python -m copernican --revalidate-dataset <dataset_id>`
- `python -m copernican --list-manifests`
- `python -m copernican --show-manifest <path>`
- `python -m copernican --run-validation`
- `python -m copernican --analysis-summary <run_dir>`
- `python -m copernican --analysis-compare <base_run> <alternate_run>`
- `python -m copernican --analysis-posterior <run_dir>`

## Repository Policy
Read `AGENTS.md` before making changes, keep the package docs mirrored, and
follow the gate workflow so edited files, manifests, and generated artifacts
stay in sync.
