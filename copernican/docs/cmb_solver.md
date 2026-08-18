# Native CMB Solver Convention
**Last Updated:** 2026-08-18
**Project Version:** 12.0.26

## Overview
This document is the canonical physical convention for Copernican's native
CMB solver path. Every bundled production CMB model uses one route-neutral
declared graph with no solver selector or backend fallback.

The production execution identity is
`ccmbs_numpy`, displayed as CCMBS — Copernican Cosmic Microwave Background
Solver. CLI and GUI workflows select control and test model
contracts and a sampler; they do not expose a CMB-sampler choice.

The scalar, vector, and tensor sectors follow this contract. Implementations
must preserve the meaning of states, source terms, gauge labels, and public
spectra defined here. A model may declare a native contract while marking CMB
output unavailable when its theory has no defensible linear perturbation
closure.

## Ordered batch contract

Callers that evaluate several cosmologies may use
`compute_cmb_spectrum_batch(contracts, ells, requested_spectra=...)`. Results
retain input order and carry the input index, either a native spectrum or a
typed failure, a performance envelope, and cache provenance. A domain or
solver failure is isolated to its item and cannot replace a neighboring
result. The initial implementation is an exact scalar-to-batch adapter; it
therefore shares only identities that the native cache already proves safe and
does not approximate parameter-dependent background or transfer state.

The MCMC sampler accepts `cmb_batch_size` as an explicit setting. `0` is the
default and preserves the scalar reference path. A value greater than one
uses bounded ordered worker batches while retaining the scalar fallback for
serial execution and any unsupported batch capability.

The native route uses conformal time `tau`, conformal distance
`chi = eta0 - eta`, and comoving wave number `k` in inverse Mpc. All
dimensionless perturbations are Fourier amplitudes in the same plane-wave
convention used by the declared graph compiler and the native line-of-sight
integrator.

## Native Model Declarations
`copernican/models/model_lcdm.yml` is the canonical native LambdaCDM
declaration. It defines the background and recombination inputs, scalar
species and hierarchy families, Thomson coupling, adiabatic initial data,
projection typing, and a bounded numerical envelope. The model compiles its
generated scalar hierarchy through the same native runtime described below.

Its perturbation contract contains physical declarations only. Keys such as
`backend`, `standard`, and `backend_mapping` are rejected rather than used to
select another solver. Production output comes from the native declared graph
and does not call CAMB or CLASS.

Available bundled CMB models use the same contract shape. Their model files
preserve theory-specific parameters, priors, distance equations, sound-
horizon expressions, and declared background functions. Each file declares a
scalar sector, its physical species, hierarchy families, Thomson coupling,
conservation rule, regular adiabatic initial family, projection typing, and
numerical controls. The compiler materializes the common scalar hierarchy
from this metadata, so model-specific background expressions feed one solver
without a backend branch. USMF declares its physical species and
native contract and now carries a complete theory-facing shrinking-matter
closure specification. Its CMB output is enabled only through the native
declared graph after the equations, limits, and runtime controls pass the
Slice Twelve acceptance tests.

The bundled ontology is explicit at the model boundary:

* LambdaCDM declares photons, baryons, cold dark matter, and massless
  neutrinos.
* LambdaCDM+Mnu and the Planck reference add a q-resolved massive-neutrino
  family.
* wCDM, w0waCDM, QAUC, and TOG retain their declared cold-matter and neutrino
  species while replacing only their theory-specific background expansion.
* QRSF and TORG declare photons, baryons, and neutrinos without a CDM species.
  Their named matter-density, matter-momentum, and baryon-Euler closures
  provide baryon-locked relational sources.
* USMF2 declares photons, baryons, and massless neutrinos plus one explicit
  shrink-field scalar degree of freedom. Its sourced equations, constraints,
  initial family, projections, and observables execute through the native
  declared graph without importing LCDM matter or a reference backend.

The compiler materializes equations, common line-of-sight sources,
observables, and regular initial data only for those declared species and
hierarchy families. It does not create `omch2`, `Omega_c0`, `delta_c`, or
`theta_c` for a contract without CDM, and it does not create q-resolved
massive-neutrino states for a contract without that family.

## USMF2 Production Closure

`model_usmf2.yml` is the production closure record for the Unified Shrinking
Matter Framework version 2. It sets `valid_for_cmb: true` only because the
proposed equations are independently implemented and tested through the
native declared graph. The contract does not invent missing physics or route
to a reference backend.

The graph declares one dimensionless `shrink_field` and its conformal-time
rate, baryon density and velocity, photon temperature moments, massless
neutrino moments, and the two conformal-Newtonian metric potentials. Its
`shrink_metric_constraint` is the only metric constraint target, while the
`zero_scalar_shear` closure supplies the lapse potential. Photon and neutrino
hierarchy tails are explicit closure nodes rather than implicit zeros.

Every USMF2 equation, constraint, closure, source, and initial condition has
a provenance note. Ma and Bertschinger (1995) is cited for the shared gauge,
fluid, collisionless hierarchy, and line-of-sight conventions; the
shrink-field response, metric normalization, and finite tails are marked as
USMF2 closure choices. The `accuracy_controls.analytic_limits` entries name
the homogeneous, no-shrink, and zero-shear limits covered by the production
contract tests. The native runtime also checks finite source histories,
parameter response, declared conservation balance, and coarse-to-reference
history agreement.

## Capability Audit
`copernican.lib.cmb_contract.audit_cmb_capabilities` is the authoritative
machine-readable audit for a compiled perturbation contract. It reports the
declared gauge, sectors, species, hierarchy families, collision operators,
interactions, closures, initial-condition families, projection typing, and
observable names. `build_cmb_capability_matrix` applies the same audit to the
bundled model corpus, keyed by the declared model name. Neither function uses
a theory name, model filename, model family, or assumed species to select a
route; the execution route is the universal native declared graph recorded in
the compiled manifest. The audit also records background references, numerical
controls, and which standard hierarchies were legitimately materialized from
the declaration.

The public capability matrix has one row for each requested spectrum. The
minimum transfer roles are:

* `TT`: temperature with temperature.
* `TE`: temperature with even polarization.
* `EE`: even polarization with even polarization.
* `BB`: odd polarization with odd polarization.
* `PP`: scalar lensing potential with scalar lensing potential.
* `TP`: temperature with scalar lensing potential.
* `EP`: even polarization with scalar lensing potential.

The compiled angular-spectrum observable must bind those transfer roles, and
`PP`, `TP`, and `EP` must remain scalar-sector combinations. A missing
observable, transfer component, role, or compatible sector produces an
unavailable matrix row. `require_cmb_capability` rejects that row before
execution with the model name and the concrete missing declaration; unknown
public names receive the supported-spectrum list. These diagnostics define
the unsupported-combination boundary without substituting a standard model.
The solver invokes this audit before constructing the declared background, so
an unsupported request cannot spend work on background tables or mode
evolution before failing.

For a non-radiation-dominated early background, the generated regular scalar
series scales its leading time powers with the local conformal-Hubble time.
This keeps the declared Einstein constraints valid for theory-specific
expansion laws while retaining the same adiabatic mode contract.

## Scalar Runtime Authority
The production scalar runtime has one execution authority: the compiled
declared equation graph. Compilation materializes state slots from declared
equations, resolves derived values, constraints, closures, interactions, and
collision metadata in dependency order, and builds the validated equation
program used at every Runge-Kutta stage. The same program supplies scalar
state histories, declared source histories, transfer components, and spectra.

No scalar mode uses a second hand-written hierarchy, alternate scalar batch
equations, empirical output scale, injected reference spectrum, or hidden
damping term. Projection batching applies only to radial kernels and does not
alter scalar state evolution. Cache entries store compiled graph structure and
reusable numerical data; they never store or substitute observable results.

The runtime envelope governs requested work before evolution begins. It
rejects unbounded or under-declared state, grid, hierarchy, source, and
projection work rather than lowering the declared physical calculation.
Scalar execution remains native and does not import or call CAMB or CLASS;
those packages appear only in independent scientific reference tests.

## Scalar Metric Convention
The canonical scalar convention is the conformal-Newtonian convention
used by Ma and Bertschinger for the physical meaning of the two scalar
metric potentials:

```text
ds^2 = a(tau)^2 [
  -(1 + 2 Psi) d tau^2
  + (1 - 2 Phi) delta_ij d x^i d x^j
]
```

`Psi` is the lapse or Newtonian potential.
`Phi` is the spatial-curvature potential.
Both are dimensionless.

The native contract names and units are:

- `Phi`, `phi_aux`, `metric_potential_phi`
  Curvature potential `Phi`. Units: dimensionless.
- `Psi`, `psi_aux`, `metric_potential_psi`
  Newtonian potential `Psi`. Units: dimensionless.
- `Phi_gi`
  Gauge-invariant curvature potential used by the synchronous route as its
  stable physical metric state. Units: dimensionless.
- `Phi_tau`, `Psi_tau`
  Conformal-time derivatives of the corresponding potentials.
  Units: `1/Mpc`.
- `theta_s`
  Velocity divergence `theta_s = i k_j v_s^j`. Units: `1/Mpc`.
- `sigma_s`
  Scalar anisotropic stress or quadrupole shear variable. Units:
  dimensionless.

The synchronous convention used for internal history checks is the
Ma-Bertschinger synchronous form with `h` and `eta`:

```text
ds^2 = a(tau)^2 [
  -d tau^2
  + ((1 + h/3) delta_ij + D_ij eta) d x^i d x^j
]
```

The native placeholder names are:

- `h_sync_metric`
  Synchronous trace perturbation `h`. Units: dimensionless.
- `eta_sync_metric`
  Synchronous shear perturbation `eta`. Units: dimensionless.

## Native State Convention
The native state meanings are fixed as follows.

### Photon Temperature
`theta_gamma0`, `theta_gamma1`, `theta_gamma2`, ... are the photon
temperature multipoles `Theta_gamma,l` from

```text
Theta_gamma(mu) = Sum_l (-i)^l (2l + 1) Theta_gamma,l P_l(mu)
```

All photon temperature multipoles are dimensionless.

### Photon Polarization
`e_gamma0`, `e_gamma1`, `e_gamma2`, ... are the even-parity polarization
multipoles `E_gamma,l`.
Vector and tensor sectors use the same dimensionless spin-2 polarization
convention and may carry odd-parity multipoles.

`polarization_moment` means

```text
polarization_moment = Theta_gamma,2 / 10 + 3 E_gamma,2 / 5
```

This is the dimensionless scalar Thomson source moment.

`polarization_b_mode_seed` is the declared odd-parity transfer seed carried
through exact lensing when a model declares primordial or sourced `B`.
Units: dimensionless.

### Declared Matter Species
- `delta_b`, `theta_b`
  Baryon density contrast and velocity divergence. Units: dimensionless and
  `1/Mpc`.
- `delta_c`, `theta_c`
  CDM density contrast and velocity divergence. These states exist only when
  the model declares a `cdm` species; the compiler never adds them to satisfy
  a hierarchy requirement.

The scalar Einstein sources assemble only the declared matter terms. Models
without CDM must provide a theory-specific matter source closure when their
background adds relational or effective inertia. QRSF and TORG declare
baryon-locked density and momentum sources and baryon Euler closures. Their
source expressions use the model background factors rather than relabeling
those effects as CDM.

### Massless Neutrinos
- `delta_nu`
  Density contrast. Units: dimensionless.
- `theta_nu`
  Velocity divergence. Units: `1/Mpc`.
- `sigma_nu`
  Anisotropic stress. Units: dimensionless.
- `nu_l3`, `nu_l4`, ...
  Higher multipoles `F_nu,l`. Units: dimensionless.

The scalar anisotropic-stress state is the declared `F_nu,2` hierarchy
member. Its hierarchy therefore starts with
`sigma_nu' = 4 theta_nu / 15 - 3 k nu_l3 / 5` and
`nu_l3' = 3 k sigma_nu / 7 - 4 k nu_l4 / 7`. The same normalization is used
by the metric shear source and the generated initial series.

### Massive Neutrinos
The canonical massive-neutrino momentum variable is the dimensionless
comoving momentum label `q = p / T_nu0`, treated as one resolved
momentum-grid coordinate. For a photon temperature `Tcmb_K`, the present
neutrino temperature is `T_nu0 = (4/11)^(1/3) Tcmb_K`, expressed in eV for
the runtime mass conversion. The canonical energy in the grid convention is

```text
epsilon(q, a) = sqrt(q^2 + (a m_nu / T_nu0)^2)
```

and the canonical background distribution is the thermal Fermi-Dirac shape.
When `sum_mnu` or `mnu` is declared, the q family uses the per-species mass
`sum_mnu / num_massive_neutrinos`; the total present density uses the declared
mass sum for normalization.

The authoritative evolved states are the q-resolved hierarchy members:

- `delta_nu_massive_q<i>`
- `theta_nu_massive_q<i>`
- `sigma_nu_massive_q<i>`
- `nu_massive_q<i>_l<j>`

All q-resolved hierarchy amplitudes are dimensionless. The q-resolved
`theta_nu_massive_q<i>` members store the regularized physical dipoles
`v(q,a) Psi_1`. The physical velocity-divergence source appears after the
matching q-integrated momentum weight multiplies them by `k`; this keeps the
nonrelativistic limit finite instead of evolving a singular `1 / v` source
term.

The aggregate names
`delta_nu_massive`, `theta_nu_massive`, `sigma_nu_massive`,
and `nu_massive_l<j>` are reserved for strict q-integrated aliases built
from the same resolved hierarchy. They must not become an independently
drifting evolution path.

The runtime also exports physical density, pressure, momentum, and shear
fractions for each q family. These fractions follow the background
`Omega_nu(a)` scaling and feed the Einstein source with the required `a^2`
factor. Because the stored dipole includes `v(q,a)`, the momentum fraction
uses `q^4 f_0 / v(q,a)`; the momentum source still carries the relativistic
`4/3` inertial factor.
The declared q surface uses a second-order composite trapezoid rule in
`log(q)`. The runtime rejects non-finite or non-monotonic nodes, non-positive
weights, invalid bounds, and unsupported quadrature orders before caching the
grid. A momentum-grid declaration remains inert unless the contract declares
the massive-neutrino species and hierarchy family.

### Vector States
The canonical vector metric amplitude is `sigma_vector`, the transverse shear
variable propagated by the vector Einstein system. It is dimensionless.

The native matter and radiation vector states are:

- `v_b_vector`, `v_c_vector`
  Baryon and optional CDM vorticity amplitudes. `v_c_vector` is present only
  when the model declares CDM. Units: dimensionless.
- `q_gamma_vector`, `q_nu_vector`
  Photon and massless-neutrino vector heat fluxes. Units: dimensionless.
- `pi_gamma_vector`, `pi_nu_vector`
  Photon and massless-neutrino vector anisotropic stress. Units:
  dimensionless.
- `theta_gamma_v3`, `theta_gamma_v4`, ...
  Higher photon vector temperature multipoles. Units: dimensionless.
- `e_gamma_v2`, `e_gamma_v3`, ...
  Vector even-parity polarization multipoles. Units: dimensionless.
- `b_gamma_v2`, `b_gamma_v3`, ...
  Vector odd-parity polarization multipoles. Units: dimensionless.
- `nu_v3`, `nu_v4`, ...
  Higher massless-neutrino vector multipoles. Units: dimensionless.

The generated vector route also carries the algebraic source moments
`vector_polarization_moment = pi_gamma_vector / 10 + 3 E_gamma,2 / 5`
and `vector_visibility_polarization_moment = g * vector_polarization_moment`.
Vector temperature uses the two flat-space radial families
`sqrt(l(l+1)/2) j_l(x)/x` and
`sqrt(3l(l+1)/2) (j_l'(x)/x - j_l(x)/x^2)`; vector E and B use
the corresponding spin-1 radial limits.
The scalar E-mode line-of-sight source applies the CAMB `15/2` normalization
to the declared `visibility * polarization_moment` before the native spin-2
projection. The scalar radial window carries the standard spin-2 factorial
prefactor and `j_l / x^2`.

### Vector Hierarchy Acceptance

The vector graph is a physical transverse Einstein-Boltzmann sector, not a
generic variable family. `sigma_vector` supplies the metric shear, while
matter vorticity, photon heat flux and anisotropic stress, massless-neutrino
moments, and photon E/B multipoles provide the declared state roles. Thomson
vector drag carries the momentum exchange between photons and baryons, and
each vector hierarchy uses its declared free-streaming terminal closure.

The compiled manifest identifies these roles through `vector_hierarchy`. It
records the metric state, vector hierarchy state groups, even/odd parity, the
free-streaming closure, and the four radial kernels
`vector_temperature_1`, `vector_temperature_2`, `vector_e`, and `vector_b`.
The vector transfer payload exposes temperature, E, and B components, and
the public vector `TT`, `TE`, `EE`, and `BB` surfaces are formed from those
components without scalar source substitution.

Vector acceptance uses independent flat-space radial-kernel limits, finite
transfer and spectrum checks, and exact-remapper checks for primordial
odd-parity `BB`. The manifest test also compiles a scalar-only contract and
requires its vector role summary, state names, source names, and radial
kernels to remain empty.

### Tensor States
The canonical tensor metric amplitude is `h_tensor`, with the explicit
conformal-time derivative `h_tensor_tau = d h_tensor / d eta`.
`h_tensor` is dimensionless and `h_tensor_tau` has units `1/Mpc`.

The native tensor radiation states are:

- `pi_gamma_tensor`, `pi_nu_tensor`
  Photon and massless-neutrino tensor temperature quadrupoles. Units:
  dimensionless.
- `theta_gamma_t3`, `theta_gamma_t4`, ...
  Higher photon tensor temperature multipoles. Units: dimensionless.
- `e_gamma_t2`, `e_gamma_t3`, ...
  Tensor even-parity polarization multipoles. Units: dimensionless.
- `b_gamma_t2`, `b_gamma_t3`, ...
  Tensor odd-parity polarization multipoles. Units: dimensionless.
- `nu_t3`, `nu_t4`, ...
  Higher massless-neutrino tensor multipoles. Units: dimensionless.

Tensor intensity and polarization are spin-2 hierarchies, so no tensor
temperature monopole or dipole and no tensor E-polarization `l = 0` or
`l = 1` state exists. The metric wave obeys

```text
h_tensor' = h_tensor_tau
h_tensor_tau' = -2 Hconf h_tensor_tau - k^2 h_tensor
                + 3 (H0/c)^2 tensor_total_shear_source / a^2
```

Photon and massless-neutrino temperature moments use the spin-2 free-streaming
recurrence. Photon E/B moments use the coupled spin-2 polarization recurrence,
with Thomson scattering supplied by the declared exact collision operator.
The final temperature and neutrino moments use the flat-space outgoing-wave
closure. The final E/B moments set the unavailable `l + 1` moment to zero
while retaining their parity coupling.

The regular superhorizon series uses

```text
R_nu = Omega_nu0 / (Omega_gamma0 + Omega_nu0)
D_tensor = 15 + 4 R_nu
h_tensor_tau = -5 k^2 eta h_tensor / D_tensor
pi_nu_tensor = 4 k^2 eta^2 h_tensor / (3 D_tensor)
collision_rate pi_gamma_tensor = -(32/45) h_tensor_tau
```

These three relations are declared as named initial constraints and are
evaluated before each generated tensor mode enters the integrator. The tensor
Thomson source moment is
`tensor_polarization_moment = 0.1 pi_gamma_tensor + 0.6 e_gamma_t2`.
The line-of-sight sources are

```text
S_T = -exp(-tau) h_tensor_tau
      + (15/8) visibility tensor_polarization_moment
S_E = (15/2) sqrt(3/8) visibility tensor_polarization_moment
S_B = (15/2) sqrt(3/8) visibility tensor_polarization_moment
```

Temperature, E, and B sources use their tensor radial kernels rather than
scalar or vector windows. Signed-kernel parity and zero-argument limits are
finite and independently tested. Increasing the generated photon,
polarization, and neutrino hierarchy depths from the working depth changes
accepted tensor source histories by less than one percent.

## Optical Depth And Visibility
The native optical-depth convention is:

```text
tau(tau_obs) = integral from tau_obs to eta0 of a n_e sigma_T d tau
tau_dot = d tau / d eta = -a n_e sigma_T
g = visibility = -tau_dot exp(-tau)
```

`tau` is dimensionless.
`tau_dot` has units `1/Mpc`.
`visibility` has units `1/Mpc` and must stay non-negative for physical
histories.

## Gauge Transformations
The canonical scalar gauge bridge between synchronous variables and the
observable Newtonian basis is the standard first-order scalar transformation
with shift generator `alpha`:

```text
alpha = (h' + 6 eta') / (2 k^2)
Phi = eta - Hconf alpha
Psi = alpha' + Hconf alpha
delta_N = delta_S + 3 (1 + w) Hconf alpha
theta_N = theta_S + k^2 alpha
```

Gauge-invariant observable construction uses the Newtonian-basis `Phi`,
`Psi`, and their matching matter and radiation combinations after those
transformations.

The generated synchronous route evolves `gauge_shift_alpha`,
`h_sync_metric`, and `eta_sync_metric` as explicit synchronous histories and
also evolves the declared `Phi_gi` physical curvature state. `Phi_gi` uses
the same declared Einstein equation as the observable curvature state, which
prevents numerical synchronous gauge modes from contaminating the physical
source history. The `Phi` and `Psi` closures use `Phi_gi` and the declared
shear correction. `Phi_from_synchronous` and `Psi_from_synchronous` remain
explicit diagnostics of the synchronous metric transform.

The generated gauge-invariant route compiles through dedicated
`Phi_gi` and `Psi_gi` variables together with observable-basis aliases,
instead of reaching the observable construction surface only by relabeling
the Newtonian branch.

Custom declared graphs may supply their own gauge bridge when the
standard first-order transform above does not apply, but gauge labels
alone do not satisfy this contract.

### Gauge Equivalence Acceptance

The compiled perturbation manifest records `gauge_equivalence` for every
scalar route. Its `observable_basis` is the Newtonian observable basis, and
its `transformation` identifies one of the explicit bridges:

- `observable_identity` for the conformal-Newtonian route;
- `scalar_first_order` for synchronous `h`, `eta`, and `alpha` states;
- `bardeen_invariant` for the gauge-invariant `Phi_gi` and `Psi_gi` states.

The manifest also names the metric and derived transformation nodes, so a
gauge label cannot claim equivalence without a corresponding compiled graph.
The fixed scalar acceptance surface uses one cosmology, grid, regular
adiabatic mode, and source-role set for all three routes. Visible source
histories and `TT`, `TE`, `EE`, `BB`, `PP`, `TP`, `EP`, and lensed scalar
surfaces must agree to `0.1%`. The synchronous route is checked through
the explicit first-order transformation, while the invariant route is
checked through its Bardeen states. The test surface rejects agreement that
comes only from shared aliases or gauge labels.

## Regular Scalar Initial Modes
The generated scalar route materializes the following regular scalar
families with explicit leading super-horizon series:

- `adiabatic_scalar`
- `baryon_isocurvature`
- `cdm_isocurvature`
- `neutrino_density_isocurvature`
- `neutrino_velocity_isocurvature`

The generated series seed every affected scalar state through the same
declared graph surface used during evolution.
Density-like states keep the documented adiabatic and isocurvature
amplitudes, velocity-like states use the matching leading `k` or `k^2
tau` scaling, and the generated synchronous bridge seeds `h`, `eta`,
`alpha`, and `Phi_gi` from the same observable-basis metric constraint
surface.

One metadata-only contract selects exactly one auto-generated scalar family.
Declaring multiple automatic families is rejected rather than silently
choosing one. A contract may provide explicit start expressions alongside
the family declaration; those expressions override generated seeds for the
named state while the remaining hierarchy seeds are materialized normally.
Massive-neutrino momentum bins use the thermal distribution derivative for
the adiabatic family and the selected isocurvature series for each
isocurvature family, so q-resolved states do not inherit an unrelated
adiabatic seed.

Before integrating one generated scalar mode, the native runtime
evaluates the starting Einstein energy, momentum, and shear residuals and
the declared fast-manifold collision expressions. It also evaluates the
declared conservation rules at the start surface. Non-finite or
out-of-tolerance initial data are rejected before evolution.

## Regular Tensor Initial Mode
The generated tensor route materializes the regular `tensor_mode` family.
It seeds `h_tensor` from the declared primordial tensor amplitude, seeds
`h_tensor_tau` from the leading `k^2 tau` super-horizon series corrected by
the free-streaming neutrino fraction, and seeds the neutrino tensor quadrupole
from the matching `k^2 tau^2` term. Photon shear and all polarization
multipoles begin on the regular tight-coupling surface.

## Canonical Scalar Equations
The standard generated scalar hierarchy must use the following physical
equation families in this convention.

### Einstein System

```text
k^2 Phi + 3 Hconf (Phi' + Hconf Psi)
  = -4 pi G a^2 delta rho

k^2 (Phi' + Hconf Psi)
  = 4 pi G a^2 (rho + p) theta

k^2 (Phi - Psi)
  = 12 pi G a^2 (rho + p) sigma
```

### Generated Einstein Inputs
`copernican/lib/perturbation_contract.py` materializes the scalar
Einstein inputs as named derived quantities so runtime checks and evolution
components share one source surface:

- `matter_density_source = (Omega_c0 delta_c + Omega_b0 delta_b) / a`
- `radiation_density_source = (4 Omega_gamma0 Theta_gamma,0
  + f_nu delta_nu) / a^2`
- `total_density_source` is the sum of the matter and radiation pieces
  plus `massive_neutrino_density_source` when the massive hierarchy is
  active.
- `photon_velocity_divergence = 3 k Theta_gamma,1`
- `total_momentum_source` combines CDM and baryon velocities with
  photon and neutrino inertial terms using the same `/a` and `/a^2`
  scaling.
- `total_shear_source` carries photon anisotropic stress and twice the
  massless or massive neutrino hierarchy shear, matching the CAMB
  `pi_gamma` and `pi_nu` normalization.
- `einstein_energy_residual`, `einstein_momentum_residual`, and
  `einstein_shear_residual` are the runtime diagnostics for the three
  Einstein equations above.

When the massive hierarchy is active, the generated scalar graph feeds
the metric system through `massive_neutrino_density_source`,
`massive_neutrino_momentum_source`, and
`massive_neutrino_shear_source`. Those source slots consume the
thermally weighted physical `q` integrals, including their time-dependent
background fractions, without renaming the public Einstein-input surface.

The generated hierarchy evolves `Phi` from the Einstein momentum equation
and reconstructs `Psi` from the shear constraint. Its
`metric_constraint_scale = k^2` is therefore the exact Fourier-space scale,
not an algebraic low-k bridge. The energy equation remains available through
`einstein_energy_residual` as an independent runtime diagnostic.
The shear diagnostic evaluates the equivalent declared
`metric_constraint_scale * metric_shear_correction` term rather than
subtracting nearly equal `Phi` and `Psi` values, so the constraint check does
not lose its physical significance to floating-point cancellation.

Conformal time retains its physical radiation-era origin. The background
adds the analytic integral below `a_min` to both `eta` and the sound horizon,
so superhorizon initial conditions and hierarchy closures use the same clock.
The generated scalar initial state obtains `Theta_gamma,2` and
`E_gamma,2` from declared regular-series expressions that use the compiled
collision rate before `Phi` is seeded from the regular Einstein shear
relation. The generated hierarchy therefore starts from the same declared
quadrupole collision block used during evolution. The regular adiabatic seed
converts primordial curvature into the radiation-era curvature and lapse
potentials using the declared relativistic-neutrino fraction.
The default hydrogen recombination quantities use the photon temperature
before Compton decoupling and the adiabatically cooled matter temperature
afterward. The native case-B coefficient includes the standard RECFAST
multilevel-atom correction, while declared recombination quantity hooks
remain authoritative.
Helium Saha fractions are iterated to convergence so neutral helium does not
leave a numerical free-electron floor in the post-recombination tail.

### Photon Temperature Hierarchy

```text
Theta_gamma,0' = -k Theta_gamma,1 + Phi'
Theta_gamma,1' = k (Theta_gamma,0 + Psi - 2 Theta_gamma,2) / 3
                 - tau_dot (Theta_gamma,1 - v_b / 3)
Theta_gamma,2' = 2 k Theta_gamma,1 / 5
                 - 3 k Theta_gamma,3 / 5
                 - tau_dot (Theta_gamma,2 - polarization_moment)
```

Higher photon multipoles use the standard free-streaming recurrence with a
physical terminal closure and the Thomson damping that survives above the
quadrupole:

```text
Theta_gamma,l' = k [l Theta_gamma,l-1 - (l + 1) Theta_gamma,l+1] / (2 l + 1)
                 - tau_dot Theta_gamma,l
for 3 <= l < l_max

Theta_gamma,l_max' = k Theta_gamma,l_max-1
                     - k (l_max + 1) Theta_gamma,l_max
                       / sqrt[(k eta)^2 + (l_max + 1)^2]
                     - tau_dot Theta_gamma,l_max
```

That terminal form approaches `k Theta_gamma,l_max-1 - k Theta_gamma,l_max`
outside the horizon and the standard `-(l_max + 1) Theta_gamma,l_max / eta`
free-streaming limit once `k eta` is large.

### Polarization Hierarchy

```text
E_gamma,2' = - k E_gamma,3 / 3
             - tau_dot (E_gamma,2 - polarization_moment)
```

Higher `E` multipoles use the spin-2 free-streaming recurrence and retain
the corresponding `- tau_dot E_gamma,l` damping:

```text
E_gamma,l' = k l E_gamma,l-1 / (2 l + 1)
             - k (l + 3) (l - 1) E_gamma,l+1
               / ((l + 1) (2 l + 1))
             - tau_dot E_gamma,l
for 3 <= l < l_max

E_gamma,l_max' = k l_max E_gamma,l_max-1 / (l_max - 2)
                 - k (l_max + 3) E_gamma,l_max
                   / sqrt[(k eta)^2 + (l_max + 3)^2]
                 - tau_dot E_gamma,l_max
```

The terminal coefficient is the CAMB spin-2 truncation factor, while the
horizon-aware denominator regularizes the closure before horizon entry.

### Matter And Neutrinos
The baryon, CDM, massless-neutrino, and massive-neutrino continuity, Euler,
and hierarchy equations follow the same Ma-Bertschinger sign convention.
The generated baryon Euler equation uses the thermodynamic baryon sound
speed, while the photon-baryon acoustic sound speed remains a separate
background quantity for the tightly coupled radiation-baryon system.
Massless-neutrino higher multipoles use the same horizon-aware terminal
closure family as the photon and polarization ladders. Massive-neutrino
density, pressure, momentum, and shear are fixed to thermal physical
`q` integrals with the matching `epsilon` factors and background-moment
normalization.

### Tight Coupling And Collision Splitting
The native scalar integrator compiles every declared collision operator
into one runtime block before evolution begins.
Each block resolves its target state slots, `rate_expression`, linear
coefficients or matrix entries, counterpart bookkeeping, and the declared
integration strategy.
The generated `thomson_drag` operator is an always-active exact block. Its
compiled matrix contains the declared photon-baryon dipole block and the
declared photon temperature/E-polarization quadrupole block, so the
Thomson collision history is advanced analytically rather than placed in an
explicit Runge-Kutta stage. The generated scalar equations omit those
collision terms, preventing direct and split updates from being counted
twice. The tight-coupling threshold controls the declaration-driven
fast-manifold projection:

```text
collision_rate >= k * tight_coupling_ratio
```

with hysteretic exit at
`collision_rate <= k * tight_coupling_ratio * tight_coupling_exit_ratio`.

The entry multiplier is `cmb.perturbations.numerics.tight_coupling_ratio`;
the exit multiplier is the separately declared
`cmb.perturbations.numerics.tight_coupling_exit_ratio`, which must be
strictly between zero and one. The native runtime does not infer an exit
threshold from a hidden scalar constant.

Within and outside the fast-manifold regime,
`copernican/lib/perturbation_contract.py` and
`copernican/lib/likelihoods/cmb/native_projection.py` advance the exact
Thomson block from the compiled `exact_form`.
While the threshold is active, the runtime derives first-order fast states
from the declared collision matrix and the forcing produced by the same
compiled equation program. Singular blocks preserve their declared
conserved left-null combinations; full-rank blocks and declared damping
targets use the compiled linear operator directly. No photon, baryon,
polarization, or multipole name is special-cased by the runtime. The same
exact split damps the generated `l >= 3` temperature and polarization
multipoles throughout the collision history.
The generated tensor hierarchy declares its photon quadrupole Thomson block
as a second exact operator, coupling `pi_gamma_tensor` and `e_gamma_t2` and
damping the higher temperature and polarization ladders. This keeps the
early-time tensor hierarchy from treating the large Thomson rate as an
explicit Runge-Kutta term.
Outside the scalar tight-coupling regime, the scalar operator falls back to
its ordinary declared expression in the explicit RHS; the tensor photon
block remains exact over its full collision history.

Other declared operators may remain `explicit`, or they may opt into one
compiled split block with `integration_strategy: exact` and a declared
`exact_form`, or with `integration_strategy: implicit` and a declared
`linear_block`.
Several compiled collision blocks may run in the same evolution interval, and
the native solver suppresses only the selected split-operator outputs
instead of zeroing shared symbols such as `collision_rate`.
After every exact, implicit, or fast-manifold update, the runtime evaluates
the conservation rules attached to that collision block. A non-finite or
out-of-tolerance invariant aborts the mode before the updated state can
reach projection.
Unsupported exact or implicit declarations therefore fail before evolution
instead of being silently ignored.

Generated scalar hierarchy families must declare
`closure: free_streaming_scalar`. The materializer uses that declaration for
the horizon-aware terminal temperature, polarization, massless-neutrino, and
q-resolved massive-neutrino closures; an unknown closure name fails during
contract preparation rather than selecting a hidden fallback.

## Line-Of-Sight Source Convention
The canonical scalar source decomposition is:

```text
S_T = g (Theta_gamma,0 + Psi)
    + (5/2) g polarization_moment
    + (15/2) (g polarization_moment)'' / k^2
    + d/d eta [g v_b] / k
    + exp(-tau) (Phi' + Psi')

S_E = 15 g polarization_moment / 2

S_B = 0 for scalar modes

S_phi = Phi + Psi
```

The native photon multipoles use the same temperature and polarization
normalization as the declared reference hierarchy. The generated scalar
source uses the factors above, and the photon contribution to the Einstein
shear source is `4 Omega_gamma0 Theta_gamma,2 / a^2`.

The source-role contract is the authoritative dispatch table for these
terms. Monopole, ISW, and additive roles use the ordinary spherical-Bessel
window. Doppler uses the first radial derivative, and additive_derivative
uses the second radial derivative. The temperature projection does not
infer a role from a source name and does not combine unbound source
histories. E polarization uses the declared spin-2 E kernel, while a
potential role uses the signed lensing-potential geometry. This keeps source
normalization and radial-kernel selection separate from the evolution code.

## Independent Projection Kernels

The radial projection layer evaluates each declared kernel independently of
source evolution. Its bounded batches expose ordinary spherical-Bessel
values, first and second radial derivatives, scalar spin-2 E/B windows,
vector temperature and spin-2 windows, tensor temperature and spin-2
windows, and the signed lensing-potential geometry. A batch shares the
spherical-Bessel recurrence for fixed ell and radial inputs, then derives
the sector kernels from those values without re-evolving a mode.

The kernel acceptance surface includes SciPy comparisons for nonzero values,
analytic zero-argument limits, signed-argument parity, and equality between
mode-batched and scalar radial evaluations. Projection declarations carry
allowed sector metadata. The compiler and runtime reject an incompatible
sector or radial kernel before line-of-sight integration rather than
silently using a scalar window for a vector or tensor source.

Every transfer component must bind a non-empty set of declared source terms.
The runtime resolves those bindings before projection and raises an
availability error when a referenced history is absent. It never replaces a
missing source with zero or evaluates an unrelated sector. The runtime
envelope records the resolved component-role set, source-grid sample count,
number of evolved modes, finite-history status, per-role maximum magnitude,
and the source-history convergence estimate. Source-grid refinement therefore
audits the histories that feed projection rather than only the final power
arrays.

The generated scalar Doppler source uses the baryon velocity
`v_b = theta_b / k` and projects `g v_b` through the derivative spherical-
Bessel kernel. The temperature polarization-quadrupole source is split into
a local `(5/2) g polarization_moment` term and a `(15/2) (g
polarization_moment)'' / k^2` term. The latter is evaluated from the declared
second conformal-time derivative history and projected with the ordinary
spherical-Bessel kernel. This preserves the canonical line-of-sight source
normalization.

During tight coupling, the native hierarchy evolves one photon-baryon
velocity by the declared momentum-weighted combination of the photon and
baryon Euler equations. The closure then restores `theta_b = 3 k
theta_gamma,1` and the declared first-order quadrupole relations. This keeps
the constrained evolution independent of the arbitrary transition threshold.

The canonical vector source decomposition is:

```text
P_V = pi_gamma_vector / 10 + 3 E_gamma,2 / 5
x = k chi

S_T^V = 4 g (v_b_vector + sigma_vector)
       + 15 d/d eta (g P_V) / (2 k)
       + 4 exp(-tau) sigma_vector'

S_E^V = 15 g P_V + 15 d/d eta (g P_V) / (2 k)

S_B^V = -15 g P_V / 2
```

`copernican/lib/perturbation_contract.py` materializes these as
`vector_temperature_source`, `vector_polarization_e_source`, and
`vector_polarization_b_source`; the radial factors are applied by the
sector-specific projection kernels. The vector temperature route selects the
`T1` family, while the vector `E` and `B` routes select their dedicated
spin-2 families. The native line-of-sight projector keeps the same sign and
parity convention across sectors: temperature sources are even, `E` is even,
`B` is odd, and lensing uses the Weyl-potential sum `Phi + Psi`.

The canonical tensor source decomposition is written in the CAMB tensor
transfer convention. The generated hierarchy evolves the tensor metric
wave and uses the physical photon polarization moment
`P_T = 0.1 pi_gamma_tensor + 0.6 E_gamma,2`. The tensor source
normalizations below also include the native primordial-power conversion
(`P_h = P_T/6`) and the tensor radial-kernel normalization:

```text
S_T^T = -exp(-tau) h_tensor' + (15/8) g P_T

S_E^T = (15/2) sqrt(3/8) g P_T

S_B^T = (15/2) sqrt(3/8) g P_T
```

`copernican/lib/perturbation_contract.py` materializes these as
`tensor_temperature_source`, `tensor_polarization_e_source`, and
`tensor_polarization_b_source`. The tensor temperature radial kernel carries
the complementary `sqrt(3/8)` and spin-2 factorial factor. The `15/8`
temperature coefficient is the native tensor polarization-moment conversion
used by the transfer convention. The photon quadrupole and E/B hierarchy
coefficients follow the tensor equations used by CAMB, including the exact
two-state Thomson block for `pi_gamma_tensor` and `E_gamma,2`. The terminal
tensor temperature recurrence uses the physical `l/(l-2)` closure, while
terminal tensor polarization multipoles omit the unavailable `l + 1` term
as in the reference
hierarchy.

## Public Spectrum Convention
The native projection layer integrates raw transfer functions into raw
`C_ell` values. The public solver then applies the native output convention.

For `TT`, `TE`, `EE`, `BB`, and the exact lensed temperature-like spectra:

```text
D_ell^XY = ell (ell + 1) C_ell^XY Tcmb^2 / (2 pi)
```

with units `muK^2`.

For `TP` and `EP`:

```text
Native TP or EP = ell (ell + 1) C_ell^{X phi} / (2 pi)
```

These cross-surfaces are dimensionless, matching the independent
lensing-potential reference returned without a CMB-unit conversion.

When exact lensing is requested, any declared unlensed `BB` input remains in
the remapping basis. The vector `polarization_b` transfer therefore survives
through `lensed_BB` instead of being dropped before the remapper runs.
If no odd-parity transfer is declared, the remapper uses a zero unlensed `BB`
baseline and still generates lensing-induced `lensed_BB` from the declared
E-mode and lensing-potential spectra.

For `PP`, the public solver returns the exact `clpp` normalization consumed
by the native curved-sky remapper:

```text
PP = clpp = [ell (ell + 1)]^2 C_ell^{phiphi} / (2 pi)
```

`PP` is dimensionless.

`lensed_TT`, `lensed_TE`, `lensed_EE`, and `lensed_BB` stay in the same
`muK^2` `D_ell` convention as their unlensed counterparts.

Lensed public requests are expanded to one contiguous zero-based analysis
surface before remapping. The exact lensed assembler passes only native
unlensed `TT`, `TE`, `EE`, optional `BB`, and native `PP` into the remapper;
it then selects the requested multipoles. Sparse requests therefore cannot
change the remapping calculation or bypass declared odd-parity input.

The remapper consumes a finite four-column unlensed surface in the order
`TT`, `EE`, `BB`, `TE`, together with `clpp` in the declared `PP` convention.
It rejects incomplete or non-finite surfaces and requires ordered interior
Gauss-Legendre nodes. Its `sampling_factor` is a declared quadrature control;
raising it refines interpolation without changing spectrum normalization or
injecting a reference output.

Native projection quadrature starts from the declared numerical ell range and
reference multipoles rather than from the subset requested by one caller.
This keeps a spectrum's k integration stable when callers request a sparse
surface such as `TT` at selected multipoles or request the complete lensed
family. Contracts that need a controlled accuracy tier can enable the
adaptive transfer, source, and line-of-sight projection surfaces described
below.

For high-resolution scalar requests, a contract can enable
`phase_aware_k_quadrature`. The fixed k envelope then uses the phase-aware
quadrature helper from declared multipole anchors, conformal distance, and
sound-horizon scales. The helper keeps acoustic phase coverage inside the
declared node budget instead of adding an unbounded high-k tail. Reference-ell
anchors remain explicit inputs to the bounded grid. Contracts without this
control retain the bounded anchor-and-gap grid.

Single-sector native routes also expose matching component aliases such as
`scalar_TT`, `vector_BB`, `tensor_BB`, and `total_TT`.
The tensor route reads `r` as the tensor-to-scalar amplitude ratio and `nt`
as the tensor spectral index when constructing the primordial tensor power
law.

Public names use the canonical
`[lensed_][scalar_|vector_|tensor_|total_]SPECTRUM` form. Exact component
observables declared by a graph retain that name; they are not collapsed into
an unprefixed surface. A component alias is valid only when the matching
sector owns the underlying observable. A `total_` alias can refer to a
single-sector output, an explicit total observable, or an observable whose
metadata identifies a mixed surface. It cannot relabel one sector of a
multi-sector graph as the total.

Every internal spectrum payload records one availability state for each
relevant name. `computed` identifies a requested declared result,
`unrequested` identifies a declared result outside the request dependency
closure, and `physical_zero` identifies the absent primordial `BB` baseline
accepted by the exact remapper. A name outside the declared graph is
unavailable and raises an error. Accessors never convert any of these states
into an empty array, and cached arrays are read-only.

Long-form CMB data uses the columns `ell`, `spectrum`, and `Dl_obs`.
Likelihood evaluation preserves the original table and covariance order,
including repeated and noncontiguous multipoles. Plot, diagnostic, and CSV
surfaces use the same canonical metadata, so scalar, vector, tensor, total,
lensed, unlensed, lensing-potential, and diagnostic outputs remain separate.

## Native Execution Pipeline

The native route is a declared-graph execution path with five physical
stages:

1. `copernican/lib/likelihoods/cmb/cmb.py` accepts only a prepared native
   runtime or a route-neutral declared contract and validates it before any
   spectrum is built.
2. `native_background.py` resolves the expansion, conformal-time, optical-
   depth, visibility, recombination, and sound-horizon tables. The table
   carries interpolation functions for the quantities sampled by evolution
   and projection.
3. `native_evolution.py` compiles hierarchy families, declared collision
   blocks, initial modes, gauge roles, and q-resolved momentum grids. Each
   wave number is evolved through the declared state surface; production
   execution does not call an external Boltzmann backend.
4. `native_projection.py` evaluates transfer histories on the fixed numerical
   envelope, applies the scalar, vector, or tensor line-of-sight kernels, and
   integrates the transfer products into raw angular spectra. The source
   history and radial-kernel conventions in this document determine the sign,
   parity, and normalization of each transfer.
5. `copernican_cmb_solver.py` converts raw spectra to the public units,
   resolves requested-spectrum dependencies, and sends the four-component
   temperature/polarization surface through `native_lensing.py` when lensed
   outputs are requested. `native_cache.py` owns bounded structural,
   parameter-dependent, and result caches with explicit invalidation rules.

The runtime has three cache classes. Structural work contains the compiled
expression and equation plans, dependency closure, state-slot and hierarchy
layouts, collision plan, and momentum-grid topology. Parameter-dependent
work contains the background, recombination, visibility, coordinate-rate,
collision, q-mass, and projection-kernel data for one bound cosmology. Result
work contains transfer matrices and public spectrum surfaces for one exact
request. `NativeRuntimeCacheIdentity` records contract-static,
cosmology-static, and request-specific key portions separately, so a new
multipole request does not invalidate structural entries.

The spectrum cache identity freezes the full execution-relevant contract view:
graph structure, bound `param_map` and model-parameter values, declared grids,
numerical and accuracy controls, background-provider identity, requested
canonical spectra, and the ordered multipole sequence. Repeated multipoles
remain part of that sequence. Changing any one of these inputs produces a
different request identity, while an identical request returns the same
read-only payload.

Primordial-only parameter rebounds use a separate bounded transfer cache.
Changes to the scalar amplitude, scalar tilt, tensor ratio, or tensor tilt
reuse the bound background, evolution, and projection products, then rerun
only the declared primordial power integration. The transfer identity retains
all non-primordial parameter values, multipoles, sectors, and requested
spectra, so a changed cosmology cannot inherit stale transfer functions.
Adaptive refinement controls disable this reuse and retain their complete
convergence path. Transfer-cache hits report zero evolution and projection
work while preserving the source-history and numerical provenance from the
cached product.

Each scalar mode uses the compiled equation program. A graph without split
collision operators uses one implicit BDF solve on the declared continuous
evolution grid, so stiff declared terms do not require an unbounded explicit
substep count. Graphs with split operators use an explicit Runge-Kutta
schedule for the remaining declared RHS together with the compiled exact or
implicit collision blocks. Both routes execute the same equation program;
neither introduces alternate scalar physics. Background values,
tight-coupling masks, collision updates, and source dependencies are resolved
from the declared graph at every stage. Projection kernels may use radial work
batches, but no batched scalar RHS or alternate scalar hierarchy is permitted.
q-resolved momentum states and declared vector or tensor states retain their
declared state-slot layouts.

The envelope records wall time for `compilation`, `background`,
`initial_data`, `evolution`, `projection`, `lensing`, and
`likelihood_assembly`. Every successful or failed request retains all phase
slots, governed work units, workload identity, cache state, stop phase, and
structured failure context. `native_cmb_performance_stats()` exposes bounded
recent records and aggregate phase totals. An identical request returns the
bounded spectrum cache directly with zero evolution and projection work.

### Runtime Lifecycle And Failure Semantics

`model_coder.py` owns structural compilation. A `NativeCMBRuntime` carries
recursively read-only model declarations, compiled perturbation data,
background expression plans, and a complete runtime signature. Binding one
proposal creates only fresh parameter mappings; it reuses the immutable
structure without deep-copying or recompiling it.

Spawn-based MCMC pools install one active posterior in each worker through a
worker initializer. The initializer prepares each enabled model's process-
local runtime assets once. Proposal tasks call the installed posterior rather
than serializing the model bundle per task. Structural assets are keyed by
process and runtime signature. Parameter changes retain expression plans,
graph topology, hierarchy metadata, state indexes, and q-grid nodes while
recomputing every affected background, source, transfer, and spectrum value.

The cache inventory classifies each bounded family:

* Structural caches hold declared symbol plans, graph execution plans,
  process-local runtime assets, and momentum quadrature topology.
* Parameter caches hold bound momentum metadata, backgrounds, and radial-
  kernel inputs and values.
* The result cache holds complete spectrum payloads under full model,
  parameter, gauge, sector, observable, numerical, multipole, and requested-
  spectrum identity.

Native failures cross the likelihood boundary as typed errors. A sampled
point outside a scientifically valid parameter domain returns negative
infinity and contributes to rate-limited rejection diagnostics. Unsupported
capabilities, invalid contracts, convergence failures, non-finite evolution,
constraint violations, performance-budget failures, and implementation
faults abort execution. Their diagnostics retain model and parameter values,
gauge, numerical tier, requested spectra, workload, and k or conformal-time
location when available. The nominal model point is evaluated before walker
creation or multiprocessing startup, so an invalid initial state cannot
become an ensemble of rejected proposals.

The pipeline distinguishes declared transfer components from derived angular
spectra. A transfer component supplies a source role and a projection kernel;
an angular spectrum combines two compatible transfers. Thus `PP` is a
lensing-potential spectrum, `TP` and `EP` are cross-surfaces, and lensed
spectra depend on both the unlensed CMB surface and `PP`. A request for one
component does not authorize a substitute source or an external-reference
result.

## Numerical Controls And Runtime Envelope

The native numerical defaults are `ell_min = 2`, `ell_max = 2500`,
`k_min = 1.0e-5`, `k_max = 0.4`, `k_sample_count = 64`, and
`eta_sample_count = 1024`. The photon and massless-neutrino hierarchy caps
default to eight multipoles. Generated scalar evolution uses deterministic
explicit Runge-Kutta substeps shared by every supported scalar gauge. The
substep count follows the declared wave-number phase history, while exact
symmetric collision half-steps absorb collision stiffness without redundant
microsteps after the declared tight-coupling transition. The tight-coupling
entry ratio is `50.0`, and the exit ratio defaults to `0.1`; both are declared
numerical controls. The
`source_grid_multiplier = 2` setting refines the line-of-sight grid. The
optional
`evolution_eta_sample_count` controls the maximum number of conformal-time
samples used by generated hierarchy evolution independently of source-grid
refinement; leaving it undeclared retains the bounded default evolution
grid. `evolution_phase_step` is a positive phase-length target for the
declaration-driven Runge-Kutta schedule on split-collision hierarchies. It
controls integration substep density, not the equations or collision rates,
and is validated as part of the numerical contract. Contracts may also
declare `ode_rtol` and `ode_atol` as
numerical-control metadata; those
values do not select a gauge-specific adaptive trajectory. All values are
declared through `cmb.perturbations.numerics` and are subject to
`cmb.perturbations.accuracy_controls` minimums.

Generated tensor projections reserve a wider fixed k envelope than scalar
requests because spin-2 radial kernels retain an oscillatory high-k tail.
This reserves quadrature coverage without rescaling or replacing a declared
tensor source.

Accuracy controls can require minimum ell, k, eta, hierarchy, source-grid,
and momentum-grid coverage. A declared `runtime_envelope` can also cap
evolution, projection, and total work units. The native runtime validates
those limits before the expensive per-wave-number integration begins. A
momentum-grid declaration supplies the q nodes and weights for a massive or
other momentum-resolved hierarchy; minimum counts are checked against the
accuracy controls before the grid enters the cache.

Split collision operators use the declaration-driven staged integrator by
default. A contract may opt into the continuous stiff collision integrator
with `continuous_collision_solver: true` when all declared collision blocks
are exact or implicit and the hierarchy has no momentum-resolved massive
neutrino family. The option is an explicit numerical control, not a change
to the equations or collision declarations; contracts that omit it retain
the staged route and its gauge-equivalent trajectory behavior.

The `bounded` runtime envelope includes separate native performance acceptance
budgets for cold full-spectrum, warm-parameter, and exact-cache requests. The
reference limits are 180 seconds, 5 seconds, and 1 second respectively. A
contract may state the values explicitly when it needs to make the acceptance
policy visible:

```yaml
accuracy_controls:
  runtime_envelope: bounded
  performance_budget:
    full_spectrum_seconds: 180
    warm_parameter_seconds: 5
    exact_cache_hit_seconds: 1
```

Measured full-spectrum time is checked against the declared cold budget after
output assembly. Production CMB likelihood calls classify each request as
`cold`, `warm`, or `exact`, so structural worker initialization, parameter
rebound, and complete cache reuse are governed independently. The native
performance report records deterministic median and p95 samples for each
workload; a budget overrun raises a typed performance error rather than
publishing a partial or misleading spectrum.

### Ensemble acceptance and resource envelope

The MCMC sampler records an `ensemble_performance` payload for every fit,
including failed fits. It reports total elapsed time, initialization,
burn-in, and production timings, the requested and effective pool sizes,
the CPU-derived worker limit, nominal proposal evaluations, and failed
proposal requests. The effective pool is bounded by
`min(requested_pool, max(cpu_count - 1, 0), n_walkers)`; an unset or unit pool
runs in the parent process. Spawned workers request one numerical thread so
process and BLAS/OpenMP parallelism cannot multiply into host oversubscription.
The payload marks `oversubscribed` and `budget_passed` explicitly and uses the
1800-second end-to-end ensemble acceptance budget.

The governed reference manifest is deterministic: it compares LambdaCDM with
TORG using Union3, compound BAO, and Planck 2018 Lite, seed 0, five burn-in
steps, ten production steps, 32 walkers, and a three-worker pool. The copied
manifest and parameter summary retain this workload identity and its timing
record. Rejected stiff collision rows suppress expected floating-point
overflow warnings; finite-result checks still determine whether a request is
accepted or fails with typed native diagnostics.

Generated scalar contracts first audit the requested k grid against the
declared numerical limits, then preflight every sorted k mode before any ODE
work. The preflight solves the coupled energy, momentum, and shear metric
system and binds its curvature solution into the declared gauge state. Each
initial residual records every signed equation term and uses the sum of their
absolute magnitudes as its dimensionally matched normalization scale. The
initial normalized tolerance is fixed at `0.01` for every supported gauge;
the solver cannot relax it per mode or skip a high-k request. Runtime envelopes
expose this evidence as `scalar_initial_constraint_preflight`, including the
ordered k values, maximum residuals, normalization terms, and provenance.

Generated scalar contracts also validate Einstein energy, momentum, and shear
residuals across the accepted evolution history.
`scalar_constraint_anchors` maps names such as `early`, `recombination`, and
`late` to normalized evolution-grid positions for diagnostics;
`scalar_constraint_normalization` is fixed to
`sum_abs_declared_einstein_terms`. At every eta point, a generated residual
uses the dimensionless measure
`abs(sum(term_i)) / sum(abs(term_i))`; every denominator term has the same
units as its residual. `scalar_constraint_tolerances` therefore sets
normalized residual tolerances by residual name. An explicitly declared
conservation rule retains its own absolute-tolerance semantics, and the
runtime records that distinction as `tolerance_kind` and `tolerance_source`.

Set `scalar_constraint_reference_eta_samples` to the eta-grid size associated
with the normalized tolerances. Under-resolved grids report their residuals
as deferred rather than claiming a physical acceptance verdict, while
explicitly declared conservation rules remain enforced. Before declared
line-of-sight sources are evaluated, generated source histories reconstruct
the observable metric on the coupled Einstein surface. The runtime records
the reconstruction count and largest relative metric correction in
`scalar_constraint_projection`.

Runtime envelopes expose `scalar_constraint_diagnostics` with the full-history
absolute and normalized maxima, eta location, grid fraction, physical regime,
signed normalization terms, normalization scale and source, tolerance
provenance, anchor values, and source-grid and evolution refinement evidence.
Generated state and residual units are checked before projection.

Adaptive refinement is opt-in through `accuracy_controls`. The canonical
sections are `adaptive_transfer`, `adaptive_source`, `adaptive_projection`,
and `adaptive_evolution`:

```yaml
accuracy_controls:
  phase_points_per_cycle: 8
  phase_aware_k_quadrature: true
  adaptive_transfer:
    minimum_nodes: 32
    maximum_nodes: 128
    relative_tolerance: 0.05
    absolute_tolerance: 1.0e-12
  adaptive_source:
    minimum_nodes: 512
    maximum_nodes: 2048
    relative_tolerance: 0.05
    absolute_tolerance: 1.0e-12
  adaptive_projection:
    minimum_nodes: 512
    maximum_nodes: 2048
    relative_tolerance: 0.05
    absolute_tolerance: 1.0e-12
  adaptive_evolution:
    minimum_nodes: 64
    maximum_nodes: 256
    relative_tolerance: 0.01
    absolute_tolerance: 1.0e-12
```

Transfer refinement places nodes from the requested radial phase, acoustic
sound-horizon phase, and declared reference multipoles. Source refinement
subdivides conformal-time intervals according to the largest requested
Fourier phase and the visibility peak and shoulders. Projection refinement
compares the full Simpson line-of-sight result with a coarsened Simpson
surface while using the same exact sector kernel and declared source
histories. Source history interpolation remains the bounded default; a
contract may set `adaptive_source.direct_source_quadrature: true` when it
explicitly budgets re-evolution on the refined k surface. Evolution refinement
runs coarse, intermediate, and reference histories for the same cosmology and
compares both adjacent pairs at early, recombination, and late anchor regions.
The reference verdict uses the intermediate-to-reference comparison; the
runtime envelope retains both comparisons, measured errors, anchor values,
sample counts, and refinement levels for all enabled surfaces.

The native projection request resolves the dependency closure of the selected
`requested_spectra`. It evaluates only the transfer components and source
terms needed by those spectra. An unavailable requested spectrum raises an
explicit availability error before evolution rather than returning an empty
surface or borrowing another sector.

`adaptive_evolution` requires `evolution_eta_sample_count` and a declared
scalar evolution graph. Its node bounds apply to the declared fine history,
and the runtime envelope charges the coarse, intermediate, and reference
integrations. A
strict request raises a named under-resolution error when any physical anchor
fails the declared absolute or relative tolerance; it never substitutes a
grid-size response or an empirical spectrum correction.

The controls are convergence guards, not output corrections. Each enabled
surface compares successive physical approximations and raises a named
under-resolution error when its declared tolerance cannot be met. Set
`fail_on_nonconvergence: false` only for an exploratory request whose runtime
envelope explicitly accepts the reported error. Adaptive work remains bounded
by the declared node and runtime limits; it never replaces unavailable
observables or introduces an empirical spectrum scale.

### Final Convergence Tier

Set `accuracy_tier: final` to request the bounded cross-sector acceptance
envelope. `final` is the only named tier. An unknown tier or an incomplete
bounded runtime envelope fails before background integration. The resolved
envelope records the active sectors, every background and projection
control, hierarchy depths, momentum-grid definitions, runtime limits, and
acceptance thresholds.

When an explicit graph omits a sector registry, the resolved envelope infers
its active sectors from the compiler's observable-sector and variable
tensor-character metadata. This preserves the executed graph identity in
runtime and manifest records without adding undeclared hierarchy families.

The final scalar floor uses photon temperature and polarization depths of
ten and a massless-neutrino depth of seven. The vector floor uses depths
eight, eight, and five. The tensor floor uses depths twelve, twelve, and
nine. An active massive-neutrino hierarchy uses at least seven multipoles
and a q grid with at least 16 nodes over `0.05 <= q <= 15`, using the declared
second-order quadrature. A contract containing more than one sector receives
the strongest applicable floor for each shared hierarchy.

The bounded numerical floor also requires `ell_max >= 2000`, `k_max >= 0.3`,
at least 18 k nodes, at least 192 background eta samples, at least 128
evolution eta samples, a source-grid multiplier of at least two, and a
lensing sampling factor of at least `1.4`. The initial redshift is at least
`2e4`; integration tolerances, phase steps, lower bounds, and tight-coupling
exit controls have explicit upper bounds. These checks reject a request that
labels reduced work as final.

Final refinement uses relative L-infinity errors for auto-spectra and the
correlation coefficient `TE / sqrt(abs(TT * EE))` for normalized `TE`.
Successive refinements must change `TT` and `EE` by less than 1%, normalized
`TE` by less than 2%, `PP` by less than 3%, and lensed `BB` by less than 5%.
Massive-neutrino q refinement must remain below 2%, and every accepted
hierarchy refinement must remain below 1%. Zero crossings remain finite
because the L-infinity metric uses the refined surface peak as its scale.

The acceptance ladder varies background resolution, source-grid density,
fixed k anchors, scalar/vector/tensor hierarchy depths, massive-neutrino q
nodes, and curved-sky lensing sampling independently. Physical k anchors are
retained across production-sized refinements, vector and tensor polarization
terminals use their flat-space free-streaming closures, and clustered
line-of-sight anchors use positive local trapezoid panels when generalized
Simpson weights would become negative.

Run manifests expose the complete result as
`native_cmb_numerical_envelope` and repeat it under the native runtime
summary as `numerical_envelope`. Runtime spectrum payloads carry the same
resolved envelope, the selected tier, and the lensing sampling factor. This
keeps validation output tied to the controls that produced it.

The native LCDM absolute-parity contract uses
`tight_coupling_ratio: 1600.0`. This value keeps the generated scalar route
on the exact split Thomson evolution and declaration-driven fast-manifold
projection for the reference surface. Lower values are
valid for exploratory runs, but they are not sufficient evidence for the
absolute scalar thresholds in `PLAN.md`.

Numerical controls define a reproducible execution envelope, and adaptive
surfaces provide local convergence evidence for k, source histories, and
line-of-sight quadrature. Cross-sector acceptance independently refines the
background, hierarchies, q grid, and lensing quadrature. The
massive-neutrino acceptance surface compares absolute density, pressure,
momentum, and shear source spectra at fixed relativistic and
nonrelativistic cosmologies against direct log-q quadrature; it does not use
a mass response ratio.

## Scalar Absolute Parity

The absolute reference contract uses one fixed native LCDM-family cosmology
and an independently constructed CAMB reference. The native call is made
through the production declared-graph route; the reference helper is confined
to the scientific test module and never calls the production solver. The
comparison is absolute over the declared multipole surfaces rather than a
response ratio or an externally calibrated output.

The acceptance surface includes native `TT`, `TE`, and `EE`, the lensing
potential `PP`, and the declared `TP` and `EP` cross-surfaces. The acceptance
contract also defines acoustic peak locations and `TE` zero crossings.
Auto-spectra use fractional median and 90th-percentile errors with a
reference-relative floor; cross-spectra use RMS errors normalized by the
independent reference RMS so their sign changes and zero crossings remain
well-defined.

The scalar source uses the independent-reference coefficients `5 / 2` and
`15 / 2` for the temperature quadrupole terms and polarization source. The
native `polarization_moment` uses the same normalization as the declared
reference hierarchy, so no post-projection conversion is applied.

Native projection preparation reuses bounded Bessel and projection-kernel
caches. Projection kernels are evaluated in ell batches so a high-multipole
request does not allocate one unbounded ell-by-time work array. The cache
budget is part of runtime behavior, while the absolute parity thresholds
remain the scientific acceptance boundary. Missing radial kernels for
compatible Fourier modes share one batched spherical-Bessel recurrence before
line-of-sight projection. The runtime envelope reports the number of radial
work batches and the number of modes covered by them.

Scalar-only high-ell requests use a streaming projection path. It materializes
one bounded mode batch, projects it, and releases its radial kernels before
the next batch. Vector and tensor kernels are not allocated for a scalar-only
request, while mixed-sector requests retain the complete sector kernel set.
This keeps projection memory proportional to the active mode batch rather
than to every sector and every Fourier mode in the request.

## Tensor Absolute Parity

The tensor acceptance surface uses the fixed reference cosmology with
`r = 0.1`, `nt = 0`, and massless neutrinos. Native tensor evolution and
projection run through the production declared graph with 96 k nodes on the
contiguous `ell = 0..70` remapping surface. Absolute comparisons use the
declared reference multipoles `ell = 40, 50, 70`; they are not tensor-amplitude
responses or synthetic source probes.

The independent test helper constructs CAMB directly and reads unlensed
tensor `TT`, `EE`, and `BB` from `get_tensor_cls`. CAMB 1.6 defines its total
CMB surface as lensed scalar plus tensor. The tensor contribution to that
lensed total is therefore isolated as
`get_total_cls - get_lensed_scalar_cls`. The helper neither imports the native
projection layer nor calls the production CMB facade.

The native lensed comparison remaps tensor-only `TT`, `TE`, `EE`, and `BB`
with the independently evolved native scalar `PP` surface. Every native
unlensed and lensed tensor auto-spectrum has a median fractional error at or
below ten percent against its independent CAMB surface. The accepted
`lensed_BB` remains finite and positive, so exact remapping cannot discard the
declared primordial tensor B-mode.


## Reference Cosmology And Acceptance Boundary

The native absolute-reference checks use a route-neutral declared contract
with the following cosmological inputs: `H0 = 67.4`, `ombh2 = 0.02237`,
`omch2 = 0.12`, `Tcmb = 2.7255 K`, `YHe = 0.245`, `Neff = 3.046`,
`As = 2.1e-9`, `ns = 0.965`, and `tau = 0.054`. CAMB is constructed only
inside the scientific tests; production native modules do not import or call
it.

The accepted background-reference limits are conformal age and sound-horizon
relative error at `0.2%`, visibility-peak redshift error at `0.5%`, visibility
width error at `3%`, recombination median and 90th-percentile electron-fraction
errors at `2%` and `5%`, and reionization optical-depth error at `1%`.
The reference surface also contains independent CAMB fixtures,
projection-kernel limits, and exact-remapper normalization checks. The tensor
fixture establishes the native unlensed and lensed `TT`, `EE`, and `BB`
ten-percent median boundary. Scalar, lensing-potential, massive-neutrino,
gauge, and vector acceptance use their own declared surfaces and thresholds
in `PLAN.md`.

## References
The canonical meanings and target equations above follow the standard
first-order CMB perturbation literature used by CAMB and CLASS:

- Ma and Bertschinger (1995) for scalar Newtonian and synchronous
  perturbation conventions;
- Seljak and Zaldarriaga (1996) for line-of-sight source decomposition;
- Lewis and Challinor (2006) and the CAMB correlation-remapping convention
  for exact curved-sky lensing normalization.
