# Native CMB Solver Convention
**Last Updated:** 2026-07-29
**Project Version:** 12.0.26

## Overview
This document is the canonical physical convention for Copernican's native
CMB solver path. Every bundled production CMB model uses
`cmb.perturbations.standard: false` and the declared native graph.

The scalar, vector, and tensor sectors follow this contract. Implementations
must preserve the meaning of states, source terms, gauge labels, and public
spectra defined here. A model may declare a native contract while marking CMB
output unavailable when its theory has no defensible linear perturbation
closure.

The native route uses conformal time `tau`, conformal distance
`chi = eta0 - eta`, and comoving wave number `k` in inverse Mpc. All
dimensionless perturbations are Fourier amplitudes in the same plane-wave
convention used by the declared graph compiler and the native line-of-sight
integrator.

## Native Model Declarations
`copernican/models/model_lcdm_ccmbs.yml` is the production native LambdaCDM
declaration. It defines the background and recombination inputs, scalar
species and hierarchy families, Thomson coupling, adiabatic initial data,
projection typing, and a bounded numerical envelope. The model compiles its
generated scalar hierarchy through the same native runtime described below.

Its perturbation contract uses `standard: false` and marks the `camb` mapping
as `native_solver_required`. The `backend: camb` value names the adapter
mapping in the model schema; it does not select a production backend. The
production output is the native declared-graph result and does not call CAMB
or CLASS.

Available bundled CMB models use the same contract shape. Their model files
preserve theory-specific parameters, priors, distance equations, sound-
horizon expressions, and declared background functions. Each file declares a
scalar sector, its physical species, hierarchy families, Thomson coupling,
conservation rule, regular adiabatic initial family, projection typing, and
native mapping. The compiler materializes the common scalar hierarchy from
this metadata, so model-specific background expressions feed one solver
without a standard-backend branch. USMF declares its physical species and
native contract but marks CMB output unavailable until its shrinking-matter
perturbation closure is specified.

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
`theta_nu_massive_q<i>` members are momentum-bin dipoles, and the
physical velocity-divergence source only appears after the matching
q-integrated momentum weight multiplies them by `k`.

The aggregate names
`delta_nu_massive`, `theta_nu_massive`, `sigma_nu_massive`,
and `nu_massive_l<j>` are reserved for strict q-integrated aliases built
from the same resolved hierarchy. They must not become an independently
drifting evolution path.

The runtime also exports physical density, pressure, momentum, and shear
fractions for each q family. These fractions follow the background
`Omega_nu(a)` scaling and feed the Einstein source with the required `a^2`
factor; the momentum source carries the relativistic `4/3` inertial factor.

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

### Tensor States
The canonical tensor metric amplitude is `h_tensor`, with the explicit
conformal-time derivative `h_tensor_tau = d h_tensor / d eta`.
`h_tensor` is dimensionless and `h_tensor_tau` has units `1/Mpc`.

The native tensor radiation states are:

- `delta_gamma_tensor`, `theta_gamma_tensor`, `pi_gamma_tensor`
  Photon tensor temperature monopole, dipole, and shear. Units:
  dimensionless, dimensionless, and dimensionless respectively.
- `delta_nu_tensor`, `theta_nu_tensor`, `pi_nu_tensor`
  Massless-neutrino tensor monopole, dipole, and shear. Units:
  dimensionless.
- `theta_gamma_t3`, `theta_gamma_t4`, ...
  Higher photon tensor temperature multipoles. Units: dimensionless.
- `e_gamma_t0`, `e_gamma_t1`, `e_gamma_t2`, ...
  Tensor even-parity polarization multipoles. Units: dimensionless.
- `b_gamma_t2`, `b_gamma_t3`, ...
  Tensor odd-parity polarization multipoles. Units: dimensionless.
- `nu_t3`, `nu_t4`, ...
  Higher massless-neutrino tensor multipoles. Units: dimensionless.

The generated tensor route also carries the full Thomson polarization source
moment
`-(delta_gamma_tensor / 10 + 2 pi_gamma_tensor / 7 +
3 theta_gamma_t4 / 70 - 3 e_gamma_t0 / 5 + 6 e_gamma_t2 / 7 -
3 e_gamma_t4 / 70) / sqrt(6)`.
Odd-parity transfer content remains odd and even-parity transfer content
remains even.

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
the free-streaming neutrino fraction, and seeds the neutrino tensor monopole
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
vector temperature recurrence uses the physical `l/(l-1)` closure, while
terminal vector polarization multipoles are held at zero as in the reference
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

For `PP`, the public solver returns the exact `clpp` normalization consumed
by the native curved-sky remapper:

```text
PP = clpp = [ell (ell + 1)]^2 C_ell^{phiphi} / (2 pi)
```

`PP` is dimensionless.

`lensed_TT`, `lensed_TE`, `lensed_EE`, and `lensed_BB` stay in the same
`muK^2` `D_ell` convention as their unlensed counterparts.

Native projection quadrature starts from the declared numerical ell range and
reference multipoles rather than from the subset requested by one caller.
This keeps a spectrum's k integration stable when callers request a sparse
surface such as `TT` at selected multipoles or request the complete lensed
family. Contracts that need a controlled accuracy tier can enable the
adaptive transfer, source, and line-of-sight projection surfaces described
below.

For high-resolution scalar requests, the fixed k envelope combines a
logarithmic low-k segment with a linear high-k segment. The low-k segment
resolves reionization and the first acoustic scales; the linear segment
resolves the rapidly oscillating high-ell radial kernels. The declared node
count remains exact. Reference-ell anchors are retained and the remaining
nodes are inserted into the widest physical k gaps.

Single-sector native routes also expose matching component aliases such as
`scalar_TT`, `vector_BB`, `tensor_BB`, and `total_TT`.
The tensor route reads `r` as the tensor-to-scalar amplitude ratio and `nt`
as the tensor spectral index when constructing the primordial tensor power
law.

Unavailable spectra stay unavailable. Physically zero and unavailable are not
the same state.

## Native Execution Pipeline

The native route is a declared-graph execution path with five physical
stages:

1. `copernican/lib/likelihoods/cmb/cmb.py` selects native execution for
   `standard: false` and validates the contract before any spectrum is built.
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
   outputs are requested. `native_cache.py` stores only results keyed by the
   declared contract and numerical controls.

The runtime has three cache layers. Contract-static work contains the
compiled graph, dependency closure, state-slot layout, collision plan, and
momentum-grid structure. Cosmology-static work contains the background,
recombination, visibility, coordinate-rate, and collision histories for one
bound cosmology. Request-specific work contains the selected multipoles,
source histories, transfer matrices, and public spectrum surfaces. The
`NativeRuntimeCacheIdentity` records those three key portions separately, so a
new multipole request does not invalidate structural or cosmology-static
entries.

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

The envelope also records wall time for `compilation`, `background`,
`preparation`, `evolution`, `projection`, and `power_spectrum`. Aggregate
phase totals, including the separate `lensing` phase, are available through
`native_cmb_performance_stats()`. Cache-hit fields identify whether the graph,
background, and projection-kernel layers were reused for a fresh request.
An identical request returns the bounded spectrum cache directly and records a
cache hit without repeating static or request-specific work.

The pipeline distinguishes declared transfer components from derived angular
spectra. A transfer component supplies a source role and a projection kernel;
an angular spectrum combines two compatible transfers. Thus `PP` is a
lensing-potential spectrum, `TP` and `EP` are cross-surfaces, and lensed
spectra depend on both the unlensed CMB surface and `PP`. A request for one
component does not authorize a substitute source or a standard-backend result.

## Numerical Controls And Runtime Envelope

The native numerical defaults are `ell_min = 2`, `ell_max = 2500`,
`k_min = 1.0e-5`, `k_max = 0.4`, `k_sample_count = 64`, and
`eta_sample_count = 1024`. The photon and massless-neutrino hierarchy caps
default to eight multipoles. Generated scalar evolution uses deterministic
explicit Runge-Kutta substeps shared by every supported scalar gauge. The
substep count follows the declared wave-number phase and collision-rate
histories, so gauge-equivalent routes do not select different numerical
trajectories. The tight-coupling entry ratio is `50.0`, and the exit ratio
defaults to `0.1`; both are declared numerical controls. The
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

The `bounded` runtime envelope includes the native performance acceptance
budgets: 180 seconds for a full native spectrum and 60 seconds for the joint
MCMC smoke workload. A contract may state the values explicitly when it needs
to make the acceptance policy visible:

```yaml
accuracy_controls:
  runtime_envelope: bounded
  performance_budget:
    full_spectrum_seconds: 180
    joint_mcmc_seconds: 60
```

Measured full-spectrum time is checked against the declared full-spectrum
budget after projection. The joint-MCMC limit is applied by its bounded smoke
benchmark because that workload spans multiple likelihood evaluations. A
budget overrun raises a named performance error rather than publishing a
partial or misleading spectrum.

Generated scalar contracts validate Einstein energy, momentum, and shear
residuals across the accepted evolution history. `scalar_constraint_anchors`
maps names such as `early`, `recombination`, and `late` to normalized
evolution-grid positions for diagnostics; `scalar_constraint_tolerances`
sets residual tolerances by residual name. Set
`scalar_constraint_reference_eta_samples` to the eta-grid size associated
with those tolerances. Under-resolved grids report diagnostics without
claiming reference-resolution acceptance, while explicitly declared
conservation rules remain enforced. Runtime envelopes expose
`scalar_constraint_diagnostics` with full-history maxima and anchor values.
Generated state and residual units are checked before projection.

Adaptive refinement is opt-in through `accuracy_controls`. The canonical
sections are `adaptive_transfer`, `adaptive_source`, `adaptive_projection`,
and `adaptive_evolution`:

```yaml
accuracy_controls:
  phase_points_per_cycle: 8
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
histories. Evolution refinement
re-runs declared scalar state and source histories at half the declared
evolution sample count, then compares finite values at early, recombination,
and late anchor regions. The runtime envelope records the measured errors,
anchor values, sample counts, and refinement levels for all enabled surfaces.

The native projection request resolves the dependency closure of the selected
`requested_spectra`. It evaluates only the transfer components and source
terms needed by those spectra. An unavailable requested spectrum raises an
explicit availability error before evolution rather than returning an empty
surface or borrowing another sector.

`adaptive_evolution` requires `evolution_eta_sample_count` and a declared
scalar evolution graph. Its node bounds apply to the declared fine history,
and the runtime envelope charges both the fine and coarse integrations. A
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

The native LCDM absolute-parity contract uses
`tight_coupling_ratio: 1600.0`. This value keeps the generated scalar route
on the exact split Thomson evolution and declaration-driven fast-manifold
projection for the reference surface. Lower values are
valid for exploratory runs, but they are not sufficient evidence for the
absolute scalar thresholds in `PLAN.md`.

Numerical controls define a reproducible execution envelope, and the adaptive
surfaces provide local convergence evidence for k, source histories, and
line-of-sight quadrature. Scientific parity still requires controlled changes
to the background, hierarchy, q-grid, and lensing resolutions with stable
observables in the later acceptance slices.

## Scalar Absolute Parity

The absolute reference contract uses one fixed native LCDM-family cosmology
and an independently constructed CAMB reference. The native call is made
through the production declared-graph route; the reference helper is confined
to the scientific test module and never calls the production solver. The
comparison is absolute over the declared multipole surfaces rather than a
response ratio or a calibrated standard-backend output.

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


## Reference Cosmology And Acceptance Boundary

The native absolute-reference checks use a neutral `standard: false` contract
with the following cosmological inputs: `H0 = 67.4`, `ombh2 = 0.02237`,
`omch2 = 0.12`, `Tcmb = 2.7255 K`, `YHe = 0.245`, `Neff = 3.046`,
`As = 2.1e-9`, `ns = 0.965`, and `tau = 0.054`. CAMB is constructed only
inside the scientific tests; production native modules do not import or call
it.

The accepted background-reference limits are conformal age and sound-horizon
relative error at `0.2%`, visibility-peak redshift error at `0.5%`, visibility
width error at `3%`, recombination median and 90th-percentile electron-fraction
errors at `2%` and `5%`, and reionization optical-depth error at `1%`.
The implemented reference surface also contains independent CAMB fixtures,
projection-kernel limits, exact-remapper normalization checks, and narrow
native tensor anchor checks. Those checks do not establish the full native
scalar, lensing, massive-neutrino, gauge, or vector absolute-parity thresholds
listed in PLAN.md. Numerical convergence of native outputs is a separate
Slice Nineteen requirement.

## References
The canonical meanings and target equations above follow the standard
first-order CMB perturbation literature used by CAMB and CLASS:

- Ma and Bertschinger (1995) for scalar Newtonian and synchronous
  perturbation conventions;
- Seljak and Zaldarriaga (1996) for line-of-sight source decomposition;
- Lewis and Challinor (2006) and the CAMB correlation-remapping convention
  for exact curved-sky lensing normalization.
