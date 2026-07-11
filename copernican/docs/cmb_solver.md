# Native CMB Solver Convention
**Last Updated:** 2026-07-11
**Project Version:** 12.0.26

## Overview
This document is the canonical physical convention for Copernican's native
CMB solver path when `cmb.perturbations.standard: false`.

The scalar, vector, and tensor sectors now follow this contract, and later
work must not redefine the meaning of states, source terms, gauge labels, or
public spectra.

The native route uses conformal time `tau`, conformal distance
`chi = eta0 - eta`, and comoving wave number `k` in inverse Mpc. All
dimensionless perturbations are Fourier amplitudes in the same plane-wave
convention used by the declared graph compiler and the native line-of-sight
integrator.

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
- `Phi_tau`, `Psi_tau`
  Conformal-time derivatives of the corresponding potentials.
  Units: `1/Mpc`.
- `theta_s`
  Velocity divergence `theta_s = i k_j v_s^j`. Units: `1/Mpc`.
- `sigma_s`
  Scalar anisotropic stress or quadrupole shear variable. Units:
  dimensionless.

The synchronous convention used for internal history checks in later slices is
the Ma-Bertschinger synchronous form with `h` and `eta`:

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
Later vector and tensor slices may add odd-parity multipoles, but the
dimensionless spin-2 polarization convention is fixed now.

`polarization_moment` means

```text
Pi = Theta_gamma,2 + 6 E_gamma,2
```

and is dimensionless.

`polarization_b_mode_seed` is the declared odd-parity transfer seed carried
through exact lensing when a model declares primordial or sourced `B`.
Units: dimensionless.

### Baryons And Cold Dark Matter
- `delta_b`, `delta_c`
  Density contrasts. Units: dimensionless.
- `theta_b`, `theta_c`
  Velocity divergences. Units: `1/Mpc`.

### Massless Neutrinos
- `delta_nu`
  Density contrast. Units: dimensionless.
- `theta_nu`
  Velocity divergence. Units: `1/Mpc`.
- `sigma_nu`
  Anisotropic stress. Units: dimensionless.
- `nu_l3`, `nu_l4`, ...
  Higher multipoles `F_nu,l`. Units: dimensionless.

### Massive Neutrinos
The canonical massive-neutrino momentum variable is the comoving momentum
label `q`, treated as one resolved momentum-grid coordinate.
The canonical energy is

```text
epsilon(q, a) = sqrt(q^2 + a^2 m_nu^2)
```

and the canonical background distribution is the thermal Fermi-Dirac shape.

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

### Vector States
The canonical vector metric amplitude is `sigma_vector`, the transverse shear
variable propagated by the vector Einstein system. It is dimensionless.

The native matter and radiation vector states are:

- `v_b_vector`, `v_c_vector`
  Baryon and CDM vorticity amplitudes. Units: dimensionless.
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

### Tensor States
The canonical tensor metric amplitude is `h_tensor`, with the explicit
conformal-time derivative `h_tensor_tau = d h_tensor / d eta`.
`h_tensor` is dimensionless and `h_tensor_tau` has units `1/Mpc`.

The native tensor radiation states are:

- `pi_gamma_tensor`, `pi_nu_tensor`
  Photon and massless-neutrino tensor anisotropic stress amplitudes.
  Units: dimensionless.
- `theta_gamma_t3`, `theta_gamma_t4`, ...
  Higher photon tensor temperature multipoles. Units: dimensionless.
- `e_gamma_t2`, `e_gamma_t3`, ...
  Tensor even-parity polarization multipoles. Units: dimensionless.
- `b_gamma_t2`, `b_gamma_t3`, ...
  Tensor odd-parity polarization multipoles. Units: dimensionless.
- `nu_t3`, `nu_t4`, ...
  Higher massless-neutrino tensor multipoles. Units: dimensionless.

The generated tensor route also carries the algebraic source moment
`tensor_polarization_moment = pi_gamma_tensor / 10 + 3 E_gamma,2 / 5`.
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

The generated synchronous route now carries `gauge_shift_alpha`,
`h_sync_metric`, and `eta_sync_metric` as explicit internal histories.
The observable scalar sources stay on the canonical `Phi` / `Psi`
basis, while `Phi_from_synchronous` and `Psi_from_synchronous` record
the explicit reconstructed transform used by the internal-history tests.

The generated gauge-invariant route now compiles through dedicated
`Phi_gi` and `Psi_gi` variables together with observable-basis aliases,
instead of reaching the observable construction surface only by relabeling
the Newtonian branch.

Custom declared graphs may still supply their own gauge bridge when the
standard first-order transform above does not apply, but gauge labels
alone do not satisfy this contract.

## Regular Scalar Initial Modes
The generated scalar route now materializes the following regular scalar
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
and `alpha` from the same observable-basis metric constraint surface.

Before integrating one generated scalar mode, the native runtime now
evaluates the starting Einstein energy, momentum, and shear residuals and
rejects non-finite or out-of-tolerance initial data.

## Regular Tensor Initial Mode
The generated tensor route materializes the regular `tensor_mode` family.
It seeds `h_tensor` from the declared primordial tensor amplitude, seeds
`h_tensor_tau` from the leading `k^2 tau` super-horizon series, and starts the
tensor photon, polarization, and neutrino multipoles from the same regular
physical surface.

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
Einstein inputs as named derived quantities so runtime checks and later
slices reuse one source surface:

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
- `total_shear_source` carries the massless and massive neutrino shear
  terms.
- `einstein_energy_residual`, `einstein_momentum_residual`, and
  `einstein_shear_residual` are the runtime diagnostics for the three
  Einstein equations above.

When the massive hierarchy is active, the generated scalar graph feeds
the metric system through `massive_neutrino_density_source`,
`massive_neutrino_momentum_source`, and
`massive_neutrino_shear_source`. Those source slots now consume the
thermally weighted physical `q` integrals without renaming the public
Einstein-input surface.

Low-k stabilization must stay separate from the physical equations. The
current generated hierarchy therefore keeps
`metric_constraint_scale = k^2 + 3 Hconf^2` as an explicit algebraic
bridge used by `phi_constraint` and the matching `psi_closure`, rather
than folding that bridge back into the canonical Einstein equations
themselves.

### Photon Temperature Hierarchy

```text
Theta_gamma,0' = -k Theta_gamma,1 - Phi'
Theta_gamma,1' = k (Theta_gamma,0 + Psi - 2 Theta_gamma,2) / 3
                 - tau_dot (Theta_gamma,1 - v_b / 3)
Theta_gamma,2' = 2 k Theta_gamma,1 / 5
                 - 3 k Theta_gamma,3 / 5
                 - tau_dot (Theta_gamma,2 - Pi / 10)
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
E_gamma,2' = 2 k E_gamma,1 / 5
             - 3 k E_gamma,3 / 5
             - tau_dot (E_gamma,2 - Pi / 10)
```

Higher `E` multipoles use the same generated free-streaming recurrence family
and the same horizon-aware terminal closure as the temperature hierarchy,
with the corresponding `- tau_dot E_gamma,l` damping kept on every
generated `l >= 3` multipole.

### Matter And Neutrinos
The baryon, CDM, massless-neutrino, and massive-neutrino continuity, Euler,
and hierarchy equations follow the same Ma-Bertschinger sign convention.
Massless-neutrino higher multipoles use the same horizon-aware terminal
closure family as the photon and polarization ladders. Massive-neutrino
density, pressure, momentum, and shear are fixed to thermal physical
`q` integrals with the matching `epsilon` factors and background-moment
normalization.

### Tight Coupling And Collision Splitting
The native scalar integrator now compiles every declared collision operator
into one runtime block before evolution begins.
Each block resolves its target state slots, `rate_expression`, linear
coefficients or matrix entries, counterpart bookkeeping, and the declared
integration strategy.
The generated `thomson_drag` operator remains the built-in exact block, and
its `activation_strategy: tight_coupling` metadata treats the Thomson dipole
and quadrupole sub-block as stiff only while

```text
collision_rate >= k * tight_coupling_ratio
```

and exits that regime once

```text
collision_rate <= 0.1 * k * tight_coupling_ratio
```

Within the active regime, `copernican/lib/perturbation_contract.py` and
`copernican/lib/likelihoods/cmb/native_projection.py` advance that exact
Thomson block analytically from the compiled `exact_form`.
The same exact split damps the generated `l >= 3` temperature and
polarization multipoles while that regime stays active.
Outside the active regime, the same compiled operator falls back to its
ordinary declared expression in the explicit RHS.

Other declared operators may remain `explicit`, or they may opt into one
compiled split block with `integration_strategy: exact` and a declared
`exact_form`, or with `integration_strategy: implicit` and a declared
`linear_block`.
Several compiled collision blocks may run in the same evolution interval, and
the native solver now suppresses only the selected split-operator outputs
instead of zeroing shared symbols such as `collision_rate`.
Unsupported exact or implicit declarations therefore fail before evolution
instead of being silently ignored.

## Line-Of-Sight Source Convention
The canonical scalar source decomposition is:

```text
S_T = g (Theta_gamma,0 + Psi + Pi/4)
    + d/d eta [g v_b / k]
    + exp(-tau) (Psi' - Phi')

S_E = 3 g Pi / 4

S_B = 0 for scalar modes

S_phi = exp(-tau) (Phi + Psi)
```

The generated scalar Doppler source uses the baryon velocity
`v_b = theta_b / k` and projects `g v_b` through the derivative spherical-
Bessel kernel. That derivative-kernel form is the implemented equivalent of
the canonical `d/d eta [g v_b / k]` contribution above.

The canonical vector source decomposition is:

```text
P_V = pi_gamma_vector / 10 + 3 E_gamma,2 / 5
x = k chi

S_T^V = [4 g (v_b_vector + sigma_vector)
       + 15 d/d eta (g P_V) / (2 k)
       + 4 exp(-tau) sigma_vector'] / x

S_E^V = 15 g P_V / x^2 + 15 d/d eta (g P_V) / (2 k x)

S_B^V = -15 g P_V / (2 x)
```

`copernican/lib/perturbation_contract.py` materializes these as
`vector_temperature_source`, `vector_polarization_e_source`, and
`vector_polarization_b_source`. The native line-of-sight projector keeps
the same sign and parity convention across sectors: temperature sources are
even, `E` is even, `B` is odd, and lensing uses the Weyl-potential sum
`Phi + Psi`.

The canonical tensor source decomposition is:

```text
P_T = pi_gamma_tensor / 10 + 3 E_gamma,2 / 5

S_T^T = -exp(-tau) h_tensor' + 2 g P_T

S_E^T = g P_T

S_B^T = g B_gamma,2
```

`copernican/lib/perturbation_contract.py` materializes these as
`tensor_temperature_source`, `tensor_polarization_e_source`, and
`tensor_polarization_b_source`.

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
Native TP or EP = ell (ell + 1) C_ell^{X phi} Tcmb / (2 pi)
```

with units `muK`.

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

Single-sector native routes also expose matching component aliases such as
`scalar_TT`, `vector_BB`, `tensor_BB`, and `total_TT`.
The tensor route reads `r` as the tensor-to-scalar amplitude ratio and `nt`
as the tensor spectral index when constructing the primordial tensor power
law.

Unavailable spectra stay unavailable. Physically zero and unavailable are not
the same state.

## References
The canonical meanings and target equations above follow the standard
first-order CMB perturbation literature used by CAMB and CLASS:

- Ma and Bertschinger (1995) for scalar Newtonian and synchronous
  perturbation conventions;
- Seljak and Zaldarriaga (1996) for line-of-sight source decomposition;
- Lewis and Challinor (2006) and the CAMB correlation-remapping convention
  for exact curved-sky lensing normalization.
