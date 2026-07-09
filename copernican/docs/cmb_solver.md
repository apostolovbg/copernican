# Native CMB Solver Convention
**Last Updated:** 2026-07-09
**Project Version:** 12.0.26

## Overview
This document is the canonical physical convention for Copernican's native
CMB solver path when `cmb.perturbations.standard: false`.

Later roadmap slices may replace incomplete equations, closures, or numerical
approximations, but they must not redefine the meaning of states, source
terms, gauge labels, or public spectra. This page fixes that contract first.

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

All q-resolved hierarchy amplitudes are dimensionless except the velocity
divergence members, which carry `1/Mpc`.

The aggregate names
`delta_nu_massive`, `theta_nu_massive`, and `sigma_nu_massive`
are reserved for q-integrated physical moments built from the same resolved
hierarchy. They must not become an independently drifting evolution path.

### Vector And Tensor Roles
The later physical vector and tensor slices will use the same basic units:

- vector metric and matter amplitudes are dimensionless unless they are
  velocity divergences, in which case they use `1/Mpc`;
- tensor metric-wave amplitudes are dimensionless;
- vector and tensor temperature, `E`, and `B` multipoles are dimensionless.

Odd-parity transfer content remains odd under parity and even-parity transfer
content remains even under parity.

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

Slice Six must implement either these explicit transformations or an exactly
equivalent gauge-invariant route. Gauge labels alone do not satisfy this
contract.

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
`massive_neutrino_shear_source`. Slice Four will replace their current
intermediate mapping with the final physical `q` integration without
renaming those source slots.

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
physical high-l closure chosen later by Slice Three.

### Polarization Hierarchy

```text
E_gamma,2' = 3 k E_gamma,3 / 5 - tau_dot (E_gamma,2 - Pi / 10)
```

Higher `E` multipoles use the matching spin-2 free-streaming recurrence with
the same closure family as the temperature hierarchy.

### Matter And Neutrinos
The baryon, CDM, massless-neutrino, and massive-neutrino continuity, Euler,
and hierarchy equations follow the same Ma-Bertschinger sign convention.
Massive-neutrino metric moments are fixed to physical q integrals with the
appropriate `q`, `epsilon`, density, pressure, momentum, and shear weights.

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

Later vector and tensor slices must use the same sign and parity convention:
temperature sources are even, `E` is even, `B` is odd, and lensing uses the
Weyl-potential sum `Phi + Psi`.

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

For `PP`, the public solver returns the exact `clpp` normalization consumed
by the native curved-sky remapper:

```text
PP = clpp = [ell (ell + 1)]^2 C_ell^{phiphi} / (2 pi)
```

`PP` is dimensionless.

`lensed_TT`, `lensed_TE`, `lensed_EE`, and `lensed_BB` stay in the same
`muK^2` `D_ell` convention as their unlensed counterparts.

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
