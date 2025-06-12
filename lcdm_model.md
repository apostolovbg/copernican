---
title: "The Standard Flat Lambda-CDM Cosmological Model"
version: "1.4rc"
date: "2025-06-12"
model_plugin: "lcdm_model.py"
---

# The Standard Flat Lambda-CDM (ΛCDM) Model

## Abstract

The Lambda-CDM (ΛCDM) or Lambda Cold Dark Matter model is the standard model of Big Bang cosmology. It serves as the concordance model because it successfully explains a wide range of cosmological observations, including the cosmic microwave background (CMB), the large-scale structure of galaxies, and the observed accelerating expansion of the universe. This plugin implements the "flat" ΛCDM model, which assumes the universe has zero spatial curvature, meaning that the total energy density is equal to the critical density. The model is defined by a cosmological constant (Λ) representing dark energy and Cold Dark Matter (CDM), in addition to ordinary baryonic matter.

---

## Quantitative Model Specification for Copernican Suite

This section provides the specific data and equations required by the Copernican Suite's computational engine to utilize this model.

### Key Equations

The theoretical predictions of the ΛCDM model are derived from the Friedmann equations.

**For Supernovae (SNe Ia) Fitting:**

The primary observable is the distance modulus, $\mu$, which depends on the luminosity distance, $d_L$. This, in turn, depends on the expansion history of the universe, $H(z)$.

$$H(z) = H_0 \sqrt{\Omega_{m0}(1+z)^3 + \Omega_{\Lambda0}}$$
$$d_L(z) = (1+z) \int_0^z \frac{c}{H(z')} dz'$$
$$\mu = 5 \log_{10}(d_L/1\,\mathrm{Mpc}) + 25$$

**For Baryon Acoustic Oscillation (BAO) Analysis:**

BAO analysis relies on standard rulers, which are characterized by the angular diameter distance, $D_A(z)$, the Hubble parameter, $H(z)$, and the volume-averaged distance, $D_V(z)$.

$$D_A(z) = \frac{1}{1+z} \int_0^z \frac{c}{H(z')} dz'$$
$$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$$

### Model Parameters

The model includes both cosmological parameters that define the universe and nuisance parameters related to the observational data.

| Parameter    | LaTeX Symbol        | Role          | Description                                 |
| :----------- | :------------------ | :------------ | :------------------------------------------ |
| `H0`         | $H_0$               | Cosmological  | Hubble Constant at z=0                      |
| `Omega_m0`   | $\Omega_{m0}$       | Cosmological  | Matter Density Parameter at z=0             |
| `Omega_b0`   | $\Omega_{b0}$       | Cosmological  | Baryon Density Parameter at z=0             |
| `M`          | $M$                 | Nuisance      | Absolute Magnitude of a Type Ia Supernova   |
| `alpha`      | $\alpha$            | Nuisance      | SNe Light Curve Stretch Nuisance Parameter  |
| `beta`       | $\beta$             | Nuisance      | SNe Light Curve Color Nuisance Parameter    |

### Fixed Constants

These physical constants are used in the model calculations and are held fixed.

| Constant              | Description                           | Value         |
| :-------------------- | :------------------------------------ | :------------ |
| `C_LIGHT_KM_S`        | Speed of Light in km/s                | 299792.458    |
| `T_CMB0_K`            | CMB Temperature at z=0 in Kelvin      | 2.7255        |
| `OMEGA_G_H2`          | Photon Density Parameter ($Ω_γ h^2$)    | 2.472e-5      |
| `NEUTRINO_MASS_eV`    | Sum of Neutrino Masses in eV          | 0.06          |