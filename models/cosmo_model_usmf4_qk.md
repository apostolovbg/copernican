---
title: "The Unified Shrinking Matter Framework (USMF) Version 4 - Quantum Kinematic"
version: "4.0"
date: "2025-06-11"
model_plugin: "usmf4_qk.py"
---

# The Unified Shrinking Matter Framework (USMF) Version 4 "Quantum Kinematic"

## Abstract

This paper introduces the "Quantum Kinematic" version of the Unified Shrinking Matter Framework (USMFv4), a cosmological model that resolves the observational inconsistencies of its predecessor (USMFv3b). The model is founded on a new physical postulate: the rest mass of fundamental particles evolves in inverse proportion to the cosmic scale factor, $m(t) \propto 1/a(t)$. This leads to a matter-energy density evolution of $\rho_m(z) \propto (1+z)^4$. This simple change has profound implications, creating a model with a direct conceptual link to quantum properties of matter, while yielding a new cosmic expansion history, $H(z) = H_0 \sqrt{\Omega_{m0}(1+z)^4 + \Omega_{\Lambda0}}$. USMFv4 is computationally robust, providing excellent fits to both Type Ia Supernovae (SNe Ia) and Baryon Acoustic Oscillation (BAO) data, and remains consistent with Newtonian mechanics in the low-redshift limit.

---

## Quantitative Model Specification for Copernican Suite

This section provides the specific data required by the `cosmo_engine.py` to generate and fit the model plugin.

### Key Equations

**For Supernovae (SNe Ia) and Baryon Acoustic Oscillation (BAO) Fitting:**
$$H(z) = H_0 \sqrt{\Omega_{m0}(1+z)^4 + \Omega_{\Lambda0}}$$
$$\Omega_{\Lambda0} = 1 - \Omega_{m0}$$
$$d_L(z) = (1+z) \int_0^z \frac{c}{H(z')} dz'$$
$$\mu = 5 \log_{10}(d_L/1\,\mathrm{Mpc}) + 25$$
$$D_A(z) = \frac{1}{1+z} \int_0^z \frac{c}{H(z')} dz'$$
$$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$$

### Model Parameters

This table provides the parameter specification in the format required by the Copernican Suite.

| Parameter Name | Python Variable | Initial Guess | Bounds | Unit | LaTeX Name |
| :--- | :--- | :--- | :--- | :--- | :--- |
| H0 | `H0` | 68.0 | (50, 80) | km/s/Mpc | `$H_0$` |
| Omega_m0 | `Omega_m0` | 0.3 | (0.1, 0.5) | | `$\Omega_{m0}$` |
| Omega_b0 | `Omega_b0` | 0.0486 | (0.04, 0.06)| | `$\Omega_{b0}$` || Omega_b0 | `Omega_b0` | 0.0486 | (0.04, 0.06)| | `$\Omega_{b0}$` |

> ### **Internal Formatting Guide for Model Definition Files**
> 1. Begin with YAML front matter containing `title`, `version`, `date`, and
>    `model_plugin`.
> 2. Provide a section titled `## Quantitative Model Specification for Copernican Suite`.
>    Include a `### Model Parameters` table with headers `Parameter Name`, `Python Variable`,
>    `Initial Guess`, `Bounds`, `Unit`, `LaTeX Name`.
> 3. Additional theory and discussion may follow using standard Markdown.

| Omega_b0 | `Omega_b0` | 0.0486 | (0.04, 0.06)| | `$\Omega_{b0}$` |
