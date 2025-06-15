<!-- DEV NOTE (v1.5e): Removed duplicated bullet line in documentation. -->
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
>
> This document establishes the `.md` format standard for defining cosmological models for the Copernican Suite. This structure is designed to be both human-readable and machine-parsable for generating Python model plugins.
>
> 1.  **YAML Front Matter:** The file must begin with a YAML front matter block (enclosed by `---`). It should contain basic metadata:
>     -   `title`: The full, human-readable name of the model.
>     -   `version`: The version of the model definition.
>     -   `date`: The date of the last update.
>     -   `model_plugin`: The filename of the corresponding Python implementation (e.g., `usmf2.py`).
>
> 2.  **Quantitative Model Specification Section:**
>     -   This section is **critical for machine parsing** and must begin with the heading: `## Quantitative Model Specification for Copernican Suite`.
>     -   It must contain a subsection titled `### Model Parameters`.
>     -   This subsection must contain a Markdown table with the following exact headers: `Parameter Name`, `Python Variable`, `Initial Guess`, `Bounds`, `Unit`, `LaTeX Name`.
>     -   This table's content will be parsed to automatically generate the `PARAMETER_NAMES`, `INITIAL_GUESSES`, `PARAMETER_BOUNDS`, and `PARAMETER_LATEX_NAMES` lists in the Python plugin script.
>
> 3.  **Theoretical Framework Section:**
>     -   The remainder of the document contains the detailed theoretical write-up of the model. It should be formatted using standard Markdown headings, lists, and LaTeX for equations. This section is intended for human readers and for providing context.


