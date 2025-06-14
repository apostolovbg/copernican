<!-- DEV NOTE (v1.4.2): Removed duplicated bullet line in documentation. -->
---
title: "The Unified Shrinking Matter Framework (USMF) Version 3b - Kinematic"
version: "3.0b"
date: "2025-06-11"
model_plugin: "usmf3b.py"
---

# The Unified Shrinking Matter Framework (USMF) Version 3b "Kinematic"

## Abstract

This paper details the "Kinematic" version of the Unified Shrinking Matter Framework (USMF), a cosmological model that posits the apparent accelerated expansion of the Universe is an observational effect arising from matter shrinking over cosmic time. USMFv3b replaces the complex, computationally expensive formulation of previous versions with a simple, elegant power-law describing the shrinking process. This "kinematic law" is defined by a single index, `$p_{kin}$`, and leads to fully analytic solutions for cosmological distance measures, dramatically increasing computational speed and model robustness. By coupling this late-time kinematic model with a standard, physically-motivated calculation for the sound horizon (`$r_s$`) at the drag epoch, USMFv3b provides a consistent and powerful framework for testing against both Type Ia Supernovae (SNe Ia) and Baryon Acoustic Oscillation (BAO) data.

---

## Quantitative Model Specification for Copernican Suite

This section provides the specific data required by the `cosmo_engine.py` to generate and fit the model plugin.

### Key Equations

**For Supernovae (SNe Ia) Fitting:**
$$\alpha(t) = \left( \frac{t_0}{t} \right)^{p_{kin}}$$
$$1+z = \alpha(t_e) \implies t_e(z) = \frac{t_0}{(1+z)^{1/p_{kin}}}$$
$$r(z) = \frac{c \cdot t_0}{p_{kin}+1} \left[ 1 - (1+z)^{-\frac{p_{kin}+1}{p_{kin}}} \right]$$
$$d_L(z) = |r(z)| \cdot (1+z)^2 \cdot \frac{70.0}{H_A}$$
$$\mu = 5 \log_{10}(d_L) + 25$$

**For Baryon Acoustic Oscillation (BAO) Analysis:**
$$D_A(z) = |r(z)| \cdot \frac{70.0}{H_A}$$
$$H_{USMF}(z) = \frac{p_{kin}}{t_0} (1+z)^{1/p_{kin}}$$
$$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$$
$$r_s \approx \frac{55.154 \exp[-72.3(\Omega_{m0}h^2 - 0.14)^2]}{\sqrt{(\Omega_{m0}h^2)^{0.25351}(\Omega_{b0}h^2)^{0.12807}}} \quad (\text{Eisenstein & Hu 1998})$$


### Model Parameters

The following table defines the cosmological parameters for the USMF V3b model.

| Parameter Name | Python Variable | Initial Guess | Bounds | Unit | LaTeX Name |
| :--- | :--- | :--- | :--- | :--- | :--- |
| H_A | `H_A` | 70.0 | (50.0, 100.0) | | `$H_A$` |
| t0_age_Gyr | `t0_age_Gyr` | 14.0 | (10.0, 20.0) | Gyr | `$t_{0,age}$` |
| p_kin | `p_kin` | 0.8 | (0.1, 2.0) | | `$p_{kin}$` |
| Omega_m0_fid | `Omega_m0_fid` | 0.31 | (0.2, 0.4) | | `$\Omega_{m0,fid}$` |
| Omega_b0_fid | `Omega_b0_fid` | 0.0486| (0.03, 0.07)| | `$\Omega_{b0,fid}$` |
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


