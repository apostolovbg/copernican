---
title: "Lambda Cold Dark Matter (\u039bCDM) Reference Model"
version: "1.0"
date: "2025-06-13"
model_plugin: "lcdm_model.py"
---

# Lambda Cold Dark Matter (\u039bCDM)

## Abstract
The \u039bCDM model describes a flat universe dominated by cold dark matter
and a cosmological constant. It serves as the reference model for the
Copernican Suite.

---

## Quantitative Model Specification for Copernican Suite

### Key Equations
**For Supernovae (SNe Ia) and BAO Fitting:**
$$H(z) = H_0 \sqrt{\Omega_{m0}(1+z)^3 + \Omega_{\Lambda0}}$$
$$d_L(z) = (1+z) \int_0^z \frac{c}{H(z')} dz'$$
$$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$$

### Model Parameters
| Parameter Name | Python Variable | Initial Guess | Bounds | Unit | LaTeX Name |
| :--- | :--- | :--- | :--- | :--- | :--- |
| H0 | `H0` | 67.7 | (50.0, 100.0) | km/s/Mpc | `$H_0$` |
| Omega_m0 | `Omega_m0` | 0.31 | (0.05, 0.7) | | `$\Omega_{m0}$` |
| Omega_b0 | `Omega_b0` | 0.0486 | (0.01, 0.1) | | `$\Omega_{b0}$` |

### Example Fit Results
- Dataset: UniStra_FixedNuis_h1
- Best-fit $\chi^2$: 1700

> ### **Internal Formatting Guide for Model Definition Files**
> 1. Begin with YAML front matter containing `title`, `version`, `date`, and
>    `model_plugin`.
> 2. Provide a section titled `## Quantitative Model Specification for Copernican Suite`.
>    Include a `### Model Parameters` table with headers `Parameter Name`, `Python Variable`,
>    `Initial Guess`, `Bounds`, `Unit`, `LaTeX Name`.
> 3. Additional theory and discussion may follow using standard Markdown.
