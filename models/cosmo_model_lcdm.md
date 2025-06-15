<!-- DEV NOTE (v1.5e): Split LCDM into two-file format using lcdm.py -->
<!-- DEV NOTE (v1.5e): Removed duplicated bullet line in documentation. -->
---
title: "Lambda Cold Dark Matter (\u039bCDM) Reference Model"
version: "1.0"
date: "2025-06-13"
model_plugin: "lcdm.py"
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


