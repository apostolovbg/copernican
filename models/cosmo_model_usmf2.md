<!-- DEV NOTE (v1.5a): Removed duplicated bullet line in documentation. -->
---
title: "The Unified Shrinking Matter Framework (USMF) Version 2"
version: "2.0"
date: "2025-05-26"
model_plugin: "usmf2.py"
---

# The Unified Shrinking Matter Framework (USMF) Version 2

## Abstract

The Unified Shrinking Matter Framework (USMF) is a cosmological model proposing that the observed accelerated expansion of the Universe and phenomena attributed to dark matter and dark energy are apparent effects. These arise from observations made from within dense cosmic structures (like galaxies) that are themselves shrinking over cosmic time relative to a static universal coordinate system. This paper details the core concepts of USMF, its refined mathematical formulation including considerations for early universe evolution, the mechanisms for inhomogeneous shrinking, its implications for key cosmological observations such as Type Ia supernovae, Big Bang Nucleosynthesis, Baryon Acoustic Oscillations, and the matter-antimatter asymmetry. The framework suggests a dynamic evolution of fundamental scales and the effective gravitational constant ($G_{univ}(t)$), proposing specific modifications to gravitational field equations and offering novel explanations and unique testable predictions for cosmic puzzles without invoking new undiscovered particles or energy fields.

---

## Quantitative Model Specification for Copernican Suite

This section provides the specific data required by the `cosmo_engine.py` to generate and fit the model plugin.

### Key Equations

**For Supernovae (SNe Ia) Fitting:**
$$1+z = \frac{\alpha(t_e)}{\alpha(t_0)}$$
$$r = \int_{t_e}^{t_0} \frac{c}{\alpha(t')} dt'$$
$$d_L(z) = |r| \cdot (1+z)^2 \cdot \frac{70.0}{H_A}$$
$$\mu = 5 \log_{10}(d_L) + 25$$

**For Baryon Acoustic Oscillation (BAO) Analysis:**
$$D_A(z) = |r| \cdot \frac{70.0}{H_A}$$
$$H_{USMF}(z) = - \frac{1}{\alpha(t_e)} \left. \frac{d\alpha}{dt} \right|_{t_e}$$
$$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$$

### Model Parameters

The following table defines the cosmological parameters for the USMF V2 model. This table is the single source of truth for generating the metadata block in the associated `model_plugin` file.

| Parameter Name | Python Variable | Initial Guess | Bounds        | Unit                 | LaTeX Name         |
| :------------- | :-------------- | :------------ | :------------ | :------------------- | :----------------- |
| H_A            | `H_A`           | 77.111        | (50.0, 100.0) |                      | $H_A$              |
| p_alpha        | `p_alpha`       | 0.4313        | (0.1, 1.5)    |                      | $p_{\alpha}$       |
| k_exp          | `k_exp`         | -0.3268       | (-2.0, 2.0)   |                      | $k_{exp}$          |
| s_exp          | `s_exp`         | 1.1038        | (0.5, 2.0)    |                      | $s_{exp}$          |
| t0_age_Gyr     | `t0_age_Gyr`    | 13.397        | (10.0, 20.0)  | Gyr                  | $t_{0,age}$        |
| A_osc          | `A_osc`         | 0.0027088     | (0.0, 0.05)   |                      | $A_{osc}$          |
| omega_osc      | `omega_osc`     | 2.3969        | (0.1, 10.0)   | rad/log(time_ratio)  | $\omega_{osc}$     |
| ti_osc_Gyr     | `ti_osc_Gyr`    | 7.1399        | (1.0, 20.0)   | Gyr                  | $t_{i,osc}$        |
| phi_osc        | `phi_osc`       | 0.10905       | (-$\pi$, $\pi$) | rad                  | $\phi_{osc}$       |

### Example Fit Results

-   **Dataset:** UniStra_FixedNuis_h1
-   **Best-fit $\chi^2$:** 1662.54

---

## Theoretical Framework and Elaboration

### 1. Introduction

Modern cosmology, while remarkably successful, faces fundamental challenges centered around the concepts of dark energy and dark matter, hypothetical entities comprising ~95% of the Universe's energy density. The observed accelerated expansion of the Universe [1, 2] and anomalous galactic and cluster dynamics [3, 4] are the primary evidence for their existence. The Unified Shrinking Matter Framework (USMF) offers an alternative perspective, positing that these phenomena are not due to new fundamental entities but are rather illusions. These illusions arise from a continuous, inhomogeneous shrinking of matter and physical scales within dense regions of the Universe, when observations are made with locally shrinking instruments and interpreted against a presumed static background.

This framework aims to explain the apparent cosmic acceleration and the effects attributed to dark matter by re-evaluating the nature of our reference frames and the evolution of physical scales over cosmic time. This updated document (Version 2) incorporates further elaborations on the model's theoretical underpinnings, mathematical extensions for early universe compatibility, and more detailed unique predictions.

### 2. Core Concepts of the Unified Shrinking Matter Framework

USMF is built upon several foundational ideas that distinguish it from standard cosmological models.

#### 2.1. Dual Reference Frames

A central tenet of USMF is the existence of two distinct types of reference frames:
* A **Universal Coordinate System** ($X, Y, Z, T_{univ}$): This is a static, non-shrinking, four-dimensional Cartesian framework representing the Universe at its largest scales. Universal time is denoted as $t_{univ}$ or simply $t$.
* **Local Coordinate Systems** ($x, y, z, t_{local}$): These are tied to dense regions of matter, such as galaxies and galaxy clusters. Within these local frames, spacetime itself, along with all matter, energy, and measuring instruments, undergoes a process of shrinking over cosmic time relative to the universal coordinates.

Observations made from Earth are inherently from within such a local, shrinking frame.

#### 2.2. Inhomogeneous Shrinking and its Quantification

The process of shrinking is not uniform throughout the Universe. It is posited to occur primarily within regions of significant matter density. Cosmic voids, being largely devoid of matter, are assumed to retain their original scales (established at a "Universal Creation Event" or UDE) in the universal coordinate system. As dense regions contract, voids effectively appear to expand or "empty out" relative to these shrinking zones. This inhomogeneity is crucial for explaining the apparent distribution of "dark matter" effects.

**Quantifying Inhomogeneous Shrinking**: The Universal Scaling Factor $\alpha_U$ becomes a function of local environment: $\alpha_U(\rho, t_{univ}, \mathbf{x}) = \alpha_{U,bg}(t_{univ}) \cdot f(\rho_{local}(\mathbf{x}, t_{univ}), \text{history})$. Here, $\alpha_{U,bg}(t_{univ})$ is the background cosmological shrinking factor, and $f$ describes the modulation due to local density $\rho_{local}$ and potentially its history.

#### 2.3. The Metric in Dense Regions

Within a local, dense, shrinking region, the spacetime metric is proposed to take the form:
$$ds^2 = -c^2 dt_{local}^2 + \alpha_U(t_{univ}, env)^2 \delta_{ij} dx^i dx^j$$
where $c$ is the speed of light, $dt_{local}$ is the local time interval, $dx^i$ are local Cartesian spatial coordinate intervals, $\delta_{ij}$ is the Kronecker delta, and $\alpha_U(t_{univ}, env)$ is the Universal Scaling Factor. This factor $\alpha_U$ describes the magnitude of spatial shrinking as a function of universal time $t_{univ}$ and potentially local environmental factors ($env$) such as density $\rho$. For the background cosmological model, we primarily consider its temporal evolution $\alpha_U(t_{univ})$. The shrinking implies that $\alpha_U(t_{univ})$ is a decreasing function of cosmic time.

*(... The rest of the theoretical sections from the JSON would follow here, formatted with Markdown headings ...)*

---
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

