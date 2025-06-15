---
title: "Fixed-Size Filament Contraction Model (USMF) Version 5"
version: "5.0"
date: "2025-06-14"
model_plugin: "usmf5.py"
---
<!-- DEV NOTE (v1.5b): Added a Key Equations section and corrected formatting so the model parses correctly. -->


# Fixed-Size Filament Contraction Model (USMF) Version 5

## Abstract
We propose a cosmological model in which the Universe has always possessed a fixed global scale, while local spacetime within dense filaments undergoes time-dependent contraction. This preserves the exceptional Type Ia supernovae fits of USMF v2 but introduces an adjustable early-universe exponent to improve the BAO fit. Quantum ripples are still encoded as oscillations in the scale factor, now tied to vacuum energy scales. Newtonian mechanics is recovered on short timescales, and the model naturally explains the very early universe without a singular “bang.”

## Quantitative Model Specification for Copernican Suite

### Key Equations

**For Supernovae (SNe Ia) Fitting:**
$$\alpha(t)=\begin{cases}(t/t_0)^{-m_e},&t\le t_{\rm eq}\\[6pt]
\tfrac{1}{2}\bigl[1+\tanh((t-t_{\rm eq})/\Delta)\bigr](t/t_0)^{-p_\alpha}
e^{k_{\exp}((t/t_0)^{s_{\exp}}-1)}\bigl[1+A_{\rm osc}\sin(\omega_{\rm osc}
\ln(t/t_{i,\rm osc})+\phi_{\rm osc})\bigr],&t>t_{\rm eq}\end{cases}$$
$$r(z)=\int_{t_e(z)}^{t_0}\frac{c}{\alpha(t)}\,dt$$
$$d_L(z)=|r(z)|(1+z)^2\frac{70.0}{H_A}$$
$$\mu=5\log_{10}(d_L)+25$$

**For Baryon Acoustic Oscillation (BAO) Analysis:**
$$D_A(z)=|r(z)|\frac{70.0}{H_A}$$
$$H_{\rm USMF}(z)=-\frac{1}{\alpha(t_e)}\left.\frac{d\alpha}{dt}\right|_{t_e}$$
$$D_V(z)=\left[(1+z)^2D_A(z)^2\frac{cz}{H(z)}\right]^{1/3}$$
$$r_s=\int_{z_d}^{\infty}\frac{c_s(z)}{H_{\rm early}(z)}\,dz$$

### Model Parameters

| Parameter Name                | Python Variable | Initial Guess | Bounds         | Unit       | LaTeX Name                          |
|-------------------------------|-----------------|---------------|----------------|------------|-------------------------------------|
| Hubble-like Scale             | H_A             | 70.0          | [50, 90]       | km s⁻¹ Mpc⁻¹ | \(H_A\)                             |
| Late-time shrink exponent     | p_alpha         | 1.0           | [0, 5]         | —          | \(p_\alpha\)                        |
| Exponential strength          | k_exp           | 0.1           | [0, 1]         | —          | \(k_{\exp}\)                        |
| Exponential power             | s_exp           | 1.0           | [0, 5]         | —          | \(s_{\exp}\)                        |
| Oscillation amplitude         | A_osc           | 0.01          | [0, 0.1]       | —          | \(A_{\rm osc}\)                     |
| Oscillation frequency         | omega_osc       | 5.0           | [0, 20]        | —          | \(\omega_{\rm osc}\)                |
| Oscillation start time        | t_i_osc         | 0.1           | [0, 1]         | Gyr        | \(t_{i,\rm osc}\)                   |
| Oscillation phase             | phi_osc         | 0.0           | [0, 2π]        | rad        | \(\phi_{\rm osc}\)                  |
| Early-universe exponent       | m_e             | 0.5           | [0, 2]         | —          | \(m_e\)                             |
| Equality time                 | t_eq            | 0.01          | [10⁻⁴, 0.1]    | Gyr        | \(t_{\rm eq}\)                      |
| Transition width              | delta           | 0.1           | [0.01, 1]      | Gyr        | \(\Delta\)                          |
| Gravity scaling exponent      | n               | 2.0           | [0, 4]         | —          | \(n\)                               |

#### Model Functions

Define the piecewise universal scale factor \(\alpha(t)\):
\[
\alpha(t) =
\begin{cases}
\bigl(\tfrac{t}{t_0}\bigr)^{-m_e}, & t \le t_{\rm eq},\\[8pt]
\frac{1}{2}\Bigl[1+\tanh\!\bigl((t-t_{\rm eq})/\Delta\bigr)\Bigr]\,
\bigl(\tfrac{t}{t_0}\bigr)^{-p_\alpha}
\exp\!\bigl[k_{\exp}\bigl((t/t_0)^{s_{\exp}}-1\bigr)\bigr]\,
\bigl[1 + A_{\rm osc}\sin\bigl(\omega_{\rm osc}\ln(t/t_{i,\rm osc})+\phi_{\rm osc}\bigr)\bigr]\\[6pt]
\quad
+\frac{1}{2}\Bigl[1-\tanh\!\bigl((t-t_{\rm eq})/\Delta\bigr)\Bigr]\,
\bigl(\tfrac{t}{t_0}\bigr)^{-m_e}, & t > t_{\rm eq}.
\end{cases}
\]

Distance measures and sound horizon integrals follow USMF v2 but now with
\[
r_s \;=\; \int_{z_d}^{\infty}\!\frac{c_s(z)}{H_{\rm early}(z)}\,dz,
\quad
H_{\rm early}(z)\propto(1+z)^{1/m_e},
\]
and an effective gravitational “constant”
\[
G_{\rm eff}(t)=G_0\Bigl[\tfrac{\alpha(t)}{\alpha(t_0)}\Bigr]^{n}.
\]

## Theoretical Framework
This framework replaces global shrinking with localized filament contraction on a fixed cosmic stage. Dense structures contract, mimicking dark‐matter effects; voids remain at the primordial scale. The free early exponent \(m_e\) allows BAO data to be fit more flexibly, while oscillatory ripples in \(\alpha(t)\) encode quantum vacuum effects. Short‐timescale dynamics recover Newtonian gravity.

> ### **Internal Formatting Guide for Model Definition Files**
>
> This document establishes the `.md` format standard for DEFINING valid and machine-parsable model definition files and generating Python plugins.
>
> 1.  **YAML Front Matter:** The file must begin with a YAML front matter block (enclosed by `---`). It should contain basic metadata:
>     -   `title`: The full, human-readable name of the model.
>     -   `version`: The version of the model definition.
>     -   `date`: The date of the last update.
>     -   `model_plugin`: The filename of the corresponding Python implementation (e.g., `usmf2.py`).
>
> 2.  **Quantitative Model Specification Section:**
>     -   This section is **critical for machine parsing** and must begin with the header `## Quantitative Model Specification for Copernican Suite`.
>     -   It must contain a subsection `### Model Parameters` with a Markdown table having headers: `Parameter Name`, `Python Variable`, `Initial Guess`, `Bounds`, `Unit`, `LaTeX Name`.
>     -   The table is parsed to generate `PARAMETER_NAMES`, `INITIAL_GUESSES`, `PARAMETER_BOUNDS`, and `PARAMETER_LATEX_NAMES` lists in the Python plugin.
>
> 3.  **Theoretical Framework Section:**
>     -   The remainder of the document contains the detailed theoretical write-up, intended for human readers and context.
