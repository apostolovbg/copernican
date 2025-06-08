---
title: "Lambda-CDM Model"
version: "1.0"
date: "2025-06-08"
model_plugin: "lcdm_model.py"
---

# Standard Flat Lambda-CDM Model

## Abstract

The Lambda-CDM (ΛCDM) or Lambda-Cold Dark Matter model is the standard model of Big Bang cosmology. It attempts to explain the observed properties of our Universe on large scales, including the cosmic microwave background (CMB), the large-scale structure of galaxy clusters, and the accelerated expansion of the Universe observed in the light from distant galaxies and supernovae.

---

## Quantitative Model Specification for Copernican Suite

### Key Equations

**For Supernovae (SNe Ia) Fitting:**
$$H(z) = H_0 \sqrt{\Omega_{m0}(1+z)^3 + \Omega_{\Lambda0}}$$
$$d_L(z) = (1+z) \int_0^z \frac{c}{H(z')} dz'$$
$$\mu = 5 \log_{10}(d_L/1\mathrm{Mpc}) + 25$$

**For Baryon Acoustic Oscillation (BAO) Analysis:**
$$D_A(z) = \frac{1}{1+z} \int_0^z \frac{c}{H(z')} dz'$$
$$D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3}$$

### Model Parameters

This table defines the parameters for the ΛCDM model.

| Parameter Name | Python Variable | Initial Guess | Bounds        | Unit       | LaTeX Name       |
| :------------- | :-------------- | :------------ | :------------ | :--------- | :--------------- |
| H0             | `H0`            | 67.7          | (50.0, 100.0) | km/s/Mpc   | $H_0$            |
| Omega_m0       | `Omega_m0`      | 0.31          | (0.05, 0.7)   |            | $\Omega_{m0}$    |
| Omega_b0       | `Omega_b0`      | 0.0486        | (0.01, 0.1)   |            | $\Omega_{b0}$    |

---
---

> ### TEMPLATE FOR NEW MODEL DEFINITION FILE
>
> ---
> title: "Title of Your Model"
> version: "1.0"
> date: "YYYY-MM-DD"
> model_plugin: "your_model_name.py"
> ---
>
> # Your Model Title
>
> ## Abstract
>
> A brief abstract describing the core concepts of your cosmological model.
>
> ---
>
> ## Quantitative Model Specification for Copernican Suite
>
> This section is critical for machine parsing and defines the model's parameters for the suite.
>
> ### Key Equations
>
> **For Supernovae (SNe Ia) Fitting:**
> $$\mu = ...$$
>
> **For Baryon Acoustic Oscillation (BAO) Analysis:**
> $$D_V(z) = ...$$
>
> ### Model Parameters
>
> This table will be parsed to generate the metadata block in the Python plugin.
>
> | Parameter Name | Python Variable | Initial Guess | Bounds        | Unit      | LaTeX Name  |
> | :------------- | :-------------- | :------------ | :------------ | :-------- | :---------- |
> | Param1 Name    | `param1`        | 1.0           | (0.0, 2.0)    | Unit1     | $p_1$       |
> | Param2 Name    | `param2`        | 50.0          | (25.0, 75.0)  | km/s/Mpc  | $p_2$       |
> | ...            | `...`           | ...           | ...           | ...       | ...         |
>
> ---
>
> ## Theoretical Framework
>
> A detailed, human-readable write-up of the model's theoretical basis, motivations, and predictions.