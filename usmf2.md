---
title: "The Unified Shrinking Matter Framework (USMF) Version 2"
version: "2.0 (Updated for 1.4rc)"
date: "2025-06-12"
model_plugin: "usmf2.py"
---

# The Unified Shrinking Matter Framework (USMF) Version 2

## Abstract

The Unified Shrinking Matter Framework (USMF) is an alternative cosmological model which proposes that the observed redshift of distant objects is not due to the metric expansion of space, but is rather an observational effect caused by the properties of matter (e.g., mass, size) changing over cosmic time. In this model, atoms and all gravitationally bound structures shrink, and the speed of light changes proportionally.

Version 2 (USMFv2) simplifies this concept into a testable, phenomenological model based on a linear shrinking law. It is primarily designed to be tested against Type Ia Supernovae data.

---

## Quantitative Model Specification for Copernican Suite

This section provides the specific data required by the Copernican Suite's computational engine to utilize this model.

### Key Equations

The core of USMFv2 is a linear law for the shrinking factor, $\alpha(t)$, which relates the size of an object at time $t$ to its size at present time $t_0$.

$$\alpha(t) = 1 + \frac{H_A}{c}(t_0 - t)$$

From this, the redshift $z$ and luminosity distance $d_L$ can be derived.

$$1+z = \alpha(t_e)$$
$$d_L(z) = c \cdot (t_0-t_e) \cdot (1+z) \cdot \frac{70}{H_A}$$
$$\mu = 5 \log_{10}(d_L) + 25$$

### Model Parameters

| Parameter      | LaTeX Symbol   | Role         | Description                                              |
| :------------- | :------------- | :----------- | :------------------------------------------------------- |
| `H_A`          | $H_A$          | Cosmological | Apparent Hubble Constant, related to the shrinking rate  |
| `t0_age_Gyr`   | $t_{0,age}$    | Cosmological | Age of the Universe at present time                      |

### Fixed Constants

| Constant       | Description                           | Value         |
| :------------- | :------------------------------------ | :------------ |
| `GYR_TO_S`     | Conversion factor from Gigayears to seconds | 3.15576e16    |
| `C_LIGHT_KM_S` | Speed of Light in km/s                | 299792.458    |