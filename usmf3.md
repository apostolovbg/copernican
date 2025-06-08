---
title: "Entangled-Geometrodynamic Model (USMF v3, Rev. 2)"
version: "1.1"
date: "2025-06-08"
model_plugin: "usmf3.py"
---

# Entangled-Geometrodynamic Model (EGM / USMF v3)

## Abstract

The Entangled-Geometrodynamic Model (EGM) is a cosmological framework wherein dark matter and dark energy are emergent phenomena arising from a dynamic relationship between spacetime geometry and the quantum entanglement of matter. This revised model introduces a key parameter, $\gamma_E$, representing the coupling between the entanglement field and matter, which modulates matter's influence on the Hubble expansion. This allows for a more nuanced cosmic history, improving fits to observational data. The model assumes a flat universe and uses an effective equation of state for the Quantum Potential Field (QPF) to drive late-time acceleration.

---

## Quantitative Model Specification for Copernican Suite

### Key Equations

**For Supernovae (SNe Ia) Fitting:**
$$ H(z) = H_0 \sqrt{\Omega_{m0}(1+z)^{3(1-\gamma_E)} + (1-\Omega_{m0})(1+z)^{3(1+w_\phi)}} $$
$$ d_L(z) = (1+z) \int_0^z \frac{c}{H(z')} dz' $$
$$ \mu = 5 \log_{10}(d_L/1\mathrm{Mpc}) + 25 $$

**For Baryon Acoustic Oscillation (BAO) Analysis:**
$$ D_A(z) = \frac{1}{1+z} \int_0^z \frac{c}{H(z')} dz' $$
$$ D_V(z) = \left[ (1+z)^2 D_A(z)^2 \frac{cz}{H(z)} \right]^{1/3} $$

### Model Parameters

| Parameter Name          | Python Variable | Initial Guess | Bounds         | Unit     | LaTeX Name     |
| :---------------------- | :-------------- | :------------ | :------------- | :------- | :------------- |
| Hubble Constant         | `H0`            | 69.0          | (60.0, 80.0)   | km/s/Mpc | $H_0$          |
| Matter Density          | `Omega_m0`      | 0.3           | (0.1, 0.5)     |          | $\Omega_{m0}$  |
| QPF Equation of State   | `w_phi`         | -1.1          | (-1.5, -0.7)   |          | $w_\phi$       |
| G-Entanglement Coupling | `gamma_E`       | 0.05          | (-0.1, 0.2)    |          | $\gamma_E$     |