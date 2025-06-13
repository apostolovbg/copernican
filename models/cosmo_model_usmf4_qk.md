:::model-meta
name = USMFv4_QK
version = 4.0
author = Copernican
date = 2025-06-12
:::

:::parameters
H0 = 70 # base parameter
Omega_m0 = 0.3 # density
M = -19.3 # nuisance
:::

:::constants
C_LIGHT = 299792.458
:::

:::components
Hz = H0 * ((Omega_m0*(1+z)**4 + (1-Omega_m0))**0.5)
:::

:::equations
\[
mu = 5 * np.log10((1+z) * C_LIGHT/Hz) + M
\]
:::

:::datasets
sne = tablef3.dat
bao = bao1.json
:::

:::description
Quantum Kinematic USMF variant supporting BAO.
:::
