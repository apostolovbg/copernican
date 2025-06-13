:::model-meta
name = LambdaCDM
version = 1.0
author = Copernican
date = 2025-06-12
:::

:::parameters
H0 = 70 # Hubble constant
Omega_m0 = 0.3 # Matter density
M = -19.3 # SN nuisance
alpha = 0.14 # stretch
beta = 3.1 # color
:::

:::constants
C_LIGHT = 299792.458
:::

:::components
H_z = H0 * ((Omega_m0*(1+z)**3 + (1-Omega_m0))**0.5)
:::

:::equations
\[
d_L = (1+z) * C_LIGHT/H_z
mu = 5 * np.log10(d_L) + M
\]
:::

:::datasets
sne = tablef3.dat
bao = bao1.json
:::

:::description
Simple LambdaCDM model expressed in CosmoDSL.
:::
