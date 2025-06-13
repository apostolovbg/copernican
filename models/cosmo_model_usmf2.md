:::model-meta
name = USMFv2
version = 2.0
author = Copernican
date = 2025-06-12
:::

:::parameters
H_A = 70 # Apparent Hubble
t0_age_Gyr = 14 # age
M = -19.3 # nuisance
:::

:::constants
C_LIGHT = 299792.458
GYR_TO_S = 3.15576e16
:::

:::components
lookback_s = z * C_LIGHT / H_A
:::

:::equations
\[
d_L = C_LIGHT * lookback_s * (1+z) * (70/H_A)
mu = 5 * np.log10(d_L/3.086e19) + M
\]
:::

:::datasets
sne = tablef3.dat
:::

:::description
Unified Shrinking Matter Framework test model.
:::
