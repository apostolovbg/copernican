:::model-meta
name = USMFv3b
version = 3.0
author = Copernican
date = 2025-06-12
:::

:::parameters
H_B = 70 # Parameter B
M = -19.3 # nuisance
:::

:::constants
C_LIGHT = 299792.458
:::

:::components
scale = (1+z)
:::

:::equations
\[
mu = 5 * np.log10(scale * C_LIGHT/H_B) + M
\]
:::

:::datasets
sne = tablef3.dat
:::

:::description
Third USMF variant minimal example.
:::
