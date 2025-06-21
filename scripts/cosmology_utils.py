# cosmology_utils.py
"""Cosmological helper functions for the Copernican Suite."""
# DEV NOTE (hotfix): added compute_sound_horizon for standard r_s calculation.

import numpy as np
from scipy.integrate import quad


def compute_sound_horizon(z_recomb, hz_func, params, Ob_val, Og_val):
    """Compute sound horizon r_s [Mpc] using the baryon-photon plasma integral.

    Parameters
    ----------
    z_recomb : float
        Redshift of recombination.
    hz_func : callable
        Function returning ``H(z)`` in ``km/s/Mpc`` given ``(z, *params)``.
    params : tuple
        Parameter values passed to ``hz_func``.
    Ob_val : float
        Baryon density parameter.
    Og_val : float
        Photon density parameter.
    """

    h0_val = hz_func(0.0, *params)

    def sound_speed(z):
        return 299792.458 / np.sqrt(3 * (1 + 3 * Ob_val / (4 * Og_val) / (1 + z)))

    def hz_with_radiation(z):
        base = hz_func(z, *params)
        rad_sq = (h0_val ** 2) * Og_val * (1 + z) ** 4
        return np.sqrt(base ** 2 + rad_sq)

    integrand = lambda zv: sound_speed(zv) / hz_with_radiation(zv)
    result, _ = quad(integrand, z_recomb, np.inf, limit=100)
    return result
