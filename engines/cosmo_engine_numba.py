# Copernican Suite Numba Engine
"""Numba-accelerated engine wrapper."""
# DEV NOTE (v1.5e): Introduced in Phase 5 to provide a faster backend.

from numba import njit
import logging
from types import SimpleNamespace
from . import cosmo_engine_1_4b as cpu_engine


def _jit_plugin(plugin):
    """Return a copy of `plugin` with JIT-compiled functions where possible."""
    attrs = {k: getattr(plugin, k) for k in dir(plugin) if not k.startswith('__')}
    compiled = SimpleNamespace(**attrs)
    try:
        compiled.distance_modulus_model = njit(plugin.distance_modulus_model)
        compiled.get_comoving_distance_Mpc = njit(plugin.get_comoving_distance_Mpc)
        compiled.get_luminosity_distance_Mpc = njit(plugin.get_luminosity_distance_Mpc)
        compiled.get_angular_diameter_distance_Mpc = njit(plugin.get_angular_diameter_distance_Mpc)
        compiled.get_Hz_per_Mpc = njit(plugin.get_Hz_per_Mpc)
        compiled.get_DV_Mpc = njit(plugin.get_DV_Mpc)
        compiled.get_sound_horizon_rs_Mpc = njit(plugin.get_sound_horizon_rs_Mpc)
    except Exception as e:
        logging.getLogger().warning(f"Numba JIT compilation failed: {e}")
        return plugin
    return compiled


def fit_sne_parameters(sne_data_df, model_plugin):
    """Fit SNe parameters using JIT-compiled model functions."""
    logger = logging.getLogger()
    logger.info("Using Numba-accelerated engine for SNe fit.")
    jitted = _jit_plugin(model_plugin)
    return cpu_engine.fit_sne_parameters(sne_data_df, jitted)


def calculate_bao_observables(bao_data_df, model_plugin, cosmo_params, z_smooth=None):
    """Calculate BAO observables using JIT-compiled model functions."""
    logger = logging.getLogger()
    logger.info("Using Numba-accelerated BAO calculations.")
    jitted = _jit_plugin(model_plugin)
    return cpu_engine.calculate_bao_observables(bao_data_df, jitted, cosmo_params, z_smooth=z_smooth)
