# Copernican Suite Numba Engine
"""Numba-accelerated engine wrapper."""

from numba import njit
import logging
from types import SimpleNamespace
from . import cosmo_engine_1_4b as cpu_engine


def _try_njit(func, name):
    """Attempt to JIT compile ``func``; return original on failure."""
    logger = logging.getLogger()
    try:
        return njit(func)
    except Exception as exc:  # pragma: no cover - runtime dependent
        logger.warning(f"JIT compile failed for {name}: {exc}")
        return func


def _jit_plugin(plugin):
    """Return a copy of `plugin` with JIT-compiled functions where possible."""
    attrs = {k: getattr(plugin, k) for k in dir(plugin) if not k.startswith('__')}
    compiled = SimpleNamespace(**attrs)

    func_names = [
        'distance_modulus_model',
        'get_comoving_distance_Mpc',
        'get_luminosity_distance_Mpc',
        'get_angular_diameter_distance_Mpc',
        'get_Hz_per_Mpc',
        'get_DV_Mpc',
        'get_sound_horizon_rs_Mpc',
    ]

    for fname in func_names:
        func = getattr(plugin, fname, None)
        if callable(func):
            setattr(compiled, fname, _try_njit(func, fname))

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
