# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Markov Chain Monte Carlo engine using :mod:`emcee`.

The combined engine performs deterministic optimisation.  This module
complements it with a lightweight sampler so users can explore posterior
distributions of cosmological parameters.  Only the supernova distance
modulus is sampled for brevity, but chi-squared helpers from the combined
engine are re-exported so that BAO and CMB calculations remain available
through the same interface.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Iterable

import emcee
import numpy as np

from copernican_lib import engine_interface

# Reuse established chi-squared utilities so all engines share the same
# statistical calculations.
from .cosmo_engine_comb import (  # noqa: F401
    calculate_bao_observables,
    chi_squared_bao,
    chi_squared_cmb,
    chi_squared_sne,
    compute_cmb_spectrum,
)


def _log_probability(
    params: np.ndarray,
    model_plugin: Any,
    sne_data_df: Any,
) -> float:
    """Return the log-posterior for ``params``.

    A uniform prior derived from ``PARAMETER_BOUNDS`` ensures that walkers stay
    within the physically allowed region.  The likelihood is based on the
    supernova chi-squared metric from the combined engine.
    """

    bounds = getattr(model_plugin, "PARAMETER_BOUNDS", [])
    for val, (low, high) in zip(params, bounds):
        if val < low or val > high:
            return -np.inf

    chi2 = chi_squared_sne(
        params,
        model_plugin.distance_modulus_model,
        sne_data_df,
    )
    if not np.isfinite(chi2):
        return -np.inf
    return -0.5 * chi2


def fit_sne_parameters(
    sne_data_df: Any,
    model_plugin: Any,
    *,
    n_walkers: int = 32,
    n_steps: int = 200,
    pool_size: int | None = None,
) -> dict[str, Any]:
    """Sample SNe parameters with :mod:`emcee`.

    The routine validates ``model_plugin`` using the shared interface before
    launching an ensemble sampler.  Sampling occurs in parallel using a
    multiprocessing pool.  The returned dictionary mirrors the structure of
    optimisation-based engines and includes the raw chain for further
    analysis.
    """

    logger = logging.getLogger()
    engine_interface.validate_plugin(model_plugin)
    names: Iterable[str] = getattr(model_plugin, "PARAMETER_NAMES", [])
    initial = np.asarray(getattr(model_plugin, "INITIAL_GUESSES", []), float)
    bounds = list(getattr(model_plugin, "PARAMETER_BOUNDS", []))

    ndim = len(initial)
    if ndim == 0 or len(bounds) != ndim:
        logger.error("Model plugin missing parameter definitions")
        return {"success": False, "samples": None}

    # emcee requires at least 2 * ndim walkers for decent mixing.
    n_walkers = max(n_walkers, 2 * ndim)
    p0 = initial + 1e-4 * np.random.randn(n_walkers, ndim)

    pool = None
    if pool_size and pool_size > 1:
        pool = mp.get_context("spawn").Pool(processes=pool_size)
    try:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim,
            _log_probability,
            args=(model_plugin, sne_data_df),
            pool=pool,
        )
        sampler.run_mcmc(p0, n_steps, progress=False)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    chain = sampler.get_chain()
    start = int(n_steps / 2)
    mean_params = np.mean(chain[start:], axis=(0, 1))
    fitted = {n: v for n, v in zip(names, mean_params)}

    return {
        "success": True,
        "samples": chain,
        "fitted_cosmological_params": fitted,
        "model_name": getattr(model_plugin, "MODEL_NAME", "Unknown"),
        "param_names": list(names),
    }


__all__ = [
    "calculate_bao_observables",
    "chi_squared_bao",
    "chi_squared_cmb",
    "chi_squared_sne",
    "compute_cmb_spectrum",
    "fit_sne_parameters",
]
