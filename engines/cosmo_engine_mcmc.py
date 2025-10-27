# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Markov Chain Monte Carlo engine using :mod:`emcee`.

**Last Updated:** 2025-11-09

The combined engine performs deterministic optimisation. This module
complements it with a lightweight sampler so users can explore posterior

distributions of cosmological parameters. Only the supernova distance
modulus is sampled for brevity. Shared chi-squared helpers now live in
``copernican_lib.statistics`` so the sampler and the legacy optimiser
consume the same functions without cross-importing engine modules.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Iterable

import emcee
import numpy as np

from copernican_lib import engine_interface
from copernican_lib.statistics import (
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

    Walkers outside the declared parameter bounds are rejected immediately
    by returning ``-np.inf``. The likelihood uses the Supernova χ² helper from
    :mod:`copernican_lib.statistics` so that both engines evaluate identical
    statistics.
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


def _reseed_invalid_walkers(
    coords: np.ndarray,
    log_prob: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    model_plugin: Any,
    sne_data_df: Any,
    max_attempts: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace non-finite walker states with fresh proposals.

    The :mod:`emcee` stretch move occasionally propagates ``nan`` coordinates
    when proposals sample highly implausible regions.  Those invalid walkers
    subsequently trigger ``RuntimeWarning`` messages inside emcee's internal
    subtraction logic.  To maintain a clean log and avoid undefined
    transitions we reseed any problematic walkers by drawing small Gaussian
    jitters around the mean of the valid ensemble before continuing the run.
    """

    logger = logging.getLogger()
    coords = np.asarray(coords, dtype=float).copy()
    log_prob = np.asarray(log_prob, dtype=float).copy()

    invalid = (~np.isfinite(coords).all(axis=1)) | (~np.isfinite(log_prob))
    if not np.any(invalid):
        return coords, log_prob

    logger.warning(
        "Detected %d invalid walkers after burn-in; reseeding them.",
        int(np.sum(invalid)),
    )

    valid_coords = coords[~invalid]
    if valid_coords.size == 0:
        initial = np.asarray(
            getattr(model_plugin, "INITIAL_GUESSES", []),
            dtype=float,
        )
        if initial.size == 0:
            raise RuntimeError("No baseline available for reseeding walkers.")
        valid_coords = initial[None, :]

    centre = np.mean(valid_coords, axis=0)
    spread = np.std(valid_coords, axis=0)
    spread = np.where(spread > 0, spread, 1.0)

    bad_idx = np.flatnonzero(invalid)
    attempts = 0
    while bad_idx.size and attempts < max_attempts:
        attempts += 1
        jitter = rng.standard_normal((bad_idx.size, centre.size))
        proposals = centre + jitter * np.maximum(spread, 1e-3)
        proposals = np.clip(proposals, lower, upper)
        new_log_prob = np.array(
            [
                _log_probability(pos, model_plugin, sne_data_df)
                for pos in proposals
            ]
        )
        finite = np.isfinite(new_log_prob)
        coords[bad_idx[finite]] = proposals[finite]
        log_prob[bad_idx[finite]] = new_log_prob[finite]
        bad_idx = bad_idx[~finite]

    if bad_idx.size:
        raise RuntimeError(
            "Unable to reseed %d walkers with finite log probability"
            % bad_idx.size
        )

    return coords, log_prob


def fit_sne_parameters(
    sne_data_df: Any,
    model_plugin: Any,
    *,
    n_walkers: int = 32,
    n_steps: int = 200,
    pool_size: int | None = None,
) -> dict[str, Any]:
    """Sample SNe parameters with :mod:`emcee`.

    The routine initialises walkers within the declared parameter bounds, runs
    a configurable burn-in stage and returns summary statistics alongside the
    raw chain. The result mirrors the combined engine API so higher-level code
    can report χ² values consistently.
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

    lower = np.array(
        [-np.inf if low is None else float(low) for low, _ in bounds]
    )
    upper = np.array(
        [np.inf if high is None else float(high) for _, high in bounds]
    )

    n_walkers = max(n_walkers, 2 * ndim)
    rng = np.random.default_rng()
    if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
        p0 = rng.uniform(lower, upper, size=(n_walkers, ndim))
    else:
        jitter = rng.standard_normal((n_walkers, ndim)) * 1e-3
        p0 = initial + jitter
        p0 = np.clip(p0, lower, upper)
    p0[0] = initial

    logp = np.array(
        [_log_probability(pos, model_plugin, sne_data_df) for pos in p0]
    )
    attempts = 0
    while np.any(~np.isfinite(logp)) and attempts < 10:
        bad = ~np.isfinite(logp)
        count = int(np.sum(bad))
        if count == 0:
            break
        if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
            p0[bad] = rng.uniform(lower, upper, size=(count, ndim))
        else:
            jitter = rng.standard_normal((count, ndim)) * 1e-3
            p0[bad] = np.clip(initial + jitter, lower, upper)
        logp[bad] = [
            _log_probability(pos, model_plugin, sne_data_df) for pos in p0[bad]
        ]
        attempts += 1

    if not np.all(np.isfinite(logp)):
        logger.error(
            "Unable to initialise walkers with finite log probability"
        )
        return {"success": False, "samples": None}

    pool = None
    if pool_size and pool_size > 1:
        pool = mp.get_context("spawn").Pool(processes=pool_size)
    burn_in = max(100, n_steps // 5)
    try:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim,
            _log_probability,
            args=(model_plugin, sne_data_df),
            pool=pool,
        )
        sampler.run_mcmc(p0, burn_in, progress=False)
        last = sampler.get_last_sample()
        try:
            coords, log_prob = _reseed_invalid_walkers(
                last.coords,
                last.log_prob,
                lower=lower,
                upper=upper,
                rng=rng,
                model_plugin=model_plugin,
                sne_data_df=sne_data_df,
            )
        except RuntimeError as exc:
            logger.error("%s", exc)
            return {"success": False, "samples": None}
        sampler.reset()
        sampler.run_mcmc(coords, n_steps, progress=False)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    chain = sampler.get_chain()
    log_prob_chain = sampler.get_log_prob()
    flat_chain = sampler.get_chain(flat=True)
    flat_log_prob = sampler.get_log_prob(flat=True)

    best_index = int(np.argmax(flat_log_prob))
    best_params = flat_chain[best_index]
    mean_params = np.mean(flat_chain, axis=0)

    covariance = np.cov(flat_chain, rowvar=False)
    errors = np.sqrt(np.diag(covariance))
    error_dict = {n: e for n, e in zip(names, errors)}

    fitted = {n: v for n, v in zip(names, best_params)}
    posterior_mean = {n: v for n, v in zip(names, mean_params)}

    chi2_best = chi_squared_sne(
        best_params,
        model_plugin.distance_modulus_model,
        sne_data_df,
    )
    dof = len(sne_data_df) - ndim
    reduced = chi2_best / dof if dof > 0 else np.nan

    acceptance = sampler.acceptance_fraction
    logger.info(
        "MCMC acceptance for %s: mean=%.3f, min=%.3f, max=%.3f",
        getattr(model_plugin, "MODEL_NAME", "Unknown"),
        float(np.mean(acceptance)),
        float(np.min(acceptance)),
        float(np.max(acceptance)),
    )

    try:
        autocorr = sampler.get_autocorr_time()
    except Exception:
        autocorr = None

    return {
        "success": np.isfinite(chi2_best),
        "samples": chain,
        "log_probability": log_prob_chain,
        "fitted_cosmological_params": fitted,
        "posterior_mean_params": posterior_mean,
        "model_name": getattr(model_plugin, "MODEL_NAME", "Unknown"),
        "param_names": list(names),
        "parameter_errors": error_dict,
        "covariance_matrix": covariance,
        "chi2_min": chi2_best,
        "chi2_sne": chi2_best,
        "chi2_total": chi2_best,
        "dof": dof,
        "reduced_chi2": reduced,
        "acceptance_fraction": acceptance,
        "burn_in_steps": burn_in,
        "production_steps": n_steps,
        "autocorrelation_time": autocorr,
    }


__all__ = [
    "calculate_bao_observables",
    "chi_squared_bao",
    "chi_squared_cmb",
    "chi_squared_sne",
    "compute_cmb_spectrum",
    "fit_sne_parameters",
]
