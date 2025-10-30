# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Markov Chain Monte Carlo engine using :mod:`emcee`.

**Last Updated:** 2025-10-30

The combined optimiser has been retired entirely, leaving this sampler as the
sole runtime engine.  It continues to focus on Supernova Ia posteriors while
delegating shared χ² helpers to :mod:`copernican_lib.statistics` so the module
acts as the canonical engine façade.  Future backends can slot in beside it
without changing the orchestration code.  Verbose progress logging tracks both
burn-in and production phases with percentage updates so long chains always
report their status.  Version 6.2.0 routes all likelihood evaluations through
the :class:`copernican_lib.likelihoods.JointLike` aggregator and the
new :func:`copernican_lib.engine_interface.make_logposterior` helper so that
posterior calculations automatically honour per-parameter priors, declared
bounds and optional reparameterisation transforms while exposing diagnostic
metadata alongside sampled chains.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
from typing import Any, Callable, Iterable, Sequence

import emcee
import numpy as np

from copernican_lib import engine_interface
from copernican_lib.likelihoods import JointLike, SNeLike
from copernican_lib.statistics import (
    calculate_bao_observables,
    chi_squared_bao,
    chi_squared_cmb,
    chi_squared_sne,
    compute_cmb_spectrum,
    compute_cmb_spectrum_from_dict,
)

ENGINE_KIND = "mcmc"
ENGINE_LABEL = "Ensemble MCMC sampler"

# ``emcee`` triggers its condition number guard when walkers occupy an almost
# degenerate subspace.  The suite accepts wildly different model definitions,
# so the sampler must adaptively identify fixed or near-fixed parameters and
# spread walkers enough to avoid singular ensembles.  These heuristics rely on
# a small mix of absolute and relative tolerances.  The defaults below flag
# intervals narrower than roughly one billionth of the parameter scale while
# still allowing legitimate, tight priors to remain active.
_FIXED_BOUNDS_RTOL = 1e-9
_FIXED_BOUNDS_ATOL = 1e-12
_MAX_INITIAL_CONDITION = 1e12
_MAX_INITIAL_ATTEMPTS = 12


def _build_sne_logposterior(
    model_plugin: Any,
    sne_data_df: Any,
) -> tuple[
    Callable[[Sequence[float]], float],
    Callable[[Sequence[float]], float],
    JointLike,
]:
    """Return posterior, likelihood and diagnostics for Supernova data.

    Engines evaluate the returned posterior repeatedly during sampling.  The
    helper therefore pre-computes the reusable :class:`JointLike` aggregator
    once, attaches the plugin's bounds and optional transformations to the
    underlying log-likelihood callable and finally hands everything to
    :func:`engine_interface.make_logposterior` so priors and Jacobian
    adjustments remain consistent across engines.
    """

    sne_like = SNeLike(model_plugin.distance_modulus_model, sne_data_df)
    joint_like = JointLike({"sne": sne_like})

    def loglike(params: Sequence[float]) -> float:
        """Return the Supernova log-likelihood for ``params``."""

        return float(joint_like.loglike(params))

    # Attach optional metadata so ``make_logposterior`` can enforce bounds and
    # apply reparameterisation transforms without the engine reimplementing
    # those mechanics locally.
    loglike.parameter_bounds = getattr(model_plugin, "PARAMETER_BOUNDS", [])
    transforms = getattr(model_plugin, "PARAMETER_TRANSFORMS", None)
    if transforms is not None:
        loglike.parameter_transforms = transforms

    priors = getattr(model_plugin, "PARAMETER_PRIORS", [])
    posterior = engine_interface.make_logposterior(loglike, priors)
    return posterior, loglike, joint_like


def _reseed_invalid_walkers(
    coords: np.ndarray,
    log_prob: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    log_probability_fn: Callable[[np.ndarray], float],
    reference_position: np.ndarray | None = None,
    max_attempts: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace non-finite walker states with fresh proposals.

    The :mod:`emcee` stretch move occasionally propagates ``nan`` coordinates
    when proposals sample highly implausible regions.  Those invalid walkers
    subsequently trigger ``RuntimeWarning`` messages inside emcee's internal
    subtraction logic.  To maintain a clean log and avoid undefined
    transitions we reseed any problematic walkers by drawing small Gaussian
    jitters around the mean of the valid ensemble before continuing the run.
    ``log_probability_fn`` evaluates the sampler's objective for the proposed
    coordinates so fixed-parameter expansions can remain encapsulated inside a
    caller-provided closure.  ``reference_position`` supplies a fallback
    centroid when every walker is invalid so reseeding still succeeds even if
    the ensemble collapses entirely.
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
        if reference_position is None:
            raise RuntimeError("No baseline available for reseeding walkers.")
        valid_coords = np.asarray(reference_position, dtype=float)[None, :]

    centre = np.mean(valid_coords, axis=0)
    spread = np.std(valid_coords, axis=0)
    finite_width = np.where(
        np.isfinite(lower) & np.isfinite(upper), upper - lower, np.nan
    )
    fallback = np.where(np.isfinite(finite_width), finite_width / 6.0, 1.0)
    spread = np.where(spread > 0, spread, fallback)
    spread = np.where(spread > 0, spread, 1.0)

    bad_idx = np.flatnonzero(invalid)
    attempts = 0
    while bad_idx.size and attempts < max_attempts:
        attempts += 1
        jitter = rng.standard_normal((bad_idx.size, centre.size))
        proposals = centre + jitter * np.maximum(spread, 1e-3)
        proposals = np.clip(proposals, lower, upper)
        new_log_prob = np.array([log_probability_fn(pos) for pos in proposals])
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


def _run_stage_with_progress(
    sampler: emcee.EnsembleSampler,
    initial_state: np.ndarray,
    n_steps: int,
    *,
    stage_name: str,
    logger: logging.Logger,
    progress_granularity: int = 20,
):
    """Iterate ``sampler.sample`` while logging percentage progress."""

    if n_steps <= 0:
        logger.info("Skipping %s stage; zero steps requested.", stage_name)
        return sampler.get_last_sample()

    logger.info("Starting MCMC %s stage for %d steps...", stage_name, n_steps)

    interval = max(1, n_steps // progress_granularity)
    state = None
    for idx, state in enumerate(
        sampler.sample(initial_state, iterations=n_steps, progress=False),
        start=1,
    ):
        if idx == 1 or idx % interval == 0 or idx == n_steps:
            percent = int(round(idx / n_steps * 100))
            logger.info(
                "MCMC %s progress: %3d%% (%d/%d steps)",
                stage_name,
                percent,
                idx,
                n_steps,
            )

    if state is None:
        raise RuntimeError("Sampler produced no states during %s" % stage_name)

    logger.info("Completed MCMC %s stage.", stage_name)
    return state


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
    raw chain. Its return structure remains stable so higher-level code can
    report χ² values consistently even after the combined optimiser removal.
    """

    logger = logging.getLogger()
    engine_interface.validate_plugin(model_plugin)

    posterior_full, loglike_full, joint_like = _build_sne_logposterior(
        model_plugin,
        sne_data_df,
    )
    names: Iterable[str] = getattr(model_plugin, "PARAMETER_NAMES", [])
    names = list(names)
    initial = np.asarray(getattr(model_plugin, "INITIAL_GUESSES", []), float)
    bounds = list(getattr(model_plugin, "PARAMETER_BOUNDS", []))

    ndim_total = len(initial)
    if ndim_total == 0 or len(bounds) != ndim_total:
        logger.error("Model plugin missing parameter definitions")
        return {"success": False, "samples": None}

    try:
        lower_all, upper_all, fixed_mask = _classify_parameter_bounds(
            bounds, logger=logger
        )
    except ValueError:
        return {"success": False, "samples": None}
    active_mask = ~fixed_mask
    active_indices = np.flatnonzero(active_mask)
    fixed_indices = np.flatnonzero(fixed_mask)

    if active_indices.size == 0:
        logger.error("All parameters are fixed; cannot run the sampler.")
        return {"success": False, "samples": None}

    if fixed_indices.size:
        fixed_names = ", ".join(names[idx] for idx in fixed_indices)
        logger.info(
            "Treating %d parameter(s) as fixed or numerically locked: %s",
            int(fixed_indices.size),
            fixed_names,
        )

    template_params = np.clip(initial, lower_all, upper_all)
    initial_active = template_params[active_indices]
    lower = lower_all[active_indices]
    upper = upper_all[active_indices]

    rng = np.random.default_rng()

    ndim_active = active_indices.size
    n_walkers = max(n_walkers, 2 * ndim_active)

    def assemble_full(position: np.ndarray) -> np.ndarray:
        """Return the full parameter vector with fixed entries restored."""
        full = template_params.copy()
        full[active_indices] = position
        return full

    def log_probability_active(position: np.ndarray) -> float:
        full = assemble_full(position)
        return posterior_full(full)

    try:
        p0, logp = _initialise_active_walkers(
            initial_active,
            lower,
            upper,
            n_walkers,
            rng,
            log_probability_active,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return {"success": False, "samples": None}

    pool = None
    if pool_size and pool_size > 1:
        pool = mp.get_context("spawn").Pool(processes=pool_size)
    burn_in = max(100, n_steps // 5)
    try:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim_active,
            log_probability_active,
            pool=pool,
        )
        last = _run_stage_with_progress(
            sampler,
            p0,
            burn_in,
            stage_name="burn-in",
            logger=logger,
        )
        try:
            coords, log_prob = _reseed_invalid_walkers(
                last.coords,
                last.log_prob,
                lower=lower,
                upper=upper,
                rng=rng,
                log_probability_fn=log_probability_active,
                reference_position=initial_active,
            )
        except RuntimeError as exc:
            logger.error("%s", exc)
            return {"success": False, "samples": None}
        sampler.reset()
        _run_stage_with_progress(
            sampler,
            coords,
            n_steps,
            stage_name="production",
            logger=logger,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    chain_active = sampler.get_chain()
    log_prob_chain = sampler.get_log_prob()
    flat_log_prob = sampler.get_log_prob(flat=True)

    n_production, n_effective_walkers, _ = chain_active.shape
    chain = np.empty(
        (n_production, n_effective_walkers, ndim_total),
        dtype=chain_active.dtype,
    )
    chain[:] = template_params
    chain[:, :, active_indices] = chain_active

    flat_chain = chain.reshape(-1, ndim_total)

    best_index = int(np.argmax(flat_log_prob))
    best_params = flat_chain[best_index]
    mean_params = np.mean(flat_chain, axis=0)

    covariance = np.cov(flat_chain, rowvar=False)
    errors = np.sqrt(np.diag(covariance))
    error_dict = {n: e for n, e in zip(names, errors)}

    fitted = {n: v for n, v in zip(names, best_params)}
    posterior_mean = {n: v for n, v in zip(names, mean_params)}

    loglike_best = float(loglike_full(best_params))
    log_posterior_best = float(posterior_full(best_params))
    likelihood_state = dict(joint_like.state)
    chi2_best = float(likelihood_state.get("chi2", float("inf")))
    dof = len(sne_data_df) - ndim_total
    reduced = chi2_best / dof if dof > 0 else np.nan

    log_prior_best = float("-inf")
    if math.isfinite(log_posterior_best) and math.isfinite(loglike_best):
        log_prior_best = log_posterior_best - loglike_best

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
        "success": np.isfinite(chi2_best)
        and math.isfinite(log_posterior_best),
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
        "log_likelihood_best": loglike_best,
        "log_posterior_best": log_posterior_best,
        "log_prior_best": log_prior_best,
        "dof": dof,
        "reduced_chi2": reduced,
        "acceptance_fraction": acceptance,
        "burn_in_steps": burn_in,
        "production_steps": n_steps,
        "autocorrelation_time": autocorr,
        "likelihood_state": likelihood_state,
    }


__all__ = [
    "ENGINE_KIND",
    "ENGINE_LABEL",
    "calculate_bao_observables",
    "chi_squared_bao",
    "chi_squared_cmb",
    "chi_squared_sne",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_from_dict",
    "fit_sne_parameters",
]


def _estimate_condition_number(samples: np.ndarray) -> float | None:
    """Return the condition number of ``samples`` or ``None`` when undefined.

    ``emcee`` inspects the condition number of the initial walker ensemble to
    ensure the stretch move can generate proposals effectively.  The function
    below mirrors that logic without importing private ``emcee`` helpers so the
    engine can deliberately inflate the walker spread before the library raises
    ``ValueError``.  When the ensemble contains fewer than two walkers the
    condition number is undefined; in that situation we return ``None`` and let
    the caller continue with additional attempts.
    """

    if samples.shape[0] < 2:
        return None
    centred = samples - np.mean(samples, axis=0, keepdims=True)
    try:
        singular_values = np.linalg.svd(
            centred, full_matrices=False, hermitian=False
        )[1]
    except np.linalg.LinAlgError:
        return float("inf")
    positive = singular_values[singular_values > 0]
    if positive.size == 0:
        return float("inf")
    return float(positive.max() / positive.min())


def _classify_parameter_bounds(
    bounds: Iterable[tuple[float | None, float | None]],
    *,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper bounds and a mask of effectively fixed parameters.

    Each entry in ``bounds`` is converted to a floating interval.  ``None``
    values map to ``-np.inf`` or ``np.inf`` as appropriate.  Bounds where the
    upper edge falls below the lower edge signal malformed model definitions
    and trigger an error log.  Parameters whose admissible range shrinks to a
    single point—or a numerically indistinguishable sliver—are flagged as
    fixed so the active sampling subspace contains only degrees of freedom
    that ``emcee`` can explore without tripping its linear-independence
    checks.
    """

    lower = np.empty(len(bounds), dtype=float)
    upper = np.empty(len(bounds), dtype=float)
    for idx, (low, high) in enumerate(bounds):
        lower[idx] = -np.inf if low is None else float(low)
        upper[idx] = np.inf if high is None else float(high)
        if (
            np.isfinite(lower[idx])
            and np.isfinite(upper[idx])
            and upper[idx] < lower[idx]
        ):
            logger.error(
                "Parameter %d declares inverted bounds [%f, %f]",
                idx,
                lower[idx],
                upper[idx],
            )
            raise ValueError("invalid parameter bounds: lower exceeds upper")

    with np.errstate(invalid="ignore"):
        widths = upper - lower
        centres = (upper + lower) / 2.0
        scale = np.maximum(np.abs(centres), 1.0)
        threshold = scale * _FIXED_BOUNDS_RTOL + _FIXED_BOUNDS_ATOL
        fixed_mask = np.isfinite(widths) & (widths <= threshold)

    return lower, upper, fixed_mask


def _initialise_active_walkers(
    initial_active: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_walkers: int,
    rng: np.random.Generator,
    log_probability_fn: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return initial walker positions with finite log probabilities.

    The generator gradually inflates the proposal scatter whenever the
    resulting ensemble either falls outside the declared bounds, yields
    non-finite log probabilities or remains dangerously close to a
    degenerate hyperplane.  The strategy favours uniform draws when both
    bounds are finite because those intervals already encode acceptable
    ranges.  Otherwise walkers jitter around the initial guess with adaptive
    Gaussian noise that widens on every retry.  The first walker remains
    anchored to ``initial_active`` so the sampler always includes the
    model's nominal parameter set.
    """

    ndim_active = initial_active.size
    uniform_mask = np.isfinite(lower) & np.isfinite(upper)
    width = upper - lower
    jitter = np.maximum(np.abs(initial_active), 1.0) * 1e-3
    jitter = np.where(
        np.isfinite(width), np.maximum(width / 10.0, jitter), jitter
    )

    attempts = 0
    scatter_multiplier = 1.0
    while attempts < _MAX_INITIAL_ATTEMPTS:
        attempts += 1
        if uniform_mask.all():
            proposals = rng.uniform(
                lower, upper, size=(n_walkers, ndim_active)
            )
        else:
            noise = rng.standard_normal((n_walkers, ndim_active))
            proposals = initial_active + noise * jitter * scatter_multiplier
            proposals = np.clip(proposals, lower, upper)
        proposals[0] = np.clip(initial_active, lower, upper)

        logp = np.array([log_probability_fn(pos) for pos in proposals])
        if not np.all(np.isfinite(logp)):
            scatter_multiplier *= 2.0
            continue

        cond = _estimate_condition_number(proposals)
        if cond is None or cond <= _MAX_INITIAL_CONDITION:
            return proposals, logp

        scatter_multiplier *= 5.0

    raise RuntimeError(
        "Unable to initialise walkers with stable condition number"
    )
