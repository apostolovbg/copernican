# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Nested sampling cosmology engine.


This backend implements a lightweight nested-sampling routine that remains
compatible with the Copernican plugin architecture.  The sampler focuses on
robustness and reproducibility rather than asymptotic optimality: it draws
initial live points uniformly from declared parameter bounds, replaces the
lowest-likelihood point with constrained proposals and accumulates evidence
estimates using a simple log-sum-exp accumulator.  The goal is to provide a
complementary alternative to the ensemble MCMC engine so operators can compare
posterior summaries produced by markedly different inference strategies while
sharing the same likelihood, prior and transform helpers supplied by
``copernican_lib.engine_plugin_validation``.

The implementation intentionally mirrors the result dictionary produced by the
MCMC backend so downstream tooling—Stage 3 diagnostics, NetCDF exporters and
summary writers—continue to operate without special cases.  Nested-sampling
specific diagnostics such as the estimated log-evidence and information gain
are stored under the ``diagnostics`` key alongside familiar statistics.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from copernican_lib import engine_plugin_validation
from copernican_lib.likelihoods import BAOLike, CMBLike, JointLike, SNeLike
from copernican_lib.progress import BatchProgressBar
from copernican_lib.statistics import (
    calculate_bao_observables,
    chi_squared_bao,
    chi_squared_cmb,
    chi_squared_sne,
    compute_cmb_spectrum,
    compute_cmb_spectrum_from_dict,
)
from copernican_lib.utils import get_random_seed

ENGINE_KIND = "nested"
ENGINE_LABEL = "Nested sampling engine"
ENGINE_VERSION = "1.1.0"

_DEFAULT_LIVE_POINTS = 400
_DEFAULT_MAX_ITERATIONS = 20000
_DEFAULT_EVIDENCE_TOLERANCE = 1e-3
_DEFAULT_ENLARGEMENT_FRACTION = 1.6
_MAX_INITIAL_ATTEMPTS = 2000
_MAX_REPLACEMENT_ATTEMPTS = 5000
_MIN_WEIGHT_FLOOR = 1e-12


@dataclass(slots=True)
class _Sample:
    """Container storing a single nested-sampling state."""

    params: np.ndarray
    log_posterior: float
    log_likelihood: float
    log_prior: float
    state: Mapping[str, Any]


class _JointLogLikelihood:
    """Picklable adapter exposing bounds and transforms for the joint like."""

    __slots__ = ("_joint_like", "parameter_bounds", "parameter_transforms")

    def __init__(
        self,
        joint_like: JointLike,
        parameter_bounds: Iterable[tuple[float | None, float | None]] | None,
        parameter_transforms: Iterable[Any] | None,
    ) -> None:
        self._joint_like = joint_like
        self.parameter_bounds = list(parameter_bounds or [])
        if parameter_transforms is not None:
            self.parameter_transforms = list(parameter_transforms)

    def __call__(self, params: Sequence[float]) -> float:
        """Return the combined log-likelihood for ``params``."""

        return float(self._joint_like.loglike(params))


def _logsumexp_pair(a: float, b: float) -> float:
    """Return ``log(exp(a) + exp(b))`` with numerical stability."""

    if not math.isfinite(a):
        return b
    if not math.isfinite(b):
        return a
    maximum = max(a, b)
    return maximum + math.log(math.exp(a - maximum) + math.exp(b - maximum))


def _build_joint_logposterior(
    model_plugin: Any,
    sne_data_df: Any,
    bao_data_df: Any | None,
    cmb_data_df: Any | None,
) -> tuple[
    engine_plugin_validation.PosteriorEvaluator, JointLike, Sequence[str]
]:
    """Return the posterior evaluator and joint likelihood helper."""

    sne_like = SNeLike(model_plugin.distance_modulus_model, sne_data_df)

    if bao_data_df is not None:
        bao_z = bao_data_df.get("redshift")
        bao_types = bao_data_df.get("observable_type")
        bao_val = bao_data_df.get("value")
        bao_err = bao_data_df.get("error")
    else:
        bao_z = bao_types = bao_val = bao_err = None

    bao_enabled = bool(
        bao_data_df is not None
        and getattr(model_plugin, "valid_for_bao", True)
        and hasattr(bao_data_df, "__len__")
        and len(bao_data_df) > 0
    )
    bao_like = BAOLike(
        np.asarray(bao_z if bao_z is not None else [], dtype=float),
        np.asarray(bao_types if bao_types is not None else [], dtype=object),
        np.asarray(bao_val if bao_val is not None else [], dtype=float),
        np.asarray(bao_err if bao_err is not None else [], dtype=float),
        model_plugin,
        covariance_matrix_inv=(
            None
            if bao_data_df is None
            else bao_data_df.attrs.get("covariance_matrix_inv")
        ),
        enabled=bao_enabled,
    )

    cmb_enabled = bool(
        cmb_data_df is not None
        and getattr(model_plugin, "valid_for_cmb", True)
        and not getattr(cmb_data_df, "empty", True)
        and "covariance_matrix_inv" in getattr(cmb_data_df, "attrs", {})
    )
    cmb_like = CMBLike(
        cmb_data_df if cmb_data_df is not None else pd.DataFrame(),
        model_plugin,
        enabled=cmb_enabled,
    )

    likelihood_config = dict(
        getattr(model_plugin, "LIKELIHOOD_CONFIG", {}) or {}
    )
    likelihood_config.setdefault(
        "sne",
        sne_data_df is not None
        and hasattr(sne_data_df, "__len__")
        and len(sne_data_df) > 0,
    )
    likelihood_config.setdefault("bao", bao_enabled)
    likelihood_config.setdefault("cmb", cmb_enabled)

    joint_like = JointLike(
        {"sne": sne_like, "bao": bao_like, "cmb": cmb_like},
        config=likelihood_config,
    )

    transforms = getattr(model_plugin, "PARAMETER_TRANSFORMS", None)
    loglike = _JointLogLikelihood(
        joint_like,
        getattr(model_plugin, "PARAMETER_BOUNDS", []),
        transforms,
    )
    priors = getattr(model_plugin, "PARAMETER_PRIOR_OBJECTS", None)
    if priors is None:
        priors = getattr(model_plugin, "PARAMETER_PRIORS", [])
    posterior = engine_plugin_validation.make_logposterior(loglike, priors)
    names = list(getattr(model_plugin, "PARAMETER_NAMES", ()))
    return posterior, joint_like, names


def _initial_live_point(
    rng: np.random.Generator,
    lower: np.ndarray,
    upper: np.ndarray,
    centre: np.ndarray,
) -> np.ndarray:
    """Return a candidate sampled uniformly within finite bounds."""

    sample = np.empty_like(lower, dtype=float)
    for idx, (lo, hi) in enumerate(zip(lower, upper, strict=False)):
        if math.isfinite(lo) and math.isfinite(hi):
            sample[idx] = rng.uniform(lo, hi)
        else:
            width = max(abs(centre[idx]), 1.0)
            draw = rng.normal(centre[idx], width)
            if math.isfinite(lo):
                draw = max(draw, lo)
            if math.isfinite(hi):
                draw = min(draw, hi)
            sample[idx] = draw
    return sample


def _replacement_sample(
    rng: np.random.Generator,
    live_points: Sequence[_Sample],
    lower: np.ndarray,
    upper: np.ndarray,
    enlargement: float,
) -> np.ndarray:
    """Return a proposal drawn around the existing live point cloud."""

    stacked = np.array([entry.params for entry in live_points], dtype=float)
    centre = stacked[rng.integers(len(stacked))]
    spread = np.std(stacked, axis=0)
    fallback = np.maximum(np.abs(centre), 1.0)
    spread = np.where(spread > 0, spread, fallback)
    spread = spread * max(enlargement, 1.0)
    proposal = centre + rng.standard_normal(centre.shape) * spread
    proposal = np.clip(proposal, lower, upper)
    return proposal


def _evaluate_point(
    posterior: engine_plugin_validation.PosteriorEvaluator,
    joint_like: JointLike,
    params: np.ndarray,
) -> _Sample | None:
    """Return a populated :class:`_Sample` when ``params`` are valid."""

    log_post = float(posterior(params))
    if not math.isfinite(log_post):
        return None
    state = joint_like.state
    log_like = float(state.get("loglike", float("-inf")))
    if not math.isfinite(log_like):
        return None
    log_prior = log_post - log_like
    return _Sample(
        params=np.asarray(params, dtype=float),
        log_posterior=log_post,
        log_likelihood=log_like,
        log_prior=log_prior,
        state=state,
    )


def _prepare_bounds(
    bounds: Iterable[tuple[float | None, float | None]] | None,
    initial: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower and upper bounds along with an initial centre."""

    bounds = list(bounds or [])
    if not bounds:
        ndim = len(initial or [])
        bounds = [(None, None)] * ndim
    lower = np.empty(len(bounds), dtype=float)
    upper = np.empty(len(bounds), dtype=float)
    for idx, (lo, hi) in enumerate(bounds):
        lower[idx] = -math.inf if lo is None else float(lo)
        upper[idx] = math.inf if hi is None else float(hi)
        if (
            math.isfinite(lower[idx])
            and math.isfinite(upper[idx])
            and upper[idx] < lower[idx]
        ):
            raise ValueError("parameter bounds inverted")
    if initial is None or len(initial) != len(bounds):
        initial_arr = np.zeros(len(bounds), dtype=float)
    else:
        initial_arr = np.asarray(initial, dtype=float)
    return lower, upper, initial_arr


def _weights_from_logs(log_weights: np.ndarray) -> np.ndarray:
    """Return normalised weights derived from ``log_weights``."""

    max_logw = np.max(log_weights)
    shifted = np.exp(log_weights - max_logw)
    total = shifted.sum()
    if total <= 0:
        return np.full_like(shifted, 1.0 / max(len(shifted), 1), dtype=float)
    return shifted / total


def fit_cosmology_parameters(
    sne_data_df: pd.DataFrame,
    model_plugin: Any,
    *,
    bao_data_df: pd.DataFrame | None = None,
    cmb_data_df: pd.DataFrame | None = None,
    n_live_points: int = _DEFAULT_LIVE_POINTS,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    evidence_tolerance: float = _DEFAULT_EVIDENCE_TOLERANCE,
    enlargement_fraction: float = _DEFAULT_ENLARGEMENT_FRACTION,
    display_progress: bool = True,
) -> Mapping[str, Any]:
    """Return posterior samples and diagnostics using nested sampling.

    ``display_progress`` mirrors the MCMC engine's toggle so Stage 2 can
    disable console updates during scripted runs while still wiring the nested
    sampler through the shared progress helpers.  Live progress remains enabled
    by default for interactive sessions.
    """

    logger = logging.getLogger(__name__)
    posterior, joint_like, param_names = _build_joint_logposterior(
        model_plugin,
        sne_data_df,
        bao_data_df,
        cmb_data_df,
    )

    lower, upper, initial = _prepare_bounds(
        getattr(model_plugin, "PARAMETER_BOUNDS", None),
        getattr(model_plugin, "INITIAL_GUESSES", None),
    )
    ndim = lower.size

    rng = np.random.default_rng(get_random_seed())
    live_points: list[_Sample] = []
    attempts = 0
    while (
        len(live_points) < max(1, n_live_points)
        and attempts < _MAX_INITIAL_ATTEMPTS
    ):
        attempts += 1
        candidate = _initial_live_point(rng, lower, upper, initial)
        evaluated = _evaluate_point(posterior, joint_like, candidate)
        if evaluated is None:
            continue
        live_points.append(evaluated)
    if len(live_points) < max(5, int(0.5 * n_live_points)):
        logger.error(
            "Nested sampler failed to initialise %d live points "
            "(obtained %d).",
            n_live_points,
            len(live_points),
        )
        return {"success": False, "samples": None}

    log_width = 0.0
    log_evidence = float("-inf")
    samples: list[_Sample] = []
    log_weights: list[float] = []
    iterations = 0

    progress_label = (
        f"{getattr(model_plugin, 'MODEL_NAME', 'Model')} nested sampling"
    )
    # ``BatchProgressBar`` expects a positive upper bound.  Clamp the maximum
    # to at least one step so the helper remains initialisable when callers
    # request zero iterations (for example during dry-run tests).
    progress_total = max(int(max_iterations), 1)
    progress_bar = BatchProgressBar(
        progress_label,
        progress_total,
        display=bool(display_progress),
        subunit_labels=("iteration", "iterations"),
    )
    progress_active = max_iterations > 0
    if progress_active:
        # Treat the entire nested run as a single progress step so the console
        # keeps repainting the same line instead of announcing a new step for
        # every iteration.  The fractional progress is injected through the
        # per-update ``step_progress`` argument below.
        progress_bar.start_batch(1, 1)

    try:
        while iterations < max_iterations and live_points:
            iterations += 1
            worst_index = int(
                np.argmin([p.log_likelihood for p in live_points])
            )
            worst_point = live_points[worst_index]
            log_width -= 1.0 / max(n_live_points, 1)
            log_weight = log_width + worst_point.log_likelihood
            log_evidence = _logsumexp_pair(log_evidence, log_weight)
            samples.append(worst_point)
            log_weights.append(log_weight)

            if progress_active:
                # Clamp the fractional completion so early convergence still
                # renders a sensible percentage and never exceeds 100% during
                # over-specified iteration budgets.
                fraction = min(iterations / progress_total, 1.0)
                progress_bar.update(
                    1,
                    processed=iterations,
                    total=progress_total,
                    step_progress=fraction,
                )

            target = worst_point.log_likelihood
            replaced = False
            for attempt in range(_MAX_REPLACEMENT_ATTEMPTS):
                proposal = _replacement_sample(
                    rng,
                    live_points,
                    lower,
                    upper,
                    enlargement_fraction,
                )
                evaluated = _evaluate_point(posterior, joint_like, proposal)
                if evaluated is None:
                    continue
                if evaluated.log_likelihood <= target and attempt < (
                    _MAX_REPLACEMENT_ATTEMPTS // 2
                ):
                    continue
                live_points[worst_index] = evaluated
                replaced = True
                break
            if not replaced:
                logger.warning(
                    (
                        "Nested sampler terminated early after %d iterations; "
                        "no valid replacement found."
                    ),
                    iterations,
                )
                break

            remaining_best = max(p.log_likelihood for p in live_points)
            if (
                iterations > n_live_points
                and math.isfinite(remaining_best)
                and math.isfinite(log_evidence)
            ):
                ratio = remaining_best + log_width - log_evidence
                if ratio < math.log(
                    max(evidence_tolerance, _MIN_WEIGHT_FLOOR)
                ):
                    break
    finally:
        progress_bar.finish_batch()

    if live_points:
        log_width -= 1.0 / max(n_live_points, 1)
        tail_weight = log_width + max(p.log_likelihood for p in live_points)
        for entry in live_points:
            samples.append(entry)
            log_weights.append(tail_weight)
            log_evidence = _logsumexp_pair(log_evidence, tail_weight)

    if not samples:
        return {"success": False, "samples": None}

    points = np.array([s.params for s in samples], dtype=float)
    log_posterior = np.array([s.log_posterior for s in samples], dtype=float)
    log_likelihoods = np.array(
        [s.log_likelihood for s in samples],
        dtype=float,
    )
    log_weights_arr = np.array(log_weights, dtype=float)
    weights = _weights_from_logs(log_weights_arr)

    mean_vector = weights @ points
    centred = points - mean_vector
    covariance = centred.T @ (centred * weights[:, None])
    covariance = covariance / max(weights.sum(), _MIN_WEIGHT_FLOOR)
    std_dev = np.sqrt(np.clip(np.diag(covariance), a_min=0.0, a_max=None))

    best_index = int(np.argmax(log_posterior))
    best_sample = samples[best_index]
    best_state = best_sample.state
    chi2_total = float(best_state.get("chi2", float("inf")))
    components = best_state.get("metadata", {}).get("components", {})
    chi2_sne = float(components.get("sne", {}).get("chi2", chi2_total))
    chi2_bao = float(components.get("bao", {}).get("chi2", 0.0))
    chi2_cmb = float(components.get("cmb", {}).get("chi2", 0.0))

    fitted = {
        name: float(best_sample.params[idx])
        for idx, name in enumerate(param_names)
    }
    posterior_mean = {
        name: float(mean_vector[idx]) for idx, name in enumerate(param_names)
    }
    parameter_errors = {
        name: float(std_dev[idx]) for idx, name in enumerate(param_names)
    }

    sne_points = int(len(sne_data_df) if sne_data_df is not None else 0)
    bao_points = int(len(bao_data_df) if bao_data_df is not None else 0)
    cmb_points = int(len(cmb_data_df) if cmb_data_df is not None else 0)
    total_points = sne_points + bao_points + cmb_points
    dof = max(total_points - ndim, 1)
    reduced = chi2_total / dof

    if math.isfinite(log_evidence):
        info_nats = float(np.dot(weights, log_likelihoods) - log_evidence)
    else:
        info_nats = float("nan")
    diagnostics = {
        "log_evidence": float(log_evidence),
        "information_nats": info_nats,
        "effective_samples": float(1.0 / np.sum(weights**2)),
        "iterations_completed": int(iterations),
    }

    chain = points[:, None, :]
    log_prob_chain = log_posterior[:, None]

    return {
        "success": (
            math.isfinite(chi2_total) and math.isfinite(log_posterior.max())
        ),
        "samples": chain,
        "log_probability": log_prob_chain,
        "fitted_cosmological_params": fitted,
        "posterior_mean_params": posterior_mean,
        "model_name": getattr(model_plugin, "MODEL_NAME", "Unknown"),
        "param_names": list(param_names),
        "parameter_errors": parameter_errors,
        "covariance_matrix": covariance,
        "chi2_min": chi2_total,
        "chi2_sne": chi2_sne,
        "chi2_bao": chi2_bao,
        "chi2_cmb": chi2_cmb,
        "chi2_total": chi2_total,
        "log_likelihood_best": float(best_sample.log_likelihood),
        "log_posterior_best": float(best_sample.log_posterior),
        "log_prior_best": float(best_sample.log_prior),
        "dof": dof,
        "reduced_chi2": reduced,
        "acceptance_fraction": None,
        "burn_in_steps": 0,
        "production_steps": iterations,
        "n_walkers": int(n_live_points),
        "n_live_points": int(n_live_points),
        "max_iterations": int(max_iterations),
        "iterations_completed": int(iterations),
        "evidence_tolerance": float(evidence_tolerance),
        "enlargement_fraction": float(enlargement_fraction),
        "pool_workers": 0,
        "diagnostics": diagnostics,
        "progress_granularity": None,
        "likelihood_state": best_state,
        "chi2_components": {
            "sne": chi2_sne,
            "bao": chi2_bao,
            "cmb": chi2_cmb,
        },
        "data_points": {
            "sne": sne_points,
            "bao": bao_points,
            "cmb": cmb_points,
            "total": total_points,
        },
    }


def fit_sne_parameters(
    sne_data_df: pd.DataFrame,
    model_plugin: Any,
    *,
    bao_data_df: pd.DataFrame | None = None,
    cmb_data_df: pd.DataFrame | None = None,
    n_live_points: int = _DEFAULT_LIVE_POINTS,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    evidence_tolerance: float = _DEFAULT_EVIDENCE_TOLERANCE,
    enlargement_fraction: float = _DEFAULT_ENLARGEMENT_FRACTION,
    display_progress: bool = True,
) -> Mapping[str, Any]:
    """Compatibility wrapper for :func:`fit_cosmology_parameters`.

    Nested-sampling now supports the same multi-probe datasets as the MCMC
    engine, so the legacy SNe-focused name remains only for backwards
    compatibility.  Prefer :func:`fit_cosmology_parameters` to avoid the
    misleading scope and to align with the CLI terminology.
    """

    warnings.warn(
        (
            "fit_sne_parameters is deprecated; "
            "use fit_cosmology_parameters instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return fit_cosmology_parameters(
        sne_data_df,
        model_plugin,
        bao_data_df=bao_data_df,
        cmb_data_df=cmb_data_df,
        n_live_points=n_live_points,
        max_iterations=max_iterations,
        evidence_tolerance=evidence_tolerance,
        enlargement_fraction=enlargement_fraction,
        display_progress=display_progress,
    )


__all__ = [
    "ENGINE_KIND",
    "ENGINE_LABEL",
    "ENGINE_VERSION",
    "calculate_bao_observables",
    "chi_squared_bao",
    "chi_squared_cmb",
    "chi_squared_sne",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_from_dict",
    "fit_cosmology_parameters",
    "fit_sne_parameters",
]
