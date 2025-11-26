"""Logging-oriented diagnostics helpers.


The utilities below translate raw numerical arrays into short, stable log
messages so long-running analyses can surface meaningful progress without
flooding the console.  They intentionally avoid performing any logging
themselves, allowing callers to decide where and how the messages should be
displayed or captured during tests.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def _residual_statistics(
    residuals: np.ndarray,
) -> tuple[float, float, float, int]:
    """Return RMS, max absolute value, median and sample size.

    ``residuals`` may contain NaNs from masked measurements.  Only finite
    entries contribute to the statistics so diagnostics remain robust even when
    parts of the dataset are disabled for a specific model.
    """

    mask = np.isfinite(residuals)
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan"), 0

    cleaned = residuals[mask]
    rms = float(np.sqrt(np.mean(cleaned**2)))
    max_abs = float(np.max(np.abs(cleaned)))
    median = float(np.median(cleaned))
    return rms, max_abs, median, int(cleaned.size)


def bao_residual_diagnostics(
    predictions: pd.DataFrame | None,
    *,
    model_name: str,
) -> list[str]:
    """Return formatted BAO residual diagnostics for ``model_name``.

    The first entry summarises the aggregate behaviour while subsequent lines
    (when available) break the metrics down by ``observable_type`` so users can
    monitor specific ratios such as ``DM_over_rs`` or ``DV_over_rs``.
    """

    if predictions is None or getattr(predictions, "empty", True):
        return [f"{model_name} BAO residuals unavailable (no data)."]

    if "model_prediction" not in predictions or "value" not in predictions:
        return [
            (
                f"{model_name} BAO residuals unavailable "
                "(missing model_prediction/value columns)."
            )
        ]

    model_vals = predictions["model_prediction"].to_numpy(dtype=float)
    obs_vals = predictions["value"].to_numpy(dtype=float)
    residuals = model_vals - obs_vals
    rms, max_abs, median, n_points = _residual_statistics(residuals)
    if n_points == 0:
        return [f"{model_name} BAO residuals unavailable (non-finite values)."]

    lines = [
        (
            f"{model_name} BAO residual RMS={rms:.3g}, "
            f"max={max_abs:.3g}, median={median:+.3g} (N={n_points})"
        )
    ]

    if "observable_type" in predictions:
        for obs_type, group in predictions.groupby("observable_type"):
            group_vals = group["model_prediction"].to_numpy(dtype=float)
            group_obs = group["value"].to_numpy(dtype=float)
            group_resid = group_vals - group_obs
            g_rms, g_max, g_median, g_n = _residual_statistics(group_resid)
            if g_n == 0:
                continue
            lines.append(
                (
                    f"    {obs_type}: rms={g_rms:.3g}, "
                    f"max={g_max:.3g}, median={g_median:+.3g} (N={g_n})"
                )
            )

    return lines


def cmb_residual_diagnostics(
    cmb_data: pd.DataFrame | None,
    theory: Mapping[str, Iterable[float]] | np.ndarray,
    *,
    model_name: str,
) -> list[str]:
    """Return residual diagnostics for each available CMB component."""

    if cmb_data is None or getattr(cmb_data, "empty", True):
        return [f"{model_name} CMB residuals unavailable (no data)."]

    if isinstance(theory, np.ndarray):
        theory_map: Mapping[str, np.ndarray] = {}
        theory_map["TT"] = np.asarray(theory, dtype=float)
    else:
        theory_map = {}
        for key, val in theory.items():
            theory_map[key] = np.asarray(val, dtype=float)

    component_columns = {
        "TT": "Dl_obs",
        "TE": "Dl_te_obs",
        "EE": "Dl_ee_obs",
    }

    lines: list[str] = []
    for component, obs_col in component_columns.items():
        if component not in theory_map or obs_col not in cmb_data:
            continue

        observed = cmb_data[obs_col].to_numpy(dtype=float)
        predicted = theory_map[component]
        size = min(observed.size, predicted.size)
        if size == 0:
            continue
        observed = observed[:size]
        predicted = predicted[:size]
        residuals = observed - predicted
        rms, max_abs, median, n_points = _residual_statistics(residuals)
        if n_points == 0:
            continue
        lines.append(
            (
                f"{model_name} CMB {component} rms={rms:.3g}, "
                f"max={max_abs:.3g}, median={median:+.3g} (N={n_points})"
            )
        )

    if not lines:
        lines.append(
            f"{model_name} CMB residuals unavailable (mismatched components)."
        )

    return lines


__all__ = [
    "bao_residual_diagnostics",
    "cmb_residual_diagnostics",
]
