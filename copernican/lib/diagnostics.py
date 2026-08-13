"""Logging-oriented diagnostics helpers.

The utilities below translate raw numerical arrays into short, stable log
messages so long-running analyses can surface meaningful progress without
flooding the console.  They intentionally avoid performing any logging
themselves, allowing callers to decide where and how the messages should be
displayed or captured during tests.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy
import pandas

from .cmb_output import cmb_observation_blocks, cmb_theory_values_for_block


def _residual_statistics(
    residuals: numpy.ndarray,
) -> tuple[float, float, float, int]:
    """Return RMS, max absolute value, median and sample size.

    ``residuals`` may contain NaNs from masked measurements.  Only finite
    entries contribute to the statistics so diagnostics remain robust even when
    parts of the dataset are disabled for a specific model.
    """

    mask = numpy.isfinite(residuals)
    if not numpy.any(mask):
        return float("nan"), float("nan"), float("nan"), 0

    cleaned = residuals[mask]
    rms = float(numpy.sqrt(numpy.mean(cleaned**2)))
    max_abs = float(numpy.max(numpy.abs(cleaned)))
    median = float(numpy.median(cleaned))
    return rms, max_abs, median, int(cleaned.size)


def bao_residual_diagnostics(
    predictions: pandas.DataFrame | None,
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
    observed_values = predictions["value"].to_numpy(dtype=float)
    residuals = model_vals - observed_values
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
            group_observed = group["value"].to_numpy(dtype=float)
            group_resid = group_vals - group_observed
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
    cmb_data: pandas.DataFrame | None,
    theory: Mapping[str, Iterable[float]] | numpy.ndarray,
    *,
    model_name: str,
) -> list[str]:
    """Return residual diagnostics for each available CMB component."""

    if cmb_data is None or getattr(cmb_data, "empty", True):
        return [f"{model_name} CMB residuals unavailable (no data)."]

    lines: list[str] = []
    for block in cmb_observation_blocks(cmb_data):
        try:
            predicted = cmb_theory_values_for_block(
                theory,
                block,
                total_row_count=len(cmb_data),
            )
        except (KeyError, TypeError, ValueError):
            continue
        residuals = block.observed - predicted
        rms, max_abs, median, n_points = _residual_statistics(residuals)
        if n_points == 0:
            continue
        lines.append(
            (
                f"{model_name} CMB "
                f"{block.metadata.canonical_name} rms={rms:.3g}, "
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
