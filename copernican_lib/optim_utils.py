# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# optim_utils.py
"""Optimization utilities for the Copernican Suite.

This module centralizes common wrappers used during numerical
optimisation. Engines can import these helpers to keep the engine code
focused strictly on mathematical calculations without bookkeeping.  The
helpers here also standardise progress reporting and resiliency so that
optimisers always yield the best-so-far solution even when they terminate
early or encounter errors.
"""

# At the moment the main helper ``minimize_with_progress`` wraps SciPy's
# ``minimize`` function and prints a live progress indicator.
# It also tracks the best solution seen so that a reasonable result is
# returned even if the optimiser fails.

import logging
import time
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from . import console_output as console


def minimize_with_progress(
    func: Callable[[Iterable, Any], float],
    initial_guess: Iterable,
    bounds: Iterable[Tuple[float, float]],
    args: Tuple = (),
    options: Optional[dict] = None,
    method: str = "L-BFGS-B",
    label: str = "Fit",
) -> Tuple[Optional[object], int, float, List[float]]:
    """Run :func:`scipy.optimize.minimize` while tracking evaluations.

    A tiny NumPy/SciPy operation is attempted first to ensure the compiled
    extensions load correctly. Failures often stem from CPU feature
    mismatches and the log advises reinstalling compatible wheels.

    Parameters
    ----------
    func : callable
        Objective function returning a numeric value.
    initial_guess : sequence
        Initial parameter guesses for the optimiser.
    bounds : sequence of tuple
        Bounds for each parameter.
    args : tuple, optional
        Extra positional arguments for ``func``.
    options : dict, optional
        Options forwarded to :func:`~scipy.optimize.minimize`.
    method : str, optional
        Optimisation method.  ``"L-BFGS-B"`` is used by default.
    label : str, optional
        Human readable label shown in the live progress display.

    Notes
    -----
    Progress updates are throttled so the console is not flooded with
    output.  The status line refreshes only after ten new evaluations or
    when at least half a second has elapsed since the previous update,
    whichever comes first.

    Returns
    -------
    result : OptimizeResult or ``None``
        The object returned by ``minimize`` or ``None`` when an exception
        occurred.
    evals : int
        Total number of function evaluations performed.
    best_objective_value : float
        Lowest finite objective value seen during the search.
    best_parameter_vector : list of float
        Parameter vector associated with ``best_objective_value``.
    """

    logger = logging.getLogger()
    try:
        np.dot(np.ones(1), np.ones(1))
        from scipy import linalg as _linalg

        _linalg.det([[1.0]])
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.error(
            "Basic NumPy/SciPy check failed. This may indicate CPU feature "
            "mismatches or a corrupted install. Reinstall with wheels "
            "built for your system.",
            exc_info=exc,
        )
        return None, 0, float("inf"), list(initial_guess)
    eval_count = {"count": 0}
    best_objective_value = [np.inf]
    best_parameter_vector = [list(initial_guess)]
    start_time = time.time()
    last_update = start_time
    last_eval = 0

    def wrapped(parameter_vector, *wrapped_args):
        """Internal function that records progress."""

        nonlocal last_update, last_eval

        eval_count["count"] += 1
        objective_value = func(parameter_vector, *wrapped_args)
        if not np.isfinite(objective_value):
            objective_value = np.inf
        if objective_value < best_objective_value[0]:
            best_objective_value[0] = float(objective_value)
            best_parameter_vector[0] = list(parameter_vector)

        now = time.time()
        need_update = False
        if eval_count["count"] - last_eval >= 10:
            need_update = True
        if now - last_update >= 0.5:
            need_update = True
        if need_update:
            elapsed = now - start_time
            rate = (
                f"{eval_count['count'] / elapsed:.1f} evals/s"
                if elapsed > 1e-6
                else "--- evals/s"
            )
            console.write(
                f"  {label} Evals: {eval_count['count']:<5} | Best Chi2: "
                f"{best_objective_value[0]:.4f} | Speed: {rate:<15}",
                end="\r",
                error=False,
            )
            last_update = now
            last_eval = eval_count["count"]

        return (
            objective_value if np.isfinite(objective_value) else 1e12
        )

    result = None
    try:
        result = minimize(
            wrapped,
            initial_guess,
            args=args,
            method=method,
            bounds=bounds,
            options=options or {},
        )
    except Exception as exc:  # pragma: no cover - hard to trigger in tests
        logger.error(
            f"Exception during {label.lower()} minimize call: {exc}",
            exc_info=True,
        )
    finally:
        # Clear the progress line so subsequent prints start on a clean line
        console.write(" " * 80, end="\r", error=False)
        logger.info(
            "%s optimization finished. Total evals: %s.",
            label,
            eval_count["count"],
        )

    return (
        result,
        eval_count["count"],
        best_objective_value[0],
        best_parameter_vector[0],
    )
