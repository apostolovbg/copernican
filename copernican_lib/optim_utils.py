# optim_utils.py
"""Optimization utilities for the Copernican Suite.

This module centralizes common wrappers used during numerical
optimisation. Engines can import these helpers to keep the engine code
focused strictly on mathematical calculations without bookkeeping.
"""

# At the moment the main helper ``minimize_with_progress`` wraps SciPy's
# ``minimize`` function and prints a live progress indicator. It also tracks the
# best solution seen so that a reasonable result is returned even if the
# optimiser fails.

from typing import Iterable, Tuple, Callable, Any, Optional, List
import sys
import time
import logging
from scipy.optimize import minimize
import numpy as np


def minimize_with_progress(
    func: Callable[[Iterable, Any], float],
    x0: Iterable,
    bounds: Iterable[Tuple[float, float]],
    args: Tuple = (),
    options: Optional[dict] = None,
    method: str = "L-BFGS-B",
    label: str = "Fit",
) -> Tuple[Optional[object], int, float, List[float]]:
    """Run :func:`scipy.optimize.minimize` while tracking evaluations.

    Parameters
    ----------
    func : callable
        Objective function returning a numeric value.
    x0 : sequence
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

    Returns
    -------
    result : OptimizeResult or ``None``
        The object returned by ``minimize`` or ``None`` when an exception
        occurred.
    evals : int
        Total number of function evaluations performed.
    best_val : float
        Lowest finite objective value seen during the search.
    best_params : list of float
        Parameter vector associated with ``best_val``.
    """

    logger = logging.getLogger()
    eval_count = {"count": 0}
    best_val = [np.inf]
    best_params = [list(x0)]
    start_time = time.time()

    def wrapped(p, *wrapped_args):
        """Internal function that records progress."""
        eval_count["count"] += 1
        val = func(p, *wrapped_args)
        if not np.isfinite(val):
            val = np.inf
        if val < best_val[0]:
            best_val[0] = float(val)
            best_params[0] = list(p)
        elapsed = time.time() - start_time
        rate = (
            f"{eval_count['count'] / elapsed:.1f} evals/s" if elapsed > 1e-6 else "--- evals/s"
        )
        print(
            f"  {label} Evals: {eval_count['count']:<5} | Best Chi2: {best_val[0]:.4f} | Speed: {rate:<15}",
            end="\r",
            file=sys.stderr,
        )
        return val if np.isfinite(val) else 1e12

    result = None
    if options is None:
        options = {}
    if 'eps' not in options:
        eps = np.maximum(1e-8, np.abs(x0) * 1e-4)
        options['eps'] = eps
    try:
        result = minimize(
            wrapped,
            x0,
            args=args,
            method=method,
            bounds=bounds,
            options=options,
        )
    except Exception as exc:  # pragma: no cover - hard to trigger in tests
        logger.error(f"Exception during {label.lower()} minimize call: {exc}", exc_info=True)
    finally:
        # Clear the progress line so subsequent prints start on a clean line
        print(" " * 80, end="\r", file=sys.stderr)
        logger.info(f"{label} optimization finished. Total evals: {eval_count['count']}.")

    return result, eval_count["count"], best_val[0], best_params[0]
