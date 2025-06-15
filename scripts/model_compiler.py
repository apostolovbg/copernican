"""Compile sanitized model data into executable Python callables."""
# DEV NOTE (v1.5e): Initial SymPy compiler with basic
# validation and safety checks from Phase 2.

import json
import numpy as np
from sympy import symbols, sympify, lambdify
from . import error_handler


def compile_cached_model(cache_path):
    """Return dictionary of callables compiled from a cached model JSON."""
    try:
        with open(cache_path, 'r', encoding='utf-8') as handle:
            model = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        error_handler.report_error(f"Failed to read cache '{cache_path}': {exc}")
        return {}

    param_names = [p.get("name") for p in model.get("parameters", [])]
    param_syms = symbols(param_names)
    z = symbols('z')
    constants = model.get("constants", {})

    locals_dict = {name: sym for name, sym in zip(param_names, param_syms)}
    locals_dict.update(constants)

    def _safe_callable(expr):
        """Wrap lambdified expression with error and NaN handling."""
        func = lambdify((z, *param_syms), expr, 'numpy')

        def wrapper(z_array, *params):
            try:
                result = func(z_array, *params)
                arr = np.asarray(result, dtype=float)
                if np.any(np.isnan(arr) | np.isinf(arr)):
                    raise ZeroDivisionError("Numerical instability detected")
                return arr
            except Exception as exc:
                error_handler.report_error(
                    f"Computation failed for expression '{expr}': {exc}")
                return np.full_like(np.asarray(z_array, dtype=float), np.nan)

        return wrapper

    compiled = {"sne": [], "bao": []}

    for label in ("sne", "bao"):
        for eq in model.get("equations", {}).get(label, []):
            try:
                rhs = eq.split('=', 1)[1] if '=' in eq else eq
                expr = sympify(rhs, locals=locals_dict)
                # simple numerical test using initial guesses
                guesses = [p.get("guess") for p in model.get("parameters", [])]
                test_func = _safe_callable(expr)
                test_val = test_func(0.1, *guesses)
                if np.any(np.isnan(test_val) | np.isinf(test_val)):
                    raise ValueError("invalid numeric result")
                compiled[label].append(_safe_callable(expr))
            except Exception as exc:
                error_handler.report_error(
                    f"Failed to compile equation '{eq}' in {label}: {exc}")
    return compiled
