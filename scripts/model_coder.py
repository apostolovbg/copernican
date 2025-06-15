"""Model coder that turns validated JSON into callable Python functions."""
# DEV NOTE (v1.5e): Loads sanitized models from ``models/cache`` and stores the
# generated SymPy expressions back to that cache.

import json
from pathlib import Path
import sympy as sp
from . import error_handler


def generate_callables(cache_path):
    """Create callables from the cached model and update the cache file.

    Parameters
    ----------
    cache_path : str or Path
        Path to the sanitized model JSON produced by :func:`parse_model_json`.

    Returns
    -------
    tuple(dict, dict)
        Dictionary of callables and the loaded JSON data.
    """
    cache_path = Path(cache_path)
    with cache_path.open("r") as f:
        model_data = json.load(f)

    z = sp.symbols('z')
    param_syms = [sp.symbols(p['python_var']) for p in model_data['parameters']]
    local_dict = {p['python_var']: sym for p, sym in zip(model_data['parameters'], param_syms)}
    local_dict['z'] = z

    funcs = {}
    code_dict = {}
    for name, expr in model_data.get('equations', {}).items():
        try:
            sym_expr = sp.sympify(expr, locals=local_dict)
            fn = sp.lambdify((z, *param_syms), sym_expr, 'numpy')
            # Quick sanity evaluation to catch division by zero or bad symbols
            try:
                test_args = (0.5,) + tuple(p['initial_guess'] for p in model_data['parameters'])
                fn(*test_args)
            except Exception as eval_e:
                error_handler.report_error(
                    f"Generated function '{name}' raised an error when tested: {eval_e}"
                )
            funcs[name] = fn
            code_dict[name] = str(sym_expr)
        except Exception as e:
            error_handler.report_error(f"Failed to parse equation '{name}': {e}")
            raise ValueError(f"Failed to parse equation '{name}': {e}") from e

    model_data['generated_code'] = code_dict
    with cache_path.open("w") as f:
        json.dump(model_data, f, indent=2)

    return funcs, model_data
