"""Model coder that turns validated JSON into callable Python functions."""
# DEV NOTE (v1.5f hotfix 7): Parses ``Hz_expression`` from each model and
# generates a callable ``get_Hz_per_Mpc`` along with optional distance
# functions. The generated SymPy expressions are written back to the cache.
# Previous hotfix notes: loads sanitized models from ``models/cache`` and allows
# "sympy." prefix in JSON equations.
# DEV NOTE (v1.5f hotfix 8): Adds ``rs_expression`` handling and a fallback
# numerical integral for the sound horizon when model parameters ``Ob``,
# ``Og`` and ``z_recomb`` are present.

import json
from pathlib import Path
import sympy as sp
import numpy as np
from scipy.integrate import quad
import logging
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

    logger = logging.getLogger()

    z = sp.symbols('z')
    param_syms = [sp.symbols(p['python_var']) for p in model_data['parameters']]
    local_dict = {p['python_var']: sym for p, sym in zip(model_data['parameters'], param_syms)}
    local_dict['z'] = z
    # Allow JSON equations to reference the full 'sympy' prefix as well as shorthand
    local_dict['sympy'] = sp

    funcs = {}
    code_dict = {}

    hz_expr_str = model_data.get('Hz_expression')
    if hz_expr_str:
        try:
            hz_sym = sp.sympify(hz_expr_str, locals=local_dict)
            used_syms = {str(s) for s in hz_sym.free_symbols if s != z}
            param_names = {p['python_var'] for p in model_data['parameters']}
            missing = used_syms - param_names
            if missing:
                raise ValueError(
                    "Parameter '" + "', '".join(missing) + "' used in Hz_expression is not defined in model parameters."
                )
            hz_fn = sp.lambdify((z, *param_syms), hz_sym, 'numpy')
            funcs['get_Hz_per_Mpc'] = hz_fn
            code_dict['get_Hz_per_Mpc'] = str(hz_sym)
            model_data['valid_for_distance_metrics'] = True

            def _dm(z_val, *params):
                integrand = lambda zp: 299792.458 / hz_fn(zp, *params)
                return quad(integrand, 0, z_val, limit=100)[0]

            if 'get_comoving_distance_Mpc' not in funcs:
                funcs['get_comoving_distance_Mpc'] = _dm
                code_dict['get_comoving_distance_Mpc'] = 'integral(c/H(z))'
            if 'get_luminosity_distance_Mpc' not in funcs:
                funcs['get_luminosity_distance_Mpc'] = lambda zv, *p: (1 + zv) * _dm(zv, *p)
                code_dict['get_luminosity_distance_Mpc'] = '(1+z)*DC'
            if 'get_angular_diameter_distance_Mpc' not in funcs:
                funcs['get_angular_diameter_distance_Mpc'] = lambda zv, *p: _dm(zv, *p) / (1 + zv)
                code_dict['get_angular_diameter_distance_Mpc'] = 'DC/(1+z)'
            if 'get_DV_Mpc' not in funcs:
                funcs['get_DV_Mpc'] = lambda zv, *p: ((
                    _dm(zv, *p) ** 2 * 299792.458 * zv / hz_fn(zv, *p)
                ) ** (1 / 3) if zv > 0 and hz_fn(zv, *p) != 0 else 0.0)
                code_dict['get_DV_Mpc'] = '((DC^2 * c*z/H)^1/3)'
            logger.info("Derived distance functions from symbolic Hz_expression in model JSON.")

            # --- Derive sound horizon at recombination (r_s) ---
            rs_expr_str = model_data.get('rs_expression')
            param_names = {p['python_var'] for p in model_data['parameters']}
            param_index = {p['python_var']: i for i, p in enumerate(model_data['parameters'])}

            if rs_expr_str:
                try:
                    rs_sym = sp.sympify(rs_expr_str, locals=local_dict)
                    used = {str(s) for s in rs_sym.free_symbols} - {'z'}
                    missing_rs = used - param_names
                    if missing_rs:
                        raise ValueError(
                            "Parameter '" + "', '".join(missing_rs) + "' used in rs_expression is not defined in model parameters."
                        )
                    rs_fn_sym = sp.lambdify(tuple(param_syms), rs_sym, 'numpy')
                    funcs['get_sound_horizon_rs_Mpc'] = lambda *p: float(rs_fn_sym(*p))
                    code_dict['get_sound_horizon_rs_Mpc'] = str(rs_sym)
                    model_data['valid_for_bao'] = True
                    logger.info("Derived r_s from symbolic rs_expression in model JSON.")
                except Exception as e:
                    error_handler.report_error(f"Failed to parse rs_expression: {e}")
                    raise ValueError(f"Failed to parse rs_expression: {e}") from e
            elif {'Ob', 'Og', 'z_recomb'}.issubset(param_names) and 'get_Hz_per_Mpc' in funcs:
                ob_i = param_index['Ob']
                og_i = param_index['Og']
                zr_i = param_index['z_recomb']

                def _rs(*params):
                    Ob_val = params[ob_i]
                    Og_val = params[og_i]
                    zrec = params[zr_i]

                    def sound_speed(zv):
                        return 299792.458 / np.sqrt(3 * (1 + 3 * Ob_val / (4 * Og_val) / (1 + zv)))

                    integrand = lambda zv: sound_speed(zv) / hz_fn(zv, *params)
                    result, _ = quad(integrand, zrec, np.inf, limit=100)
                    return result

                funcs['get_sound_horizon_rs_Mpc'] = _rs
                code_dict['get_sound_horizon_rs_Mpc'] = 'quad(c_s/H(z))'
                model_data['valid_for_bao'] = True
                logger.info("Derived r_s using fallback integral from Hz_expression.")
            else:
                print(
                    "\u26A0\uFE0F  Model does not define all necessary parameters for computing r_s. BAO scaling may be unavailable."
                )
                model_data['valid_for_bao'] = False
        except Exception as e:
            error_handler.report_error(f"Failed to parse Hz_expression: {e}")
            raise ValueError(f"Failed to parse Hz_expression: {e}") from e
    else:
        print(
            "\u26A0\uFE0F  Model does not define H(z). Distance-based observables such as BAO, comoving distances, and luminosity distances will be unavailable."
        )
        model_data['valid_for_distance_metrics'] = False
        model_data['valid_for_bao'] = False
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
