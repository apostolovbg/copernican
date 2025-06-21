"""Model coder that turns validated JSON into callable Python functions."""
# DEV NOTE (hotfix): uses parameter bounds to test equations and ignores
# non-string entries under ``equations``.
# DEV NOTE (v1.5f hotfix 7): Parses ``Hz_expression`` from each model and
# generates a callable ``get_Hz_per_Mpc`` along with optional distance
# functions. The generated SymPy expressions are written back to the cache.
# Previous hotfix notes: loads sanitized models from ``models/cache`` and allows
# "sympy." prefix in JSON equations.
# DEV NOTE (v1.5f hotfix 8): Adds ``rs_expression`` handling and a fallback
# numerical integral for the sound horizon when model parameters ``Ob``,
# ``Og`` and ``z_recomb`` are present.
# DEV NOTE (v1.5f hotfix 10): ``_dm`` now accepts array inputs for BAO smooth
# curve generation.
# DEV NOTE (v1.5f hotfix 11): Automatically derives ``distance_modulus_model``
# from ``get_luminosity_distance_Mpc`` when not provided by the JSON model.
# DEV NOTE (v1.5f hotfix 12): ``get_DV_Mpc`` now handles NumPy arrays for
# smooth BAO curve generation without runtime errors.
# DEV NOTE (v1.5f hotfix 13): The fallback ``get_sound_horizon_rs_Mpc``
# integral now adds radiation density to the model's ``H(z)`` to produce
# realistic sound horizon values.

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
                """Comoving distance integral valid for scalars or arrays."""
                integrand = lambda zp: 299792.458 / hz_fn(zp, *params)
                if np.isscalar(z_val):
                    # ``quad`` expects scalar limits; cast to float explicitly.
                    return quad(integrand, 0, float(z_val), limit=100)[0]

                # For arrays, compute the integral element-wise and
                # preserve the input shape.
                z_flat = np.ravel(z_val)
                results = [quad(integrand, 0, float(z), limit=100)[0] for z in z_flat]
                return np.reshape(results, np.shape(z_val))

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
                def _dv(z_val, *params):
                    """Volume-averaged distance D_V valid for scalars or arrays."""
                    dm_val = _dm(z_val, *params)
                    hz_val = hz_fn(z_val, *params)

                    term = dm_val ** 2 * 299792.458 * z_val / hz_val

                    if np.isscalar(z_val):
                        if z_val > 0 and hz_val != 0:
                            return term ** (1 / 3) if term >= 0 else np.nan
                        return 0.0

                    result = np.zeros_like(z_val, dtype=float)
                    mask = (z_val > 0) & (hz_val != 0)
                    term_arr = term[mask]
                    result[mask] = np.where(term_arr >= 0, np.power(term_arr, 1/3), np.nan)
                    return result

                funcs['get_DV_Mpc'] = _dv
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
                    """Numerically compute the sound horizon r_s in Mpc."""
                    Ob_val = params[ob_i]
                    Og_val = params[og_i]
                    zrec = params[zr_i]

                    def sound_speed(zv):
                        return 299792.458 / np.sqrt(3 * (1 + 3 * Ob_val / (4 * Og_val) / (1 + zv)))

                    h0_val = hz_fn(0.0, *params)

                    def hz_with_radiation(zv):
                        base = hz_fn(zv, *params)
                        rad_sq = (h0_val ** 2) * Og_val * (1 + zv) ** 4
                        return np.sqrt(base ** 2 + rad_sq)

                    integrand = lambda zv: sound_speed(zv) / hz_with_radiation(zv)
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
        if not isinstance(expr, str):
            # Textual equations are preserved but not parsed into functions
            continue
        try:
            sym_expr = sp.sympify(expr, locals=local_dict)
            fn = sp.lambdify((z, *param_syms), sym_expr, 'numpy')
            # Quick sanity evaluation using midpoints of parameter bounds
            try:
                mid_params = tuple(sum(p['bounds']) / 2.0 for p in model_data['parameters'])
                test_args = (0.5,) + mid_params
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

    if (
        'distance_modulus_model' not in funcs
        and 'get_luminosity_distance_Mpc' in funcs
    ):
        def _mu(zv, *params):
            dl = funcs['get_luminosity_distance_Mpc'](zv, *params)
            with np.errstate(divide='ignore', invalid='ignore'):
                mu = 5 * np.log10(dl) + 25.0
            mu = np.where(np.asarray(dl) > 0, mu, np.nan)
            return mu

        funcs['distance_modulus_model'] = _mu
        code_dict['distance_modulus_model'] = '5*log10(DL_Mpc)+25'
        logger.info("Derived distance_modulus_model from luminosity distance.")

    model_data['generated_code'] = code_dict
    with cache_path.open("w") as f:
        json.dump(model_data, f, indent=2)

    return funcs, model_data
