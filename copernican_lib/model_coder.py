# """Model coder that turns validated YAML into callable Python functions."""

# Every cosmological model is stored as YAML.  This module reads the sanitized
# YAML and uses SymPy to translate mathematical expressions into efficient
# NumPy-friendly functions.

import yaml
from pathlib import Path
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
import numpy as np
from scipy.integrate import quad
import logging
from sympy.printing.numpy import NumPyPrinter
from . import error_handler
from . import engine_interface
from . import latex_utils


class QuadPrinter(NumPyPrinter):
    """NumPy printer that expands ``Integral`` nodes into ``scipy`` quad calls."""

    def _print_Integral(self, expr):
        # Currently support single-variable integrals of the form (var, a, b).
        var, a, b = expr.limits[0]
        integrand = expr.function
        var_code = self._print(var)
        a_code = self._print(a)
        b_code = self._print(b)
        integrand_code = self._print(integrand)
        return f"quad(lambda {var_code}: {integrand_code}, {a_code}, {b_code})[0]"


def _latex_to_sympy_str(expr: str) -> str:
    """Convert a LaTeX-style expression to a SymPy-friendly string."""
    return latex_utils.latex_to_sympy(expr)


def _compile_sympy_expr(sym_expr, args):
    """Compile a SymPy expression into a callable handling ``Integral`` nodes."""
    # SymPy can convert expressions directly to Python functions, but it does
    # not evaluate ``Integral`` objects by default. When an integral is present
    # we expand it into a call to ``scipy.integrate.quad`` so the resulting
    # callable is fully numerical.
    if sym_expr.atoms(sp.Integral):
        printer = QuadPrinter({'strict': False})
        code = printer.doprint(sym_expr)
        lambda_src = f"lambda {', '.join(str(a) for a in args)}: {code}"
        return eval(lambda_src, {'np': np, 'quad': quad})
    return sp.lambdify(args, sym_expr, 'numpy')


def generate_callables(cache_path):
    """Create callables from the cached model and update the cache file.

    Parameters
    ----------
    cache_path : str or Path
        Path to the sanitized model YAML produced by :func:`parse_model`.

    Returns
    -------
    tuple(dict, dict)
        Dictionary of callables and the loaded YAML data.
    """
    cache_path = Path(cache_path)
    with cache_path.open("r") as f:
        model_data = yaml.safe_load(f)

    logger = logging.getLogger()

    z = sp.symbols('z')
    param_syms = [sp.symbols(p['python_var']) for p in model_data['parameters']]
    local_dict = {p['python_var']: sym for p, sym in zip(model_data['parameters'], param_syms)}
    local_dict['z'] = z
    # Allow YAML equations to reference the full 'sympy' prefix as well as shorthand
    local_dict['sympy'] = sp

    funcs = {}
    code_dict = {}

    hz_expr_str = model_data.get('Hz_expression')
    if hz_expr_str:
        try:
            parsed_hz = _latex_to_sympy_str(hz_expr_str)
            hz_sym = parse_expr(
                parsed_hz,
                local_dict,
                transformations=standard_transformations
                + (implicit_multiplication_application,),
            )
            used_syms = {str(s) for s in hz_sym.free_symbols if s != z}
            param_names = {p['python_var'] for p in model_data['parameters']}
            missing = used_syms - param_names
            if missing:
                raise ValueError(
                    "Parameter '" + "', '".join(missing) + "' used in Hz_expression is not defined in model parameters."
                )
            # Convert SymPy expression to NumPy callable. Any ``Integral`` terms
            # are replaced with numerical quad evaluations.
            hz_fn = _compile_sympy_expr(hz_sym, (z, *param_syms))
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
            logger.info("Derived distance functions from symbolic Hz_expression in model YAML.")

            # --- Derive sound horizon at recombination (r_s) ---
            rs_expr_str = model_data.get('rs_expression')
            param_names = {p['python_var'] for p in model_data['parameters']}
            param_index = {p['python_var']: i for i, p in enumerate(model_data['parameters'])}

            if rs_expr_str:
                try:
                    parsed_rs = _latex_to_sympy_str(rs_expr_str)
                    rs_sym = parse_expr(
                        parsed_rs,
                        local_dict,
                        transformations=standard_transformations
                        + (implicit_multiplication_application,),
                    )
                    used = {str(s) for s in rs_sym.free_symbols} - {'z'}
                    missing_rs = used - param_names
                    if missing_rs:
                        raise ValueError(
                            "Parameter '" + "', '".join(missing_rs) + "' used in rs_expression is not defined in model parameters."
                        )
                    # ``Integral`` terms here are also expanded to calls to ``quad``.
                    rs_fn_sym = _compile_sympy_expr(rs_sym, tuple(param_syms))
                    funcs['get_sound_horizon_rs_Mpc'] = lambda *p: float(rs_fn_sym(*p))
                    code_dict['get_sound_horizon_rs_Mpc'] = str(rs_sym)
                    model_data['valid_for_bao'] = True
                    logger.info("Derived r_s from symbolic rs_expression in model YAML.")
                except Exception as e:
                    error_handler.report_error(f"Failed to parse rs_expression: {e}")
                    raise ValueError(f"Failed to parse rs_expression: {e}") from e
            elif (
                'Omega_b' in param_names
                and 'Omega_gamma' in param_names
                and ('z_rec' in param_names or 'z_recomb' in param_names)
                and 'get_Hz_per_Mpc' in funcs
            ):
                ob_i = param_index['Omega_b']
                og_i = param_index['Omega_gamma']
                zr_key = 'z_rec' if 'z_rec' in param_names else 'z_recomb'
                zr_i = param_index[zr_key]

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
            sym_expr = parse_expr(
                expr,
                local_dict,
                transformations=standard_transformations
                + (implicit_multiplication_application,),
            )
            # Convert SymPy expression to a callable, numerically evaluating
            # ``Integral`` constructs if present.
            fn = _compile_sympy_expr(sym_expr, (z, *param_syms))
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
        yaml.safe_dump(model_data, f, sort_keys=False, allow_unicode=True)

    return funcs, model_data


