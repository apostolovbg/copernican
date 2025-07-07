"""Model coder that turns validated JSON into callable Python functions."""

# Every cosmological model is stored as JSON.  This module reads the sanitized
# JSON and uses SymPy to translate mathematical expressions into efficient
# NumPy-friendly functions.

import json
from pathlib import Path
import sympy as sp
import numpy as np
import camb
from scipy.integrate import quad
import logging
from sympy.printing.numpy import NumPyPrinter
from . import error_handler
from . import engine_interface


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
                    # ``Integral`` terms here are also expanded to calls to ``quad``.
                    rs_fn_sym = _compile_sympy_expr(rs_sym, tuple(param_syms))
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
        json.dump(model_data, f, indent=2)

    return funcs, model_data


# ==============================================================================
# Chi-squared helper functions
# ==============================================================================

def chi_squared_sne(cosmo_params, mu_model_func, sne_data_df):
    """Calculate chi-squared for SNe Ia data with or without covariance."""
    logger = logging.getLogger()
    if not all(col in sne_data_df.columns for col in ["zcmb", "mu_obs"]):
        logger.error("SNe DataFrame missing required columns 'zcmb' or 'mu_obs'.")
        return np.inf

    z_data = sne_data_df["zcmb"].values
    mu_obs = sne_data_df["mu_obs"].values

    try:
        mu_model = mu_model_func(z_data, *cosmo_params)
    except Exception:
        return np.inf

    if not isinstance(mu_model, np.ndarray) or mu_model.shape != mu_obs.shape or np.any(~np.isfinite(mu_model)):
        return np.inf

    resid = mu_obs - mu_model
    C_inv = sne_data_df.attrs.get("covariance_matrix_inv")

    if C_inv is not None:
        try:
            if C_inv.shape[0] != len(resid):
                logger.error("Covariance matrix dimension mismatch for SNe data.")
                return np.inf
            chi2 = float(resid @ C_inv @ resid)
        except Exception as exc:
            logger.warning(f"Falling back to diagonal errors due to covariance issue: {exc}")
            C_inv = None

    if C_inv is None:
        if "e_mu_obs" not in sne_data_df.columns:
            logger.error("No diagonal errors available for SNe data.")
            return np.inf
        err = sne_data_df["e_mu_obs"].values
        err = np.where(err <= 0, 1e-12, err)
        chi2 = np.sum((resid / err) ** 2)

    return chi2 if np.isfinite(chi2) else np.inf


def chi_squared_bao(bao_data_df, model_plugin, cosmo_params, model_rs_Mpc):
    """Calculate chi-squared for BAO observables."""
    logger = logging.getLogger()
    engine_interface.validate_plugin(model_plugin)
    if getattr(model_plugin, "valid_for_bao", True) is False:
        logger.warning("(chi2_bao): Model flagged as invalid for BAO. Skipping calculation.")
        return np.inf
    if bao_data_df is None or bao_data_df.empty:
        logger.error("(chi2_bao): BAO data is empty.")
        return np.inf
    if not (np.isfinite(model_rs_Mpc) and model_rs_Mpc > 0):
        return np.inf

    total = 0.0
    n_valid = 0

    try:
        get_DM = getattr(model_plugin, "get_comoving_distance_Mpc")
        get_Hz = getattr(model_plugin, "get_Hz_per_Mpc")
        get_DV = getattr(model_plugin, "get_DV_Mpc", None)
        C_LIGHT = model_plugin.FIXED_PARAMS.get("C_LIGHT_KM_S", 299792.458)
    except AttributeError as e:
        logger.error(f"(chi2_bao): Model plugin missing required function: {e}")
        return np.inf

    for _, row in bao_data_df.iterrows():
        z_val = row["redshift"]
        obs_type = row["observable_type"]
        obs_val = row["value"]
        obs_err = row["error"]

        if obs_err == 0 or not np.isfinite(obs_err) or obs_err < 1e-9:
            continue

        mod_num = np.nan
        try:
            if obs_type == "DM_over_rs":
                mod_num = get_DM(z_val, *cosmo_params)
            elif obs_type == "DH_over_rs":
                hz_val = get_Hz(z_val, *cosmo_params)
                if np.isfinite(hz_val) and abs(hz_val) > 1e-9:
                    mod_num = C_LIGHT / hz_val
            elif obs_type == "DV_over_rs":
                if get_DV:
                    mod_num = get_DV(z_val, *cosmo_params)
                else:
                    dm_val = get_DM(z_val, *cosmo_params)
                    hz_val = get_Hz(z_val, *cosmo_params)
                    if np.isfinite(dm_val) and dm_val >= 0 and np.isfinite(hz_val) and abs(hz_val) > 1e-9 and z_val > 1e-9:
                        term = (dm_val ** 2) * C_LIGHT * z_val / hz_val
                        mod_num = term ** (1.0 / 3.0) if term >= 0 else np.nan
                    elif abs(z_val) < 1e-9:
                        mod_num = 0.0
        except Exception:
            continue

        if np.isfinite(mod_num):
            total += ((obs_val - mod_num / model_rs_Mpc) / obs_err) ** 2
            n_valid += 1

    if n_valid == 0:
        logger.warning("(chi2_bao): No valid BAO points to calculate chi-squared.")
        return np.inf

    return total if np.isfinite(total) else np.inf


def compute_cmb_spectrum(param_dict, ells, spectra=("TT",)):
    """Return theoretical D_ell spectra using CAMB."""
    logger = logging.getLogger()
    try:
        H0 = float(param_dict.get("H0", 67.0))
        ombh2 = float(param_dict.get("ombh2", 0.02237))
        omch2 = float(param_dict.get("omch2", 0.12))
        tau = float(param_dict.get("tau", 0.054))
        As = float(param_dict.get("As", 2.1e-9))
        ns = float(param_dict.get("ns", 0.965))
        omnuh2 = float(param_dict.get("omnuh2", 0.0))
    except Exception as exc:
        logger.error(f"(compute_cmb_spectrum): Invalid parameter mapping: {exc}")
        return np.full_like(ells, np.nan, dtype=float)

    params = camb.CAMBparams()
    params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    params.omnuh2 = omnuh2
    params.InitPower.set_params(As=As, ns=ns)
    params.set_for_lmax(int(np.max(ells)) + 300, lens_potential_accuracy=0)
    try:
        results = camb.get_results(params)
        full_dls = results.get_unlensed_scalar_cls(
            lmax=int(np.max(ells)), CMB_unit="muK"
        )

        ell_arr = np.asarray(ells, dtype=int)
        result = {}
        if "TT" in spectra:
            result["TT"] = full_dls[ell_arr, 0]
        if "EE" in spectra:
            result["EE"] = full_dls[ell_arr, 1]
        if "TE" in spectra:
            result["TE"] = full_dls[ell_arr, 3]

        if len(result) == 1:
            return next(iter(result.values()))
        return result
    except Exception as exc:
        logger.error(f"(compute_cmb_spectrum): CAMB failed: {exc}")
        return np.full_like(ells, np.nan, dtype=float)


def chi_squared_cmb(cosmo_params, cmb_data_df, plugin=None, extra_params=None):
    """Calculate chi-squared for CMB power spectrum."""
    logger = logging.getLogger()
    if cmb_data_df is None or cmb_data_df.empty:
        logger.error("(chi2_cmb): CMB data is empty.")
        return np.inf
    if "covariance_matrix_inv" not in cmb_data_df.attrs:
        logger.error("(chi2_cmb): Inverse covariance matrix missing in attrs.")
        return np.inf

    ells = cmb_data_df["ell"].values
    obs = cmb_data_df["Dl_obs"].values

    if plugin is not None:
        try:
            param_dict = plugin.get_camb_params(cosmo_params)
        except Exception as exc:
            logger.error(f"(chi2_cmb): failed to map parameters: {exc}")
            return np.inf
    else:
        if isinstance(cosmo_params, dict):
            param_dict = dict(cosmo_params)
        else:
            names = cmb_data_df.attrs.get("param_names", [])
            param_dict = {n: v for n, v in zip(names, cosmo_params)}

    if extra_params:
        param_dict.update(extra_params)

    th = compute_cmb_spectrum(param_dict, ells, spectra=("TT",))
    if th.shape != obs.shape or np.any(~np.isfinite(th)):
        return np.inf

    resid = obs - th
    C_inv = cmb_data_df.attrs["covariance_matrix_inv"]
    try:
        chi2 = float(resid @ C_inv @ resid)
    except Exception as exc:
        logger.error(f"(chi2_cmb): Linear algebra failure: {exc}")
        return np.inf

    return chi2 if np.isfinite(chi2) else np.inf
