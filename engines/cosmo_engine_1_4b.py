# copernican_suite/cosmo_engine.py
"""
Cosmological Engine for the Copernican Suite.
Relies on SciPy/NumPy for all computations.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import LinAlgError
import camb
import sys
import time
import logging
from scripts import engine_interface

# --- Constants for SNe H2-style (SALT2 nuisance parameter fitting) ---
SALT2_NUISANCE_PARAMS_INIT = {
    "M_B": -19.3,
    "alpha_salt2": 0.14,
    "beta_salt2": 3.1
}
SALT2_NUISANCE_PARAMS_BOUNDS = {
    "M_B": (-20.5, -18.0),
    "alpha_salt2": (-0.5, 0.5),
    "beta_salt2": (0.0, 5.0)
}
SIGMA_INT_SQ_DEFAULT = 0.1**2


# ==============================================================================
# --- CHI-SQUARED HELPER FUNCTIONS ---
# ==============================================================================

def chi_squared_sne_h1_fixed_nuisance(cosmo_params, mu_model_func, sne_data_df):
    r"""
    Calculates chi-squared for SNe Ia: H1-style (fixed nuisance).
    This uses pre-calculated mu_obs and diagonal errors e_mu_obs.
    $\chi^2 = \sum ((mu_data - mu_model) / e_mu_obs)^2$.
    """
    logger = logging.getLogger()
    if not all(col in sne_data_df.columns for col in ['zcmb', 'mu_obs', 'e_mu_obs']):
        logger.error("(chi2_h1): sne_data_df missing required columns 'zcmb', 'mu_obs', 'e_mu_obs'.")
        return np.inf

    z_data = sne_data_df['zcmb'].values
    mu_data = sne_data_df['mu_obs'].values
    mu_err_diag = sne_data_df['e_mu_obs'].values

    try:
        mu_model = mu_model_func(z_data, *cosmo_params)
    except Exception as e:
        return np.inf # Fitter will handle this

    if not isinstance(mu_model, np.ndarray) or mu_model.shape != mu_data.shape:
        return np.inf
    if np.any(~np.isfinite(mu_model)):
        return np.inf

    residuals = mu_data - mu_model
    safe_mu_err = np.where(np.abs(mu_err_diag) < 1e-12, 1e-12, np.abs(mu_err_diag))
    if np.any(safe_mu_err <= 0):
        return np.inf

    chi2 = np.sum((residuals / safe_mu_err)**2)
    return chi2 if np.isfinite(chi2) else np.inf


def chi_squared_sne_h2_salt2_fitting(params_full, mu_model_func, sne_data_df, num_cosmo_params):
    r"""
    Calculates chi-squared for SNe Ia: H2-style (SALT2 m_b, x1, c fitting).
    """
    logger = logging.getLogger()
    req_cols = ['zcmb', 'mb', 'x1', 'c', 'e_mb', 'e_x1', 'e_c']
    if not all(col in sne_data_df.columns for col in req_cols):
        logger.error(f"(chi2_h2_salt2): sne_data_df missing one of required columns: {req_cols}.")
        return np.inf

    z_data = sne_data_df['zcmb'].values; mb_data = sne_data_df['mb'].values
    x1_data = sne_data_df['x1'].values; c_data = sne_data_df['c'].values
    err_mb_data = sne_data_df['e_mb'].values; err_x1_data = sne_data_df['e_x1'].values
    err_c_data = sne_data_df['e_c'].values
    
    if not all(len(arr) == len(z_data) for arr in [mb_data,x1_data,c_data,err_mb_data,err_x1_data,err_c_data]):
        logger.error("(chi2_h2_salt2): Data array length mismatch.")
        return np.inf

    cosmo_params = params_full[:num_cosmo_params]
    M_B_fit, alpha_salt2_fit, beta_salt2_fit = params_full[num_cosmo_params : num_cosmo_params+3]

    try:
        mu_cosmo_model = mu_model_func(z_data, *cosmo_params)
    except Exception:
        return np.inf

    if not isinstance(mu_cosmo_model, np.ndarray) or mu_cosmo_model.shape != z_data.shape or np.any(~np.isfinite(mu_cosmo_model)):
        return np.inf

    mb_model = M_B_fit - alpha_salt2_fit * x1_data + beta_salt2_fit * c_data + mu_cosmo_model
    residuals = mb_data - mb_model
    
    sigma_eff_sq = (err_mb_data**2) + (alpha_salt2_fit * err_x1_data)**2 + (beta_salt2_fit * err_c_data)**2 + SIGMA_INT_SQ_DEFAULT
    safe_sigma_eff_sq = np.where(sigma_eff_sq < 1e-12, 1e-12, sigma_eff_sq)
    if np.any(safe_sigma_eff_sq <= 0): return np.inf
        
    chi2 = np.sum(residuals**2 / safe_sigma_eff_sq)
    return chi2 if np.isfinite(chi2) else np.inf


def chi_squared_sne_mu_covariance(cosmo_params, mu_model_func, sne_data_df):
    r"""
    Calculates chi-squared for SNe Ia: H2-style (mu_obs with full covariance matrix).
    """
    logger = logging.getLogger()
    if not all(col in sne_data_df.columns for col in ['zcmb', 'mu_obs']):
        logger.error("(chi2_mu_cov): sne_data_df missing 'zcmb' or 'mu_obs'.")
        return np.inf
    if 'covariance_matrix_inv' not in sne_data_df.attrs or sne_data_df.attrs['covariance_matrix_inv'] is None:
        logger.error("(chi2_mu_cov): Inverse covariance matrix 'covariance_matrix_inv' missing.")
        return np.inf

    z_data = sne_data_df['zcmb'].values; mu_data = sne_data_df['mu_obs'].values
    C_inv = sne_data_df.attrs['covariance_matrix_inv']

    try:
        mu_model = mu_model_func(z_data, *cosmo_params)
    except Exception:
        return np.inf

    if not isinstance(mu_model, np.ndarray) or mu_model.shape != mu_data.shape or np.any(~np.isfinite(mu_model)):
        return np.inf

    residuals = (mu_data - mu_model).flatten()

    try:
        if C_inv.ndim != 2 or C_inv.shape[0] != C_inv.shape[1] or C_inv.shape[0] != len(residuals):
            logger.error("(chi2_mu_cov): Covariance matrix dimension mismatch.")
            return np.inf
        
        term1 = np.dot(residuals, C_inv)
        chi2 = np.dot(term1, residuals)
    except (LinAlgError, ValueError) as e: 
        logger.warning(f"(chi2_mu_cov): Linear algebra error during chi2 calculation: {e}")
        return np.inf
        
    return chi2 if np.isfinite(chi2) else np.inf


def chi_squared_bao(bao_data_df, model_plugin, cosmo_params, model_rs_Mpc):
    r"""
    Calculates chi-squared for BAO data against model predictions.
    """
    logger = logging.getLogger()
    engine_interface.validate_plugin(model_plugin)
    if getattr(model_plugin, 'valid_for_bao', True) is False:
        logger.warning("(chi2_bao): Model flagged as invalid for BAO. Skipping calculation.")
        return np.inf
    if bao_data_df is None or bao_data_df.empty:
        logger.error("(chi2_bao): BAO data is empty.")
        return np.inf
    if not (np.isfinite(model_rs_Mpc) and model_rs_Mpc > 0):
        return np.inf # Invalid r_s, cannot calculate chi2

    total_chi2 = 0.0
    num_valid_points = 0

    try:
        get_DM_model = getattr(model_plugin, "get_comoving_distance_Mpc")
        get_Hz_model = getattr(model_plugin, "get_Hz_per_Mpc")
        get_DV_model_specific = getattr(model_plugin, "get_DV_Mpc", None)
        C_LIGHT = model_plugin.FIXED_PARAMS.get("C_LIGHT_KM_S", 299792.458) 
    except AttributeError as e:
        logger.error(f"(chi2_bao): Model plugin '{model_plugin.MODEL_NAME}' missing required function: {e}")
        return np.inf

    for index, row in bao_data_df.iterrows():
        z_val = row['redshift']
        obs_type = row['observable_type']
        obs_value = row['value']
        obs_error = row['error']

        if obs_error == 0 or not np.isfinite(obs_error) or obs_error < 1e-9: continue

        model_pred_numerator = np.nan
        try:
            if obs_type == "DM_over_rs": 
                model_pred_numerator = get_DM_model(z_val, *cosmo_params)
            elif obs_type == "DH_over_rs":
                hz_val = get_Hz_model(z_val, *cosmo_params)
                if np.isfinite(hz_val) and abs(hz_val) > 1e-9:
                    model_pred_numerator = C_LIGHT / hz_val
            elif obs_type == "DV_over_rs": 
                if get_DV_model_specific:
                    model_pred_numerator = get_DV_model_specific(z_val, *cosmo_params)
                else: # Fallback to calculating from DM and Hz
                    dm_val = get_DM_model(z_val, *cosmo_params)
                    hz_val = get_Hz_model(z_val, *cosmo_params)
                    if np.isfinite(dm_val) and dm_val >=0 and np.isfinite(hz_val) and abs(hz_val) > 1e-9 and z_val > 1e-9:
                        term_in_bracket = (dm_val**2) * C_LIGHT * z_val / hz_val
                        model_pred_numerator = term_in_bracket**(1.0/3.0) if term_in_bracket >=0 else np.nan
                    elif abs(z_val) < 1e-9 : model_pred_numerator = 0.0
            else:
                continue
            
            if not np.isfinite(model_pred_numerator): continue
            
            model_value_ratio = model_pred_numerator / model_rs_Mpc
            total_chi2 += ((obs_value - model_value_ratio) / obs_error)**2
            num_valid_points += 1
        
        except Exception:
            continue
            
    if num_valid_points == 0:
        logger.warning("(chi2_bao): No valid BAO points to calculate chi-squared.")
        return np.inf
        
    return total_chi2 if np.isfinite(total_chi2) else np.inf


def compute_cmb_spectrum(param_dict, ells):
    """Return the theoretical D_ell spectrum using CAMB."""
    logger = logging.getLogger()
    try:
        H0 = float(param_dict.get("H0", 67.0))
        ombh2 = float(param_dict.get("ombh2", 0.02237))
        omch2 = float(param_dict.get("omch2", 0.12))
        tau = float(param_dict.get("tau", 0.054))
        As = float(param_dict.get("As", 2.1e-9))
        ns = float(param_dict.get("ns", 0.965))
    except Exception as exc:
        logger.error(f"(compute_cmb_spectrum): Invalid parameter mapping: {exc}")
        return np.full_like(ells, np.nan, dtype=float)

    params = camb.CAMBparams()
    params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    params.InitPower.set_params(As=As, ns=ns)
    params.set_for_lmax(int(np.max(ells)) + 300, lens_potential_accuracy=0)
    try:
        results = camb.get_results(params)
        powers = results.get_cmb_power_spectra(
            params, lmax=int(np.max(ells)), CMB_unit="muK"
        )
        cl_tt = powers["total"][:, 0]
        ell_arr = np.asarray(ells, dtype=int)
        # CAMB already returns D_ell = ell(ell+1) C_ell / (2pi) when raw_cl=False
        # (the default). Simply index the array without applying the factor again.
        dl = cl_tt[ell_arr]
        return dl
    except Exception as exc:
        logger.error(f"(compute_cmb_spectrum): CAMB failed: {exc}")
        return np.full_like(ells, np.nan, dtype=float)


def chi_squared_cmb(cosmo_params, cmb_data_df):
    """Calculate chi-squared for CMB data using full covariance."""
    logger = logging.getLogger()
    if cmb_data_df is None or cmb_data_df.empty:
        logger.error("(chi2_cmb): CMB data is empty.")
        return np.inf
    if 'covariance_matrix_inv' not in cmb_data_df.attrs:
        logger.error("(chi2_cmb): Inverse covariance matrix missing in attrs.")
        return np.inf

    ells = cmb_data_df['ell'].values
    obs = cmb_data_df['Dl_obs'].values
    param_dict = {name: val for name, val in zip(cmb_data_df.attrs.get('param_names', []), cosmo_params)} if isinstance(cosmo_params, (list, tuple)) else cosmo_params
    th = compute_cmb_spectrum(param_dict, ells)
    if th.shape != obs.shape or np.any(~np.isfinite(th)):
        return np.inf

    resid = obs - th
    C_inv = cmb_data_df.attrs['covariance_matrix_inv']
    try:
        chi2 = float(resid @ C_inv @ resid)
    except Exception as exc:
        logger.error(f"(chi2_cmb): Linear algebra failure: {exc}")
        return np.inf

    return chi2 if np.isfinite(chi2) else np.inf


# ==============================================================================
# --- MAIN ENGINE FUNCTIONS ---
# ==============================================================================

def fit_sne_parameters(sne_data_df, model_plugin):
    """
    Fits cosmological (and optionally SNe nuisance) parameters to SNe Ia data.
    """
    logger = logging.getLogger()
    engine_interface.validate_plugin(model_plugin)
    fit_style = sne_data_df.attrs.get('fit_style', 'unknown')
    dataset_name = sne_data_df.attrs.get('dataset_name_attr', 'UnknownSNeDataset')
    model_name_str = getattr(model_plugin, 'MODEL_NAME', 'UnknownModel')

    logger.info(f"\n--- Fitting SNe Ia ({dataset_name}, Style: {fit_style}) for Model: {model_name_str} ---")

    cosmo_param_names = getattr(model_plugin, 'PARAMETER_NAMES', [])
    initial_cosmo_params = list(getattr(model_plugin, 'INITIAL_GUESSES', []))
    cosmo_param_bounds = list(getattr(model_plugin, 'PARAMETER_BOUNDS', []))
    num_cosmo_params = len(initial_cosmo_params)

    if not (cosmo_param_names and initial_cosmo_params and cosmo_param_bounds and len(cosmo_param_names) == num_cosmo_params and len(cosmo_param_bounds) == num_cosmo_params):
        logger.error(f"Model plugin {model_name_str} missing or has inconsistent parameter definitions.")
        return {'success': False, 'message': "Model parameter definition error.", 'chi2_min': np.inf}

    logger.info(f"Using standard Python (SciPy) function for '{model_name_str}'.")
    selected_mu_func = model_plugin.distance_modulus_model

    current_initial_params = list(initial_cosmo_params)
    current_param_bounds = list(cosmo_param_bounds)
    chi2_function_to_call = None
    args_for_chi2_func = ()
    
    fit_nuisance_params_flag = sne_data_df.attrs.get('fit_nuisance_params', False)

    if fit_style == 'h2_fit_nuisance' and fit_nuisance_params_flag:
        logger.info("Fitting cosmological parameters + SNe nuisance parameters (M_B, alpha, beta).")
        chi2_function_to_call = chi_squared_sne_h2_salt2_fitting
        
        current_initial_params.extend([SALT2_NUISANCE_PARAMS_INIT["M_B"], SALT2_NUISANCE_PARAMS_INIT["alpha_salt2"], SALT2_NUISANCE_PARAMS_INIT["beta_salt2"]])
        current_param_bounds.extend([SALT2_NUISANCE_PARAMS_BOUNDS["M_B"], SALT2_NUISANCE_PARAMS_BOUNDS["alpha_salt2"], SALT2_NUISANCE_PARAMS_BOUNDS["beta_salt2"]])
        args_for_chi2_func = (selected_mu_func, sne_data_df, num_cosmo_params)
        
    elif fit_style == 'h2_mu_covariance' and sne_data_df.attrs.get('covariance_matrix_inv') is not None:
        logger.info("Fitting cosmological parameters using mu_obs with full covariance matrix.")
        chi2_function_to_call = chi_squared_sne_mu_covariance
        args_for_chi2_func = (selected_mu_func, sne_data_df)
        
    elif fit_style == 'h1_fixed_nuisance' or (fit_style == 'h2_mu_covariance' and sne_data_df.attrs.get('covariance_matrix_inv') is None):
        if fit_style == 'h2_mu_covariance':
            logger.warning("Covariance matrix not available/invertible. Falling back to diagonal errors.")
        else:
            logger.info("Fitting cosmological parameters using mu_obs with diagonal errors (H1-style).")
        chi2_function_to_call = chi_squared_sne_h1_fixed_nuisance
        args_for_chi2_func = (selected_mu_func, sne_data_df)
    else:
        message = f"Error: Undetermined SNe fitting type or inconsistent data attributes for fit_style '{fit_style}'."
        logger.error(message)
        return {'success': False, 'message': message, 'chi2_min': np.inf, 'model_name': model_name_str}

    options = {'maxiter': 2000, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-7, 'eps': 1e-9}
    
    eval_count = {'count': 0}
    best_chi2_so_far = [np.inf]
    best_params_so_far = [list(current_initial_params)]
    
    start_time = time.time()

    def chi2_wrapper_for_minimize(params_to_test, *args_passed):
        """Wrapper to count evaluations and track best result for robust failure handling."""
        eval_count['count'] += 1
        current_chi2_val = chi2_function_to_call(params_to_test, *args_passed)
        
        if not np.isfinite(current_chi2_val): current_chi2_val = np.inf
        
        if current_chi2_val < best_chi2_so_far[0]:
            best_chi2_so_far[0] = float(current_chi2_val)
            best_params_so_far[0] = list(params_to_test) 
        
        elapsed_time = time.time() - start_time
        speed_str = f"{(eval_count['count'] / elapsed_time):.1f} evals/s" if elapsed_time > 1e-6 else "--- evals/s"
        
        print(f"  SNe Fit Evals: {eval_count['count']:<5} | Best Chi2: {best_chi2_so_far[0]:.4f} | Speed: {speed_str:<15}", end='\r', file=sys.stderr)
        
        return current_chi2_val if np.isfinite(current_chi2_val) else 1e12 

    logger.info(f"Starting SNe optimization for {model_name_str} using {len(current_initial_params)} parameters...")
    result_obj = None
    try:
        result_obj = minimize(chi2_wrapper_for_minimize, current_initial_params, args=args_for_chi2_func,
                              method='L-BFGS-B', bounds=current_param_bounds, options=options)
    except Exception as e_min:
        logger.error(f"\nException during SNe minimize call for {model_name_str}: {e_min}", exc_info=True)
    finally:
        print(" " * 80, end='\r', file=sys.stderr)
        logger.info(f"SNe Optimization for {model_name_str} finished. Total evals: {eval_count['count']}.")

    if result_obj and result_obj.success and np.isfinite(result_obj.fun):
        final_params = result_obj.x
        final_chi2 = result_obj.fun
        message = result_obj.message
        success_flag = True
    else: 
        final_params = np.array(best_params_so_far[0])
        final_chi2 = best_chi2_so_far[0]
        message = "Optimizer failed or did not improve; using best parameters found during search."
        if result_obj and hasattr(result_obj, 'message') and result_obj.message:
             message += f" (Optimizer msg: {result_obj.message})"
        success_flag = np.isfinite(final_chi2)

    fitted_cosmo_params_dict = None
    fitted_nuisance_params_dict = None
    
    if np.isfinite(final_chi2):
        fitted_cosmo_values = final_params[:num_cosmo_params]
        fitted_cosmo_params_dict = {name: val for name, val in zip(cosmo_param_names, fitted_cosmo_values)}

        if fit_nuisance_params_flag and len(final_params) == num_cosmo_params + 3:
            M_B_f, alpha_s2_f, beta_s2_f = final_params[num_cosmo_params:]
            fitted_nuisance_params_dict = {"M_B": M_B_f, "alpha_salt2": alpha_s2_f, "beta_salt2": beta_s2_f}
        
        num_data_points = len(sne_data_df)
        num_params_fitted_total = len(final_params)
        dof = num_data_points - num_params_fitted_total
        reduced_chi2 = final_chi2 / dof if dof > 0 else np.nan
        
        logger.info(f"SNe Fitting Results for {model_name_str}:")
        logger.info(f"  - Best-fit Cosmological Parameters:")
        for name, val in fitted_cosmo_params_dict.items():
            logger.info(f"    - {name}: {val:.5g}")
        if fitted_nuisance_params_dict:
            logger.info(f"  - Best-fit SNe Nuisance Parameters:")
            for name, val in fitted_nuisance_params_dict.items():
                logger.info(f"    - {name}: {val:.5g}")
        logger.info(f"  - Final Chi-squared: {final_chi2:.4f}")
        logger.info(f"  - Degrees of Freedom (DoF): {dof}")
        logger.info(f"  - Reduced Chi-squared: {reduced_chi2:.4f}" if np.isfinite(reduced_chi2) else "N/A")
        logger.info(f"  - Optimizer Success: {success_flag}, Message: {message}")
    else:
        logger.error(f"SNe Fitting for {model_name_str} FAILED catastrophically (Chi2 is Inf or NaN).")
        dof = np.nan; reduced_chi2 = np.nan

    return {
        'model_name': model_name_str,
        'fit_style_used': fit_style,
        'fitted_cosmological_params': fitted_cosmo_params_dict,
        'fitted_nuisance_params': fitted_nuisance_params_dict,
        'chi2_min': final_chi2,
        'dof': dof,
        'reduced_chi2': reduced_chi2,
        'success': success_flag and np.isfinite(final_chi2),
        'message': message,
        'n_evals_wrapper': eval_count['count']
    }


def calculate_bao_observables(bao_data_df, model_plugin, cosmo_params, z_smooth=None):
    """
    Calculates BAO observable predictions for a given model and its parameters.
    Also calculates smooth curves for plotting if z_smooth is provided.
    """
    logger = logging.getLogger()
    engine_interface.validate_plugin(model_plugin)
    model_name = model_plugin.MODEL_NAME

    # --- Part 1: Calculate for BAO data points ---
    bao_pred_df = bao_data_df.copy()
    bao_pred_df['model_prediction'] = np.nan
    if getattr(model_plugin, 'valid_for_bao', True) is False:
        logger.warning("Model flagged as invalid for BAO. Skipping calculations.")
        return bao_pred_df, np.nan, None
    
    param_str = ", ".join([f"{p:.4g}" for p in cosmo_params])
    logger.info(f"Calculating BAO observables for {model_name} with parameters: [{param_str}]")

    try:
        model_rs_Mpc = model_plugin.get_sound_horizon_rs_Mpc(*cosmo_params)
        if not (np.isfinite(model_rs_Mpc) and model_rs_Mpc > 0):
            logger.warning(f"Model '{model_name}' returned invalid r_s ({model_rs_Mpc:.3f} Mpc). BAO calculations will be NaN.")
            return bao_pred_df, np.nan, None
    except Exception as e:
        logger.error(f"Failed to calculate r_s for model '{model_name}': {e}", exc_info=True)
        return bao_pred_df, np.nan, None

    logger.info(f"Successfully calculated r_s for {model_name}: {model_rs_Mpc:.3f} Mpc")
    
    try:
        get_DM_model = getattr(model_plugin, "get_comoving_distance_Mpc")
        get_Hz_model = getattr(model_plugin, "get_Hz_per_Mpc")
        get_DV_model_specific = getattr(model_plugin, "get_DV_Mpc", None)
        get_DA_model = getattr(model_plugin, "get_angular_diameter_distance_Mpc")
        C_LIGHT = model_plugin.FIXED_PARAMS.get("C_LIGHT_KM_S", 299792.458)
    except AttributeError as e:
        logger.error(f"Model plugin '{model_name}' missing required function for BAO: {e}")
        return bao_pred_df, model_rs_Mpc, None

    for index, row in bao_pred_df.iterrows():
        z_val = row['redshift']
        obs_type = row['observable_type']
        
        model_pred_numerator = np.nan
        try:
            if obs_type == "DM_over_rs":
                model_pred_numerator = get_DM_model(z_val, *cosmo_params)
            elif obs_type == "DH_over_rs":
                hz_val = get_Hz_model(z_val, *cosmo_params)
                if np.isfinite(hz_val) and abs(hz_val) > 1e-9: model_pred_numerator = C_LIGHT / hz_val
            elif obs_type == "DV_over_rs":
                if get_DV_model_specific: model_pred_numerator = get_DV_model_specific(z_val, *cosmo_params)
                else: 
                    dm_val = get_DM_model(z_val, *cosmo_params); hz_val = get_Hz_model(z_val, *cosmo_params)
                    if np.isfinite(dm_val) and dm_val >=0 and np.isfinite(hz_val) and abs(hz_val) > 1e-9 and z_val > 1e-9:
                        term = (dm_val**2) * C_LIGHT * z_val / hz_val; model_pred_numerator = term**(1.0/3.0) if term >=0 else np.nan
                    elif abs(z_val) < 1e-9: model_pred_numerator = 0.0

            if np.isfinite(model_pred_numerator):
                bao_pred_df.loc[index, 'model_prediction'] = model_pred_numerator / model_rs_Mpc
        except Exception: pass 
            
    logger.debug(f"BAO predictions for {model_name}:\n{bao_pred_df.head().to_string()}")

    # --- Part 2: Calculate for smooth plotting curves ---
    smooth_predictions = None
    if z_smooth is not None:
        try:
            dm_smooth = get_DM_model(z_smooth, *cosmo_params)
            hz_smooth = get_Hz_model(z_smooth, *cosmo_params)
            dh_smooth = np.where(hz_smooth > 0, C_LIGHT / hz_smooth, np.nan)
            
            if get_DV_model_specific: dv_smooth = get_DV_model_specific(z_smooth, *cosmo_params)
            else:
                da_smooth = get_DA_model(z_smooth, *cosmo_params)
                term = np.power(1+z_smooth, 2) * np.power(da_smooth, 2) * C_LIGHT * z_smooth / hz_smooth
                dv_smooth = np.power(term, 1/3, where=term>=0, out=np.full_like(z_smooth, np.nan))

            smooth_predictions = {
                'z': z_smooth,
                'dm_over_rs': dm_smooth / model_rs_Mpc,
                'dh_over_rs': dh_smooth / model_rs_Mpc,
                'dv_over_rs': dv_smooth / model_rs_Mpc
            }
        except Exception as e:
            logger.error(f"Failed to calculate smooth BAO curves for {model_name}: {e}", exc_info=True)

    return bao_pred_df, model_rs_Mpc, smooth_predictions