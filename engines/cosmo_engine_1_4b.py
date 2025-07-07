# copernican_suite/cosmo_engine.py
"""
Cosmological Engine for the Copernican Suite.
Relies on SciPy/NumPy for all computations.
"""
# This is the reference CPU implementation used when no acceleration is
# available. It implements generic chi-squared calculations for SNe, BAO and CMB
# data and exposes fitting routines consumed by ``copernican.py``.

import numpy as np
from scipy.linalg import LinAlgError
import camb
import sys
import logging
from copernican_lib import engine_interface
from copernican_lib.optim_utils import minimize_with_progress
from copernican_lib.model_coder import (
    chi_squared_sne,
    chi_squared_bao,
    chi_squared_cmb,
    compute_cmb_spectrum,
)





# ==============================================================================
# --- MAIN ENGINE FUNCTIONS ---
# ==============================================================================

def fit_sne_parameters(sne_data_df, model_plugin):
    """Fit cosmological parameters to SNe Ia data."""
    logger = logging.getLogger()
    engine_interface.validate_plugin(model_plugin)
    dataset_name = sne_data_df.attrs.get('dataset_name_attr', 'UnknownSNeDataset')
    model_name_str = getattr(model_plugin, 'MODEL_NAME', 'UnknownModel')

    logger.info(f"\n--- Fitting SNe Ia ({dataset_name}) for Model: {model_name_str} ---")

    names = getattr(model_plugin, 'PARAMETER_NAMES', [])
    initial = list(getattr(model_plugin, 'INITIAL_GUESSES', []))
    bounds = list(getattr(model_plugin, 'PARAMETER_BOUNDS', []))

    if not (names and initial and bounds and len(names) == len(initial) == len(bounds)):
        logger.error(f"Model plugin {model_name_str} missing or has inconsistent parameter definitions.")
        return {'success': False, 'message': 'Model parameter definition error.', 'chi2_min': np.inf}

    options = {'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-7, 'eps': 1e-9}

    logger.info(f"Starting SNe optimization for {model_name_str} using {len(initial)} parameters...")

    result_obj, eval_total, best_chi2_so_far, best_params_so_far = minimize_with_progress(
        chi_squared_sne,
        initial,
        bounds=bounds,
        args=(model_plugin.distance_modulus_model, sne_data_df),
        options=options,
        label="SNe Fit",
    )

    if result_obj and result_obj.success and np.isfinite(result_obj.fun):
        final_params = result_obj.x
        final_chi2 = result_obj.fun
        message = result_obj.message
        success_flag = True
    else:
        final_params = np.array(best_params_so_far)
        final_chi2 = best_chi2_so_far
        message = "Optimizer failed or did not improve"
        if result_obj and hasattr(result_obj, 'message') and result_obj.message:
            message += f" (Optimizer msg: {result_obj.message})"
        success_flag = np.isfinite(final_chi2)

    fitted_cosmo_params_dict = None

    if np.isfinite(final_chi2):
        fitted_cosmo_params_dict = {n: v for n, v in zip(names, final_params)}
        dof = len(sne_data_df) - len(final_params)
        reduced_chi2 = final_chi2 / dof if dof > 0 else np.nan
        logger.info(f"SNe results for {model_name_str}: chi2={final_chi2:.4f}, DoF={dof}, reduced={reduced_chi2:.4f}")
    else:
        logger.error(f"SNe fitting for {model_name_str} failed: chi2 is NaN/Inf")
        dof = np.nan
        reduced_chi2 = np.nan

    return {
        'model_name': model_name_str,
        'fit_style_used': 'covariance' if sne_data_df.attrs.get('covariance_matrix_inv') is not None else 'diagonal',
        'fitted_cosmological_params': fitted_cosmo_params_dict,
        'fitted_nuisance_params': None,
        'chi2_min': final_chi2,
        'dof': dof,
        'reduced_chi2': reduced_chi2,
        'success': success_flag and np.isfinite(final_chi2),
        'message': message,
        'n_evals_wrapper': eval_total,
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
