# copernican_suite/cosmo_engine_1.4b.py
"""
DEV NOTE (v1.4b-fix3): This version corrects a critical SyntaxError introduced
in the previous fix.

BUG FIX (Critical): A typo on line 193 resulted in an unclosed parenthesis,
causing a SyntaxError that crashed the engine during the creation of the SNe
detailed dataframe. This has been corrected.

BUG FIX 2: The check for SNe fit success was faulty. This has been corrected
to properly check the 'success' flag within the 'fit_summary'.

BUG FIX 3: In `_calculate_bao_observables`, a `NameError` was corrected.
"""

import json
import importlib.util
import os
import logging
import time
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# --- Custom JSON Encoder for Output ---
class CustomJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder to handle special data types from NumPy and pandas,
    ensuring they can be successfully serialized into the Results JSON string.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='split')
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(CustomJSONEncoder, self).default(obj)

# --- Internal Helper Functions ---

def _load_model_module(model_filepath):
    """Dynamically loads a Python model plugin from its file path."""
    logger = logging.getLogger()
    if not os.path.isfile(model_filepath):
        logger.critical(f"Engine Error: Model file does not exist at path: {model_filepath}")
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            name=os.path.basename(model_filepath).replace('.py', ''),
            location=model_filepath
        )
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        return model_module
    except Exception as e:
        logger.critical(f"Engine Error: Failed to load model plugin from {model_filepath}: {e}", exc_info=True)
        return None

def _reconstruct_df_from_split(df_dict):
    """Helper to reconstruct a pandas DataFrame from the 'split' dict format."""
    if not df_dict or 'data' not in df_dict:
        return None
    try:
        return pd.DataFrame(df_dict['data'], index=df_dict['index'], columns=df_dict['columns'])
    except Exception as e:
        logging.getLogger().error(f"Engine Error: Could not reconstruct DataFrame from dict. Error: {e}")
        return None

def _chi2_sne(params, model_module, sne_df):
    """Calculates the chi-squared value for a given set of parameters against SNe data."""
    z_obs = sne_df['zcmb'].values
    fit_style = sne_df.attrs.get('fit_style', 'h1_fixed_nuisance')

    if fit_style in ['h1_fixed_nuisance', 'h2_mu_covariance']:
        cosmo_params = params
        mu_model = model_module.distance_modulus_model(z_obs, *cosmo_params)
        mu_obs = sne_df['mu_obs'].values
        residuals = mu_obs - mu_model
        if fit_style == 'h2_mu_covariance' and 'covariance_matrix_inv' in sne_df.attrs:
            cov_inv = np.array(sne_df.attrs['covariance_matrix_inv'])
            chi2 = residuals.T @ cov_inv @ residuals
        else:
            e_mu_obs = sne_df['e_mu_obs'].values
            chi2 = np.sum((residuals / e_mu_obs) ** 2)
    elif fit_style == 'h2_fit_nuisance':
        num_cosmo_params = len(sne_df.attrs['cosmo_param_names'])
        cosmo_params = params[:num_cosmo_params]
        M_abs, alpha, beta = params[num_cosmo_params:]
        mu_model = model_module.distance_modulus_model(z_obs, *cosmo_params)
        mb_obs = sne_df['mb'].values
        x1_obs = sne_df['x1'].values
        c_obs = sne_df['c'].values
        mb_model = mu_model + M_abs - alpha * x1_obs + beta * c_obs
        residuals = mb_obs - mb_model
        e_mb_obs = sne_df['e_mb'].values
        e_x1_obs = sne_df['e_x1'].values
        e_c_obs = sne_df['e_c'].values
        variance = e_mb_obs**2 + (alpha * e_x1_obs)**2 + (beta * e_c_obs)**2
        chi2 = np.sum(residuals**2 / variance)
    else:
        return np.inf
    return chi2 if np.isfinite(chi2) else np.inf

def _fit_sne_model(model_module, model_info, sne_df):
    """Performs the SNe data fitting for a single model using scipy.minimize."""
    logger = logging.getLogger()
    fit_style = sne_df.attrs.get('fit_style')
    initial_guesses = list(model_info['initial_guesses'])
    bounds = list(model_info['bounds'])
    param_names = list(model_info['parameters'])
    if fit_style == 'h2_fit_nuisance':
        nuisance_params = {'M_abs': -19.3, 'alpha': 0.14, 'beta': 3.1}
        nuisance_bounds = {'M_abs': (-22, -17), 'alpha': (-1, 1), 'beta': (0, 6)}
        initial_guesses.extend(nuisance_params.values())
        bounds.extend(nuisance_bounds.values())
        param_names.extend(nuisance_params.keys())
        sne_df.attrs['cosmo_param_names'] = model_info['parameters']
    logger.info(f"Starting SNe fit for {model_info['name']} with initial guesses: {initial_guesses}")
    start_time = time.time()
    fit_result = minimize(
        _chi2_sne, x0=initial_guesses, args=(model_module, sne_df),
        method='L-BFGS-B', bounds=bounds
    )
    fit_duration = time.time() - start_time
    logger.info(f"SNe fit for {model_info['name']} completed in {fit_duration:.2f}s. Success: {fit_result.success}")

    result_package = {
        'fit_summary': {
            'success': bool(fit_result.success),
            'message': str(fit_result.message),
            'nfev': int(fit_result.nfev)
        },
        'best_fit_params': dict(zip(param_names, fit_result.x)),
        'chi2': float(fit_result.fun) if fit_result.success and np.isfinite(fit_result.fun) else None,
        'dof': len(sne_df) - len(initial_guesses),
        'fit_duration_s': fit_duration
    }
    return result_package

def _calculate_bao_observables(model_module, best_fit_cosmo_params, bao_df):
    """Calculates theoretical BAO observables given best-fit parameters."""
    logger = logging.getLogger()
    results = {}
    try:
        rs_model = model_module.get_sound_horizon_rs_Mpc(*best_fit_cosmo_params)
        logger.info(f"Successfully calculated r_s for {model_module.MODEL_NAME}: {rs_model:.3f} Mpc")
        results['rs_Mpc'] = rs_model
    except Exception as e:
        logger.error(f"Could not calculate sound horizon for {model_module.MODEL_NAME}: {e}")
        results['rs_Mpc'], rs_model = np.nan, np.nan
        
    if pd.isna(rs_model):
        bao_df['model_value'], results['chi2'], results['dof'] = np.nan, np.nan, len(bao_df)
        results['detailed_df'] = bao_df
        return results
        
    model_values = []
    for _, row in bao_df.iterrows():
        z, obs_type, val = row['redshift'], row['observable_type'], np.nan
        try:
            if obs_type == 'DV_rs':
                val = model_module.get_DV_Mpc(z, *best_fit_cosmo_params) / rs_model
            elif obs_type == 'DM_rs':
                val = model_module.get_comoving_distance_Mpc(z, *best_fit_cosmo_params) / rs_model
            elif obs_type == 'DH_rs':
                c_km_s = model_module.FIXED_PARAMS.get("C_LIGHT_KM_S", 299792.458)
                hz = model_module.get_Hz_per_Mpc(z, *best_fit_cosmo_params)
                val = (c_km_s / hz) / rs_model
        except Exception as e:
            logger.error(f"ENGINE failed to calculate BAO observable {obs_type} at z={z}: {e}", exc_info=True)
        model_values.append(val)
        
    bao_df['model_value'] = model_values
    bao_df['residual'] = bao_df['value'] - bao_df['model_value']
    chi2 = np.sum((bao_df['residual'] / bao_df['error']) ** 2)
    results['chi2'], results['dof'], results['detailed_df'] = (chi2 if np.isfinite(chi2) else np.nan), len(bao_df), bao_df
    logger.info(f"{model_module.MODEL_NAME} BAO: r_s = {results['rs_Mpc']:.2f} Mpc, Chi2_BAO = {results['chi2']:.2f}")
    return results

def _create_detailed_sne_df(model_module, fit_results, sne_df):
    """Generates a detailed DataFrame for SNe results, including model predictions."""
    z_obs = sne_df['zcmb'].values
    num_cosmo_params = len(fit_results['best_fit_params'])
    if sne_df.attrs.get('fit_style') == 'h2_fit_nuisance':
        num_cosmo_params = len(sne_df.attrs['cosmo_param_names'])
        
    cosmo_params = list(fit_results['best_fit_params'].values())[:num_cosmo_params]
    # BUG FIX: Corrected the line below which was cut off
    mu_model = model_module.distance_modulus_model(z_obs, *cosmo_params)
    detailed_df = sne_df.copy()
    detailed_df['mu_model'], detailed_df['residual'] = mu_model, detailed_df.get('mu_obs', np.nan) - mu_model
    return detailed_df

# --- Main Public Entry Point ---
def execute_job(job_json_string):
    """The single, public entry point for the engine."""
    logger = logging.getLogger()
    logger.info("--- Cosmology Engine Job Started ---")
    try:
        job_dict = json.loads(job_json_string)
        run_id = job_dict['metadata']['run_id']
        logger.info(f"Processing Run ID: {run_id}")
    except (json.JSONDecodeError, KeyError) as e:
        logger.critical(f"Engine Error: Failed to parse Job JSON. Error: {e}")
        return json.dumps({"status": "error", "message": "Invalid Job JSON"})

    alt_model_filepath = job_dict['models']['alt_model']['filepath']
    lcdm_model_module, alt_model_module = _load_model_module('lcdm_model.py'), _load_model_module(alt_model_filepath)
    if not lcdm_model_module or not alt_model_module:
        return json.dumps({"status": "error", "message": "Failed to load model modules."})

    sne_df = _reconstruct_df_from_split(job_dict['datasets']['sne_data']['data'])
    if sne_df is not None: sne_df.attrs = job_dict['datasets']['sne_data']['attributes']

    bao_df = _reconstruct_df_from_split(job_dict['datasets']['bao_data']['data'])
    if bao_df is not None: bao_df.attrs = job_dict['datasets']['bao_data']['attributes']

    results_dict = {
        "metadata": job_dict['metadata'],
        "inputs": {
            "models": job_dict['models'],
            "datasets": job_dict['datasets']
        },
        "results": {"lcdm": {}, "alt_model": {}}
    }

    if sne_df is not None:
        logger.info("\n--- Stage: SNe Fitting ---")
        lcdm_sne_results = _fit_sne_model(lcdm_model_module, job_dict['models']['lcdm'], sne_df.copy())
        alt_sne_results = _fit_sne_model(alt_model_module, job_dict['models']['alt_model'], sne_df.copy())

        results_dict['results']['lcdm']['sne_fit'] = lcdm_sne_results
        results_dict['results']['alt_model']['sne_fit'] = alt_sne_results
        
        if lcdm_sne_results.get('fit_summary', {}).get('success'):
            results_dict['results']['lcdm']['sne_detailed_df'] = _create_detailed_sne_df(lcdm_model_module, lcdm_sne_results, sne_df)
        if alt_sne_results.get('fit_summary', {}).get('success'):
            results_dict['results']['alt_model']['sne_detailed_df'] = _create_detailed_sne_df(alt_model_module, alt_sne_results, sne_df)

    if bao_df is not None and sne_df is not None:
        if results_dict['results']['lcdm']['sne_fit'].get('fit_summary', {}).get('success'):
            logger.info("\n--- Stage: BAO Analysis ---")
            num_lcdm_cosmo = len(job_dict['models']['lcdm']['parameters'])
            num_alt_cosmo = len(job_dict['models']['alt_model']['parameters'])
            
            lcdm_cosmo_params = list(lcdm_sne_results['best_fit_params'].values())[:num_lcdm_cosmo]
            alt_cosmo_params = list(alt_sne_results['best_fit_params'].values())[:num_alt_cosmo]
            
            results_dict['results']['lcdm']['bao_analysis'] = _calculate_bao_observables(lcdm_model_module, lcdm_cosmo_params, bao_df.copy())
            results_dict['results']['alt_model']['bao_analysis'] = _calculate_bao_observables(alt_model_module, alt_cosmo_params, bao_df.copy())
        else:
            logger.warning("Skipping BAO analysis because SNe fit was not successful.")

    logger.info("--- Cosmology Engine Job Finished ---")
    try:
        results_json = json.dumps(results_dict, cls=CustomJSONEncoder, indent=2)
        return results_json
    except Exception as e:
        logger.critical(f"Engine Error: Failed to serialize results dictionary to JSON: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": f"Failed to serialize final results: {e}"})