# cosmo_engine_1.4g.py
# Computational Engine for Copernican Suite

"""
DEV NOTE (v1.4g): This engine refines the unfinished v1.4rc prototype.
- COLUMN FIX: uses the SNe column `mu` to compute residuals, avoiding a KeyError.
- COMPATIBLE: designed for lcdm_model.py, usmf2.py, usmf3b.py, and any plugin following the v1.4 API.
- BAO SUPPORT: retains generation of smooth curves for BAO plots.
"""

import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import importlib.util

# --- Helper Functions ---

def _load_model_plugin(path):
    """Dynamically loads a cosmetic model module from a given path."""
    try:
        spec = importlib.util.spec_from_file_location("model_plugin", path)
        model_plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_plugin)
        logging.info(f"Successfully loaded model plugin: {path}")
        return model_plugin
    except Exception as e:
        logging.error(f"Failed to load model plugin from {path}: {e}", exc_info=True)
        return None

def _get_params(params_dict, role='cosmological'):
    """Extracts parameter names based on their role (cosmological or nuisance)."""
    return [p_name for p_name, p_info in params_dict.items() if p_info['role'] == role]

# --- SNe Ia Analysis ---

def _chi_squared(params, sne_df, model_plugin):
    """Calculates the chi-squared statistic for SNe Ia data."""
    # Unpack parameters: cosmological first, then the single nuisance param M
    cosmo_params_list = params[:-1]
    M = params[-1]
    
    cosmo_param_names = _get_params(model_plugin.METADATA['parameters'], 'cosmological')
    cosmo_params_dict = dict(zip(cosmo_param_names, cosmo_params_list))
    
    # Ok0 is derived, not fitted directly
    cosmo_params_dict['Ok0'] = 0.0 # Assuming flat universe for now

    z = sne_df['z'].values
    mu = sne_df['mu'].values
    mu_err = sne_df['mu_err'].values

    mu_model = model_plugin.get_distance_modulus(z, **cosmo_params_dict, M=M)
    
    chi2 = np.sum(((mu - mu_model) / mu_err)**2)
    return chi2

def _perform_sne_fit(sne_df, model_plugin):
    """Performs the chi-squared minimization to find the best-fit parameters."""
    logging.info(f"Performing SNe Ia fit for model: {model_plugin.METADATA['model_name']}...")
    
    param_meta = model_plugin.METADATA['parameters']
    
    # Get initial guesses and bounds in the correct order: cosmological, then nuisance
    cosmo_param_names = _get_params(param_meta, 'cosmological')
    nuisance_param_names = _get_params(param_meta, 'nuisance')
    
    initial_guesses = [param_meta[p]['initial_guess'] for p in cosmo_param_names]
    initial_guesses += [param_meta[p]['initial_guess'] for p in nuisance_param_names]
    
    bounds = [param_meta[p]['bounds'] for p in cosmo_param_names]
    bounds += [param_meta[p]['bounds'] for p in nuisance_param_names]

    # Run the minimization
    result = minimize(_chi_squared, initial_guesses, args=(sne_df, model_plugin),
                      method='L-BFGS-B', bounds=bounds)

    if not result.success:
        logging.warning(f"Minimization for model {model_plugin.METADATA['model_name']} may not have converged: {result.message}")

    # Package the results
    best_fit_params_list = result.x
    all_param_names = cosmo_param_names + nuisance_param_names
    best_fit_params = dict(zip(all_param_names, best_fit_params_list))
    best_fit_params['Ok0'] = 0.0 # Add the derived curvature parameter
    
    min_chi2 = result.fun
    dof = len(sne_df) - len(initial_guesses)
    reduced_chi2 = min_chi2 / dof if dof > 0 else np.inf

    logging.info(f"Fit complete. Minimum Chi2 = {min_chi2:.2f}, Reduced Chi2 = {reduced_chi2:.2f}")
    
    fit_results = {
        'best_fit_params': best_fit_params,
        'min_chi2': min_chi2,
        'reduced_chi2': reduced_chi2,
        'dof': dof
    }
    return fit_results

def _create_detailed_sne_df(sne_df, fit_results, model_plugin):
    """Adds model-predicted columns to the SNe DataFrame."""
    params = fit_results['best_fit_params']
    cosmo_params = {k: v for k, v in params.items() if k != 'M'}
    M = params['M']

    sne_df['mu_model'] = model_plugin.get_distance_modulus(sne_df['z'].values, **cosmo_params, M=M)
    sne_df['residual'] = sne_df['mu'] - sne_df['mu_model']
    
    # Generate smooth curve for plotting
    z_smooth = np.linspace(sne_df['z'].min(), sne_df['z'].max(), 300)
    mu_model_smooth = model_plugin.get_distance_modulus(z_smooth, **cosmo_params, M=M)
    
    smooth_curve_data = {
        'z_smooth': z_smooth.tolist(),
        'mu_model_smooth': mu_model_smooth.tolist()
    }
    return sne_df, smooth_curve_data

# --- BAO Analysis ---

def _calculate_bao_observables(bao_df, fit_params, model_plugin):
    """Calculates the model-predicted BAO observables for the given data points."""
    logging.info(f"Calculating BAO observables for model: {model_plugin.METADATA['model_name']}")
    bao_df['y_model'] = np.nan # Initialize column
    
    cosmo_params = {k: v for k, v in fit_params.items() if k != 'M'}

    for index, row in bao_df.iterrows():
        try:
            z = row['z']
            obs_type = row['observable_type']
            rs_drag = row['rs_drag']
            
            y_model_val = np.nan
            if obs_type == 'DV_over_rs':
                dv_mpc = model_plugin.get_DV_Mpc(z, **cosmo_params)
                y_model_val = dv_mpc / rs_drag
            elif obs_type == 'DA_over_rs':
                da_mpc = model_plugin.get_angular_diameter_distance_Mpc(z, **cosmo_params)
                y_model_val = da_mpc / rs_drag
            # Add other BAO observable calculations here as needed
            else:
                logging.warning(f"Unsupported BAO observable type '{obs_type}' at z={z}. Skipping.")

            bao_df.loc[index, 'y_model'] = y_model_val
        except Exception as e:
            logging.error(f"Error calculating BAO observable at z={row['z']} for model {model_plugin.METADATA['model_name']}: {e}", exc_info=True)
            bao_df.loc[index, 'y_model'] = np.nan
            
    return bao_df

# NEW FUNCTION (v1.4rc)
def _generate_smooth_bao_curves(bao_df, model1_params, model2_params, model1_plugin, model2_plugin):
    """
    Generates smooth model curves for all BAO observables present in the data.
    This is the new function that provides the data needed for plotting lines.
    """
    logging.info("Generating smooth BAO model curves for plotting...")
    smooth_curves_data = {}
    
    # Determine the redshift range from the data, with a small padding
    z_min = 0
    z_max = bao_df['z'].max() * 1.05
    z_smooth = np.linspace(z_min, z_max, 300)
    
    unique_observables = bao_df['observable_type'].unique()
    
    # Get cosmological parameters only (exclude nuisance param M)
    m1_cosmo_params = {k: v for k, v in model1_params.items() if k != 'M'}
    m2_cosmo_params = {k: v for k, v in model2_params.items() if k != 'M'}
    
    for obs_type in unique_observables:
        # We need rs_drag for the calculation. Assume it's constant for a given dataset.
        rs_drag = bao_df[bao_df['observable_type'] == obs_type]['rs_drag'].iloc[0]
        
        y_smooth_1, y_smooth_2 = None, None
        
        try:
            if obs_type == 'DV_over_rs':
                dv1 = model1_plugin.get_DV_Mpc(z_smooth, **m1_cosmo_params)
                dv2 = model2_plugin.get_DV_Mpc(z_smooth, **m2_cosmo_params)
                y_smooth_1 = dv1 / rs_drag
                y_smooth_2 = dv2 / rs_drag
            elif obs_type == 'DA_over_rs':
                da1 = model1_plugin.get_angular_diameter_distance_Mpc(z_smooth, **m1_cosmo_params)
                da2 = model2_plugin.get_angular_diameter_distance_Mpc(z_smooth, **m2_cosmo_params)
                y_smooth_1 = da1 / rs_drag
                y_smooth_2 = da2 / rs_drag
            # Add other observables here if they appear in data files
            
            if y_smooth_1 is not None and y_smooth_2 is not None:
                smooth_curves_data[obs_type] = {
                    'z_smooth': z_smooth.tolist(),
                    'model1_y_smooth': y_smooth_1.tolist(),
                    'model2_y_smooth': y_smooth_2.tolist()
                }
            else:
                 logging.warning(f"Could not generate smooth curve for observable '{obs_type}'. Unsupported type.")

        except Exception as e:
            logging.error(f"Failed to generate smooth curve for {obs_type}: {e}", exc_info=True)
            
    return smooth_curves_data

# --- Main Execution Function ---

def execute_job(job_json):
    """
    The main entry point for the engine. The entire workflow is wrapped in a
    try/except block so unexpected errors are logged with full tracebacks.
    """
    logging.info("Cosmological engine execution started.")
    run_id = job_json.get('run_id', 'unknown')

    try:
        # --- Validate Job Structure ---
        if 'data' not in job_json:
            logging.error("Job JSON is missing the 'data' section.")
            return None
        if 'models' not in job_json:
            logging.error("Job JSON is missing the 'models' section with plugin paths.")
            logging.debug(f"Job JSON keys received: {list(job_json.keys())}")
            return None

        # --- Load Data and Models ---
        sne_df = pd.DataFrame(job_json['data']['sne_data']['dataframe'])
        model1_plugin = _load_model_plugin(job_json['models']['model1']['path'])
        model2_plugin = _load_model_plugin(job_json['models']['model2']['path'])

        if not all([model1_plugin, model2_plugin]):
            logging.critical("One or more model plugins failed to load. Aborting job.")
            return None

        # --- SNe Ia Fitting ---
        model1_fit_results = _perform_sne_fit(sne_df, model1_plugin)
        model2_fit_results = _perform_sne_fit(sne_df, model2_plugin)

        # Create detailed SNe dataframes with model predictions
        sne_df_model1, smooth1 = _create_detailed_sne_df(sne_df.copy(), model1_fit_results, model1_plugin)
        sne_df_model2, smooth2 = _create_detailed_sne_df(sne_df.copy(), model2_fit_results, model2_plugin)

        # Consolidate SNe results
        sne_results_df = sne_df_model1.rename(columns={'mu_model': 'model1_mu', 'residual': 'model1_residual'})
        sne_results_df['model2_mu'] = sne_df_model2['mu_model']
        sne_results_df['model2_residual'] = sne_df_model2['residual']

        # --- BAO Analysis (if data provided) ---
        bao_analysis_results = {}
        if job_json['data'].get('bao_data'):
            bao_df = pd.DataFrame(job_json['data']['bao_data']['dataframe'])

            bao_m1_df = _calculate_bao_observables(
                bao_df.copy(), model1_fit_results['best_fit_params'], model1_plugin)
            bao_m2_df = _calculate_bao_observables(
                bao_df.copy(), model2_fit_results['best_fit_params'], model2_plugin)

            bao_results_df = bao_m1_df.rename(columns={'y_model': 'model1_y'})
            bao_results_df['model2_y'] = bao_m2_df['y_model']

            smooth_curves = _generate_smooth_bao_curves(
                bao_results_df,
                model1_fit_results['best_fit_params'],
                model2_fit_results['best_fit_params'],
                model1_plugin,
                model2_plugin
            )

            bao_analysis_results = {
                "detailed_df": bao_results_df.to_dict('split'),
                "smooth_curves": smooth_curves
            }

        # --- Assemble Final Results JSON ---
        results_dict = {
            "metadata": {
                "run_id": run_id,
                "engine_name": job_json['engine_name'],
                "model1_name": model1_plugin.METADATA['model_name'],
                "model2_name": model2_plugin.METADATA['model_name'],
                "model1_metadata": model1_plugin.METADATA,
                "model2_metadata": model2_plugin.METADATA,
            },
            "sne_analysis": {
                "detailed_df": sne_results_df.to_dict('split'),
                "model1_fit_results": model1_fit_results,
                "model2_fit_results": model2_fit_results,
                "model1_smooth_curve": smooth1,
                "model2_smooth_curve": smooth2
            },
            "bao_analysis": bao_analysis_results
        }

        logging.info("Cosmological engine execution finished successfully.")
        return results_dict

    except Exception as e:
        logging.critical(f"Engine execution failed: {e}", exc_info=True)
        return None
