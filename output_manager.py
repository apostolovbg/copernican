# copernican_suite/output_manager.py
"""
DEV NOTE (v1.4a): This module has been significantly refactored as part of
the v1.4a "black box" architecture. It no longer interacts directly with
the cosmology engine or live Python objects.

- Its sole purpose is to receive the final "Results JSON" string from the
  main orchestrator.
- It has a single public entry point: generate_outputs(results_json_string).
- It parses this JSON to reconstruct all necessary data, including best-fit
  parameters, statistics, and detailed pandas DataFrames.
- The core plotting and CSV-writing logic remains the same, but it is now
  fed data exclusively from the parsed JSON object.

BUG FIX (v1.4a): Corrected a `ValueError` in the SNe plotting function. The
call to `np.histogram` was attempting to unpack 3 return values, but the
current numpy version only returns 2. The calls have been corrected to
unpack only two values, resolving the crash during plot generation.
"""

import json
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# --- Main Public Entry Point ---

def generate_outputs(results_json_string, output_dir='output'):
    """
    The single, public entry point for the output manager. It parses the
    results JSON and calls the appropriate functions to generate all plots
    and data files.
    """
    logger = logging.getLogger()
    logger.info("\n--- Stage 4: Generating Outputs ---")

    try:
        results = json.loads(results_json_string)
        if results.get('status') == 'error':
            logger.critical(f"Output generation aborted. Engine reported an error: {results.get('message')}")
            return
    except (json.JSONDecodeError, TypeError) as e:
        logger.critical(f"Output Manager Error: Failed to parse Results JSON. It might be empty or invalid. Error: {e}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_id = results.get('metadata', {}).get('run_id', 'unknown_run')
    alt_model_name = results.get('inputs', {}).get('models', {}).get('alt_model', {}).get('name', 'AltModel')
    
    if 'sne_fit' in results.get('results', {}).get('lcdm', {}):
        sne_dataset_name = results['inputs']['datasets']['sne_data']['attributes'].get('dataset_name_attr', 'SNe_data')
        _generate_sne_plot(results, run_id, sne_dataset_name, alt_model_name, output_dir)
        _save_detailed_csvs(results, run_id, sne_dataset_name, alt_model_name, output_dir, 'sne')

    if 'bao_analysis' in results.get('results', {}).get('lcdm', {}):
        bao_dataset_name = results['inputs']['datasets']['bao_data']['attributes'].get('dataset_name_attr', 'BAO_data')
        _generate_bao_plot(results, run_id, bao_dataset_name, alt_model_name, output_dir)
        _save_detailed_csvs(results, run_id, bao_dataset_name, alt_model_name, output_dir, 'bao')


# --- Internal Helper Functions ---

def _reconstruct_df_from_split(df_dict):
    """Helper to reconstruct a pandas DataFrame from the 'split' dict format."""
    if not isinstance(df_dict, dict) or 'data' not in df_dict:
        return pd.DataFrame()
    return pd.DataFrame(df_dict['data'], index=df_dict['index'], columns=df_dict['columns'])

def _save_detailed_csvs(results, run_id, dataset_name, alt_model_name, output_dir, data_type):
    """Saves detailed pandas DataFrames to a unified CSV file."""
    logger = logging.getLogger()
    
    if data_type == 'sne':
        lcdm_df_dict = results['results']['lcdm'].get('sne_detailed_df')
        alt_df_dict = results['results']['alt_model'].get('sne_detailed_df')
        # This function is primarily designed for SNe data now.
        # Can be extended for BAO if a unified CSV is desired for it too.
        if lcdm_df_dict is None or alt_df_dict is None:
            logger.warning("SNe detailed data missing, skipping CSV output.")
            return

        lcdm_df = _reconstruct_df_from_split(lcdm_df_dict)
        alt_df = _reconstruct_df_from_split(alt_df_dict)
        
        # Create a single DataFrame for easier comparison
        # Start with the observational data from one of the DFs
        obs_cols = [col for col in lcdm_df.columns if 'model' not in col and 'residual' not in col]
        combined_df = lcdm_df[obs_cols].copy()

        # Add model predictions and residuals from both models
        combined_df['mu_model_lcdm'] = lcdm_df['mu_model']
        combined_df['residual_lcdm'] = lcdm_df['residual']
        combined_df[f'mu_model_{alt_model_name}'] = alt_df['mu_model']
        combined_df[f'residual_{alt_model_name}'] = alt_df['residual']
        
        filename = f"sne-detailed-data-unified-{dataset_name}_{run_id}.csv"

    elif data_type == 'bao':
        lcdm_df_dict = results['results']['lcdm']['bao_analysis'].get('detailed_df')
        alt_df_dict = results['results']['alt_model']['bao_analysis'].get('detailed_df')
        if lcdm_df_dict is None or alt_df_dict is None:
            logger.warning("BAO detailed data missing, skipping CSV output.")
            return
            
        lcdm_df = _reconstruct_df_from_split(lcdm_df_dict)
        alt_df = _reconstruct_df_from_split(alt_df_dict)
        
        # For BAO, a simple combination is better
        obs_cols = ['redshift', 'observable_type', 'value', 'error']
        combined_df = lcdm_df[obs_cols].copy()
        combined_df['model_value_lcdm'] = lcdm_df['model_value']
        combined_df[f'model_value_{alt_model_name}'] = alt_df['model_value']

        filename = f"bao-detailed-data-unified-{dataset_name}_{run_id}.csv"
    else:
        return

    filepath = os.path.join(output_dir, filename)
    try:
        combined_df.to_csv(filepath, index=False, float_format='%.6f')
        logger.info(f"Unified detailed data saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save detailed CSV {filename}: {e}")

def _generate_sne_plot(results, run_id, dataset_name, alt_model_name, output_dir):
    """Generates and saves the Hubble Diagram (SNe plot)."""
    logger = logging.getLogger()
    logger.info(f"Generating Hubble Diagram for {dataset_name}...")

    lcdm_df = _reconstruct_df_from_split(results['results']['lcdm']['sne_detailed_df'])
    alt_df = _reconstruct_df_from_split(results['results']['alt_model']['sne_detailed_df'])

    if lcdm_df.empty or alt_df.empty:
        logger.error("Cannot generate SNe plot, detailed dataframes are missing or empty.")
        return

    diag_errors = results['inputs']['datasets']['sne_data']['attributes'].get('diag_errors_for_plot')
    if diag_errors is None:
        diag_errors = lcdm_df['e_mu_obs'].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    ax1.errorbar(lcdm_df['zcmb'], lcdm_df['mu_obs'], yerr=diag_errors, fmt='.', color='lightgray',
                 elinewidth=0.5, capsize=0, alpha=0.6, label=f'{dataset_name}')
    ax1.plot(lcdm_df['zcmb'], lcdm_df['mu_model'], 'r-', label=f'ΛCDM (χ²={results["results"]["lcdm"]["sne_fit"]["chi2"]:.2f})')
    ax1.plot(alt_df['zcmb'], alt_df['mu_model'], 'b--', label=f'{alt_model_name} (χ²={results["results"]["alt_model"]["sne_fit"]["chi2"]:.2f})')
    ax1.set_ylabel('Distance Modulus (μ)')
    ax1.set_title(f'Hubble Diagram: {dataset_name}')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.errorbar(lcdm_df['zcmb'], lcdm_df['residual'], yerr=diag_errors, fmt='r.', alpha=0.2, label='ΛCDM Res.')
    ax2.errorbar(alt_df['zcmb'], alt_df['residual'], yerr=diag_errors, fmt='b.', alpha=0.2, label=f'{alt_model_name} Res.')
    
    bins = np.linspace(lcdm_df['zcmb'].min(), lcdm_df['zcmb'].max(), 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # **BUG FIX**: Corrected the unpacking of np.histogram to expect 2 values, not 3.
    lcdm_binned_res, _ = np.histogram(lcdm_df['zcmb'], bins=bins, weights=lcdm_df['residual'])
    alt_binned_res, _ = np.histogram(alt_df['zcmb'], bins=bins, weights=alt_df['residual'])
    counts, _ = np.histogram(lcdm_df['zcmb'], bins=bins)
    
    safe_counts = np.where(counts > 0, counts, 1)
    ax2.plot(bin_centers, lcdm_binned_res / safe_counts, 'r-', label='Avg. ΛCDM Res.')
    ax2.plot(bin_centers, alt_binned_res / safe_counts, 'b--', label=f'Avg. {alt_model_name} Res.')
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Redshift (z)')
    ax2.set_ylabel('μ_obs - μ_model')
    ax2.legend(loc='lower left')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    filename = f"hubble-plot-LambdaCDM-vs-{alt_model_name}-{dataset_name}_{run_id}.png"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath, dpi=150)
        logger.info(f"Hubble diagram saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save Hubble diagram: {e}")
    plt.close(fig)

def _generate_bao_plot(results, run_id, dataset_name, alt_model_name, output_dir):
    """Generates and saves the BAO comparison plot."""
    logger = logging.getLogger()
    logger.info(f"Generating BAO plot for {dataset_name}...")
    
    lcdm_df = _reconstruct_df_from_split(results['results']['lcdm']['bao_analysis']['detailed_df'])
    alt_df = _reconstruct_df_from_split(results['results']['alt_model']['bao_analysis']['detailed_df'])

    if lcdm_df.empty or alt_df.empty:
        logger.error("Cannot generate BAO plot, detailed dataframes are missing or empty.")
        return

    plot_types = lcdm_df['observable_type'].unique()
    fig, axes = plt.subplots(len(plot_types), 1, figsize=(10, 5 * len(plot_types)), sharex=True, squeeze=False)
    axes = axes.flatten()

    for i, obs_type in enumerate(plot_types):
        ax = axes[i]
        lcdm_subset = lcdm_df[lcdm_df['observable_type'] == obs_type]
        
        ax.errorbar(lcdm_subset['redshift'], lcdm_subset['value'], yerr=lcdm_subset['error'],
                    fmt='ko', capsize=3, label='Observed Data')

        z_smooth = np.linspace(0, lcdm_df['redshift'].max() * 1.1, 200)
        lcdm_smooth_vals = _get_smooth_bao_curve(results, 'lcdm', z_smooth, obs_type)
        alt_smooth_vals = _get_smooth_bao_curve(results, 'alt_model', z_smooth, obs_type)
        
        ax.plot(z_smooth, lcdm_smooth_vals, 'r-', label=f'ΛCDM (χ²={results["results"]["lcdm"]["bao_analysis"]["chi2"]:.2f})')
        ax.plot(z_smooth, alt_smooth_vals, 'b--', alpha=0.75, label=f'{alt_model_name} (χ²={results["results"]["alt_model"]["bao_analysis"]["chi2"]:.2f})')

        ax.set_ylabel(obs_type.replace('_', '/'))
        ax.set_title(f'BAO Comparison: {obs_type}')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)

    axes[-1].set_xlabel('Redshift (z)')
    plt.tight_layout()
    filename = f"bao-plot-LambdaCDM-vs-{alt_model_name}-{dataset_name}_{run_id}.png"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath, dpi=150)
        logger.info(f"BAO plot saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save BAO plot: {e}")
    plt.close(fig)

def _get_smooth_bao_curve(results, model_key, z_smooth, obs_type):
    """
    Internal helper to calculate a smooth theoretical curve for BAO observables.
    """
    model_info = results['inputs']['models'][model_key]
    model_filepath = model_info.get('filepath', 'lcdm_model.py')
    
    spec = importlib.util.spec_from_file_location("temp_model", model_filepath)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    num_cosmo_params = len(model_info['parameters'])
    best_fit_params = list(results['results'][model_key]['sne_fit']['best_fit_params'].values())[:num_cosmo_params]
    
    rs_model = results['results'][model_key]['bao_analysis']['rs_Mpc']
    if pd.isna(rs_model): return np.full_like(z_smooth, np.nan)
    
    try:
        if obs_type == 'DV_rs':
            return model_module.get_DV_Mpc(z_smooth, *best_fit_params) / rs_model
        elif obs_type == 'DM_rs':
            return model_module.get_comoving_distance_Mpc(z_smooth, *best_fit_params) / rs_model
        elif obs_type == 'DH_rs':
            c_km_s = model_module.FIXED_PARAMS.get("C_LIGHT_KM_S", 299792.458)
            hz = model_module.get_Hz_per_Mpc(z_smooth, *best_fit_params)
            return (c_km_s / hz) / rs_model
    except Exception:
        return np.full_like(z_smooth, np.nan)
    
    return np.full_like(z_smooth, np.nan)