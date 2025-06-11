# copernican_suite/csv_writer.py
"""
DEV NOTE (v1.4b): This is a new module introduced in the v1.4b refactor to
handle all CSV output generation. Its creation follows the Single
Responsibility Principle, separating data serialization from plotting.

The logic herein was migrated and refined from the `_save_detailed_csvs`
function in the v1.4a `output_manager.py`. This module is designed to be
called by the `output_manager.py` dispatcher, which passes it the final
results data.
"""

import os
import logging
import pandas as pd

def _reconstruct_df_from_split(df_dict):
    """
    Internal helper to safely reconstruct a pandas DataFrame from the 'split'
    dictionary format that is used in the JSON data contract.

    Args:
        df_dict (dict): The dictionary in 'split' orientation.

    Returns:
        pandas.DataFrame: The reconstructed DataFrame, or an empty one on failure.
    """
    # Basic validation to prevent errors with malformed or missing data.
    if not isinstance(df_dict, dict) or 'data' not in df_dict or 'columns' not in df_dict:
        logging.getLogger().warning("Failed to reconstruct DataFrame: dictionary is malformed or empty.")
        return pd.DataFrame()
    try:
        return pd.DataFrame(df_dict['data'], index=df_dict.get('index'), columns=df_dict['columns'])
    except Exception as e:
        logging.getLogger().error(f"Error reconstructing DataFrame: {e}")
        return pd.DataFrame()

def _save_sne_csv(results, alt_model_name, dataset_name, run_id, output_dir):
    """
    Creates and saves a detailed, unified CSV for the SNe Ia analysis results.
    """
    logger = logging.getLogger()
    
    # Reconstruct the detailed DataFrames from the results dictionary
    lcdm_df_dict = results['results']['lcdm'].get('sne_detailed_df')
    alt_df_dict = results['results']['alt_model'].get('sne_detailed_df')

    if lcdm_df_dict is None or alt_df_dict is None:
        logger.warning("SNe detailed data missing, skipping unified CSV output.")
        return

    lcdm_df = _reconstruct_df_from_split(lcdm_df_dict)
    alt_df = _reconstruct_df_from_split(alt_df_dict)

    if lcdm_df.empty or alt_df.empty:
        logger.warning("Reconstructed SNe DataFrames are empty, skipping CSV output.")
        return

    # Create a single, unified DataFrame for easier comparison.
    # Start with the observational data columns from the LCDM dataframe.
    obs_cols = [col for col in lcdm_df.columns if 'model' not in col and 'residual' not in col]
    combined_df = lcdm_df[obs_cols].copy()

    # Add model predictions and residuals from both models with clear, dynamic names.
    combined_df['mu_model_lcdm'] = lcdm_df['mu_model']
    combined_df['residual_lcdm'] = lcdm_df['residual']
    combined_df[f'mu_model_{alt_model_name}'] = alt_df['mu_model']
    combined_df[f'residual_{alt_model_name}'] = alt_df['residual']

    filename = f"sne-detailed-data_LambdaCDM-vs-{alt_model_name}_{dataset_name}_{run_id}.csv"
    filepath = os.path.join(output_dir, filename)
    
    try:
        combined_df.to_csv(filepath, index=False, float_format='%.6f')
        logger.info(f"Unified SNe detailed data saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save unified SNe CSV {filename}: {e}", exc_info=True)


def _save_bao_csv(results, alt_model_name, dataset_name, run_id, output_dir):
    """
    Creates and saves a detailed, unified CSV for the BAO analysis results.
    """
    logger = logging.getLogger()

    # Reconstruct the detailed DataFrames from the results dictionary
    lcdm_df_dict = results['results']['lcdm']['bao_analysis'].get('detailed_df')
    alt_df_dict = results['results']['alt_model']['bao_analysis'].get('detailed_df')

    if lcdm_df_dict is None or alt_df_dict is None:
        logger.warning("BAO detailed data missing, skipping unified CSV output.")
        return

    lcdm_df = _reconstruct_df_from_split(lcdm_df_dict)
    alt_df = _reconstruct_df_from_split(alt_df_dict)

    if lcdm_df.empty or alt_df.empty:
        logger.warning("Reconstructed BAO DataFrames are empty, skipping CSV output.")
        return

    # For BAO, combine the observational data with model predictions from both.
    obs_cols = ['redshift', 'observable_type', 'value', 'error']
    combined_df = lcdm_df[obs_cols].copy()
    combined_df['model_value_lcdm'] = lcdm_df['model_value']
    combined_df[f'model_value_{alt_model_name}'] = alt_df['model_value']

    filename = f"bao-detailed-data_LambdaCDM-vs-{alt_model_name}_{dataset_name}_{run_id}.csv"
    filepath = os.path.join(output_dir, filename)
    
    try:
        combined_df.to_csv(filepath, index=False, float_format='%.6f')
        logger.info(f"Unified BAO detailed data saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save unified BAO CSV {filename}: {e}", exc_info=True)


def create_csv_outputs(results, output_dir='output'):
    """
    The main public entry point for this module. It coordinates the creation
    of all relevant CSV output files based on the contents of the results data.

    Args:
        results (dict): The parsed JSON results from the cosmology engine.
        output_dir (str): The directory where files will be saved.
    """
    logger = logging.getLogger()
    logger.info("--- CSV Writer Module Activated ---")

    # Safely extract metadata needed for filenames
    run_id = results.get('metadata', {}).get('run_id', 'unknown_run')
    alt_model_name_raw = results.get('inputs', {}).get('models', {}).get('alt_model', {}).get('name', 'AltModel')
    # Sanitize the model name for use in filenames
    alt_model_name_safe = alt_model_name_raw.replace(' ', '_').replace('.', '')


    # Check if SNe analysis was run and save its CSV
    if 'sne_fit' in results.get('results', {}).get('lcdm', {}):
        sne_dataset_name_raw = results['inputs']['datasets']['sne_data']['attributes'].get('dataset_name_attr', 'SNe_data')
        sne_dataset_name_safe = sne_dataset_name_raw.replace('.dat', '').replace('.txt', '')
        _save_sne_csv(results, alt_model_name_safe, sne_dataset_name_safe, run_id, output_dir)

    # Check if BAO analysis was run and save its CSV
    if 'bao_analysis' in results.get('results', {}).get('lcdm', {}):
        bao_dataset_name_raw = results['inputs']['datasets']['bao_data']['attributes'].get('dataset_name_attr', 'BAO_data')
        bao_dataset_name_safe = bao_dataset_name_raw.replace('.json', '')
        _save_bao_csv(results, alt_model_name_safe, bao_dataset_name_safe, run_id, output_dir)