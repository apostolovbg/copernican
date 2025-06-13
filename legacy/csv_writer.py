# csv_writer.py
# Handles the generation of all CSV output files.

"""
DEV NOTE (v1.4rc): This module has been fortified to be more robust. Each
function now checks for the existence of the required data in the results
dictionary before attempting to create a CSV file. If data is missing (e.g.,
no BAO data was provided for the run), the function will log an info message
and exit gracefully instead of crashing with a KeyError. This improves the
suite's overall stability and error handling.

DEV NOTE (v1.4g): Added a missing newline at the end of the file to satisfy
repository style guidelines. No functional code changes were made.
DEV NOTE (data filenames): CSV generators now derive dataset names from the
new 'filepath' fields provided by the engine.
"""

import logging
import os
import pandas as pd

# --- Helper Functions ---

def _reconstruct_df_from_split(split_dict):
    """Reconstructs a Pandas DataFrame from the 'split' dictionary format."""
    return pd.DataFrame(split_dict['data'], index=split_dict['index'], columns=split_dict['columns'])

def _save_df_to_csv(df, full_path):
    """Saves a DataFrame to a CSV file."""
    try:
        df.to_csv(full_path, index=False)
        logging.info(f"Successfully saved data to {full_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV to {full_path}: {e}", exc_info=True)

# --- Public CSV Generation Functions ---

def create_sne_csv(results, style_guide):
    """Creates the detailed CSV for the SNe Ia analysis."""
    # Fortification: Check if SNe data exists
    sne_analysis = results.get('sne_analysis')
    if not sne_analysis or 'detailed_df' not in sne_analysis:
        logging.info("No SNe analysis data found in results. Skipping SNe CSV generation.")
        return

    df = _reconstruct_df_from_split(sne_analysis['detailed_df'])
    
    # Generate filename
    m1_name = results['metadata']['model1_name']
    m2_name = results['metadata']['model2_name']
    dataset_path = sne_analysis.get('filepath', 'sne')
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    run_id = results['metadata']['run_id']
    filename = f"sne-detailed-data_{m1_name}-vs-{m2_name}_{dataset_name}_{run_id}.csv"
    
    full_path = os.path.join('output', filename)
    _save_df_to_csv(df, full_path)


def create_bao_csv(results, style_guide):
    """Creates the detailed CSV for the BAO analysis."""
    # Fortification: Check if BAO data exists
    bao_analysis = results.get('bao_analysis')
    if not bao_analysis or 'detailed_df' not in bao_analysis:
        logging.info("No BAO analysis data found in results. Skipping BAO CSV generation.")
        return
        
    df = _reconstruct_df_from_split(bao_analysis['detailed_df'])

    # Generate filename
    m1_name = results['metadata']['model1_name']
    m2_name = results['metadata']['model2_name']
    run_id = results['metadata']['run_id']
    dataset_path = bao_analysis.get('filepath', 'bao')
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    filename = f"bao-detailed-data_{m1_name}-vs-{m2_name}_{dataset_name}_{run_id}.csv"
    
    full_path = os.path.join('output', filename)
    _save_df_to_csv(df, full_path)

def create_fit_summary_csv(results, style_guide):
    """Creates a summary CSV with the best-fit parameters and Chi2 values."""
    # Fortification: Check if fit data exists
    sne_analysis = results.get('sne_analysis')
    if not sne_analysis or 'model1_fit_results' not in sne_analysis:
        logging.info("No fit summary data found in results. Skipping fit summary CSV generation.")
        return
        
    m1_name = results['metadata']['model1_name']
    m2_name = results['metadata']['model2_name']
    m1_fit = sne_analysis['model1_fit_results']
    m2_fit = sne_analysis['model2_fit_results']

    summary_data = {
        'Model': [m1_name, m2_name],
        'Chi_Squared': [m1_fit['min_chi2'], m2_fit['min_chi2']],
        'Reduced_Chi_Squared': [m1_fit['reduced_chi2'], m2_fit['reduced_chi2']],
        'DOF': [m1_fit['dof'], m2_fit['dof']]
    }
    
    # Add all fitted parameters to the dictionary
    all_params = set(m1_fit['best_fit_params'].keys()) | set(m2_fit['best_fit_params'].keys())
    for param in all_params:
        summary_data[f"{param}_best_fit"] = [
            m1_fit['best_fit_params'].get(param),
            m2_fit['best_fit_params'].get(param)
        ]

    df = pd.DataFrame(summary_data)
    
    # Generate filename
    run_id = results['metadata']['run_id']
    filename = f"fit-summary_{m1_name}-vs-{m2_name}_{run_id}.csv"
    
    full_path = os.path.join('output', filename)
    _save_df_to_csv(df, full_path)
