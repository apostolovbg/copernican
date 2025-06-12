# data_loaders.py
"""
Handles the loading and parsing of various cosmological data formats.

DEV NOTE (v1.4rc12 - CRITICAL REGRESSION FIX):
This version corrects a fatal `NameError` introduced in v1.4rc11.

1.  CRITICAL FIX (NameError): Re-inserted several missing lines in the
    `_prompt_for_data` function that were accidentally deleted. These lines
    are responsible for defining the `parser_keys` variable. Their absence
    caused the program to crash instantly on startup.

(Previous notes from v1.4rc11 preserved below)
...
"""

import os
import sys
import pandas as pd
import numpy as np
import json

# --- SNe Ia Parser Functions (Restored from v1.3 for stability) ---

def _load_unistra_fixed_nuisance_h1(filepath, **kwargs):
    """
    Parser for UniStra-like data (h1-style).
    Reads z_cmb, mu_obs, and mu_err from tablef3.dat.
    Falls back to z_hel if z_cmb is not available for a given row.
    """
    try:
        # Read z_hel, z_cmb, mu, and mu_err
        data = pd.read_csv(
            filepath,
            sep=r'\s+',
            comment='#',
            usecols=[1, 2, 4, 5],
            names=['z_hel', 'z_cmb', 'mu', 'mu_err']
        )
        
        # Coerce all columns to numeric, turning non-numbers into NaN
        data['z_hel'] = pd.to_numeric(data['z_hel'], errors='coerce')
        data['z_cmb'] = pd.to_numeric(data['z_cmb'], errors='coerce')
        data['mu'] = pd.to_numeric(data['mu'], errors='coerce')
        data['mu_err'] = pd.to_numeric(data['mu_err'], errors='coerce')

        # Create the final 'z' column: use z_cmb if it's a valid number, otherwise use z_hel.
        data['z'] = np.where(pd.notna(data['z_cmb']), data['z_cmb'], data['z_hel'])

        # Drop rows only if the essential FINAL columns are missing
        data.dropna(subset=['z', 'mu', 'mu_err'], inplace=True)
        
        # Return only the standardized columns the engine needs
        return data[['z', 'mu', 'mu_err']]
        
    except Exception as e:
        print(f"FATAL: Failed to parse UniStra (h1-style) file '{filepath}': {e}")
        raise

def _load_unistra_fit_nuisance_h2(filepath, **kwargs):
    """
    Parser for UniStra-like data (h2-style).
    Reads light-curve parameters and handles missing z_cmb.
    """
    try:
        # Read all necessary columns, including both redshift types
        data = pd.read_csv(
            filepath,
            sep=r'\s+',
            comment='#',
            usecols=[1, 2, 7, 8, 9, 10, 11, 12],
            names=['z_hel', 'z_cmb', 'mb', 'mb_err', 'x1', 'x1_err', 'c', 'c_err']
        )
        
        # Coerce all columns to numeric
        numeric_cols = ['z_hel', 'z_cmb', 'mb', 'mb_err', 'x1', 'x1_err', 'c', 'c_err']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Create the final 'z' column with fallback logic
        data['z'] = np.where(pd.notna(data['z_cmb']), data['z_cmb'], data['z_hel'])
        
        # Define the essential final columns and drop rows if they are missing
        final_cols = ['z', 'mb', 'mb_err', 'x1', 'x1_err', 'c', 'c_err']
        data.dropna(subset=final_cols, inplace=True)
        
        # Return only the standardized columns
        return data[final_cols]

    except Exception as e:
        print(f"FATAL: Failed to parse UniStra (h2-style) file '{filepath}': {e}")
        raise

def _load_pantheon_plus_h2(filepath, base_dir, **kwargs):
    """
    Parser for Pantheon+ data (h2-style).
    Requires a main data file and a separate covariance matrix file.
    """
    try:
        df = pd.read_csv(filepath, sep=r'\s+', comment='#')
        df.rename(columns={'zCMB': 'z', 'm_b_corr': 'mb', 'm_b_corr_err_DIAG': 'mb_err'}, inplace=True)
        
        numeric_cols = ['z', 'mb', 'mb_err']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=numeric_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

        print("\n  > The Pantheon+ parser requires a covariance matrix file.")
        cov_path_prompt = "  > Enter the path to the covariance matrix (e.g., Pancm.txt): "
        while True:
            cov_filepath = input(cov_path_prompt).strip()
            if cov_filepath.lower() == 'c':
                raise KeyboardInterrupt("User cancelled during covariance file selection.")

            full_cov_path = cov_filepath if os.path.isabs(cov_filepath) else os.path.join(base_dir, cov_filepath)
            if os.path.isfile(full_cov_path):
                break
            else:
                print(f"  > Error: Covariance file not found at '{full_cov_path}'. Please try again.")

        cov_matrix = np.loadtxt(full_cov_path)
        if len(cov_matrix) != len(df):
            raise ValueError("Covariance matrix dimensions do not match data file after cleaning.")

        df.attrs['cov_matrix'] = cov_matrix
        output_df = df[['z', 'mb', 'mb_err']].copy()
        output_df.attrs['cov_matrix'] = cov_matrix
        return output_df
    except Exception as e:
        print(f"FATAL: Failed to process Pantheon+ file '{filepath}': {e}")
        raise

# --- BAO Parser Functions (Restored from v1.3 for stability) ---

def _load_bao_json_v1(filepath, **kwargs):
    """Parses a generic BAO JSON file into a standard DataFrame."""
    try:
        with open(filepath, 'r') as f:
            data_json = json.load(f)

        df = pd.DataFrame(data_json['data_points'])
        
        required_cols = ['redshift', 'observable_type', 'value', 'error']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"BAO JSON file missing one or more required columns: {required_cols}")

        for col in ['redshift', 'value', 'error']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=required_cols, inplace=True)
        
        df.rename(columns={'redshift': 'z'}, inplace=True)

        if df.empty:
            raise ValueError("No valid BAO data points after parsing.")
            
        if 'rs_drag' in data_json:
             df['rs_drag'] = pd.to_numeric(data_json['rs_drag'])
        else:
             df['rs_drag'] = np.nan
             
        return df
    except Exception as e:
        print(f"FATAL: Failed to parse BAO JSON file '{filepath}': {e}")
        raise

# --- Parser Dictionaries (v1.4rc structure) ---
SNE_PARSERS = {
    'unistra_fixed_nuisance_h1': {
        'function': _load_unistra_fixed_nuisance_h1,
        'description': "UniStra-like (e.g., tablef3.dat), h1-style: uses pre-calculated mu_obs."
    },
    'unistra_fit_nuisance_h2': {
        'function': _load_unistra_fit_nuisance_h2,
        'description': "UniStra-like (e.g., tablef3.dat), h2-style: fits nuisance params from mb,x1,c."
    },
    'pantheon_plus_h2': {
        'function': _load_pantheon_plus_h2,
        'description': "Pantheon+ (e.g., Pantheon+SH0ES.txt + .cov), h2-style: fits MU_SH0ES with full Covariance Matrix."
    }
}

BAO_PARSERS = {
    'bao_json_general_v1': {
        'function': _load_bao_json_v1,
        'description': "General BAO JSON format (e.g., bao_data.json)."
    }
}


# --- User Interface Functions (from v1.4rc) ---

def _prompt_for_data(base_dir, data_type_name, parsers_dict, is_optional=False):
    """Generic helper to prompt user for a data file and a parser choice."""
    icons = {"SNe Ia": "🌠", "BAO": "🌌"}
    icon = icons.get(data_type_name, "🛰️")
    print(f"\n--- {icon} {data_type_name} Data ---")

    prompt_msg = f"Enter path to {data_type_name} data file"
    if is_optional:
        prompt_msg += " (or press Enter to skip): "
    else:
        prompt_msg += ": "

    filepath = ''
    while True:
        filepath = input(prompt_msg).strip()
        if filepath.lower() == 'c':
            return None
        if is_optional and not filepath:
            return {}

        full_path = filepath if os.path.isabs(filepath) else os.path.join(base_dir, filepath)
        if os.path.isfile(full_path):
            break
        else:
            print(f"Error: File not found at '{full_path}'. Please check the path and try again.")

    # FIX (v1.4rc12): These lines were accidentally deleted and are now restored.
    print(f"Available {data_type_name} data format parsers:")
    parser_keys = list(parsers_dict.keys())
    for i, key in enumerate(parser_keys):
        print(f"  {i+1}. {parsers_dict[key]['description']}")

    parser_id = ''
    while True:
        choice = input(f"Select a {data_type_name} parser (number) or 'c' to cancel: ").strip()
        if choice.lower() == 'c':
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(parser_keys):
                parser_id = parser_keys[idx]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    key_name = 'sne_data' if data_type_name == 'SNe Ia' else 'bao_data'

    return {'filepath': filepath, 'parser_id': parser_id, 'type': key_name}

def get_user_selections(base_dir):
    """Main UI function to get all user selections for a run."""
    print("\n--- 🪐 Select a Computational Engine ---")
    print("  1. cosmo_engine_.1.4rc.py")
    engine_choice = input("Enter the number of the engine to use (or 'c' to cancel): ").strip()
    if engine_choice.lower() == 'c': return None
    engine_name = "cosmo_engine_.1.4rc.py"

    alt_model_path = input("Enter path to alternative model .py file (or 'test'): ").strip()
    if alt_model_path.lower() == 'c': return None

    sne_data_info = _prompt_for_data(base_dir, "SNe Ia", SNE_PARSERS)
    if sne_data_info is None: return None

    bao_data_info = _prompt_for_data(base_dir, "BAO", BAO_PARSERS, is_optional=True)
    if bao_data_info is None: return None

    return {
        'engine_name': engine_name,
        'alt_model_path': alt_model_path,
        'sne_data_info': sne_data_info,
        'bao_data_info': bao_data_info
    }