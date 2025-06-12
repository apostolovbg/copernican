# data_loaders.py
"""
Handles the loading and parsing of various cosmological data formats.

DEV NOTE (v1.4g):
tkyfah-codex/fix-bug-in-data_readers.py-and-explain-long-term-solution
Restored the UniStra fixed-width parsers using the stable v1.3 logic.
They now target the correct columns, convert '---' to NaN, and load all 740 SNe.
=======
 codex/fix-bug-in-data_readers.py-and-explain-long-term-solution
Restored the UniStra fixed-width parsers using the stable v1.3 logic.
They now target the correct columns, convert '---' to NaN, and load all 740 SNe.

gggxa2-codex/fix-bug-in-data_readers.py-and-explain-long-term-solution
Restored the UniStra fixed-width parsers using the stable v1.3 logic.
They now target the correct columns, convert '---' to NaN, and load all 740 SNe.

The UniStra parsers now replicate the stable v1.3 fixed-width logic.
This restores correct column targeting and NaN handling for `tablef3.dat`.
1.4g
"""

import os
import sys
import pandas as pd
import numpy as np
import json

# --- Constants used for UniStra parsers (from v1.3) ---
DEFAULT_SALT2_M_ABS_FIXED = -19.3
DEFAULT_SALT2_ALPHA_FIXED = 0.14
DEFAULT_SALT2_BETA_FIXED = 3.1

# --- SNe Ia Parser Functions (Restored from v1.3 for stability) ---

def _load_unistra_fixed_nuisance_h1(filepath, **kwargs):
    """Parses UniStra-like data using fixed nuisance parameters (h1-style)."""
    try:
        col_specs = [
            (0, 12), (12, 21), (21, 30), (30, 31), (31, 41), (41, 50), (50, 60),
            (60, 69), (69, 79), (79, 88), (88, 98), (98, 108), (108, 121),
            (121, 130), (130, 140), (140, 150), (150, 160), (160, 161),
            (161, 172), (172, 183), (183, 193)
        ]
        col_names = [
            'name', 'z_cmb_str', 'z_hel_str', 'ez_str', 'mb_str', 'e_mb_str',
            'x1_str', 'e_x1_str', 'c_str', 'e_c_str', 'logM_str', 'e_logM_str',
            'tmax_str', 'e_tmax_str', 'cov_mb_x1_str', 'cov_mb_c_str',
            'cov_x1_c_str', 'set_str', 'ra_str', 'dec_str', 'bias_str'
        ]

        df_raw = pd.read_fwf(
            filepath, colspecs=col_specs, names=col_names,
            na_values='---', comment='#', dtype=str
        )

        df = pd.DataFrame()
        df['z_cmb'] = pd.to_numeric(df_raw['z_cmb_str'], errors='coerce')
        df['z_hel'] = pd.to_numeric(df_raw['z_hel_str'], errors='coerce')
        df['mb'] = pd.to_numeric(df_raw['mb_str'], errors='coerce')
        df['mb_err'] = pd.to_numeric(df_raw['e_mb_str'], errors='coerce')
        df['x1'] = pd.to_numeric(df_raw['x1_str'], errors='coerce')
        df['c'] = pd.to_numeric(df_raw['c_str'], errors='coerce')

        df['mu'] = df['mb'] - DEFAULT_SALT2_M_ABS_FIXED \
            + DEFAULT_SALT2_ALPHA_FIXED * df['x1'] \
            - DEFAULT_SALT2_BETA_FIXED * df['c']
        df['mu_err'] = df['mb_err']

        df['z'] = np.where(pd.notna(df['z_cmb']), df['z_cmb'], df['z_hel'])

        final = df[['z', 'mu', 'mu_err']].dropna().reset_index(drop=True)
        return final

    except Exception as e:
        print(f"FATAL: Failed to parse UniStra (h1-style) file '{filepath}': {e}")
        raise

def _load_unistra_fit_nuisance_h2(filepath, **kwargs):
    """
    Parser for UniStra-like data (h2-style).
    Reads light-curve parameters and handles missing z_cmb.
    """
    try:
        col_specs = [
            (0, 12), (12, 21), (21, 30), (30, 31), (31, 41), (41, 50), (50, 60),
            (60, 69), (69, 79), (79, 88), (88, 98), (98, 108), (108, 121),
            (121, 130), (130, 140), (140, 150), (150, 160), (160, 161),
            (161, 172), (172, 183), (183, 193)
        ]
        col_names = [
            'name', 'z_cmb_str', 'z_hel_str', 'ez_str', 'mb_str', 'e_mb_str',
            'x1_str', 'e_x1_str', 'c_str', 'e_c_str', 'logM_str', 'e_logM_str',
            'tmax_str', 'e_tmax_str', 'cov_mb_x1_str', 'cov_mb_c_str',
            'cov_x1_c_str', 'set_str', 'ra_str', 'dec_str', 'bias_str'
        ]

        df_raw = pd.read_fwf(
            filepath, colspecs=col_specs, names=col_names,
            na_values='---', comment='#', dtype=str
        )

        df = pd.DataFrame()
        df['z_cmb'] = pd.to_numeric(df_raw['z_cmb_str'], errors='coerce')
        df['z_hel'] = pd.to_numeric(df_raw['z_hel_str'], errors='coerce')
        df['mb'] = pd.to_numeric(df_raw['mb_str'], errors='coerce')
        df['mb_err'] = pd.to_numeric(df_raw['e_mb_str'], errors='coerce')
        df['x1'] = pd.to_numeric(df_raw['x1_str'], errors='coerce')
        df['x1_err'] = pd.to_numeric(df_raw['e_x1_str'], errors='coerce')
        df['c'] = pd.to_numeric(df_raw['c_str'], errors='coerce')
        df['c_err'] = pd.to_numeric(df_raw['e_c_str'], errors='coerce')

        df['z'] = np.where(pd.notna(df['z_cmb']), df['z_cmb'], df['z_hel'])

        final_cols = ['z', 'mb', 'mb_err', 'x1', 'x1_err', 'c', 'c_err']
        final = df[final_cols].dropna().reset_index(drop=True)
        return final

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
    print("  1. cosmo_engine_1.4g.py")
    print("  2. cosmo_engine_.1.4rc.py")
    engine_choice = input("Enter the number of the engine to use (or 'c' to cancel): ").strip()
    if engine_choice.lower() == 'c':
        return None
    engine_name = "cosmo_engine_1.4g.py" if engine_choice == '1' else "cosmo_engine_.1.4rc.py"

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
