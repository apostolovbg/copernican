# data_loaders.py
"""
Handles the loading and parsing of various cosmological data formats.

DEV NOTE (v1.4rc):
1.  CRITICAL FIX: Restored the 'usecols' argument to the
    '_load_unistra_fixed_nuisance_h1' parser. This argument was present
    in v1.3 and is essential for allowing this parser to extract the 3
    required columns (z, mu, mu_err) from a wide data file like
    'tablef3.dat'. Its removal was the cause of the ParserError crash.
2.  ERROR HANDLING: Removed internal try/except blocks from data loading
    functions to allow detailed ParserErrors to propagate to the main
    exception hook, ensuring full tracebacks are logged.
3.  SYNTAX UPDATE: Replaced the deprecated 'delim_whitespace' keyword with
    the modern separator for whitespace, using a raw string in the code
    to prevent SyntaxWarnings.
"""

import os
import sys
import pandas as pd
import numpy as np
import json

# --- Data Loading Functions ---

def _load_unistra_fixed_nuisance_h1(filepath):
    """
    Parser for UniStra-like data files (e.g., tablef3.dat).
    This format assumes mu_obs is pre-calculated from fixed nuisance parameters.
    It reads a wide file but only uses columns for z, mu, and mu_err.
    """
    data = pd.read_csv(
        filepath,
        sep=r'\s+',
        comment='#',
        usecols=[1, 8, 9],
        names=['z_cmb', 'mu_obs', 'mu_err']
    )
    print(f"Successfully loaded {len(data)} SNe data points from '{os.path.basename(filepath)}'.")
    return {'sne_data': data}

def _load_unistra_raw_lc_h2(filepath):
    """
    Parser for SNe data files with raw lightcurve params (e.g., tablef3.dat).
    Used when fitting for nuisance parameters M, alpha, beta.
    """
    data = pd.read_csv(
        filepath,
        sep=r'\s+',
        comment='#',
        names=['z_cmb', 'm_b', 'x1', 'c', 'm_b_err', 'x1_err', 'c_err', 'cov_mb_x1', 'cov_mb_c', 'cov_x1_c']
    )
    print(f"Successfully loaded {len(data)} SNe data points with lightcurve parameters from '{os.path.basename(filepath)}'.")
    return {'sne_data': data}

def _load_pantheon_plus_mu_cov_h2(filepath):
    """
    Parser for Pantheon+ data files, including a main data file and a .cov file.
    """
    cov_path = os.path.splitext(filepath)[0] + ".cov"
    if not os.path.exists(cov_path):
        raise FileNotFoundError(f"Covariance matrix file not found at '{cov_path}'. It must exist alongside the data file '{filepath}'.")

    data = pd.read_csv(
        filepath,
        sep=r'\s+',
        comment='#',
        names=['z_cmb', 'z_helio', 'mu_sh0es', 'mu_sh0es_err']
    )
    cov_matrix = np.loadtxt(cov_path, skiprows=1)
    print(f"Successfully loaded {len(data)} Pantheon+ SNe data points and covariance matrix.")
    return {'sne_data': data, 'sne_cov_matrix': cov_matrix}

def _load_bao_json_general_v1(filepath):
    """
    Parser for a general JSON format for BAO data.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Successfully loaded {len(data)} BAO data points from '{os.path.basename(filepath)}'.")
    return {'bao_data': data}

# --- Parser Dictionaries ---

SNE_PARSERS = {
    'unistra_fixed_nuisance_h1': {
        'function': _load_unistra_fixed_nuisance_h1,
        'description': "UniStra-like (e.g., tablef3.dat), h1-style: mu_obs from fixed M,alpha,beta."
    },
    'unistra_raw_lc_h2': {
        'function': _load_unistra_raw_lc_h2,
        'description': "UniStra-like (e.g., tablef3.dat), h2-style: fit mb,x1,c and nuisance M,alpha,beta."
    },
    'pantheon_plus_mu_cov_h2': {
        'function': _load_pantheon_plus_mu_cov_h2,
        'description': "Pantheon+ (e.g., Pantheon+SH0ES.txt + .cov), h2-style: fit MU_SH0ES with full Covariance Matrix."
    }
}

BAO_PARSERS = {
    'bao_json_general_v1': {
        'function': _load_bao_json_general_v1,
        'description': "General BAO JSON format (e.g., bao1.json)."
    }
}

# --- Interactive Data Collection Function ---

def collect_data_info(data_type_name, parsers_dict, base_dir, is_optional=False):
    """
    Interactively prompts the user to provide a data file and select a parser.
    """
    icons = {"SNe Ia": "🌠", "BAO": "🌌"}
    icon = icons.get(data_type_name, "🛰️")
    print(f"\n--- {icon} {data_type_name} Data ---")
    
    prompt_msg = f"Enter path to {data_type_name} data file"
    if is_optional:
        prompt_msg += " (or press Enter to skip): "
    else:
        prompt_msg += ": "
        
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
    
    print(f"Available {data_type_name} data format parsers:")
    parser_keys = list(parsers_dict.keys())
    for i, key in enumerate(parser_keys):
        print(f"  {i+1}. {parsers_dict[key]['description']}")

    while True:
        choice = input(f"Select a {data_type_name} parser (number) or 'c' to cancel: ").strip()
        if choice.lower() == 'c':
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(parser_keys):
                parser_key = parser_keys[idx]
                return {'path': filepath, 'parser': parser_key}
        except ValueError:
            pass
        
        print("Invalid selection. Please enter a number from the list.")