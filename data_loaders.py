# copernican_suite/data_loaders.py
"""
Modular data loading for various cosmological datasets (SNe, BAO, etc.).
...
"""
import pandas as pd
import numpy as np
import json
import os
import logging

# --- Parser Registry ---
SNE_PARSERS = {}
BAO_PARSERS = {}

# --- Decorators to register parsers ---
def register_sne_parser(name, description=""):
    """Decorator to register a SNe data parsing function."""
    def decorator(func):
        SNE_PARSERS[name] = {'function': func, 'description': description}
        return func
    return decorator

def register_bao_parser(name, description=""):
    """Decorator to register a BAO data parsing function."""
    def decorator(func):
        BAO_PARSERS[name] = {'function': func, 'description': description}
        return func
    return decorator

# --- Helper to list and select parsers ---
def _select_parser(parser_registry, data_type_name):
    """Displays available parsers and prompts user for selection."""
    logger = logging.getLogger()
    if not parser_registry:
        logger.error(f"No parsers registered for {data_type_name} data.")
        return None

    logger.info(f"\nAvailable {data_type_name} data format parsers:")
    options = list(parser_registry.keys())
    for i, key in enumerate(options):
        desc = parser_registry[key]['description']
        # Print to console directly for user interaction
        print(f"  {i+1}. {key} ({desc})")

    while True:
        try:
            choice = input(f"Select a {data_type_name} parser (number) or 'c' to cancel: ")
            if choice.lower() == 'c':
                return None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                return options[choice_idx]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'c'.")

# --- Main Loading Functions ---
def load_sne_data(filepath, format_key=None, **kwargs):
    """Loads SNe data using a registered parser."""
    logger = logging.getLogger()
    if not os.path.isfile(filepath):
        logger.error(f"SNe data file not found at {filepath}")
        return None

    if format_key is None:
        format_key = _select_parser(SNE_PARSERS, "SNe")
        if format_key is None:
            logger.info("SNe data loading canceled by user.")
            return None
    
    if format_key not in SNE_PARSERS:
        logger.error(f"No SNe parser registered for format_key '{format_key}'")
        return None
        
    try:
        logger.info(f"Attempting to load SNe data from '{os.path.basename(filepath)}' using format: {format_key}")
        parser_func = SNE_PARSERS[format_key]['function']
        data_df = parser_func(filepath, **kwargs) 
        if data_df is not None and not data_df.empty:
            data_df.attrs['filepath'] = filepath
            data_df.attrs['format_key'] = format_key
            if 'dataset_name_attr' not in data_df.attrs: 
                data_df.attrs['dataset_name_attr'] = f"SNe_{format_key.replace(' ', '_')}"
            logger.info(f"Successfully loaded {len(data_df)} SNe data points.")
        elif data_df is None:
             logger.error(f"SNe parser '{format_key}' returned None for {filepath}.")
        else: 
             logger.error(f"SNe parser '{format_key}' returned an empty DataFrame for {filepath}.")
        return data_df
    except Exception as e:
        logger.critical(f"CRITICAL Error during SNe data parsing ({filepath}, {format_key}): {e}", exc_info=True)
        return None

def load_bao_data(filepath, format_key=None, **kwargs):
    """Loads BAO data using a registered parser."""
    logger = logging.getLogger()
    if not os.path.isfile(filepath):
        logger.error(f"BAO data file not found at {filepath}")
        return None

    if format_key is None:
        format_key = _select_parser(BAO_PARSERS, "BAO")
        if format_key is None:
            logger.info("BAO data loading canceled by user.")
            return None

    if format_key not in BAO_PARSERS:
        logger.error(f"No BAO parser registered for format_key '{format_key}'")
        return None
        
    try:
        logger.info(f"Attempting to load BAO data from '{os.path.basename(filepath)}' using format: {format_key}")
        parser_func = BAO_PARSERS[format_key]['function']
        data_df = parser_func(filepath, **kwargs)
        if data_df is not None and not data_df.empty:
            data_df.attrs['filepath'] = filepath
            data_df.attrs['format_key'] = format_key
            if 'dataset_name_attr' not in data_df.attrs: 
                data_df.attrs['dataset_name_attr'] = f"BAO_{format_key.replace(' ', '_')}"
            logger.info(f"Successfully loaded {len(data_df)} BAO data points.")
        elif data_df is None:
            logger.error(f"BAO parser '{format_key}' returned None for {filepath}.")
        else: 
            logger.error(f"BAO parser '{format_key}' returned an empty DataFrame for {filepath}.")
        return data_df
    except Exception as e:
        logger.critical(f"CRITICAL Error during BAO data parsing ({filepath}, {format_key}): {e}", exc_info=True)
        return None

# --- Constants for SNe h1_style fits ---
DEFAULT_SALT2_M_ABS_FIXED = -19.3
DEFAULT_SALT2_ALPHA_FIXED = 0.14
DEFAULT_SALT2_BETA_FIXED = 3.1

# --- Specific Parsers ---

@register_sne_parser("unistra_fixed_nuisance_h1", "UniStra-like (e.g., tablef3.dat), h1-style: mu_obs from fixed M,alpha,beta.")
def parse_unistra_h1_style(filepath, salt2_m_abs_fixed=DEFAULT_SALT2_M_ABS_FIXED, salt2_alpha_fixed=DEFAULT_SALT2_ALPHA_FIXED, salt2_beta_fixed=DEFAULT_SALT2_BETA_FIXED, **kwargs):
    """Parses UniStra-like fixed-width files and calculates mu_obs with fixed nuisance parameters."""
    logger = logging.getLogger()
    col_specs = [(0,12),(12,21),(21,30),(30,31),(31,41),(41,50),(50,60),(60,69),(69,79),(79,88),(88,98),(98,108),(108,121),(121,130),(130,140),(140,150),(150,160),(160,161),(161,172),(172,183),(183,193)]
    col_names = ['Name','zcmb_str','zhel_str','e_z_str','mb_str','e_mb_str','x1_str','e_x1_str','c_str','e_c_str','logMst_str','e_logMst_str','tmax_str','e_tmax_str','cov_mb_x1_str','cov_mb_c_str','cov_x1_c_str','set_str','RAdeg_str','DEdeg_str','bias_str']
    try:
        df = pd.read_fwf(filepath, colspecs=col_specs, names=col_names, dtype=str, comment="#")
    except Exception as e:
        logger.error(f"Error reading UniStra-like file for h1_style: {e}"); return None

    parsed_data = pd.DataFrame()
    parsed_data['Name'] = df['Name'].str.strip()
    
    # Convert relevant columns to numeric
    cols_to_numeric = {'zcmb':'zcmb_str','mb':'mb_str','e_mb':'e_mb_str','x1':'x1_str','c':'c_str'}
    for new_col, old_col_str in cols_to_numeric.items():
        parsed_data[new_col] = pd.to_numeric(df[old_col_str], errors='coerce')

    try:
        if not all(col in parsed_data and not parsed_data[col].isnull().all() for col in ['mb', 'x1', 'c']):
             raise ValueError("mb, x1, or c contain all NaNs or are missing before mu_obs calculation.")
        
        # BUGFIX: Corrected the Tripp equation sign for the beta term. It's `m - M + a*x1 - b*c`.
        # The previous version had `... - a*x1 + b*c`, which was incorrect.
        parsed_data['mu_obs'] = parsed_data['mb'] - salt2_m_abs_fixed + salt2_alpha_fixed * parsed_data['x1'] - salt2_beta_fixed * parsed_data['c']
        
        # The primary error on mu_obs in this simplified scheme is the error on mb
        parsed_data['e_mu_obs'] = parsed_data['e_mb']
    except Exception as e:
        logger.error(f"Failed mu_obs calculation for UniStra h1_style: {e}"); return None

    essential_cols = ['Name','zcmb','mu_obs','e_mu_obs']
    if any(col not in parsed_data.columns or parsed_data[col].isnull().all() for col in essential_cols):
        logger.error("One or more essential columns missing/all NaN after parsing for UniStra h1_style."); return None
    
    parsed_data_filtered = parsed_data[essential_cols].dropna().copy()
    if parsed_data_filtered.empty: 
        logger.error("No valid SNe data remains after cleaning NaNs in UniStra h1_style parser."); return None
    
    parsed_data_filtered = parsed_data_filtered.sort_values(by='zcmb').reset_index(drop=True)
    
    # Set attributes to guide the fitting engine
    parsed_data_filtered.attrs['fit_style'] = 'h1_fixed_nuisance'
    parsed_data_filtered.attrs['is_mu_data'] = True
    parsed_data_filtered.attrs['fit_nuisance_params'] = False
    parsed_data_filtered.attrs['diag_errors_for_plot'] = parsed_data_filtered['e_mu_obs'].values
    parsed_data_filtered.attrs['salt2_m_abs_fixed'] = salt2_m_abs_fixed
    parsed_data_filtered.attrs['salt2_alpha_fixed'] = salt2_alpha_fixed
    parsed_data_filtered.attrs['salt2_beta_fixed'] = salt2_beta_fixed
    return parsed_data_filtered

# --- The rest of the file (other parsers) remains unchanged ---
@register_sne_parser("unistra_raw_lc_h2", "UniStra-like (e.g., tablef3.dat), h2-style: fit mb,x1,c and nuisance M,alpha,beta.")
def parse_unistra_h2_style(filepath, **kwargs):
    logger = logging.getLogger()
    col_specs = [(0,12),(12,21),(21,30),(30,31),(31,41),(41,50),(50,60),(60,69),(69,79),(79,88),(88,98),(98,108),(108,121),(121,130),(130,140),(140,150),(150,160),(160,161),(161,172),(172,183),(183,193)]
    col_names = ['Name','zcmb_str','zhel_str','e_z_str','mb_str','e_mb_str','x1_str','e_x1_str','c_str','e_c_str','logMst_str','e_logMst_str','tmax_str','e_tmax_str','cov_mb_x1_str','cov_mb_c_str','cov_x1_c_str','set_str','RAdeg_str','DEdeg_str','bias_str']
    try:
        df = pd.read_fwf(filepath, colspecs=col_specs, names=col_names, dtype=str, comment="#")
    except Exception as e:
        logger.error(f"Error reading UniStra-like file (h2_style): {e}"); return None

    cols_to_convert = {'zcmb':'zcmb_str','mb':'mb_str','e_mb':'e_mb_str','x1':'x1_str','e_x1':'e_x1_str','c':'c_str','e_c':'e_c_str',
                       'cov_mb_x1':'cov_mb_x1_str', 'cov_mb_c':'cov_mb_c_str', 'cov_x1_c':'cov_x1_c_str'}
    parsed_data = pd.DataFrame()
    parsed_data['Name'] = df['Name'].str.strip()

    for new_col, old_col_str in cols_to_convert.items():
        parsed_data[new_col] = pd.to_numeric(df[old_col_str], errors='coerce')

    essential_cols = ['zcmb','mb','e_mb','x1','e_x1','c','e_c']
    if any(col not in parsed_data.columns or parsed_data[col].isnull().all() for col in essential_cols):
        logger.error("One or more essential columns missing/all NaN in UniStra h2_style data."); return None
    
    cols_to_keep = ['Name'] + list(cols_to_convert.keys())
    parsed_data_filtered = parsed_data[cols_to_keep].dropna(subset=essential_cols).copy()

    if parsed_data_filtered.empty: 
        logger.error("No valid SNe data remains after cleaning NaNs in UniStra h2_style parser."); return None
    
    parsed_data_filtered = parsed_data_filtered.sort_values(by='zcmb').reset_index(drop=True)
    
    parsed_data_filtered.attrs['fit_style'] = 'h2_fit_nuisance'
    parsed_data_filtered.attrs['is_mu_data'] = False
    parsed_data_filtered.attrs['fit_nuisance_params'] = True
    parsed_data_filtered.attrs['diag_errors_for_plot_raw_e_mb'] = parsed_data_filtered['e_mb'].values
    return parsed_data_filtered


@register_sne_parser("pantheon_plus_mu_cov_h2", "Pantheon+ (e.g., Pantheon+SH0ES.txt + .cov), h2-style: fit MU_SH0ES with full Covariance Matrix.")
def parse_pantheon_plus_mu_cov_h2(filepath, cov_filepath=None, **kwargs):
    logger = logging.getLogger()
    if not cov_filepath or not os.path.isfile(cov_filepath):
        # Attempt to infer covariance filepath
        base, _ = os.path.splitext(filepath)
        inferred_cov = base + "_STAT+SYS.txt"
        if os.path.isfile(inferred_cov):
            logger.info(f"Inferred covariance file: {inferred_cov}")
            cov_filepath = inferred_cov
        else:
            logger.error(f"Pantheon+ covariance file not specified or found (tried {inferred_cov})."); return None

    try:
        # Load main data
        temp_df = pd.read_csv(filepath, delim_whitespace=True, comment='#')
        data_df = pd.DataFrame()
        # Map possible column names to standardized names
        col_map = {'Name':['CID','SNID','ID','NAME'],'zcmb':['zCMB','ZCMB','zcmb'],'mu_obs':['MU_SH0ES','mu'],'mu_sh0es_err_diag':['MU_SH0ES_ERR_DIAG','e_mu_diag']}
        
        # Load covariance matrix first to get expected size
        with open(cov_filepath,'r') as f: N_cov = int(f.readlines()[0].strip())

        for target_col, possible_names in col_map.items():
            found_col = next((p for p in possible_names if p in temp_df.columns), None)
            if found_col: data_df[target_col] = temp_df[found_col]
            elif target_col not in ['Name', 'mu_sh0es_err_diag']:
                logger.error(f"Column for '{target_col}' not found in Pantheon+ (mu_cov)."); return None
        
        if 'Name' not in data_df: data_df['Name'] = temp_df.get('CID', pd.Series([f"SN_PPlus_mucov_{i}" for i in range(len(temp_df))]))
        data_df['Name'] = data_df['Name'].astype(str).str.strip()
        
        essential_cols = ['zcmb','mu_obs']
        for col in essential_cols + ['mu_sh0es_err_diag']:
            if col in data_df: data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        
        if any(col not in data_df.columns or data_df[col].isnull().all() for col in essential_cols):
            logger.error("One or more essential columns missing/all NaN in Pantheon+ mu_cov_h2 data."); return None

        data_df = data_df.dropna(subset=essential_cols).reset_index(drop=True)
        if data_df.empty: logger.error("No valid Pantheon+ mu_cov_h2 SNe data after filtering."); return None
        if len(data_df) != N_cov: logger.critical(f"SNe count for mu_cov: data ({len(data_df)}) vs cov N ({N_cov})."); return None
        
        # Load and reshape covariance matrix
        cov_matrix_flat = np.loadtxt(cov_filepath, skiprows=1)
        if len(cov_matrix_flat) != N_cov*N_cov: logger.error(f"Cov matrix len ({len(cov_matrix_flat)}) != N*N ({N_cov*N_cov})."); return None
        cov_matrix_pantheon = cov_matrix_flat.reshape((N_cov,N_cov))

        # Use diagonal errors from file if present, otherwise calculate from covariance matrix
        if 'mu_sh0es_err_diag' in data_df and data_df['mu_sh0es_err_diag'].notna().any():
            data_df['e_mu_obs'] = data_df['mu_sh0es_err_diag']
        else:
            data_df['e_mu_obs'] = np.sqrt(np.diag(cov_matrix_pantheon)) 

        output_df = data_df[['Name', 'zcmb', 'mu_obs', 'e_mu_obs']].copy().sort_values(by='zcmb').reset_index(drop=True)
        
        output_df.attrs['fit_style'] = 'h2_mu_covariance'
        output_df.attrs['is_mu_data'] = True
        output_df.attrs['fit_nuisance_params'] = False
        try:
            output_df.attrs['covariance_matrix_inv'] = np.linalg.inv(cov_matrix_pantheon)
            output_df.attrs['diag_errors_for_plot'] = np.sqrt(np.diag(cov_matrix_pantheon))
        except np.linalg.LinAlgError:
            logger.warning("Could not invert Pantheon+ cov matrix. Chi2 will fallback to diagonal errors.")
            output_df.attrs['covariance_matrix_inv'] = None 
            output_df.attrs['diag_errors_for_plot'] = output_df['e_mu_obs'].values
        return output_df
    except Exception as e: 
        logger.error(f"Error processing Pantheon+ (mu_cov h2_style): {e}", exc_info=True); return None


@register_bao_parser("bao_json_general_v1", "General BAO JSON format (e.g., bao1.json).")
def parse_bao_json_v1(filepath, **kwargs):
    """Parses a generic BAO JSON file into a standard DataFrame."""
    logger = logging.getLogger()
    try:
        with open(filepath, 'r') as f: data_json = json.load(f)
        
        df = pd.DataFrame(data_json['data_points'])
        required_cols = ['redshift', 'observable_type', 'value', 'error']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"BAO JSON file {filepath} missing one or more required columns: {required_cols}"); return None 
        
        # Convert numeric columns, coercing errors to NaN
        for col in ['redshift', 'value', 'error']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=required_cols, inplace=True)
        if df.empty:
            logger.error(f"No valid BAO data points after parsing {filepath}."); return None 
            
        df.attrs['citation'] = data_json.get('citation', 'N/A')
        df.attrs['notes'] = data_json.get('notes', 'N/A')
        df.attrs['dataset_name_attr'] = data_json.get('name', f"BAO_{os.path.basename(filepath)}")
        return df
    except Exception as e:
        logger.error(f"Error reading or parsing BAO JSON file {filepath}: {e}", exc_info=True); return None