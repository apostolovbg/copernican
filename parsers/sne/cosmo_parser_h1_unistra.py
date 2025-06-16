# DEV NOTE (v1.5e): Extracted from data_loaders.py during modular refactor.
# This module registers the UniStra fixed-nuisance (h1) parser.
# DEV NOTE (v1.5f hotfix): Updated ``data_loaders`` import path.

import pandas as pd
import logging

from scripts.data_loaders import register_sne_parser

DEFAULT_SALT2_M_ABS_FIXED = -19.3
DEFAULT_SALT2_ALPHA_FIXED = 0.14
DEFAULT_SALT2_BETA_FIXED = 3.1

@register_sne_parser(
    "unistra_fixed_nuisance_h1",
    "UniStra-like (e.g., tablef3.dat), h1-style: mu_obs from fixed M,alpha,beta."
)
def parse_unistra_h1_style(filepath, salt2_m_abs_fixed=DEFAULT_SALT2_M_ABS_FIXED,
                           salt2_alpha_fixed=DEFAULT_SALT2_ALPHA_FIXED,
                           salt2_beta_fixed=DEFAULT_SALT2_BETA_FIXED, **kwargs):
    """Parses UniStra-like fixed-width files and calculates mu_obs with fixed nuisance parameters."""
    logger = logging.getLogger()
    col_specs = [(0,12),(12,21),(21,30),(30,31),(31,41),(41,50),(50,60),(60,69),(69,79),
                 (79,88),(88,98),(98,108),(108,121),(121,130),(130,140),(140,150),(150,160),
                 (160,161),(161,172),(172,183),(183,193)]
    col_names = ['Name','zcmb_str','zhel_str','e_z_str','mb_str','e_mb_str','x1_str','e_x1_str',
                 'c_str','e_c_str','logMst_str','e_logMst_str','tmax_str','e_tmax_str',
                 'cov_mb_x1_str','cov_mb_c_str','cov_x1_c_str','set_str','RAdeg_str','DEdeg_str',
                 'bias_str']
    try:
        df = pd.read_fwf(filepath, colspecs=col_specs, names=col_names, dtype=str, comment="#")
    except Exception as e:
        logger.error(f"Error reading UniStra-like file for h1_style: {e}"); return None

    parsed_data = pd.DataFrame()
    parsed_data['Name'] = df['Name'].str.strip()

    cols_to_numeric = {'zcmb':'zcmb_str','mb':'mb_str','e_mb':'e_mb_str','x1':'x1_str','c':'c_str'}
    for new_col, old_col_str in cols_to_numeric.items():
        parsed_data[new_col] = pd.to_numeric(df[old_col_str], errors='coerce')

    try:
        if not all(col in parsed_data and not parsed_data[col].isnull().all() for col in ['mb', 'x1', 'c']):
            raise ValueError("mb, x1, or c contain all NaNs or are missing before mu_obs calculation.")
        parsed_data['mu_obs'] = parsed_data['mb'] - salt2_m_abs_fixed + salt2_alpha_fixed * parsed_data['x1'] - salt2_beta_fixed * parsed_data['c']
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

    parsed_data_filtered.attrs['fit_style'] = 'h1_fixed_nuisance'
    parsed_data_filtered.attrs['is_mu_data'] = True
    parsed_data_filtered.attrs['fit_nuisance_params'] = False
    parsed_data_filtered.attrs['diag_errors_for_plot'] = parsed_data_filtered['e_mu_obs'].values
    parsed_data_filtered.attrs['salt2_m_abs_fixed'] = salt2_m_abs_fixed
    parsed_data_filtered.attrs['salt2_alpha_fixed'] = salt2_alpha_fixed
    parsed_data_filtered.attrs['salt2_beta_fixed'] = salt2_beta_fixed
    return parsed_data_filtered
