# DEV NOTE (v1.5e): Extracted from data_loaders.py for modular architecture.
# Registers the UniStra raw light-curve (h2) parser.
# DEV NOTE (v1.5f hotfix): Updated ``data_loaders`` import path.

import pandas as pd
import logging

from scripts.data_loaders import register_sne_parser

@register_sne_parser(
    "unistra_raw_lc_h2",
    "UniStra-like (e.g., tablef3.dat), h2-style: fit mb,x1,c and nuisance M,alpha,beta."
)
def parse_unistra_h2_style(filepath, **kwargs):
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
