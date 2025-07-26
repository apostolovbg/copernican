"""Parser for the JLA 2014 supernova sample with full covariance matrix."""

import os
import pandas as pd
import numpy as np
import logging
from astropy.io import fits

from copernican_lib.data_loaders import register_sne_parser
from copernican_lib.utils import load_metadata_from_dir

DATA_DIR = os.path.dirname(__file__)
META = load_metadata_from_dir(DATA_DIR)

DATASET_NAME = META.get("dataset_name", "JLA 2014")
DESCRIPTION = META.get(
    "description",
    "Joint SDSS-II and SNLS supernova sample (Betoule et al. 2014).",
)

DEFAULT_SALT2_M_ABS_FIXED = -19.3
DEFAULT_SALT2_ALPHA_FIXED = 0.14
DEFAULT_SALT2_BETA_FIXED = 3.1


@register_sne_parser(
    DATASET_NAME,
    DESCRIPTION,
    data_dir=DATA_DIR,
)
def parse_jla2014(
    data_dir,
    salt2_m_abs_fixed=DEFAULT_SALT2_M_ABS_FIXED,
    salt2_alpha_fixed=DEFAULT_SALT2_ALPHA_FIXED,
    salt2_beta_fixed=DEFAULT_SALT2_BETA_FIXED,
    **kwargs,
):
    """Parse JLA 2014 light-curve parameters and full covariance matrix."""
    logger = logging.getLogger()
    filepath = os.path.join(data_dir, "tablef3.dat")
    covpath = os.path.join(data_dir, "tablef4.fit")
    col_specs = [
        (0, 12), (12, 21), (21, 30), (30, 31), (31, 41), (41, 50),
        (50, 60), (60, 69), (69, 79), (79, 88), (88, 98), (98, 108),
        (108, 121), (121, 130), (130, 140), (140, 150), (150, 160),
        (160, 161), (161, 172), (172, 183), (183, 193)
    ]
    col_names = [
        'Name','zcmb_str','zhel_str','e_z_str','mb_str','e_mb_str','x1_str','e_x1_str',
        'c_str','e_c_str','logMst_str','e_logMst_str','tmax_str','e_tmax_str',
        'cov_mb_x1_str','cov_mb_c_str','cov_x1_c_str','set_str','RAdeg_str','DEdeg_str',
        'bias_str'
    ]
    try:
        df = pd.read_fwf(filepath, colspecs=col_specs, names=col_names, dtype=str, comment="#")
    except Exception as e:
        logger.error(f"Error reading JLA data file: {e}")
        return None

    parsed = pd.DataFrame()
    parsed['Name'] = df['Name'].str.strip()
    for new_col, old in {'zcmb':'zcmb_str','mb':'mb_str','e_mb':'e_mb_str','x1':'x1_str','c':'c_str'}.items():
        parsed[new_col] = pd.to_numeric(df[old], errors='coerce')

    if parsed[['mb','x1','c']].isnull().any().any():
        logger.error("JLA data missing mb, x1 or c values")
        return None

    parsed['mu_obs'] = (
        parsed['mb'] - salt2_m_abs_fixed +
        salt2_alpha_fixed * parsed['x1'] - salt2_beta_fixed * parsed['c']
    )
    parsed['e_mu_obs'] = pd.to_numeric(df['e_mb_str'], errors='coerce')

    essential_cols = ['Name','zcmb','mu_obs','e_mu_obs']
    parsed = parsed[essential_cols].dropna().copy()
    if parsed.empty:
        logger.error("No valid SNe after parsing JLA data")
        return None

    # --- covariance matrix ---
    try:
        cov_params = fits.getdata(covpath)
    except Exception as e:
        logger.error(f"Failed reading covariance matrix: {e}")
        cov_params = None

    covariance_matrix_inv = None
    diag_errors_for_plot = parsed['e_mu_obs'].values
    if cov_params is not None:
        try:
            n_sne = len(parsed)
            A = np.zeros((n_sne, cov_params.shape[0]))
            for i in range(n_sne):
                idx = 3 * i
                A[i, idx] = 1.0
                A[i, idx + 1] = salt2_alpha_fixed
                A[i, idx + 2] = -salt2_beta_fixed
            mu_cov = A @ cov_params @ A.T
            diag_errors_for_plot = np.sqrt(np.diag(mu_cov))
            covariance_matrix_inv = np.linalg.inv(mu_cov)
        except Exception as e:
            logger.warning(f"Could not process JLA covariance matrix: {e}")
            covariance_matrix_inv = None

    sort_idx = np.argsort(parsed['zcmb'].values)
    parsed = parsed.iloc[sort_idx].reset_index(drop=True)
    diag_errors_for_plot = diag_errors_for_plot[sort_idx]
    if covariance_matrix_inv is not None:
        covariance_matrix_inv = covariance_matrix_inv[sort_idx][:, sort_idx]

    parsed.attrs['covariance_matrix_inv'] = covariance_matrix_inv
    parsed.attrs['diag_errors_for_plot'] = diag_errors_for_plot
    parsed.attrs['salt2_m_abs_fixed'] = salt2_m_abs_fixed
    parsed.attrs['salt2_alpha_fixed'] = salt2_alpha_fixed
    parsed.attrs['salt2_beta_fixed'] = salt2_beta_fixed
    parsed.attrs['dataset_long_name'] = META.get('dataset_name', 'JLA2014')
    parsed.attrs['dataset_name_attr'] = parsed.attrs['dataset_long_name'].replace(' ', '_')
    parsed.attrs['citation'] = META.get('citation', '')
    parsed.attrs['notes'] = META.get('notes', '')
    parsed.attrs['description'] = META.get('description', '')
    return parsed
