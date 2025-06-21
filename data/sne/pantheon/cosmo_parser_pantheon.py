
import os
import pandas as pd
import numpy as np
import logging

from scripts.data_loaders import register_sne_parser


@register_sne_parser(
    "Pantheon+ dataset",
    "with full covariance, but LCDM-oriented calibration",
    data_dir=os.path.dirname(__file__),
)
def parse_pantheon_plus_mu_cov_h2(data_dir, **kwargs):
    logger = logging.getLogger()
    filepath = os.path.join(data_dir, "Pan.dat")
    cov_filepath = os.path.join(data_dir, "Pancm.cov")

    try:
        temp_df = pd.read_csv(filepath, delim_whitespace=True, comment='#')
        data_df = pd.DataFrame()
        col_map = {
            'Name': ['CID','SNID','ID','NAME'],
            'zcmb': ['zCMB','ZCMB','zcmb'],
            'mu_obs': ['MU_SH0ES','mu'],
            'mu_sh0es_err_diag': ['MU_SH0ES_ERR_DIAG','e_mu_diag'],
        }

        with open(cov_filepath,'r') as f:
            N_cov = int(f.readlines()[0].strip())

        for target_col, possible_names in col_map.items():
            found_col = next((p for p in possible_names if p in temp_df.columns), None)
            if found_col:
                data_df[target_col] = temp_df[found_col]
            elif target_col not in ['Name', 'mu_sh0es_err_diag']:
                logger.error(f"Column for '{target_col}' not found in Pantheon+ (mu_cov)."); return None

        if 'Name' not in data_df:
            data_df['Name'] = temp_df.get('CID', pd.Series([f"SN_PPlus_mucov_{i}" for i in range(len(temp_df))]))
        data_df['Name'] = data_df['Name'].astype(str).str.strip()

        essential_cols = ['zcmb','mu_obs']
        for col in essential_cols + ['mu_sh0es_err_diag']:
            if col in data_df:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

        if any(col not in data_df.columns or data_df[col].isnull().all() for col in essential_cols):
            logger.error("One or more essential columns missing/all NaN in Pantheon+ mu_cov_h2 data."); return None

        data_df = data_df.dropna(subset=essential_cols).reset_index(drop=True)
        if data_df.empty:
            logger.error("No valid Pantheon+ mu_cov_h2 SNe data after filtering."); return None
        if len(data_df) != N_cov:
            logger.critical(f"SNe count for mu_cov: data ({len(data_df)}) vs cov N ({N_cov})."); return None

        cov_matrix_flat = np.loadtxt(cov_filepath, skiprows=1)
        if len(cov_matrix_flat) != N_cov*N_cov:
            logger.error(f"Cov matrix len ({len(cov_matrix_flat)}) != N*N ({N_cov*N_cov})."); return None
        cov_matrix_pantheon = cov_matrix_flat.reshape((N_cov,N_cov))

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
