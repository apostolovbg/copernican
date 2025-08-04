"""Parser for the Pantheon+SH0ES 2022 supernova sample."""

import os
import pandas as pd
import numpy as np
import logging

from copernican_lib.data_loaders import register_sne_parser
from copernican_lib.utils import load_metadata_from_dir

DATA_DIR = os.path.dirname(__file__)
META = load_metadata_from_dir(DATA_DIR)

DATASET_NAME = META.get("dataset_name", "Pantheon+ dataset").replace("\\", "")
DESCRIPTION = META.get(
    "description",
    "Supernova distances with full covariance matrix.",
)


@register_sne_parser(
    DATASET_NAME,
    DESCRIPTION,
    data_dir=DATA_DIR,
)
def parse_pantheon_plus(data_dir, **kwargs):
    """Parse Pantheon+ data and its covariance matrix."""
    logger = logging.getLogger()
    # Dynamically resolve the dataset file names so the parser does not
    # depend on a specific release naming convention. Expect exactly one
    # ``*.dat`` file for the supernova data and one ``*.cov`` file for the
    # covariance matrix inside ``data_dir``.
    dat_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".dat")]
    cov_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".cov")]
    if not dat_files or not cov_files:
        logger.error("Pantheon+ directory must contain .dat and .cov files")
        return None
    if len(dat_files) > 1 or len(cov_files) > 1:
        logger.warning("Multiple data/covariance files found; using first match")
    filepath = os.path.join(data_dir, sorted(dat_files)[0])
    cov_filepath = os.path.join(data_dir, sorted(cov_files)[0])

    try:
        temp_df = pd.read_csv(filepath, sep=r"\s+", engine="python", comment="#")
        data_df = pd.DataFrame()
        col_map = {
            "Name": ["CID", "SNID", "ID", "NAME"],
            "zcmb": ["zCMB", "ZCMB", "zcmb"],
            "mu_obs": ["MU_SH0ES", "mu"],
            "mu_sh0es_err_diag": ["MU_SH0ES_ERR_DIAG", "e_mu_diag"],
        }

        with open(cov_filepath, "r") as f:
            N_cov = int(f.readlines()[0].strip())

        for target_col, possible_names in col_map.items():
            found_col = next((p for p in possible_names if p in temp_df.columns), None)
            if found_col:
                data_df[target_col] = temp_df[found_col]
            elif target_col not in ["Name", "mu_sh0es_err_diag"]:
                logger.error(
                    f"Column for '{target_col}' not found in Pantheon+ (mu_cov)."
                )
                return None

        if "Name" not in data_df:
            data_df["Name"] = temp_df.get(
                "CID", pd.Series([f"SN_PPlus_mucov_{i}" for i in range(len(temp_df))])
            )
        data_df["Name"] = data_df["Name"].astype(str).str.strip()

        essential_cols = ["zcmb", "mu_obs"]
        for col in essential_cols + ["mu_sh0es_err_diag"]:
            if col in data_df:
                data_df[col] = pd.to_numeric(data_df[col], errors="coerce")

        if any(
            col not in data_df.columns or data_df[col].isnull().all()
            for col in essential_cols
        ):
            logger.error(
                "One or more essential columns missing/all NaN in Pantheon+ data."
            )
            return None

        data_df = data_df.dropna(subset=essential_cols).reset_index(drop=True)
        if data_df.empty:
            logger.error("No valid Pantheon+ SNe data after filtering.")
            return None
        if len(data_df) != N_cov:
            logger.critical(
                f"SNe count for mu_cov: data ({len(data_df)}) vs cov N ({N_cov})."
            )
            return None

        cov_matrix_flat = np.loadtxt(cov_filepath, skiprows=1)
        if len(cov_matrix_flat) != N_cov * N_cov:
            logger.error(
                f"Cov matrix len ({len(cov_matrix_flat)}) != N*N ({N_cov * N_cov})."
            )
            return None
        cov_matrix_pantheon = cov_matrix_flat.reshape((N_cov, N_cov))

        if (
            "mu_sh0es_err_diag" in data_df
            and data_df["mu_sh0es_err_diag"].notna().any()
        ):
            data_df["e_mu_obs"] = data_df["mu_sh0es_err_diag"]
        else:
            data_df["e_mu_obs"] = np.sqrt(np.diag(cov_matrix_pantheon))

        sort_idx = np.argsort(data_df["zcmb"].values)
        data_df = data_df.iloc[sort_idx].reset_index(drop=True)
        cov_matrix_pantheon = cov_matrix_pantheon[sort_idx][:, sort_idx]

        output_df = data_df[["Name", "zcmb", "mu_obs", "e_mu_obs"]].copy()
        try:
            output_df.attrs["covariance_matrix_inv"] = np.linalg.inv(
                cov_matrix_pantheon
            )
            output_df.attrs["diag_errors_for_plot"] = np.sqrt(
                np.diag(cov_matrix_pantheon)
            )
        except np.linalg.LinAlgError:
            logger.warning(
                "Could not invert Pantheon+ covariance matrix. Chi2 will fallback to diagonal errors."
            )
            output_df.attrs["covariance_matrix_inv"] = None
            output_df.attrs["diag_errors_for_plot"] = output_df["e_mu_obs"].values
        long_name = META.get("dataset_name", "PantheonPlus2022").replace("\\", "")
        output_df.attrs["dataset_name"] = long_name
        output_df.attrs["dataset_name_sanitized"] = long_name.replace(" ", "_")
        output_df.attrs["citation"] = META.get("citation", "")
        output_df.attrs["notes"] = META.get("notes", "")
        output_df.attrs["description"] = META.get("description", "")
        return output_df
    except Exception as e:
        logger.error(f"Error processing Pantheon+ data: {e}", exc_info=True)
        return None
