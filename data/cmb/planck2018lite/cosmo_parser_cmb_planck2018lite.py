"""Planck 2018 lite CMB parser."""

import os
import logging
import numpy as np
import pandas as pd

from scripts.data_loaders import register_cmb_parser


@register_cmb_parser(
    "planck2018lite_v1",
    "Planck 2018 lite TT spectrum.",
    data_dir=os.path.dirname(__file__),
)
def parse_planck2018lite(data_dir, **kwargs):
    """Parse Planck 2018 lite power spectrum and covariance."""

    logger = logging.getLogger()
    cl_path = os.path.join(data_dir, "cl_cmb_plik_v22.dat")
    cov_path = os.path.join(data_dir, "c_matrix_plik_v22.dat")

    try:
        df = pd.read_csv(
            cl_path,
            sep=r"\s+",
            header=None,
            usecols=[0, 1],
            names=["ell", "Dl_obs"],
        )
        n = len(df)
        cov_arr = np.fromfile(
            cov_path, dtype=np.float64, offset=4, count=n * n
        )
        if cov_arr.size != n * n:
            logger.error(
                f"Covariance matrix size mismatch: expected {n*n} values, got {cov_arr.size}"
            )
            return None
        cov_matrix = cov_arr.reshape(n, n)
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            logger.error("Planck2018lite covariance matrix is singular.")
            return None

        df.attrs["covariance_matrix_inv"] = cov_inv
        df.attrs["dataset_name_attr"] = "CMB_Planck2018lite"
        df.attrs["is_cmb"] = True
        return df
    except Exception as e:
        logger.error(f"Error parsing Planck2018lite data: {e}", exc_info=True)
        return None
