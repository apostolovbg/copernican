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
            names=["ell", "Cl_TT", "sigma_Cl_TT"],
        )

        ell_arr = df["ell"].to_numpy()
        factor = ell_arr * (ell_arr + 1) / (2 * np.pi)
        df["Dl_TT"] = factor * df["Cl_TT"]
        df["sigma_TT"] = factor * df["sigma_Cl_TT"]
        df = df.drop(columns=["Cl_TT", "sigma_Cl_TT"])
        n = len(df)

        # The covariance matrix file is stored as a Fortran unformatted
        # binary record. The first and last 4 bytes contain the record
        # length (n*n*8). Read the data as little-endian 64-bit floats
        # and reshape to ``n x n``.
        with open(cov_path, "rb") as fh:
            header = np.fromfile(fh, dtype="<i4", count=1)[0]
            cov_arr = np.fromfile(fh, dtype="<f8", count=n * n)
            trailer = np.fromfile(fh, dtype="<i4", count=1)[0]

        if cov_arr.size != n * n or header != trailer or header != n * n * 8:
            logger.error(
                "Planck2018lite covariance matrix header mismatch or size error."
            )
            return None

        cov_matrix = cov_arr.reshape(n, n)
        factors = factor
        cov_matrix = cov_matrix * np.outer(factors, factors)
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            logger.error("Planck2018lite covariance matrix is singular.")
            return None
        df.attrs["covariance_matrix_inv_TT"] = cov_inv
        df.attrs["dataset_name_attr"] = "CMB_Planck2018lite"
        df.attrs["is_cmb"] = True
        df.attrs["param_names"] = [
            "H0",
            "ombh2",
            "omch2",
            "tau",
            "As",
            "ns",
        ]
        return df
    except Exception as e:
        logger.error(f"Error parsing Planck2018lite data: {e}", exc_info=True)
        return None
