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
        raw = pd.read_csv(cl_path, sep=r"\s+", header=None)
        col_map = {0: "ell", 1: "Cl_obs"}
        if raw.shape[1] >= 3:
            col_map[2] = "Cl_te_obs"
        if raw.shape[1] >= 4:
            col_map[3] = "Cl_ee_obs"

        raw.rename(columns=col_map, inplace=True)
        df = raw[list(col_map.values())]

        ell_arr = df["ell"].values
        df["Dl_obs"] = ell_arr * (ell_arr + 1) * df["Cl_obs"] / (2 * np.pi)
        if "Cl_te_obs" in df.columns:
            df["Dl_te_obs"] = ell_arr * (ell_arr + 1) * df["Cl_te_obs"] / (2 * np.pi)
        if "Cl_ee_obs" in df.columns:
            df["Dl_ee_obs"] = ell_arr * (ell_arr + 1) * df["Cl_ee_obs"] / (2 * np.pi)
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
        # The covariance matrix is supplied for C_ell. Scale to D_ell using
        # the same ell(ell+1)/(2pi) factors applied above.
        factors = ell_arr * (ell_arr + 1) / (2 * np.pi)
        cov_matrix = cov_matrix * np.outer(factors, factors)

        # Pre-compute diagonal errors for plotting or fallback usage
        diag_errors = np.sqrt(np.diag(cov_matrix))

        try:
            cov_inv = np.linalg.inv(cov_matrix)
            # Check for NaNs or infinities after inversion
            if not np.all(np.isfinite(cov_inv)):
                raise ValueError(
                    "Inverted Planck2018lite covariance contains non-finite values."
                )

            cond_num = np.linalg.cond(cov_matrix)
            if not np.isfinite(cond_num) or cond_num > 1e12:
                raise ValueError(
                    f"Planck2018lite covariance matrix ill-conditioned (cond={cond_num:.2e})."
                )
        except (np.linalg.LinAlgError, ValueError) as e:
            # Fall back to diagonal errors if inversion fails or matrix is bad
            logger.warning(f"{e} Falling back to diagonal errors.")
            cov_inv = None

        df.attrs["covariance_matrix_inv"] = cov_inv
        df.attrs["diag_errors_for_plot"] = diag_errors
        df.attrs["dataset_name_attr"] = "CMB_Planck2018lite"
        df.attrs["is_cmb"] = True
        # Map the order of CAMB parameters used by the engine
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
