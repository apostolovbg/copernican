# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Parser for the Union3/UNITY 1.5 compressed supernova distances."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from astropy.io import fits

from copernican_lib.dataset_registry import register_sne_parser

DATA_DIR = os.path.dirname(__file__)


def _find_matching_fits(data_dir: str) -> list[str]:
    """Return FITS files that look like the compressed `mu_mat` outputs."""

    def looks_like_mu_mat(name: str) -> bool:
        """Return True when a file name matches the mu_mat pattern."""
        lower = name.lower()
        return lower.endswith(".fits") and "mu_mat" in lower

    return [
        os.path.join(data_dir, entry)
        for entry in sorted(os.listdir(data_dir))
        if looks_like_mu_mat(entry)
    ]


def _load_mu_matrix(path: str) -> np.ndarray:
    """Open the FITS file and validate that it carries the expected layout."""

    with fits.open(path, memmap=False) as hdul:
        matrix = np.asarray(hdul[0].data, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("mu_mat FITS must be a square matrix.")
    if matrix.shape[0] < 3:
        raise ValueError(
            "mu_mat FITS must include at least one node plus headers."
        )
    return matrix


@register_sne_parser(data_dir=DATA_DIR)
def parse_union3(data_dir: str, **kwargs) -> pd.DataFrame | None:
    """Read the compressed Union3 distances and attach their covariance."""

    logger = logging.getLogger(__name__)
    fits_paths = _find_matching_fits(data_dir)
    if not fits_paths:
        logger.error(
            "Union3 directory %s does not contain a mu_mat FITS.", data_dir
        )
        return None

    path = fits_paths[0]
    try:
        matrix = _load_mu_matrix(path)
    except Exception as exc:
        logger.error(
            "Unable to interpret Union3 matrix: %s",
            exc,
            exc_info=True,
        )
        return None

    redshift = matrix[0, 1:]
    mu_values = matrix[1:, 0]
    inv_covariance = matrix[1:, 1:]

    if not (np.isfinite(redshift).all() and np.isfinite(mu_values).all()):
        logger.warning(
            "Union3 mu_mat contains non-finite redshifts or mu values."
        )

    covariance = None
    diag_errors: np.ndarray
    try:
        covariance = np.linalg.inv(inv_covariance)
        diag_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    except np.linalg.LinAlgError:
        diag_errors = np.full(redshift.shape, np.nan)
        logger.warning(
            "Union3 inverse covariance could not be inverted; "
            "diag errors set to NaN."
        )

    record_names = [f"Union3_bin_{idx + 1}" for idx in range(redshift.size)]
    distance_df = pd.DataFrame(
        {
            "Name": record_names,
            "zcmb": redshift,
            "mu_obs": mu_values,
            "e_mu_obs": diag_errors,
        }
    )

    distance_df.attrs["covariance_matrix_inv"] = inv_covariance
    distance_df.attrs["covariance_matrix"] = covariance
    distance_df.attrs["diag_errors_for_plot"] = diag_errors
    distance_df.attrs["redshift_nodes"] = redshift
    distance_df.attrs["mu_matrix_path"] = path
    return distance_df
