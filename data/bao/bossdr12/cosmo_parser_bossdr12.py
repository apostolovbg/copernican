r"""Parse BOSS DR12 consensus BAO measurements.

The public BOSS DR12 release tabulates transverse distances
``D_M(r_s^{\rm fid}/r_s)`` and Hubble parameters
``H(z)(r_s/r_s^{\rm fid})`` for three redshift bins. This parser converts
those quantities into the dimensionless observables ``D_M/r_s`` and
``D_H/r_s`` (with ``D_H = c / H``) so they can be consumed by the
engine's generic BAO routines. The published covariance matrix is
propagated through this variable transformation and its inverse is
attached to the returned :class:`pandas.DataFrame`.

``D_V`` is a deterministic combination of ``D_M`` and ``D_H`` and would
render the covariance matrix singular if included explicitly. The parser
therefore omits ``D_V`` rows entirely while still allowing the engine to
reconstruct ``D_V`` from ``D_M`` and ``D_H`` during evaluation.
"""

import logging
import os

import numpy as np
import pandas as pd

from copernican_lib.data_loaders import register_bao_parser
from copernican_lib.utils import load_metadata_from_dir


# Directory containing the raw BOSS DR12 files and accompanying metadata.
DATA_DIR = os.path.dirname(__file__)
META = load_metadata_from_dir(DATA_DIR)


# Reusable constants from the public release.  The sound horizon of the
# fiducial cosmology is required to convert tabulated numbers to the
# dimensionless form used internally by the suite.
RS_FIDUCIAL_MPC = 147.78
C_LIGHT = 299_792.458  # km/s


DATASET_NAME = META.get("dataset_name", "BOSS DR12 BAO Consensus")
DESCRIPTION = META.get(
    "description",
    "BOSS DR12 consensus BAO distances with full covariance.",
)


@register_bao_parser(DATASET_NAME, DESCRIPTION, data_dir=DATA_DIR)
def parse_boss_dr12(data_dir, **kwargs):
    """Return BAO observables and covariance from the BOSS DR12 release.

    Parameters
    ----------
    data_dir:
        Directory that holds the published BOSS DR12 tables and
        covariance matrices. This function assumes that
        ``BAO_consensus_results_dM_Hz.txt`` and
        ``BAO_consensus_covtot_dM_Hz.txt`` live inside that directory.

    Returns
    -------
    pandas.DataFrame | None
        DataFrame with ``DM_over_rs`` and ``DH_over_rs`` rows. The inverse
        covariance matrix is attached to ``df.attrs['covariance_matrix_inv']``.
        ``None`` is returned when any of the input files cannot be loaded
        or when the covariance matrix fails to invert.
    """

    logger = logging.getLogger()

    results_path = os.path.join(data_dir, "BAO_consensus_results_dM_Hz.txt")
    cov_path = os.path.join(data_dir, "BAO_consensus_covtot_dM_Hz.txt")

    try:
        table = pd.read_csv(
            results_path,
            comment="#",
            sep=r"\s+",
            header=None,
            names=["z", "label", "value"],
        )
    except Exception as exc:  # pragma: no cover - file errors are logged
        logger.error(f"Failed reading BOSS DR12 results file: {exc}")
        return None

    # Extract ``dM`` and ``Hz`` pairs while preserving the redshift order.
    redshifts: list[float] = []
    dm_vals: list[float] = []
    hz_vals: list[float] = []
    for _, row in table.iterrows():
        label = row["label"]
        if isinstance(label, str) and label.startswith("dM"):
            redshifts.append(float(row["z"]))
            dm_vals.append(float(row["value"]))
        elif isinstance(label, str) and label.startswith("Hz"):
            hz_vals.append(float(row["value"]))

    if not (len(redshifts) == len(dm_vals) == len(hz_vals)):
        logger.error("BOSS DR12 results file is malformed.")
        return None

    try:
        cov_x = np.loadtxt(cov_path)
    except Exception as exc:  # pragma: no cover
        logger.error(f"Failed reading BOSS DR12 covariance: {exc}")
        return None

    if cov_x.shape != (6, 6):
        logger.error("Unexpected BOSS DR12 covariance matrix size.")
        return None

    # Vector ordering in the files: [dM1, Hz1, dM2, Hz2, dM3, Hz3]
    x_vec = np.empty(6)
    x_vec[0::2] = dm_vals
    x_vec[1::2] = hz_vals

    n_z = len(redshifts)
    y_vec = np.empty(n_z * 2)
    jac = np.zeros((n_z * 2, 6))

    # Convert the published values into DM/rs and DH/rs and build the
    # Jacobian for covariance propagation. The transformation is linear
    # in ``dM`` and nonlinear in ``Hz`` owing to ``D_H = c / H``.
    for i in range(n_z):
        dM_file = x_vec[2 * i]
        Hz_file = x_vec[2 * i + 1]

        dm_over_rs = dM_file / RS_FIDUCIAL_MPC
        dh_over_rs = C_LIGHT / (Hz_file * RS_FIDUCIAL_MPC)

        y_vec[2 * i : 2 * i + 2] = [dm_over_rs, dh_over_rs]

        jac[2 * i, 2 * i] = 1.0 / RS_FIDUCIAL_MPC
        jac[2 * i + 1, 2 * i + 1] = -C_LIGHT / (Hz_file ** 2 * RS_FIDUCIAL_MPC)

    # Propagate covariance and attempt to invert it. The transformation
    # yields a well-conditioned 6x6 matrix because only the independent
    # ``D_M`` and ``D_H`` measurements are kept.
    cov_y = jac @ cov_x @ jac.T
    diag = np.sqrt(np.diag(cov_y))
    try:
        cond = np.linalg.cond(cov_y)
        cov_inv = np.linalg.inv(cov_y) if np.isfinite(cond) and cond < 1e12 else None
    except Exception as exc:  # pragma: no cover
        logger.warning(f"BOSS DR12 covariance inversion failed: {exc}")
        cov_inv = None

    records = []
    for i, z in enumerate(redshifts):
        dm = y_vec[2 * i]
        dh = y_vec[2 * i + 1]
        records.append(
            {
                "name": f"BOSS DR12 z={z} (DM/rs)",
                "redshift": z,
                "observable_type": "DM_over_rs",
                "value": dm,
                "error": diag[2 * i],
            }
        )
        records.append(
            {
                "name": f"BOSS DR12 z={z} (DH/rs)",
                "redshift": z,
                "observable_type": "DH_over_rs",
                "value": dh,
                "error": diag[2 * i + 1],
            }
        )

    df = pd.DataFrame.from_records(records)
    df.sort_values("redshift", inplace=True, ignore_index=True)

    df.attrs["covariance_matrix_inv"] = cov_inv
    df.attrs["diag_errors_for_plot"] = diag
    df.attrs["dataset_long_name"] = META.get("dataset_name", DATASET_NAME)
    df.attrs["dataset_name_attr"] = df.attrs["dataset_long_name"].replace(" ", "_")
    df.attrs["citation"] = META.get("citation", "")
    df.attrs["notes"] = META.get("notes", "")
    df.attrs["description"] = META.get("description", "")
    return df

