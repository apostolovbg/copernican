"""Parse BOSS DR12 consensus BAO measurements.

The public BOSS DR12 release tabulates the spherically averaged distance
``D_V/rs`` and the Alcock--Paczyński parameter ``F_AP`` for three redshift
bins.  This parser converts those quantities into ``D_M/rs``, ``D_H/rs`` and
``D_V/rs`` so they can be consumed by the engine's generic BAO routines.
The accompanying covariance matrix is propagated through the variable
transformation and the inverse covariance is attached to the resulting
:class:`pandas.DataFrame`.
"""

import os
import logging
import numpy as np
import pandas as pd

from copernican_lib.data_loaders import register_bao_parser
from copernican_lib.utils import load_metadata_from_dir

DATA_DIR = os.path.dirname(__file__)
META = load_metadata_from_dir(DATA_DIR)

DATASET_NAME = META.get("dataset_name", "BOSS DR12 BAO Consensus")
DESCRIPTION = META.get(
    "description",
    "BOSS DR12 consensus BAO distances with full covariance.",
)


@register_bao_parser(DATASET_NAME, DESCRIPTION, data_dir=DATA_DIR)
def parse_boss_dr12(data_dir, **kwargs):
    """Return BAO observables and covariance from the BOSS DR12 release."""
    logger = logging.getLogger()

    results_path = os.path.join(data_dir, "BAO_consensus_results_dV_FAP.txt")
    cov_path = os.path.join(data_dir, "BAO_consensus_covtot_dV_FAP.txt")
    try:
        table = pd.read_csv(
            results_path,
            comment="#",
            sep=r"\s+",
            header=None,
            names=["z", "label", "value"],
        )
    except Exception as exc:
        logger.error(f"Failed reading BOSS DR12 results file: {exc}")
        return None

    # Extract ``D_V/rs`` and ``F_AP`` pairs while preserving redshift order.
    redshifts = []
    dv_vals = []
    fap_vals = []
    for _, row in table.iterrows():
        if isinstance(row["label"], str) and row["label"].startswith("dV/rs"):
            redshifts.append(float(row["z"]))
            dv_vals.append(float(row["value"]))
        elif isinstance(row["label"], str) and row["label"].startswith("F_AP"):
            fap_vals.append(float(row["value"]))

    if not (len(redshifts) == len(dv_vals) == len(fap_vals)):
        logger.error("BOSS DR12 results file is malformed.")
        return None

    try:
        cov_x = np.loadtxt(cov_path)
    except Exception as exc:
        logger.error(f"Failed reading BOSS DR12 covariance: {exc}")
        return None

    if cov_x.shape != (6, 6):
        logger.error("Unexpected BOSS DR12 covariance matrix size.")
        return None

    # Vector ordering in the files: [DV1, FAP1, DV2, FAP2, DV3, FAP3]
    x_vec = np.empty(6)
    x_vec[0::2] = dv_vals
    x_vec[1::2] = fap_vals

    n_z = len(redshifts)
    y_vec = np.empty(n_z * 3)
    jac = np.zeros((n_z * 3, 6))

    for i in range(n_z):
        dv = x_vec[2 * i]
        fap = x_vec[2 * i + 1]

        # Recover the transverse and radial distances.  The relationships
        # follow from :math:`D_V^3 = D_M^2 D_H` and ``F_AP = D_M / D_H``.
        dm = dv * fap ** (1.0 / 3.0)
        dh = dv / fap ** (2.0 / 3.0)
        y_vec[3 * i : 3 * i + 3] = [dm, dh, dv]

        # Jacobian for covariance propagation.  Each block maps the original
        # ``[DV, F_AP]`` pair to ``[DM, DH, DV]`` at the same redshift.
        jac[3 * i, 2 * i] = fap ** (1.0 / 3.0)
        jac[3 * i, 2 * i + 1] = (1.0 / 3.0) * dv * fap ** (-2.0 / 3.0)
        jac[3 * i + 1, 2 * i] = fap ** (-2.0 / 3.0)
        jac[3 * i + 1, 2 * i + 1] = (-2.0 / 3.0) * dv * fap ** (-5.0 / 3.0)
        jac[3 * i + 2, 2 * i] = 1.0
        jac[3 * i + 2, 2 * i + 1] = 0.0

    # Propagate covariance and attempt to invert
    cov_y = jac @ cov_x @ jac.T
    diag = np.sqrt(np.diag(cov_y))
    try:
        cond = np.linalg.cond(cov_y)
        cov_inv = np.linalg.inv(cov_y) if np.isfinite(cond) and cond < 1e12 else None
    except Exception as exc:
        logger.warning(f"BOSS DR12 covariance inversion failed: {exc}")
        cov_inv = None

    records = []
    for i, z in enumerate(redshifts):
        dm = y_vec[3 * i]
        dh = y_vec[3 * i + 1]
        dv = y_vec[3 * i + 2]
        records.append(
            {
                "name": f"BOSS DR12 z={z} (DM/rs)",
                "redshift": z,
                "observable_type": "DM_over_rs",
                "value": dm,
                "error": diag[3 * i],
            }
        )
        records.append(
            {
                "name": f"BOSS DR12 z={z} (DH/rs)",
                "redshift": z,
                "observable_type": "DH_over_rs",
                "value": dh,
                "error": diag[3 * i + 1],
            }
        )
        records.append(
            {
                "name": f"BOSS DR12 z={z} (DV/rs)",
                "redshift": z,
                "observable_type": "DV_over_rs",
                "value": dv,
                "error": diag[3 * i + 2],
            }
        )

    df = pd.DataFrame.from_records(records)
    df.sort_values('redshift', inplace=True, ignore_index=True)
    df.attrs["covariance_matrix_inv"] = cov_inv
    df.attrs["diag_errors_for_plot"] = diag
    df.attrs["dataset_long_name"] = META.get("dataset_name", DATASET_NAME)
    df.attrs["dataset_name_attr"] = df.attrs["dataset_long_name"].replace(" ", "_")
    df.attrs["citation"] = META.get("citation", "")
    df.attrs["notes"] = META.get("notes", "")
    df.attrs["description"] = META.get("description", "")
    return df
