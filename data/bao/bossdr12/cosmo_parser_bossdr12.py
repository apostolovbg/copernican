r"""Parse BOSS DR12 consensus BAO measurements.

The SDSS-III BOSS DR12 release tabulates the transverse comoving distance
``dM(rsfid/rs)`` in megaparsecs and the scaled Hubble parameter
``Hz(rs/rsfid)`` for three effective redshifts.  This parser converts those
quantities into the dimensionless observables ``DM_over_rs``, ``DH_over_rs`` and
``DV_over_rs`` expected by the engine.  The published covariance matrix is
propagated through the variable transformation so downstream code receives
uncertainties in the correct space.
"""

from __future__ import annotations

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

# Fiducial sound horizon used by the BOSS DR12 analysis (Alam et al. 2017)
RS_FIDUCIAL_MPC = 147.78
# Speed of light in km/s for converting ``H(z)`` to ``D_H``
C_LIGHT_KMS = 299792.458


@register_bao_parser(DATASET_NAME, DESCRIPTION, data_dir=DATA_DIR)
def parse_boss_dr12(data_dir: str, **kwargs) -> pd.DataFrame | None:
    """Return BAO observables and covariance from the BOSS DR12 release."""

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
    except Exception as exc:  # pragma: no cover - I/O failure
        logger.error(f"Failed reading BOSS DR12 results file: {exc}")
        return None

    # Extract ``dM(rsfid/rs)`` and ``Hz(rs/rsfid)`` pairs while preserving
    # redshift order.
    redshifts: list[float] = []
    dm_fid: list[float] = []
    hz_scaled: list[float] = []
    for _, row in table.iterrows():
        label = str(row["label"])
        if label.startswith("dM"):
            redshifts.append(float(row["z"]))
            dm_fid.append(float(row["value"]))
        elif label.startswith("Hz"):
            hz_scaled.append(float(row["value"]))

    if not (len(redshifts) == len(dm_fid) == len(hz_scaled)):
        logger.error("BOSS DR12 results file is malformed.")
        return None

    try:
        cov_x = np.loadtxt(cov_path)
    except Exception as exc:  # pragma: no cover - I/O failure
        logger.error(f"Failed reading BOSS DR12 covariance: {exc}")
        return None

    if cov_x.shape != (6, 6):
        logger.error("Unexpected BOSS DR12 covariance matrix size.")
        return None

    n_z = len(redshifts)
    y_vec = np.empty(n_z * 3)
    jac = np.zeros((n_z * 3, 6))

    for i in range(n_z):
        dm = dm_fid[i]  # ``D_M * (r_s^fid / r_s)`` in Mpc
        hz = hz_scaled[i]  # ``H(z) * (r_s / r_s^fid)`` in km/s/Mpc

        dm_over_rs = dm / RS_FIDUCIAL_MPC
        dh_over_rs = C_LIGHT_KMS / (hz * RS_FIDUCIAL_MPC)
        dv_over_rs = (dm_over_rs ** 2 * dh_over_rs) ** (1.0 / 3.0)

        y_vec[3 * i : 3 * i + 3] = [dm_over_rs, dh_over_rs, dv_over_rs]

        # Jacobian for covariance propagation.  Each 3x2 block maps the
        # original ``[dM, Hz]`` pair to ``[DM_over_rs, DH_over_rs, DV_over_rs]``
        # at the same redshift.  Analytic derivatives are used for stability.
        jac[3 * i, 2 * i] = 1.0 / RS_FIDUCIAL_MPC
        jac[3 * i + 1, 2 * i + 1] = -C_LIGHT_KMS / (hz ** 2 * RS_FIDUCIAL_MPC)
        jac[3 * i + 2, 2 * i] = (
            (2.0 / 3.0) * (dv_over_rs / dm_over_rs) / RS_FIDUCIAL_MPC
        )
        jac[3 * i + 2, 2 * i + 1] = -(
            (1.0 / 3.0) * dv_over_rs / hz
        )

    # Propagate covariance into the transformed space and try to invert.
    cov_y = jac @ cov_x @ jac.T
    diag = np.sqrt(np.diag(cov_y))
    try:
        cov_inv = np.linalg.inv(cov_y)
    except np.linalg.LinAlgError:
        logger.warning("BOSS DR12 covariance inversion failed; proceeding without inverse")
        cov_inv = None

    records: list[dict[str, float]] = []
    for i, z in enumerate(redshifts):
        dm_over_rs = y_vec[3 * i]
        dh_over_rs = y_vec[3 * i + 1]
        dv_over_rs = y_vec[3 * i + 2]
        records.append(
            {
                "name": f"BOSS DR12 z={z} (DM/rs)",
                "redshift": z,
                "observable_type": "DM_over_rs",
                "value": dm_over_rs,
                "error": diag[3 * i],
            }
        )
        records.append(
            {
                "name": f"BOSS DR12 z={z} (DH/rs)",
                "redshift": z,
                "observable_type": "DH_over_rs",
                "value": dh_over_rs,
                "error": diag[3 * i + 1],
            }
        )
        records.append(
            {
                "name": f"BOSS DR12 z={z} (DV/rs)",
                "redshift": z,
                "observable_type": "DV_over_rs",
                "value": dv_over_rs,
                "error": diag[3 * i + 2],
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

