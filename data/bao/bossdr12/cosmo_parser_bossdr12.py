# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

r"""Parse BOSS DR12 consensus BAO measurements.

The public BOSS DR12 release provides two equivalent sets of BAO-only
observables for three effective redshift bins. One file lists the transverse
distance ``dM(rsfid/rs)`` and Hubble parameter ``Hz(rs/rsfid)`` while the
other tabulates the spherically averaged distance ``D_V/rs`` and the
Alcock--Paczyński parameter ``F_AP``. Each pair comes with a published 6x6
covariance matrix. This parser combines the four inputs to produce
``D_M/rs``, ``D_H/rs`` and ``D_V/rs`` with full error propagation.

The ``dM`` and ``Hz`` measurements are first converted to ``D_M/rs`` and
``D_H/rs`` using the BOSS fiducial sound horizon. The ``D_V`` values are
read directly from the second results file while ``F_AP`` is used only to
propagate the covariance. The two covariance matrices are assembled into a
block-diagonal matrix, propagated through the transformation and the inverse
covariance is attached to the resulting :class:`pandas.DataFrame`.
"""

import logging
import os

import numpy as np
import pandas as pd

from copernican_lib.dataset_registry import register_bao_parser

# Speed of light in km/s used to convert ``H(z)`` into ``D_H``.
C_LIGHT = 299792.458

# Fiducial sound horizon (in Mpc) assumed by the BOSS DR12 analysis. The
# ``dM`` and ``Hz`` measurements are given relative to this value which makes
# it possible to compute ``D_M/rs`` and ``D_H/rs`` without knowing the true
# sound horizon of the cosmological model under test.
RS_FIDUCIAL_MPC = 147.78

DATA_DIR = os.path.dirname(__file__)


@register_bao_parser(data_dir=DATA_DIR)
def parse_boss_dr12(data_dir, **kwargs):
    """Return BAO observables and covariance from the BOSS DR12 release."""

    logger = logging.getLogger()
    logger.info("Loading BOSS DR12 data from %s", data_dir)

    # ------------------------------------------------------------------
    # Load ``dM(rsfid/rs)`` and ``Hz(rs/rsfid)`` results.
    # ------------------------------------------------------------------
    dm_results = os.path.join(data_dir, "BAO_consensus_results_dM_Hz.txt")
    dm_cov_path = os.path.join(data_dir, "BAO_consensus_covtot_dM_Hz.txt")
    try:
        table_dm = pd.read_csv(
            dm_results,
            comment="#",
            sep=r"\s+",
            header=None,
            names=["z", "label", "value"],
        )
    except Exception as exc:
        logger.error(f"Failed reading BOSS DR12 dM/Hz results file: {exc}")
        return None

    redshifts = []
    dm_obs = []  # ``dM(rsfid/rs)`` in Mpc
    hz_obs = []  # ``Hz(rs/rsfid)`` in km/s/Mpc
    for _, row in table_dm.iterrows():
        label = row["label"]
        if isinstance(label, str) and label.startswith("dM"):
            redshifts.append(float(row["z"]))
            dm_obs.append(float(row["value"]))
        elif isinstance(label, str) and label.startswith("Hz"):
            hz_obs.append(float(row["value"]))

    if not (len(redshifts) == len(dm_obs) == len(hz_obs)):
        logger.error("BOSS DR12 dM/Hz results file is malformed.")
        return None

    try:
        cov_dm = np.loadtxt(dm_cov_path)
    except Exception as exc:
        logger.error(f"Failed reading BOSS DR12 dM/Hz covariance: {exc}")
        return None

    if cov_dm.shape != (6, 6):
        logger.error("Unexpected BOSS DR12 dM/Hz covariance matrix size.")
        return None

    # ------------------------------------------------------------------
    # Load ``D_V/rs`` and ``F_AP`` results used to describe the spherically
    # averaged distance. ``F_AP`` is required only for covariance propagation
    # since the final observables contain ``D_V/rs`` directly.
    # ------------------------------------------------------------------
    dv_results = os.path.join(data_dir, "BAO_consensus_results_dV_FAP.txt")
    dv_cov_path = os.path.join(data_dir, "BAO_consensus_covtot_dV_FAP.txt")
    try:
        table_dv = pd.read_csv(
            dv_results,
            comment="#",
            sep=r"\s+",
            header=None,
            names=["z", "label", "value"],
        )
    except Exception as exc:
        logger.error(f"Failed reading BOSS DR12 D_V/F_AP results file: {exc}")
        return None

    dv_obs = []  # ``D_V/rs``
    fap_obs = []  # ``F_AP``
    redshifts_dv = []
    for _, row in table_dv.iterrows():
        label = row["label"]
        if isinstance(label, str) and label.startswith("dV/rs"):
            redshifts_dv.append(float(row["z"]))
            dv_obs.append(float(row["value"]))
        elif isinstance(label, str) and label.startswith("F_AP"):
            fap_obs.append(float(row["value"]))

    if not (len(redshifts_dv) == len(dv_obs) == len(fap_obs)):
        logger.error("BOSS DR12 D_V/F_AP results file is malformed.")
        return None

    if redshifts_dv != redshifts:
        logger.error("BOSS DR12 redshift mismatch between results files.")
        return None

    try:
        cov_dv = np.loadtxt(dv_cov_path)
    except Exception as exc:
        logger.error(f"Failed reading BOSS DR12 D_V/F_AP covariance: {exc}")
        return None

    if cov_dv.shape != (6, 6):
        logger.error("Unexpected BOSS DR12 D_V/F_AP covariance matrix size.")
        return None

    # ------------------------------------------------------------------
    # Convert ``dM`` and ``Hz`` to ``D_M/rs`` and ``D_H/rs``. The Jacobian of
    # this transformation propagates the published covariance matrix.
    # ------------------------------------------------------------------
    n_z = len(redshifts)
    dm_dh_vec = np.empty(n_z * 2)
    jac_dm = np.zeros((n_z * 2, n_z * 2))

    for i in range(n_z):
        dm_val = dm_obs[i]
        h_val = hz_obs[i]

        dm_over_rs = dm_val / RS_FIDUCIAL_MPC
        dh_over_rs = C_LIGHT / (h_val * RS_FIDUCIAL_MPC)

        dm_dh_vec[2 * i : 2 * i + 2] = [dm_over_rs, dh_over_rs]  # noqa: E203

        jac_dm[2 * i, 2 * i] = 1.0 / RS_FIDUCIAL_MPC
        jac_dm[2 * i + 1, 2 * i + 1] = -C_LIGHT / (h_val**2 * RS_FIDUCIAL_MPC)

    cov_dm_dh = jac_dm @ cov_dm @ jac_dm.T

    # ------------------------------------------------------------------
    # Extract ``D_V/rs`` uncertainties. Only the rows/columns corresponding to
    # ``D_V/rs`` are required because the final observables do not use ``F_AP``
    # explicitly. However the off-diagonal correlations between different
    # ``D_V/rs`` entries are preserved.
    # ------------------------------------------------------------------
    dv_vec = np.array(dv_obs)
    jac_dv = np.zeros((n_z, n_z * 2))
    for i in range(n_z):
        jac_dv[i, 2 * i] = 1.0  # derivative of D_V/rs w.r.t. itself
    cov_dv_only = jac_dv @ cov_dv @ jac_dv.T

    # ------------------------------------------------------------------
    # Assemble the final data vector and combined covariance matrix. The
    # structure of ``y_vec`` is [DM1, DH1, DV1, DM2, DH2, DV2, DM3, DH3, DV3].
    # ``cov_y`` is constructed block-diagonally because cross-covariances
    # between the dM/Hz and D_V/F_AP measurements are not provided in the
    # public release. Treating them as uncorrelated is a common approximation
    # when combining these consensus results.
    # ------------------------------------------------------------------
    y_vec = np.empty(n_z * 3)
    y_vec[0::3] = dm_dh_vec[0::2]
    y_vec[1::3] = dm_dh_vec[1::2]
    y_vec[2::3] = dv_vec

    cov_y = np.zeros((n_z * 3, n_z * 3))

    idx_dm_dh = []
    for i in range(n_z):
        idx_dm_dh.extend([3 * i, 3 * i + 1])
    cov_y[np.ix_(idx_dm_dh, idx_dm_dh)] = cov_dm_dh

    idx_dv = [3 * i + 2 for i in range(n_z)]
    cov_y[np.ix_(idx_dv, idx_dv)] = cov_dv_only

    # Attempt to invert the covariance for chi-squared calculations.
    diag = np.sqrt(np.diag(cov_y))
    try:
        cond = np.linalg.cond(cov_y)
        if np.isfinite(cond) and cond < 1e12:
            cov_inv = np.linalg.inv(cov_y)
        else:
            cov_inv = None
    except Exception as exc:
        logger.warning(f"BOSS DR12 covariance inversion failed: {exc}")
        cov_inv = None

    # ------------------------------------------------------------------
    # Build the final DataFrame expected by the engine.
    # ------------------------------------------------------------------
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
    df.sort_values("redshift", inplace=True, ignore_index=True)
    df.attrs["covariance_matrix_inv"] = cov_inv
    df.attrs["diag_errors_for_plot"] = diag
    # Metadata such as dataset name and citation is attached by
    # ``load_bao_data`` after this function returns.
    return df
