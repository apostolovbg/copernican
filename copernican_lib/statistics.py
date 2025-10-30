"""Shared statistical helpers for cosmological engines.

**Last Updated:** 2025-10-30

This module now delegates dataset-specific likelihood calculations to
:mod:`copernican_lib.likelihoods`.  The thin wrappers exposed here preserve the
public API while the new package stores the covariance-aware implementations
used by all engines.  CAMB spectrum helpers remain available for backward
compatibility.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np

from copernican_lib import engine_interface
from copernican_lib.likelihoods import (
    BAOLike,
    CMBLike,
    SNeLike,
    compute_cmb_spectrum,
    compute_cmb_spectrum_cached,
    compute_cmb_spectrum_from_dict,
)

__all__ = [
    "calculate_bao_observables",
    "chi_squared_bao",
    "chi_squared_cmb",
    "chi_squared_sne",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_from_dict",
]


def chi_squared_sne(
    cosmo_params: Sequence[float],
    mu_model_func,
    sne_data_df,
) -> float:
    """Return the χ² value for Supernovae Ia data."""

    like = SNeLike(mu_model_func, sne_data_df)
    loglike = like.loglike(cosmo_params)
    chi2 = float(like.state.get("chi2", float("inf")))
    if not np.isfinite(loglike):
        return float("inf")
    return chi2 if np.isfinite(chi2) else float("inf")


def chi_squared_bao(
    z: np.ndarray,
    obs_type: np.ndarray,
    obs_val: np.ndarray,
    obs_err: np.ndarray,
    model_plugin,
    cosmo_params: Sequence[float],
    model_rs_Mpc: float,
    *,
    covariance_matrix_inv=None,
) -> float:
    """Return the χ² value for BAO observations."""

    like = BAOLike(
        z=z,
        obs_type=obs_type,
        obs_val=obs_val,
        obs_err=obs_err,
        model_plugin=model_plugin,
        covariance_matrix_inv=covariance_matrix_inv,
        rs_override=model_rs_Mpc,
    )
    loglike = like.loglike(cosmo_params)
    chi2 = float(like.state.get("chi2", float("inf")))
    if not np.isfinite(loglike):
        return float("inf")
    return chi2 if np.isfinite(chi2) else float("inf")


def chi_squared_cmb(
    cosmo_params: Sequence[float],
    cmb_data_df,
    plugin,
    extra_params: Mapping[str, float] | None = None,
) -> float:
    """Return the χ² value for CMB spectra."""

    like = CMBLike(cmb_data_df, plugin, extra_params=extra_params or {})
    loglike = like.loglike(cosmo_params)
    chi2 = float(like.state.get("chi2", float("inf")))
    if not np.isfinite(loglike):
        return float("inf")
    return chi2 if np.isfinite(chi2) else float("inf")


def calculate_bao_observables(
    bao_data_df,
    model_plugin,
    cosmo_params: Sequence[float],
    *,
    z_smooth: np.ndarray | None = None,
):
    """Return BAO predictions and optional smooth curves for plotting."""

    logger = logging.getLogger()
    engine_interface.validate_plugin(model_plugin)
    model_name = model_plugin.MODEL_NAME

    bao_pred_df = bao_data_df.copy()
    bao_pred_df["model_prediction"] = np.nan
    if getattr(model_plugin, "valid_for_bao", True) is False:
        logger.warning("Model invalid for BAO; skipping.")
        return bao_pred_df, np.nan, None

    param_str = ", ".join([f"{p:.4g}" for p in cosmo_params])
    logger.info(
        "Calculating BAO observables for %s with parameters: [%s]",
        model_name,
        param_str,
    )

    try:
        model_rs_Mpc = model_plugin.get_sound_horizon_rs_Mpc(*cosmo_params)
        if not (np.isfinite(model_rs_Mpc) and model_rs_Mpc > 0):
            logger.warning(
                "Model '%s' returned invalid r_s (%.3f Mpc).",
                model_name,
                model_rs_Mpc,
            )
            return bao_pred_df, np.nan, None
    except Exception as exc:
        logger.error(
            "Failed to calculate r_s for model '%s': %s",
            model_name,
            exc,
            exc_info=True,
        )
        return bao_pred_df, np.nan, None

    logger.info(
        "Successfully calculated r_s for %s: %.3f Mpc",
        model_name,
        model_rs_Mpc,
    )

    try:
        get_DM_model = getattr(model_plugin, "get_comoving_distance_Mpc")
        get_Hz_model = getattr(model_plugin, "get_Hz_per_Mpc")
        get_DV_model_specific = getattr(model_plugin, "get_DV_Mpc", None)
        get_DA_model = getattr(
            model_plugin,
            "get_angular_diameter_distance_Mpc",
        )
        C_LIGHT = model_plugin.FIXED_PARAMS.get("C_LIGHT_KM_S", 299792.458)
    except AttributeError as exc:
        logger.error(
            "Model plugin '%s' missing required function for BAO: %s",
            model_name,
            exc,
        )
        return bao_pred_df, model_rs_Mpc, None

    for index, row in bao_pred_df.iterrows():
        z_val = row["redshift"]
        obs_type = row["observable_type"]
        model_pred_numerator = np.nan
        try:
            if obs_type == "DM_over_rs":
                model_pred_numerator = get_DM_model(z_val, *cosmo_params)
            elif obs_type == "DH_over_rs":
                hz_val = get_Hz_model(z_val, *cosmo_params)
                if np.isfinite(hz_val) and abs(hz_val) > 1e-9:
                    model_pred_numerator = C_LIGHT / hz_val
            elif obs_type == "DV_over_rs":
                if get_DV_model_specific:
                    model_pred_numerator = get_DV_model_specific(
                        z_val,
                        *cosmo_params,
                    )
                else:
                    dm_val = get_DM_model(z_val, *cosmo_params)
                    hz_val = get_Hz_model(z_val, *cosmo_params)
                    if (
                        np.isfinite(dm_val)
                        and dm_val >= 0
                        and np.isfinite(hz_val)
                        and abs(hz_val) > 1e-9
                        and z_val > 1e-9
                    ):
                        term = (dm_val**2) * C_LIGHT * z_val / hz_val
                        model_pred_numerator = (
                            term ** (1.0 / 3.0) if term >= 0 else np.nan
                        )
                    elif abs(z_val) < 1e-9:
                        model_pred_numerator = 0.0

            if np.isfinite(model_pred_numerator):
                bao_pred_df.loc[index, "model_prediction"] = (
                    model_pred_numerator / model_rs_Mpc
                )
        except Exception:
            logger.exception(
                "statistics.calculate_bao_observables: "
                "BAO prediction failed for %s at z=%s in model '%s'",
                obs_type,
                z_val,
                model_name,
            )

    smooth_predictions = None
    if z_smooth is not None:
        try:
            dm_smooth = get_DM_model(z_smooth, *cosmo_params)
            hz_smooth = get_Hz_model(z_smooth, *cosmo_params)
            dh_smooth = np.where(hz_smooth > 0, C_LIGHT / hz_smooth, np.nan)

            if get_DV_model_specific:
                dv_smooth = get_DV_model_specific(z_smooth, *cosmo_params)
            else:
                da_smooth = get_DA_model(z_smooth, *cosmo_params)
                term = (
                    np.power(1 + z_smooth, 2)
                    * np.power(da_smooth, 2)
                    * C_LIGHT
                    * z_smooth
                    / hz_smooth
                )
                dv_smooth = np.power(
                    term,
                    1 / 3,
                    where=term >= 0,
                    out=np.full_like(z_smooth, np.nan),
                )

            smooth_predictions = {
                "z": z_smooth,
                "dm_over_rs": dm_smooth / model_rs_Mpc,
                "dh_over_rs": dh_smooth / model_rs_Mpc,
                "dv_over_rs": dv_smooth / model_rs_Mpc,
            }
        except Exception as exc:
            logger.error(
                "Failed to calculate smooth BAO curves for %s: %s",
                model_name,
                exc,
                exc_info=True,
            )

    return bao_pred_df, model_rs_Mpc, smooth_predictions
