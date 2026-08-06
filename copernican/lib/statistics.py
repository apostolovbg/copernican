# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Shared statistical helpers for cosmological engines.

This module delegates dataset-specific likelihood calculations to
:mod:`copernican.lib.likelihoods`. The package stores the covariance-aware
implementations used by all engines.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy

from copernican.lib import engine_adapter as engine_plugin_validation
from copernican.lib.likelihoods import (
    BAOLike,
    CMBLike,
    SNeLike,
    compute_cmb_spectrum,
    compute_cmb_spectrum_cached,
    compute_cmb_spectrum_from_contract,
)

__all__ = [
    "calculate_bao_observables",
    "chi_squared_bao",
    "chi_squared_cmb",
    "chi_squared_sne",
    "compute_cmb_spectrum",
    "compute_cmb_spectrum_cached",
    "compute_cmb_spectrum_from_contract",
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
    if not numpy.isfinite(loglike):
        return float("inf")
    return chi2 if numpy.isfinite(chi2) else float("inf")


def chi_squared_bao(
    redshifts: numpy.ndarray,
    observable_types: numpy.ndarray,
    observable_values: numpy.ndarray,
    observable_errors: numpy.ndarray,
    model_plugin,
    cosmo_params: Sequence[float],
    model_rs_Mpc: float,
    *,
    covariance_matrix_inv=None,
) -> float:
    """Return the χ² value for BAO observations."""

    like = BAOLike(
        redshifts=redshifts,
        observable_types=observable_types,
        observable_values=observable_values,
        observable_errors=observable_errors,
        model_plugin=model_plugin,
        covariance_matrix_inv=covariance_matrix_inv,
        rs_override=model_rs_Mpc,
    )
    loglike = like.loglike(cosmo_params)
    chi2 = float(like.state.get("chi2", float("inf")))
    if not numpy.isfinite(loglike):
        return float("inf")
    return chi2 if numpy.isfinite(chi2) else float("inf")


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
    if not numpy.isfinite(loglike):
        return float("inf")
    return chi2 if numpy.isfinite(chi2) else float("inf")


def calculate_bao_observables(
    bao_data_df,
    model_plugin,
    cosmo_params: Sequence[float],
    *,
    z_smooth: numpy.ndarray | None = None,
):
    """Return BAO predictions and optional smooth curves for plotting."""

    logger = logging.getLogger()
    engine_plugin_validation.validate_plugin(model_plugin)
    model_name = model_plugin.MODEL_NAME

    bao_pred_df = bao_data_df.copy()
    bao_pred_df["model_prediction"] = numpy.nan
    if getattr(model_plugin, "valid_for_bao", True) is False:
        logger.warning("Model invalid for BAO; skipping.")
        return bao_pred_df, numpy.nan, None

    param_str = ", ".join([f"{parameter:.4g}" for parameter in cosmo_params])
    logger.info(
        "Calculating BAO observables for %s with parameters: [%s]",
        model_name,
        param_str,
    )

    z_smooth_arr = None
    if z_smooth is not None:
        z_smooth_arr = numpy.asarray(z_smooth, dtype=float)
        if z_smooth_arr.size == 0:
            z_smooth_arr = None

    def _fill_from_plugin(
        rs_guess: float,
    ) -> tuple[float, dict[str, numpy.ndarray] | None]:
        """Return plugin-supplied BAO predictions when available."""
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
            return float("nan"), None

        rs_value = rs_guess
        if not (numpy.isfinite(rs_value) and rs_value > 0):
            try:
                rs_value = float(
                    model_plugin.get_sound_horizon_rs_Mpc(*cosmo_params)
                )
            except (
                AttributeError,
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.error(
                    "Failed to calculate r_s for model '%s': %s",
                    model_name,
                    exc,
                    exc_info=True,
                )
                return float("nan"), None
            if not (numpy.isfinite(rs_value) and rs_value > 0):
                logger.warning(
                    "Model '%s' returned invalid r_s (%.3f Mpc).",
                    model_name,
                    rs_value,
                )
                return float("nan"), None

        for index, row in bao_pred_df.iterrows():
            z_val = row["redshift"]
            obs = row["observable_type"]
            model_pred_numerator = numpy.nan
            try:
                if obs == "DM_over_rs":
                    model_pred_numerator = get_DM_model(z_val, *cosmo_params)
                elif obs == "DH_over_rs":
                    hz_val = get_Hz_model(z_val, *cosmo_params)
                    if numpy.isfinite(hz_val) and abs(hz_val) > 1e-9:
                        model_pred_numerator = C_LIGHT / hz_val
                elif obs == "DV_over_rs":
                    if get_DV_model_specific:
                        model_pred_numerator = get_DV_model_specific(
                            z_val,
                            *cosmo_params,
                        )
                    else:
                        dm_val = get_DM_model(z_val, *cosmo_params)
                        hz_val = get_Hz_model(z_val, *cosmo_params)
                        if (
                            numpy.isfinite(dm_val)
                            and dm_val >= 0
                            and numpy.isfinite(hz_val)
                            and abs(hz_val) > 1e-9
                            and z_val > 1e-9
                        ):
                            term = (dm_val**2) * C_LIGHT * z_val / hz_val
                            model_pred_numerator = (
                                term ** (1.0 / 3.0) if term >= 0 else numpy.nan
                            )
                        elif abs(z_val) < 1e-9:
                            model_pred_numerator = 0.0

                if numpy.isfinite(model_pred_numerator):
                    bao_pred_df.loc[index, "model_prediction"] = (
                        model_pred_numerator / rs_value
                    )
            except (
                AttributeError,
                IndexError,
                ImportError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                logger.exception(
                    "statistics.calculate_bao_observables: "
                    "BAO prediction failed for %s at z=%s in model '%s'",
                    obs,
                    z_val,
                    model_name,
                )

        smooth_preds = None
        if z_smooth_arr is not None:
            try:
                dm_smooth = get_DM_model(z_smooth_arr, *cosmo_params)
                hz_smooth = get_Hz_model(z_smooth_arr, *cosmo_params)
                dh_smooth = numpy.where(
                    hz_smooth > 0, C_LIGHT / hz_smooth, numpy.nan
                )

                if get_DV_model_specific:
                    dv_smooth = get_DV_model_specific(
                        z_smooth_arr,
                        *cosmo_params,
                    )
                else:
                    da_smooth = get_DA_model(z_smooth_arr, *cosmo_params)
                    term = (
                        numpy.power(1 + z_smooth_arr, 2)
                        * numpy.power(da_smooth, 2)
                        * C_LIGHT
                        * z_smooth_arr
                        / hz_smooth
                    )
                    dv_smooth = numpy.power(
                        term,
                        1 / 3,
                        where=term >= 0,
                        out=numpy.full_like(z_smooth_arr, numpy.nan),
                    )

                smooth_preds = {
                    "z": z_smooth_arr,
                    "dm_over_rs": dm_smooth / rs_value,
                    "dh_over_rs": dh_smooth / rs_value,
                    "dv_over_rs": dv_smooth / rs_value,
                }
            except (
                AttributeError,
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.error(
                    "Failed to calculate smooth BAO curves for %s: %s",
                    model_name,
                    exc,
                    exc_info=True,
                )
        return rs_value, smooth_preds

    rs_mpc, smooth_predictions = _fill_from_plugin(float("nan"))

    if not (numpy.isfinite(rs_mpc) and rs_mpc > 0):
        return bao_pred_df, float("nan"), None

    logger.info(
        "Successfully calculated r_s for %s: %.3f Mpc",
        model_name,
        rs_mpc,
    )
    return bao_pred_df, rs_mpc, smooth_predictions
