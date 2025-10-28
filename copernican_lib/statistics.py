"""Shared statistical helpers for cosmological engines.

**Last Updated:** 2025-10-28

The retirement of the combined optimiser elevated this module to the single
source of truth for χ² and spectrum helpers.  All engines—including the
default MCMC sampler—import these routines so numerical behaviour stays
consistent regardless of the backend plugged into :mod:`copernican.py`.  The
helpers cover Supernovae Ia, BAO and CMB likelihood calculations alongside
CAMB-based spectrum generation and BAO plotting utilities.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import camb
import numpy as np

from copernican_lib import engine_interface

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
    """Return the χ² value for Supernovae Ia data.

    Parameters
    ----------
    cosmo_params:
        Ordered sequence of cosmological parameters.
    mu_model_func:
        Callable returning the distance modulus predictions for an array of
        redshifts.  The callable usually lives on the validated model plugin.
    sne_data_df:
        DataFrame returned by
        :func:`copernican_lib.data_loaders.load_sne_data`.

    Notes
    -----
    The routine mirrors the previous combined optimiser implementation and
    keeps the behaviour stable.  The logic is kept intentionally verbose so
    diagnostics in the engines can emit precise error messages when the
    dataset is malformed or the model prediction fails.
    """

    logger = logging.getLogger()
    if not all(col in sne_data_df.columns for col in ("zcmb", "mu_obs")):
        logger.error(
            "SNe DataFrame missing required columns '%s' or '%s'.",
            "zcmb",
            "mu_obs",
        )
        return np.inf

    z_data = sne_data_df["zcmb"].to_numpy(dtype=float)
    mu_obs = sne_data_df["mu_obs"].to_numpy(dtype=float)

    if np.any(~np.isfinite(z_data)) or np.any(~np.isfinite(mu_obs)):
        logger.error("SNe data contains non-finite zcmb or mu_obs values")
        return np.inf

    try:
        mu_model = mu_model_func(z_data, *cosmo_params)
    except Exception:
        return np.inf

    if (
        not isinstance(mu_model, np.ndarray)
        or mu_model.shape != mu_obs.shape
        or np.any(~np.isfinite(mu_model))
    ):
        return np.inf

    resid = mu_obs - mu_model
    C_inv = sne_data_df.attrs.get("covariance_matrix_inv")

    if C_inv is not None:
        try:
            if C_inv.shape[0] != len(resid):
                logger.error("Covariance mismatch for SNe data.")
                return np.inf
            chi2 = float(resid @ C_inv @ resid)
        except Exception as exc:
            logger.warning(
                "Falling back to diagonal errors due to covariance issue: %s",
                exc,
            )
            C_inv = None

    if C_inv is None:
        if "e_mu_obs" not in sne_data_df.columns:
            logger.error("No diagonal errors available for SNe data.")
            return np.inf
        err = sne_data_df["e_mu_obs"].to_numpy(dtype=float)
        err = np.where(~np.isfinite(err) | (err <= 0), 1e-12, err)
        chi2 = float(np.sum((resid / err) ** 2))

    return chi2 if np.isfinite(chi2) else np.inf


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
    r"""Return χ² for BAO observables.

    The calculation is vectorised so the costly distance functions operate on
    NumPy arrays directly.  The function remains dataset-agnostic: callers are
    responsible for extracting the arrays from the BAO DataFrame exactly once
    before invoking this helper inside optimisation or sampling loops.
    """

    logger = logging.getLogger()
    if getattr(model_plugin, "valid_for_bao", True) is False:
        logger.warning("(chi2_bao): Model invalid for BAO; skipping.")
        return np.inf
    if z is None or len(z) == 0:
        logger.error("(chi2_bao): BAO data arrays are empty.")
        return np.inf
    if not (np.isfinite(model_rs_Mpc) and model_rs_Mpc > 0):
        return np.inf

    try:
        get_DM = getattr(model_plugin, "get_comoving_distance_Mpc")
        get_Hz = getattr(model_plugin, "get_Hz_per_Mpc")
        get_DV = getattr(model_plugin, "get_DV_Mpc", None)
        C_LIGHT = model_plugin.FIXED_PARAMS.get("C_LIGHT_KM_S", 299792.458)
    except AttributeError as exc:
        logger.error(
            "(chi2_bao): Model plugin missing required function: %s",
            exc,
        )
        return np.inf

    pred = np.full_like(obs_val, np.nan, dtype=float)

    idx = obs_type == "DM_over_rs"
    if np.any(idx):
        pred[idx] = get_DM(z[idx], *cosmo_params) / model_rs_Mpc

    idx = obs_type == "DH_over_rs"
    if np.any(idx):
        hz = get_Hz(z[idx], *cosmo_params)
        dh = np.where(
            np.isfinite(hz) & (np.abs(hz) > 1e-9),
            C_LIGHT / hz,
            np.nan,
        )
        pred[idx] = dh / model_rs_Mpc

    idx = obs_type == "DV_over_rs"
    if np.any(idx):
        if get_DV:
            dv = get_DV(z[idx], *cosmo_params)
        else:
            dm_val = get_DM(z[idx], *cosmo_params)
            hz_val = get_Hz(z[idx], *cosmo_params)
            term = dm_val**2 * C_LIGHT * z[idx] / hz_val
            mask = (
                np.isfinite(dm_val)
                & (dm_val >= 0)
                & np.isfinite(hz_val)
                & (np.abs(hz_val) > 1e-9)
                & (z[idx] > 1e-9)
            )
            dv = np.full_like(dm_val, np.nan, dtype=float)
            dv[mask] = np.where(
                term[mask] >= 0,
                term[mask] ** (1.0 / 3.0),
                np.nan,
            )
            dv[np.abs(z[idx]) < 1e-9] = 0.0
        pred[idx] = dv / model_rs_Mpc

    if np.all(~np.isfinite(pred)):
        logger.warning("(chi2_bao): Model returned no finite BAO predictions.")
        return np.inf

    resid = obs_val - pred
    if np.any(~np.isfinite(resid)):
        logger.warning("(chi2_bao): Non-finite residuals in BAO data.")
        return np.inf

    C_inv = covariance_matrix_inv
    if C_inv is not None:
        try:
            if C_inv.shape[0] != len(resid):
                raise ValueError("Covariance size mismatch")
            chi2 = float(resid @ C_inv @ resid)
        except Exception as exc:
            logger.warning(
                "Falling back to diagonal BAO errors due to covariance issue:"
                " %s",
                exc,
            )
            C_inv = None

    if C_inv is None:
        valid = np.isfinite(obs_err) & (obs_err > 1e-9)
        if not np.any(valid):
            logger.warning("(chi2_bao): No valid BAO points for chi-squared.")
            return np.inf
        chi2 = float(np.sum((resid[valid] / obs_err[valid]) ** 2))

    return chi2 if np.isfinite(chi2) else np.inf


@lru_cache(maxsize=128)
def _cached_cmb(
    key: tuple[str, tuple[tuple[str, float], ...], int, tuple[str, ...]]
):
    r"""Return unlensed CAMB spectra for a given cache key."""

    _, param_tuple, lmax, spectra = key
    param_dict = dict(param_tuple)
    params = camb.CAMBparams()
    params.set_cosmology(
        H0=param_dict["H0"],
        ombh2=param_dict["ombh2"],
        omch2=param_dict["omch2"],
        tau=param_dict["tau"],
    )
    params.omnuh2 = param_dict.get("omnuh2", 0.0)
    params.InitPower.set_params(As=param_dict["As"], ns=param_dict["ns"])
    params.set_for_lmax(lmax + 300, lens_potential_accuracy=0)
    results = camb.get_results(params)
    cls = results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit="muK")
    out: dict[str, np.ndarray] = {}
    if "TT" in spectra:
        out["TT"] = cls[:, 0]
    if "EE" in spectra:
        out["EE"] = cls[:, 1]
    if "TE" in spectra:
        out["TE"] = cls[:, 3]
    return out


def compute_cmb_spectrum_from_dict(
    param_dict: Mapping[str, float],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
):
    r"""Return theoretical :math:`D_\ell` spectra using CAMB with caching."""

    logger = logging.getLogger()
    try:
        pairs: list[tuple[str, float]] = []
        for key, value in sorted(param_dict.items()):
            pairs.append((key, float(f"{float(value):.6g}")))
        key_tuple = tuple(pairs)
        lmax = int(np.max(ells))
        cache_key = ("dict", key_tuple, lmax, tuple(sorted(spectra)))
        full = _cached_cmb(cache_key)
    except Exception as exc:
        logger.error("(compute_cmb_spectrum_from_dict): %s", exc)
        return np.full_like(np.asarray(list(ells)), np.nan, dtype=float)

    ell_arr = np.asarray(list(ells), dtype=int)
    result = {spec: full[spec][ell_arr] for spec in spectra}
    if len(result) == 1:
        return next(iter(result.values()))
    return result


def compute_cmb_spectrum_cached(
    plugin,
    cosmo_params: Sequence[float],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
):
    r"""Return theoretical :math:`D_\ell` spectra using the model plugin."""

    logger = logging.getLogger()
    try:
        camb_params = plugin.get_camb_params(cosmo_params)
    except Exception as exc:
        logger.error("(compute_cmb_spectrum_cached): %s", exc)
        return np.full_like(np.asarray(list(ells)), np.nan, dtype=float)

    return compute_cmb_spectrum_from_dict(camb_params, ells, spectra=spectra)


def compute_cmb_spectrum(
    param_dict: Mapping[str, float],
    ells: Iterable[int],
    *,
    spectra: Sequence[str] = ("TT",),
):
    """Backward-compatible wrapper accepting a CAMB parameter dictionary."""

    dummy = type(
        "_Dummy",
        (),
        {
            "MODEL_NAME": "direct",
            "get_camb_params": lambda self, _: param_dict,
        },
    )()
    return compute_cmb_spectrum_cached(dummy, [], ells, spectra=spectra)


def chi_squared_cmb(
    cosmo_params: Sequence[float],
    cmb_data_df,
    plugin,
    extra_params: Mapping[str, float] | None = None,
) -> float:
    """Return χ² for CMB temperature and polarisation spectra."""

    logger = logging.getLogger()
    if cmb_data_df is None or cmb_data_df.empty:
        logger.error("(chi2_cmb): CMB data is empty.")
        return np.inf
    if "covariance_matrix_inv" not in cmb_data_df.attrs:
        logger.error("(chi2_cmb): Inverse covariance matrix missing in attrs.")
        return np.inf

    ells = cmb_data_df["ell"].to_numpy(dtype=int)
    obs = cmb_data_df["Dl_obs"].to_numpy(dtype=float)
    camb_params = plugin.get_camb_params(cosmo_params)
    if extra_params:
        camb_params.update(extra_params)

    theory = compute_cmb_spectrum_from_dict(camb_params, ells, spectra=("TT",))
    if theory.shape != obs.shape or np.any(~np.isfinite(theory)):
        return np.inf

    resid = obs - theory
    C_inv = cmb_data_df.attrs["covariance_matrix_inv"]
    try:
        chi2 = float(resid @ C_inv @ resid)
    except Exception as exc:
        logger.error("(chi2_cmb): Linear algebra failure: %s", exc)
        return np.inf

    return chi2 if np.isfinite(chi2) else np.inf


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
