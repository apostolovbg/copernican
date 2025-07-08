"""Chi-squared calculation helpers for the Copernican Suite.

This module collects generic statistical routines used by multiple engines.
They were previously located in ``model_coder.py`` but now live here so that
``model_coder.py`` focuses solely on transforming validated JSON models into
callable functions. The helpers are intentionally lightweight and rely only on
NumPy, CAMB and the engine interface.
"""

import logging
from typing import Sequence, Iterable, Optional, Dict

import numpy as np
import camb

from . import engine_interface


def chi_squared_sne(
    cosmo_params: Sequence[float],
    mu_model_func,
    sne_data_df,
) -> float:
    """Calculate chi-squared for supernovae Ia distance modulus data.

    Parameters
    ----------
    cosmo_params : sequence of float
        Cosmological parameters to pass to ``mu_model_func``.
    mu_model_func : callable
        Function computing the theoretical distance modulus for an array of
        redshifts.
    sne_data_df : pandas.DataFrame
        Table containing ``zcmb`` and ``mu_obs`` columns and optionally a
        ``covariance_matrix_inv`` attribute or ``e_mu_obs`` errors.
    """
    logger = logging.getLogger()
    if not all(col in sne_data_df.columns for col in ("zcmb", "mu_obs")):
        logger.error("SNe DataFrame missing required columns 'zcmb' or 'mu_obs'.")
        return np.inf

    z_data = sne_data_df["zcmb"].values
    mu_obs = sne_data_df["mu_obs"].values

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
                logger.error("Covariance matrix dimension mismatch for SNe data.")
                return np.inf
            chi2 = float(resid @ C_inv @ resid)
        except Exception as exc:
            logger.warning(
                f"Falling back to diagonal errors due to covariance issue: {exc}"
            )
            C_inv = None

    if C_inv is None:
        if "e_mu_obs" not in sne_data_df.columns:
            logger.error("No diagonal errors available for SNe data.")
            return np.inf
        err = sne_data_df["e_mu_obs"].values
        err = np.where(err <= 0, 1e-12, err)
        chi2 = np.sum((resid / err) ** 2)

    return chi2 if np.isfinite(chi2) else np.inf


def chi_squared_bao(
    bao_data_df,
    model_plugin,
    cosmo_params: Sequence[float],
    model_rs_Mpc: float,
) -> float:
    """Return chi-squared for BAO observables.

    Parameters
    ----------
    bao_data_df : pandas.DataFrame
        Data with columns ``redshift``, ``observable_type`` and ``value``.
    model_plugin : object
        Validated model plugin providing distance functions.
    cosmo_params : sequence of float
        Cosmological parameter values passed to the plugin.
    model_rs_Mpc : float
        Sound horizon used to normalise the observables.
    """
    logger = logging.getLogger()
    engine_interface.validate_plugin(model_plugin)
    if getattr(model_plugin, "valid_for_bao", True) is False:
        logger.warning("(chi2_bao): Model flagged as invalid for BAO. Skipping calculation.")
        return np.inf
    if bao_data_df is None or bao_data_df.empty:
        logger.error("(chi2_bao): BAO data is empty.")
        return np.inf
    if not (np.isfinite(model_rs_Mpc) and model_rs_Mpc > 0):
        return np.inf

    total = 0.0
    n_valid = 0

    try:
        get_DM = getattr(model_plugin, "get_comoving_distance_Mpc")
        get_Hz = getattr(model_plugin, "get_Hz_per_Mpc")
        get_DV = getattr(model_plugin, "get_DV_Mpc", None)
        C_LIGHT = model_plugin.FIXED_PARAMS.get("C_LIGHT_KM_S", 299792.458)
    except AttributeError as e:
        logger.error(f"(chi2_bao): Model plugin missing required function: {e}")
        return np.inf

    for _, row in bao_data_df.iterrows():
        z_val = row["redshift"]
        obs_type = row["observable_type"]
        obs_val = row["value"]
        obs_err = row["error"]

        if obs_err == 0 or not np.isfinite(obs_err) or obs_err < 1e-9:
            continue

        mod_num = np.nan
        try:
            if obs_type == "DM_over_rs":
                mod_num = get_DM(z_val, *cosmo_params)
            elif obs_type == "DH_over_rs":
                hz_val = get_Hz(z_val, *cosmo_params)
                if np.isfinite(hz_val) and abs(hz_val) > 1e-9:
                    mod_num = C_LIGHT / hz_val
            elif obs_type == "DV_over_rs":
                if get_DV:
                    mod_num = get_DV(z_val, *cosmo_params)
                else:
                    dm_val = get_DM(z_val, *cosmo_params)
                    hz_val = get_Hz(z_val, *cosmo_params)
                    if (
                        np.isfinite(dm_val)
                        and dm_val >= 0
                        and np.isfinite(hz_val)
                        and abs(hz_val) > 1e-9
                        and z_val > 1e-9
                    ):
                        term = (dm_val ** 2) * C_LIGHT * z_val / hz_val
                        mod_num = term ** (1.0 / 3.0) if term >= 0 else np.nan
                    elif abs(z_val) < 1e-9:
                        mod_num = 0.0
        except Exception:
            continue

        if np.isfinite(mod_num):
            total += ((obs_val - mod_num / model_rs_Mpc) / obs_err) ** 2
            n_valid += 1

    if n_valid == 0:
        logger.warning("(chi2_bao): No valid BAO points to calculate chi-squared.")
        return np.inf

    return total if np.isfinite(total) else np.inf


def compute_cmb_spectrum(
    param_dict: Dict[str, float],
    ells: Iterable[int],
    spectra: tuple = ("TT",),
):
    """Return theoretical D_ell spectra using CAMB.

    Parameters
    ----------
    param_dict : dict
        CAMB parameter dictionary such as produced by ``plugin.get_camb_params``.
    ells : iterable of int
        Multipole moments at which to evaluate the spectra.
    spectra : tuple of str
        Combination of ``"TT"``, ``"TE"`` and ``"EE"`` to compute.
    """
    logger = logging.getLogger()
    try:
        H0 = float(param_dict.get("H0", 67.0))
        ombh2 = float(param_dict.get("ombh2", 0.02237))
        omch2 = float(param_dict.get("omch2", 0.12))
        tau = float(param_dict.get("tau", 0.054))
        As = float(param_dict.get("As", 2.1e-9))
        ns = float(param_dict.get("ns", 0.965))
        omnuh2 = float(param_dict.get("omnuh2", 0.0))
    except Exception as exc:
        logger.error(f"(compute_cmb_spectrum): Invalid parameter mapping: {exc}")
        return np.full_like(ells, np.nan, dtype=float)

    params = camb.CAMBparams()
    params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    params.omnuh2 = omnuh2
    params.InitPower.set_params(As=As, ns=ns)
    params.set_for_lmax(int(np.max(ells)) + 300, lens_potential_accuracy=0)
    try:
        results = camb.get_results(params)
        full_dls = results.get_unlensed_scalar_cls(
            lmax=int(np.max(ells)), CMB_unit="muK"
        )

        ell_arr = np.asarray(ells, dtype=int)
        result = {}
        if "TT" in spectra:
            result["TT"] = full_dls[ell_arr, 0]
        if "EE" in spectra:
            result["EE"] = full_dls[ell_arr, 1]
        if "TE" in spectra:
            result["TE"] = full_dls[ell_arr, 3]

        if len(result) == 1:
            return next(iter(result.values()))
        return result
    except Exception as exc:
        logger.error(f"(compute_cmb_spectrum): CAMB failed: {exc}")
        return np.full_like(ells, np.nan, dtype=float)


def chi_squared_cmb(
    cosmo_params: Sequence[float] | Dict[str, float],
    cmb_data_df,
    plugin=None,
    extra_params: Optional[Dict[str, float]] = None,
) -> float:
    """Calculate chi-squared for CMB power spectrum data."""

    logger = logging.getLogger()
    if cmb_data_df is None or cmb_data_df.empty:
        logger.error("(chi2_cmb): CMB data is empty.")
        return np.inf
    if "covariance_matrix_inv" not in cmb_data_df.attrs:
        logger.error("(chi2_cmb): Inverse covariance matrix missing in attrs.")
        return np.inf

    ells = cmb_data_df["ell"].values
    obs = cmb_data_df["Dl_obs"].values

    if plugin is not None:
        try:
            param_dict = plugin.get_camb_params(cosmo_params)
        except Exception as exc:
            logger.error(f"(chi2_cmb): failed to map parameters: {exc}")
            return np.inf
    else:
        if isinstance(cosmo_params, dict):
            param_dict = dict(cosmo_params)
        else:
            names = cmb_data_df.attrs.get("param_names", [])
            param_dict = {n: v for n, v in zip(names, cosmo_params)}

    if extra_params:
        param_dict.update(extra_params)

    th = compute_cmb_spectrum(param_dict, ells, spectra=("TT",))
    if th.shape != obs.shape or np.any(~np.isfinite(th)):
        return np.inf

    resid = obs - th
    C_inv = cmb_data_df.attrs["covariance_matrix_inv"]
    try:
        chi2 = float(resid @ C_inv @ resid)
    except Exception as exc:
        logger.error(f"(chi2_cmb): Linear algebra failure: {exc}")
        return np.inf

    return chi2 if np.isfinite(chi2) else np.inf

