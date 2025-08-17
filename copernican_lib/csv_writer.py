# Copernican Suite CSV Writer
"""CSV writing utilities for the Copernican Suite.

These helpers convert fitting results into comma-separated value files so
they can be examined with external tools.  Filenames are normalised via
``utils.generate_filename`` to capture the dataset identifier, model name
and execution timestamp.  Dedicated functions exist for each supported
dataset so that the columns reflect their domain-specific outputs.
"""
import os
from typing import Any

import numpy as np

from .logger import get_logger
from .utils import ensure_dir_exists, generate_filename


def save_sne_results_detailed_csv(
    sne_data_df: Any,
    lcdm_fit_results: Any,
    alt_model_fit_results: Any,
    lcdm_plugin: Any,
    alt_model_plugin: Any,
    csv_dir: str = ".",
    timestamp: str | None = None,
) -> None:
    """Save a detailed, point-by-point breakdown of the SNe Ia fitting
    results."""
    ensure_dir_exists(csv_dir)
    logger = get_logger()

    cols_to_keep = [
        col
        for col in ["Name", "zcmb", "mu_obs", "e_mu_obs"]
        if col in sne_data_df.columns
    ]
    df_out = sne_data_df[cols_to_keep].copy()

    z_data = df_out["zcmb"].values
    mu_data = df_out["mu_obs"].values

    if lcdm_fit_results and lcdm_fit_results.get("success"):
        p_lcdm = list(lcdm_fit_results["fitted_cosmological_params"].values())
        mu_model_lcdm = lcdm_plugin.distance_modulus_model(z_data, *p_lcdm)
        df_out["mu_model_lcdm"] = mu_model_lcdm
        df_out["residual_lcdm"] = mu_data - mu_model_lcdm
    else:
        df_out["mu_model_lcdm"] = np.nan
        df_out["residual_lcdm"] = np.nan

    alt_model_name = alt_model_plugin.MODEL_NAME.replace(" ", "_")
    alt_model_name = alt_model_name.replace(".", "")
    if alt_model_fit_results and alt_model_fit_results.get("success"):
        p_alt = list(
            alt_model_fit_results["fitted_cosmological_params"].values(),
        )
        mu_model_alt = alt_model_plugin.distance_modulus_model(z_data, *p_alt)
        df_out[f"mu_model_{alt_model_name}"] = mu_model_alt
        df_out[f"residual_{alt_model_name}"] = mu_data - mu_model_alt
    else:
        df_out[f"mu_model_{alt_model_name}"] = np.nan
        df_out[f"residual_{alt_model_name}"] = np.nan

    dataset_id = sne_data_df.attrs.get("dataset_id", "sne_data")
    model_comparison_name = f"vs-{alt_model_name}"
    filename = generate_filename(
        "sne-data",
        dataset_id,
        "csv",
        model_name=model_comparison_name,
        timestamp=timestamp,
    )
    try:
        df_out.to_csv(
            os.path.join(csv_dir, filename),
            index=False,
            float_format="%.8g",
        )
        logger.info(f"SNe detailed results CSV saved to {filename}")
    except Exception as exc:
        logger.error(f"Error saving SNe detailed results CSV: {exc}")


def save_bao_results_csv(
    bao_data_df: Any,
    lcdm_results: Any,
    alt_model_results: Any,
    alt_model_name: str,
    csv_dir: str = ".",
    timestamp: str | None = None,
) -> None:
    """Save a detailed breakdown of the BAO results to a CSV file."""
    ensure_dir_exists(csv_dir)
    logger = get_logger()
    if bao_data_df is None or bao_data_df.empty:
        logger.warning("BAO data is empty, skipping CSV save.")
        return

    df_out = bao_data_df.copy()

    if (
        lcdm_results
        and lcdm_results.get("pred_df") is not None
        and not lcdm_results["pred_df"].empty
    ):
        df_out["pred_lcdm"] = lcdm_results["pred_df"]["model_prediction"]
        df_out["chi2_contrib_lcdm"] = (
            (df_out["value"] - df_out["pred_lcdm"]) / df_out["error"]
        ) ** 2

    alt_model_name_safe = alt_model_name.replace(" ", "_").replace(".", "")
    if (
        alt_model_results
        and alt_model_results.get("pred_df") is not None
        and not alt_model_results["pred_df"].empty
    ):
        df_out[f"pred_{alt_model_name_safe}"] = alt_model_results["pred_df"][
            "model_prediction"
        ]
        diff = df_out["value"] - df_out[f"pred_{alt_model_name_safe}"]
        ratio = diff / df_out["error"]
        df_out[f"chi2_contrib_{alt_model_name_safe}"] = ratio**2

    dataset_id = bao_data_df.attrs.get("dataset_id", "bao_data")
    model_comparison_name = f"vs-{alt_model_name}"
    filename = generate_filename(
        "bao-data",
        dataset_id,
        "csv",
        model_name=model_comparison_name,
        timestamp=timestamp,
    )
    try:
        df_out.to_csv(
            os.path.join(csv_dir, filename),
            index=False,
            float_format="%.6g",
        )
        logger.info(f"BAO detailed results CSV saved to {filename}")
    except Exception as exc:
        logger.error(f"Error saving BAO detailed results CSV: {exc}")


def save_cmb_results_csv(
    cmb_data_df: Any,
    lcdm_results: Any,
    alt_model_results: Any,
    alt_model_name: str,
    csv_dir: str = ".",
    timestamp: str | None = None,
) -> None:
    """Save CMB spectrum predictions and residuals to a CSV file."""
    ensure_dir_exists(csv_dir)
    logger = get_logger()
    if cmb_data_df is None or cmb_data_df.empty:
        logger.warning("CMB data is empty, skipping CSV save.")
        return

    df_out = cmb_data_df[["ell", "Dl_obs"]].copy()
    if "Dl_te_obs" in cmb_data_df.columns:
        df_out["Dl_te_obs"] = cmb_data_df["Dl_te_obs"]
    if "Dl_ee_obs" in cmb_data_df.columns:
        df_out["Dl_ee_obs"] = cmb_data_df["Dl_ee_obs"]

    if lcdm_results and lcdm_results.get("theory_spectrum") is not None:
        th_lcdm = lcdm_results["theory_spectrum"]
        if isinstance(th_lcdm, dict):
            if "TT" in th_lcdm:
                df_out["Dl_lcdm_tt"] = th_lcdm["TT"]
                df_out["residual_lcdm_tt"] = df_out["Dl_obs"] - th_lcdm["TT"]
            if "TE" in th_lcdm and "Dl_te_obs" in df_out.columns:
                df_out["Dl_lcdm_te"] = th_lcdm["TE"]
                te_diff = df_out["Dl_te_obs"] - th_lcdm["TE"]
                df_out["residual_lcdm_te"] = te_diff
            if "EE" in th_lcdm and "Dl_ee_obs" in df_out.columns:
                df_out["Dl_lcdm_ee"] = th_lcdm["EE"]
                ee_diff = df_out["Dl_ee_obs"] - th_lcdm["EE"]
                df_out["residual_lcdm_ee"] = ee_diff
        else:
            df_out["Dl_lcdm"] = th_lcdm
            df_out["residual_lcdm"] = df_out["Dl_obs"] - th_lcdm
    else:
        df_out[
            [
                col
                for col in [
                    "Dl_lcdm",
                    "residual_lcdm",
                    "Dl_lcdm_tt",
                    "residual_lcdm_tt",
                    "Dl_lcdm_te",
                    "residual_lcdm_te",
                    "Dl_lcdm_ee",
                    "residual_lcdm_ee",
                ]
                if col not in df_out.columns
            ]
        ] = np.nan

    alt_name_safe = alt_model_name.replace(" ", "_").replace(".", "")
    has_theory = False
    if alt_model_results:
        has_theory = alt_model_results.get("theory_spectrum") is not None
    if has_theory:
        th_alt = alt_model_results["theory_spectrum"]
        if isinstance(th_alt, dict):
            if "TT" in th_alt:
                df_out[f"Dl_{alt_name_safe}_tt"] = th_alt["TT"]
                tt_diff = df_out["Dl_obs"] - th_alt["TT"]
                df_out[f"residual_{alt_name_safe}_tt"] = tt_diff
            if "TE" in th_alt and "Dl_te_obs" in df_out.columns:
                df_out[f"Dl_{alt_name_safe}_te"] = th_alt["TE"]
                te_diff = df_out["Dl_te_obs"] - th_alt["TE"]
                df_out[f"residual_{alt_name_safe}_te"] = te_diff
            if "EE" in th_alt and "Dl_ee_obs" in df_out.columns:
                df_out[f"Dl_{alt_name_safe}_ee"] = th_alt["EE"]
                ee_diff = df_out["Dl_ee_obs"] - th_alt["EE"]
                df_out[f"residual_{alt_name_safe}_ee"] = ee_diff
        else:
            df_out[f"Dl_{alt_name_safe}"] = th_alt
            df_out[f"residual_{alt_name_safe}"] = df_out["Dl_obs"] - th_alt
    else:
        df_out[
            [
                col
                for col in [
                    f"Dl_{alt_name_safe}",
                    f"residual_{alt_name_safe}",
                    f"Dl_{alt_name_safe}_tt",
                    f"residual_{alt_name_safe}_tt",
                    f"Dl_{alt_name_safe}_te",
                    f"residual_{alt_name_safe}_te",
                    f"Dl_{alt_name_safe}_ee",
                    f"residual_{alt_name_safe}_ee",
                ]
                if col not in df_out.columns
            ]
        ] = np.nan

    dataset_id = cmb_data_df.attrs.get("dataset_id", "cmb_data")
    model_comparison_name = f"vs-{alt_name_safe}"
    filename = generate_filename(
        "cmb-data",
        dataset_id,
        "csv",
        model_name=model_comparison_name,
        timestamp=timestamp,
    )
    try:
        df_out.to_csv(
            os.path.join(csv_dir, filename),
            index=False,
            float_format="%.6g",
        )
        logger.info(f"CMB detailed results CSV saved to {filename}")
    except Exception as exc:
        logger.error(f"Error saving CMB detailed results CSV: {exc}")
