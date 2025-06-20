# Copernican Suite CSV Writer
"""CSV writing utilities for the Copernican Suite."""
# DEV NOTE (v1.6): Implementations migrated from ``output_manager.py`` to make
# this module self-contained. CSV output formatting remains unchanged.

from typing import Any
import os
import numpy as np

from .utils import generate_filename, ensure_dir_exists
from .logger import get_logger


def save_sne_results_detailed_csv(
    sne_data_df: Any,
    lcdm_fit_results: Any,
    alt_model_fit_results: Any,
    lcdm_plugin: Any,
    alt_model_plugin: Any,
    csv_dir: str = ".",
) -> None:
    """Save a detailed, point-by-point breakdown of the SNe Ia fitting results."""
    ensure_dir_exists(csv_dir)
    logger = get_logger()

    cols_to_keep = [col for col in ["Name", "zcmb", "mu_obs", "e_mu_obs"] if col in sne_data_df.columns]
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

    alt_model_name = alt_model_plugin.MODEL_NAME.replace(" ", "_").replace(".", "")
    if alt_model_fit_results and alt_model_fit_results.get("success"):
        p_alt = list(alt_model_fit_results["fitted_cosmological_params"].values())
        mu_model_alt = alt_model_plugin.distance_modulus_model(z_data, *p_alt)
        df_out[f"mu_model_{alt_model_name}"] = mu_model_alt
        df_out[f"residual_{alt_model_name}"] = mu_data - mu_model_alt
    else:
        df_out[f"mu_model_{alt_model_name}"] = np.nan
        df_out[f"residual_{alt_model_name}"] = np.nan

    dataset_name = sne_data_df.attrs.get("dataset_name_attr", "SNe_data")
    model_comparison_name = f"LCDM-vs-{alt_model_name}"
    filename = generate_filename("sne-detailed-data", dataset_name, "csv", model_name=model_comparison_name)
    try:
        df_out.to_csv(os.path.join(csv_dir, filename), index=False, float_format="%.8g")
        logger.info(f"SNe detailed results CSV saved to {filename}")
    except Exception as exc:
        logger.error(f"Error saving SNe detailed results CSV: {exc}")


def save_bao_results_csv(
    bao_data_df: Any,
    lcdm_results: Any,
    alt_model_results: Any,
    alt_model_name: str,
    csv_dir: str = ".",
) -> None:
    """Save a detailed breakdown of the BAO results to a CSV file."""
    ensure_dir_exists(csv_dir)
    logger = get_logger()
    if bao_data_df is None or bao_data_df.empty:
        logger.warning("BAO data is empty, skipping CSV save.")
        return

    df_out = bao_data_df.copy()

    if lcdm_results and lcdm_results.get("pred_df") is not None and not lcdm_results["pred_df"].empty:
        df_out["pred_lcdm"] = lcdm_results["pred_df"]["model_prediction"]
        df_out["chi2_contrib_lcdm"] = ((df_out["value"] - df_out["pred_lcdm"]) / df_out["error"]) ** 2

    alt_model_name_safe = alt_model_name.replace(" ", "_").replace(".", "")
    if alt_model_results and alt_model_results.get("pred_df") is not None and not alt_model_results["pred_df"].empty:
        df_out[f"pred_{alt_model_name_safe}"] = alt_model_results["pred_df"]["model_prediction"]
        df_out[f"chi2_contrib_{alt_model_name_safe}"] = ((df_out["value"] - df_out[f"pred_{alt_model_name_safe}"]) / df_out["error"]) ** 2

    dataset_name = bao_data_df.attrs.get("dataset_name_attr", "BAO_data")
    model_comparison_name = f"LCDM-vs-{alt_model_name}"
    filename = generate_filename("bao-detailed-data", dataset_name, "csv", model_name=model_comparison_name)
    try:
        df_out.to_csv(os.path.join(csv_dir, filename), index=False, float_format="%.6g")
        logger.info(f"BAO detailed results CSV saved to {filename}")
    except Exception as exc:
        logger.error(f"Error saving BAO detailed results CSV: {exc}")
