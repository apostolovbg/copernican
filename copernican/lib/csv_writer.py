# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

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

import numpy

from .likelihoods.sne import compute_sne_intercept_delta
from .logger import get_logger
from .utils import ensure_dir_exists, generate_filename


def _safe_model_label(name: str) -> str:
    """Return a model identity suitable for CSV columns and filenames."""

    return name.replace(" ", "_").replace(".", "") or "model"


def save_sne_results_detailed_csv(
    sne_data_df: Any,
    lcdm_fit_results: Any,
    alt_model_fit_results: Any,
    lcdm_plugin: Any,
    alt_model_plugin: Any,
    csv_dir: str = ".",
    timestamp: str | None = None,
    *,
    control_model_name: str | None = None,
    test_model_name: str | None = None,
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
    diag_errors = (
        df_out["e_mu_obs"].to_numpy(dtype=float, copy=True)
        if "e_mu_obs" in df_out
        else None
    )
    requires_intercept = bool(
        sne_data_df.attrs.get("requires_sne_intercept_marginalization")
    )

    control_label = _safe_model_label(control_model_name or "lcdm")
    test_label = _safe_model_label(
        test_model_name or getattr(alt_model_plugin, "MODEL_NAME", "alt")
    )
    if lcdm_fit_results and lcdm_fit_results.get("success"):
        p_lcdm = list(lcdm_fit_results["fitted_cosmological_params"].values())
        mu_model_lcdm = lcdm_plugin.distance_modulus_model(z_data, *p_lcdm)
        df_out[f"mu_model_{control_label}"] = mu_model_lcdm
        residual_lcdm = mu_data - mu_model_lcdm
        if requires_intercept:
            delta_lcdm = compute_sne_intercept_delta(
                residual_lcdm,
                covariance_matrix_inv=sne_data_df.attrs.get(
                    "covariance_matrix_inv"
                ),
                diag_errors=diag_errors,
            )
            residual_lcdm = residual_lcdm + delta_lcdm
        df_out[f"residual_{control_label}"] = residual_lcdm
    else:
        df_out[f"mu_model_{control_label}"] = numpy.nan
        df_out[f"residual_{control_label}"] = numpy.nan

    if alt_model_fit_results and alt_model_fit_results.get("success"):
        p_alt = list(
            alt_model_fit_results["fitted_cosmological_params"].values(),
        )
        mu_model_alt = alt_model_plugin.distance_modulus_model(z_data, *p_alt)
        df_out[f"mu_model_{test_label}"] = mu_model_alt
        residual_alt = mu_data - mu_model_alt
        if requires_intercept:
            delta_alt = compute_sne_intercept_delta(
                residual_alt,
                covariance_matrix_inv=sne_data_df.attrs.get(
                    "covariance_matrix_inv"
                ),
                diag_errors=diag_errors,
            )
            residual_alt = residual_alt + delta_alt
        df_out[f"residual_{test_label}"] = residual_alt
    else:
        df_out[f"mu_model_{test_label}"] = numpy.nan
        df_out[f"residual_{test_label}"] = numpy.nan

    dataset_id = sne_data_df.attrs.get("dataset_id", "sne_data")
    model_comparison_name = f"{control_label}-vs-{test_label}"
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
    except (OSError, ValueError) as exc:
        logger.error(f"Error saving SNe detailed results CSV: {exc}")


def save_bao_results_csv(
    bao_data_df: Any,
    lcdm_results: Any,
    alt_model_results: Any,
    alt_model_name: str,
    csv_dir: str = ".",
    timestamp: str | None = None,
    *,
    control_model_name: str | None = None,
    test_model_name: str | None = None,
) -> None:
    """Save a detailed breakdown of the BAO results to a CSV file."""
    ensure_dir_exists(csv_dir)
    logger = get_logger()
    if bao_data_df is None or bao_data_df.empty:
        logger.warning("BAO data is empty, skipping CSV save.")
        return

    df_out = bao_data_df.copy()

    control_label = _safe_model_label(control_model_name or "lcdm")
    test_label = _safe_model_label(test_model_name or alt_model_name)
    if (
        lcdm_results
        and lcdm_results.get("pred_df") is not None
        and not lcdm_results["pred_df"].empty
    ):
        df_out[f"pred_{control_label}"] = lcdm_results["pred_df"][
            "model_prediction"
        ]
        df_out[f"chi2_contrib_{control_label}"] = (
            (df_out["value"] - df_out[f"pred_{control_label}"])
            / df_out["error"]
        ) ** 2

    if (
        alt_model_results
        and alt_model_results.get("pred_df") is not None
        and not alt_model_results["pred_df"].empty
    ):
        df_out[f"pred_{test_label}"] = alt_model_results["pred_df"][
            "model_prediction"
        ]
        diff = df_out["value"] - df_out[f"pred_{test_label}"]
        ratio = diff / df_out["error"]
        df_out[f"chi2_contrib_{test_label}"] = ratio**2

    dataset_id = bao_data_df.attrs.get("dataset_id", "bao_data")
    model_comparison_name = f"{control_label}-vs-{test_label}"
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
    except (OSError, ValueError) as exc:
        logger.error(f"Error saving BAO detailed results CSV: {exc}")


def save_cmb_results_csv(
    cmb_data_df: Any,
    lcdm_results: Any,
    alt_model_results: Any,
    alt_model_name: str,
    csv_dir: str = ".",
    timestamp: str | None = None,
    *,
    control_model_name: str | None = None,
    test_model_name: str | None = None,
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

    control_label = _safe_model_label(control_model_name or "lcdm")
    test_label = _safe_model_label(test_model_name or alt_model_name)
    if lcdm_results and lcdm_results.get("theory_spectrum") is not None:
        th_lcdm = lcdm_results["theory_spectrum"]
        if isinstance(th_lcdm, dict):
            if "TT" in th_lcdm:
                df_out[f"Dl_{control_label}_tt"] = th_lcdm["TT"]
                df_out[f"residual_{control_label}_tt"] = (
                    df_out["Dl_obs"] - th_lcdm["TT"]
                )
            if "TE" in th_lcdm and "Dl_te_obs" in df_out.columns:
                df_out[f"Dl_{control_label}_te"] = th_lcdm["TE"]
                te_diff = df_out["Dl_te_obs"] - th_lcdm["TE"]
                df_out[f"residual_{control_label}_te"] = te_diff
            if "EE" in th_lcdm and "Dl_ee_obs" in df_out.columns:
                df_out[f"Dl_{control_label}_ee"] = th_lcdm["EE"]
                ee_diff = df_out["Dl_ee_obs"] - th_lcdm["EE"]
                df_out[f"residual_{control_label}_ee"] = ee_diff
        else:
            df_out[f"Dl_{control_label}"] = th_lcdm
            df_out[f"residual_{control_label}"] = df_out["Dl_obs"] - th_lcdm
    else:
        df_out[
            [
                col
                for col in [
                    f"Dl_{control_label}",
                    f"residual_{control_label}",
                    f"Dl_{control_label}_tt",
                    f"residual_{control_label}_tt",
                    f"Dl_{control_label}_te",
                    f"residual_{control_label}_te",
                    f"Dl_{control_label}_ee",
                    f"residual_{control_label}_ee",
                ]
                if col not in df_out.columns
            ]
        ] = numpy.nan

    has_theory = False
    if alt_model_results:
        has_theory = alt_model_results.get("theory_spectrum") is not None
    if has_theory:
        th_alt = alt_model_results["theory_spectrum"]
        if isinstance(th_alt, dict):
            if "TT" in th_alt:
                df_out[f"Dl_{test_label}_tt"] = th_alt["TT"]
                tt_diff = df_out["Dl_obs"] - th_alt["TT"]
                df_out[f"residual_{test_label}_tt"] = tt_diff
            if "TE" in th_alt and "Dl_te_obs" in df_out.columns:
                df_out[f"Dl_{test_label}_te"] = th_alt["TE"]
                te_diff = df_out["Dl_te_obs"] - th_alt["TE"]
                df_out[f"residual_{test_label}_te"] = te_diff
            if "EE" in th_alt and "Dl_ee_obs" in df_out.columns:
                df_out[f"Dl_{test_label}_ee"] = th_alt["EE"]
                ee_diff = df_out["Dl_ee_obs"] - th_alt["EE"]
                df_out[f"residual_{test_label}_ee"] = ee_diff
        else:
            df_out[f"Dl_{test_label}"] = th_alt
            df_out[f"residual_{test_label}"] = df_out["Dl_obs"] - th_alt
    else:
        df_out[
            [
                col
                for col in [
                    f"Dl_{test_label}",
                    f"residual_{test_label}",
                    f"Dl_{test_label}_tt",
                    f"residual_{test_label}_tt",
                    f"Dl_{test_label}_te",
                    f"residual_{test_label}_te",
                    f"Dl_{test_label}_ee",
                    f"residual_{test_label}_ee",
                ]
                if col not in df_out.columns
            ]
        ] = numpy.nan

    dataset_id = cmb_data_df.attrs.get("dataset_id", "cmb_data")
    model_comparison_name = f"{control_label}-vs-{test_label}"
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
    except (OSError, ValueError) as exc:
        logger.error(f"Error saving CMB detailed results CSV: {exc}")
