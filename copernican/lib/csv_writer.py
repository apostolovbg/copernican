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

from .cmb_output import assemble_cmb_theory_vector, cmb_observation_blocks
from .likelihoods.sne import compute_sne_intercept_delta
from .logger import get_logger
from .model_selection import ComparisonRequest
from .utils import ensure_dir_exists, generate_filename


def _safe_model_label(name: str) -> str:
    """Return a model identity suitable for CSV columns and filenames."""

    return name.replace(" ", "_").replace(".", "") or "model"


def _validate_plugin_pair(
    comparison: ComparisonRequest,
    control_model_plugin: Any,
    test_model_plugin: Any,
) -> None:
    """Require CSV plugins to match the declared comparison roles."""

    plugin_names = (
        str(getattr(control_model_plugin, "MODEL_NAME", "")),
        str(getattr(test_model_plugin, "MODEL_NAME", "")),
    )
    if plugin_names != comparison.model_names:
        raise ValueError(
            "CSV model plugins must match the declared control/test "
            "comparison."
        )


def save_sne_results_detailed_csv(
    sne_data_df: Any,
    control_fit_results: Any,
    test_fit_results: Any,
    control_model_plugin: Any,
    test_model_plugin: Any,
    csv_dir: str = ".",
    timestamp: str | None = None,
    *,
    comparison: ComparisonRequest,
) -> None:
    """Save a detailed, point-by-point breakdown of the SNe Ia fitting
    results."""
    ensure_dir_exists(csv_dir)
    logger = get_logger()
    _validate_plugin_pair(
        comparison,
        control_model_plugin,
        test_model_plugin,
    )

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

    control_name, test_name = comparison.model_names
    control_label = _safe_model_label(control_name)
    test_label = _safe_model_label(test_name)
    if control_fit_results and control_fit_results.get("success"):
        p_control = list(control_fit_results["fitted_model_params"].values())
        mu_model_control = control_model_plugin.distance_modulus_model(
            z_data, *p_control
        )
        df_out[f"mu_model_{control_label}"] = mu_model_control
        residual_control = mu_data - mu_model_control
        if requires_intercept:
            delta_control = compute_sne_intercept_delta(
                residual_control,
                covariance_matrix_inv=sne_data_df.attrs.get(
                    "covariance_matrix_inv"
                ),
                diag_errors=diag_errors,
            )
            residual_control = residual_control + delta_control
        df_out[f"residual_{control_label}"] = residual_control
    else:
        df_out[f"mu_model_{control_label}"] = numpy.nan
        df_out[f"residual_{control_label}"] = numpy.nan

    if test_fit_results and test_fit_results.get("success"):
        p_test = list(
            test_fit_results["fitted_model_params"].values(),
        )
        mu_model_test = test_model_plugin.distance_modulus_model(
            z_data, *p_test
        )
        df_out[f"mu_model_{test_label}"] = mu_model_test
        residual_test = mu_data - mu_model_test
        if requires_intercept:
            delta_test = compute_sne_intercept_delta(
                residual_test,
                covariance_matrix_inv=sne_data_df.attrs.get(
                    "covariance_matrix_inv"
                ),
                diag_errors=diag_errors,
            )
            residual_test = residual_test + delta_test
        df_out[f"residual_{test_label}"] = residual_test
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
    control_results: Any,
    test_results: Any,
    csv_dir: str = ".",
    timestamp: str | None = None,
    *,
    comparison: ComparisonRequest,
) -> None:
    """Save a detailed breakdown of the BAO results to a CSV file."""
    ensure_dir_exists(csv_dir)
    logger = get_logger()
    if bao_data_df is None or bao_data_df.empty:
        logger.warning("BAO data is empty, skipping CSV save.")
        return

    df_out = bao_data_df.copy()

    control_name, test_name = comparison.model_names
    control_label = _safe_model_label(control_name)
    test_label = _safe_model_label(test_name)
    if (
        control_results
        and control_results.get("pred_df") is not None
        and not control_results["pred_df"].empty
    ):
        df_out[f"pred_{control_label}"] = control_results["pred_df"][
            "model_prediction"
        ]
        df_out[f"chi2_contrib_{control_label}"] = (
            (df_out["value"] - df_out[f"pred_{control_label}"])
            / df_out["error"]
        ) ** 2

    if (
        test_results
        and test_results.get("pred_df") is not None
        and not test_results["pred_df"].empty
    ):
        df_out[f"pred_{test_label}"] = test_results["pred_df"][
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
    control_results: Any,
    test_results: Any,
    csv_dir: str = ".",
    timestamp: str | None = None,
    *,
    comparison: ComparisonRequest,
) -> None:
    """Save CMB spectrum predictions and residuals to a CSV file."""
    ensure_dir_exists(csv_dir)
    logger = get_logger()
    if cmb_data_df is None or cmb_data_df.empty:
        logger.warning("CMB data is empty, skipping CSV save.")
        return

    control_name, test_name = comparison.model_names
    control_label = _safe_model_label(control_name)
    test_label = _safe_model_label(test_name)
    if "spectrum" in cmb_data_df.columns:
        df_out = cmb_data_df[["ell", "spectrum", "Dl_obs"]].copy()
        blocks = cmb_observation_blocks(cmb_data_df)
        for results, model_label in (
            (control_results, control_label),
            (test_results, test_label),
        ):
            theory = results.get("theory_spectrum") if results else None
            try:
                values = (
                    assemble_cmb_theory_vector(
                        theory,
                        blocks,
                        total_row_count=len(cmb_data_df),
                    )
                    if theory is not None
                    else numpy.full(len(cmb_data_df), numpy.nan)
                )
            except (KeyError, TypeError, ValueError):
                values = numpy.full(len(cmb_data_df), numpy.nan)
            df_out[f"Dl_{model_label}"] = values
            df_out[f"residual_{model_label}"] = df_out["Dl_obs"] - values

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
        return

    df_out = cmb_data_df[["ell", "Dl_obs"]].copy()
    if "Dl_te_obs" in cmb_data_df.columns:
        df_out["Dl_te_obs"] = cmb_data_df["Dl_te_obs"]
    if "Dl_ee_obs" in cmb_data_df.columns:
        df_out["Dl_ee_obs"] = cmb_data_df["Dl_ee_obs"]

    if control_results and control_results.get("theory_spectrum") is not None:
        control_spectrum = control_results["theory_spectrum"]
        if isinstance(control_spectrum, dict):
            if "TT" in control_spectrum:
                df_out[f"Dl_{control_label}_tt"] = control_spectrum["TT"]
                df_out[f"residual_{control_label}_tt"] = (
                    df_out["Dl_obs"] - control_spectrum["TT"]
                )
            if "TE" in control_spectrum and "Dl_te_obs" in df_out.columns:
                df_out[f"Dl_{control_label}_te"] = control_spectrum["TE"]
                te_diff = df_out["Dl_te_obs"] - control_spectrum["TE"]
                df_out[f"residual_{control_label}_te"] = te_diff
            if "EE" in control_spectrum and "Dl_ee_obs" in df_out.columns:
                df_out[f"Dl_{control_label}_ee"] = control_spectrum["EE"]
                ee_diff = df_out["Dl_ee_obs"] - control_spectrum["EE"]
                df_out[f"residual_{control_label}_ee"] = ee_diff
        else:
            df_out[f"Dl_{control_label}"] = control_spectrum
            df_out[f"residual_{control_label}"] = (
                df_out["Dl_obs"] - control_spectrum
            )
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
    if test_results:
        has_theory = test_results.get("theory_spectrum") is not None
    if has_theory:
        test_spectrum = test_results["theory_spectrum"]
        if isinstance(test_spectrum, dict):
            if "TT" in test_spectrum:
                df_out[f"Dl_{test_label}_tt"] = test_spectrum["TT"]
                tt_diff = df_out["Dl_obs"] - test_spectrum["TT"]
                df_out[f"residual_{test_label}_tt"] = tt_diff
            if "TE" in test_spectrum and "Dl_te_obs" in df_out.columns:
                df_out[f"Dl_{test_label}_te"] = test_spectrum["TE"]
                te_diff = df_out["Dl_te_obs"] - test_spectrum["TE"]
                df_out[f"residual_{test_label}_te"] = te_diff
            if "EE" in test_spectrum and "Dl_ee_obs" in df_out.columns:
                df_out[f"Dl_{test_label}_ee"] = test_spectrum["EE"]
                ee_diff = df_out["Dl_ee_obs"] - test_spectrum["EE"]
                df_out[f"residual_{test_label}_ee"] = ee_diff
        else:
            df_out[f"Dl_{test_label}"] = test_spectrum
            df_out[f"residual_{test_label}"] = df_out["Dl_obs"] - test_spectrum
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
