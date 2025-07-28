# Copernican Suite Plotter
"""Plotting utilities for the Copernican Suite."""
# All plotting code lives here so that engines only perform computations.
# Functions create Matplotlib figures summarising SNe, BAO and CMB results.

from typing import Any
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import textwrap

from .utils import generate_filename, ensure_dir_exists, get_timestamp
from .logger import get_logger
from copernican import COPERNICAN_VERSION


def _wrap_math(text: str) -> str:
    """Return ``text`` wrapped in dollar signs for MathText rendering."""
    if text is None:
        return ""
    cleaned = re.sub(r"^\$+|\$+$", "", str(text).strip())
    # Matplotlib's MathText does not understand sizing macros like ``\bigl``
    # or ``\bigr``. Remove them to avoid parse errors when rendering.
    cleaned = re.sub(r"\\(bigl|bigr|Bigl|Bigr|biggl|biggr|left|right)", "", cleaned)
    cleaned = re.sub(r"\\,", "", cleaned)
    cleaned = re.sub(r"\\rm\s*", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return f"${cleaned}$" if cleaned else ""


def _smooth_line(
    x: np.ndarray, y: np.ndarray, points: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth interpolation of ``y`` over ``x``."""
    if len(x) < 4:
        return x, y
    try:
        from scipy.interpolate import make_interp_spline

        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]
        order = 3 if len(x_sorted) > 3 else 1
        spline = make_interp_spline(x_sorted, y_sorted, k=order)
        x_new = np.linspace(x_sorted[0], x_sorted[-1], points)
        y_new = spline(x_new)
        return x_new, y_new
    except Exception as exc:
        get_logger().warning(f"Could not smooth line: {exc}")
        return x, y


def get_binned_average(
    z: np.ndarray, residuals: np.ndarray, n_bins: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Return binned average residuals or empty arrays when unavailable."""
    if len(z) < n_bins:
        return np.array([]), np.array([])
    try:
        from scipy.stats import binned_statistic

        mean_stat, bin_edges, _ = binned_statistic(
            z, residuals, statistic="mean", bins=n_bins
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        valid_indices = ~np.isnan(mean_stat)
        return bin_centers[valid_indices], mean_stat[valid_indices]
    except ImportError:
        get_logger().warning("Scipy not found, cannot plot binned residual averages.")
        return np.array([]), np.array([])
    except Exception as exc:
        get_logger().warning(
            f"Could not calculate binned average due to an error: {exc}"
        )
        return np.array([]), np.array([])


def compose_footer(base_line: str, data_attrs: dict) -> list[str]:
    """Return footer lines with dataset information."""

    long_name = data_attrs.get(
        "dataset_long_name", data_attrs.get("dataset_name_attr", "")
    )
    notes = data_attrs.get("notes", "")
    citation = data_attrs.get("citation", "")
    second_line = f"{_wrap_math(long_name)}: {notes} {citation}".strip()
    # Allow more characters per line so lengthy citations do not wrap
    # excessively. Each wrapped line will be drawn separately with a
    # slightly smaller font size by the caller.
    if second_line:
        wrapped_lines = textwrap.wrap(second_line, width=200)
        return [base_line] + wrapped_lines
    return [base_line]


def _apply_common_style() -> None:
    """Apply a consistent white background and grey grid style."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.facecolor": "white",
            "grid.color": "#E0E0E0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
        }
    )


def format_model_summary_text(
    model_plugin: Any,
    dataset_type: str,
    fit_results: Any,
    **kwargs: Any,
) -> str:
    """Return a formatted text block with model details and fit statistics."""

    lines: list[str] = []
    model_name_raw = getattr(model_plugin, "MODEL_NAME", "N/A")
    model_name_latex = model_name_raw.replace("_", r"\_")
    lines.append(rf"**Model: {model_name_latex}**")

    lines.append("$\\mathbf{Mathematical\\ Form:}$")
    for eq_line in getattr(model_plugin, "MODEL_EQUATIONS_LATEX_SN", []):
        lines.append(f"  {_wrap_math(eq_line)}")
    if dataset_type == "bao":
        for eq_line in getattr(model_plugin, "MODEL_EQUATIONS_LATEX_BAO", []):
            lines.append(f"  {_wrap_math(eq_line)}")

    lines.append("$\\mathbf{Cosmological\\ Parameters:}$")
    param_names = getattr(model_plugin, "PARAMETER_NAMES", [])
    param_latex_names = getattr(model_plugin, "PARAMETER_LATEX_NAMES", [])
    fitted_cosmo_params = fit_results.get("fitted_cosmological_params")

    if fitted_cosmo_params:
        for i, name in enumerate(param_names):
            val = fitted_cosmo_params.get(name)
            latex_name = param_latex_names[i] if i < len(param_latex_names) else name
            latex_name = _wrap_math(latex_name)
            if val is not None:
                lines.append(rf"  {latex_name} = ${val:.4g}$")
            else:
                lines.append(rf"  {latex_name} = N/A")
    else:
        lines.append("  (Fit failed or parameters unavailable)")

    if dataset_type == "sne" and fit_results.get("fitted_nuisance_params"):
        lines.append("$\\mathbf{SNe\\ Nuisance\\ Parameters:}$")
        for name, val in fit_results["fitted_nuisance_params"].items():
            name_latex = {
                "M_B": r"M_B",
                "alpha_salt2": r"\alpha",
                "beta_salt2": r"\beta",
            }.get(name, name)
            lines.append(rf"  {_wrap_math(name_latex)} = ${val:.4g}$")

    if dataset_type == "sne":
        lines.append("$\\mathbf{SNe\\ Fit\\ Statistics:}$")
        lines.append(
            rf"  $\chi^2_{{SNe}}$ = {fit_results.get('chi2_sne', fit_results.get('chi2_min', np.nan)):.2f}"
        )
        if "chi2_total" in fit_results:
            lines.append(
                rf"  $\chi^2_{{tot}}$ = {fit_results.get('chi2_total', np.nan):.2f}"
            )
    elif dataset_type == "bao":
        lines.append("$\\mathbf{BAO\\ Fit\\ Results:}$")
        lines.append(rf"  $r_s$ = {kwargs.get('rs_Mpc', np.nan):.2f} Mpc")
        lines.append(rf"  $\chi^2_{{BAO}}$ = {kwargs.get('chi2_bao', np.nan):.2f}")
        if "chi2_total" in kwargs:
            lines.append(
                rf"  $\chi^2_{{tot}}$ = {kwargs.get('chi2_total', np.nan):.2f}"
            )
    elif dataset_type == "cmb":
        lines.append("$\\mathbf{CMB\\ Fit\\ Statistics:}$")
        lines.append(rf"  $\chi^2_{{CMB}}$ = {kwargs.get('chi2_cmb', np.nan):.2f}")
        if "chi2_total" in kwargs:
            lines.append(
                rf"  $\chi^2_{{tot}}$ = {kwargs.get('chi2_total', np.nan):.2f}"
            )

    return "\n".join(lines)


def plot_hubble_diagram(
    sne_data_df: Any,
    lcdm_fit_results: Any,
    alt_model_fit_results: Any,
    lcdm_plugin: Any,
    alt_model_plugin: Any,
    plot_dir: str = ".",
    timestamp: str | None = None,
) -> None:
    """Generate and save a Hubble diagram and residuals plot."""
    ensure_dir_exists(plot_dir)
    logger = get_logger()
    dataset_name = sne_data_df.attrs.get("dataset_name_attr", "SNe_data")
    logger.info(f"Generating Hubble Diagram for {dataset_name}...")

    _apply_common_style()

    font_sizes = {
        "title": 22,
        "label": 18,
        "legend": 14,
        "infobox": 12,
        "ticks": 12,
    }

    if "mu_obs" not in sne_data_df.columns:
        fit_res_for_mu = (
            alt_model_fit_results
            if alt_model_fit_results
            and alt_model_fit_results.get("fitted_nuisance_params")
            else lcdm_fit_results
        )
        if (
            sne_data_df.attrs.get("fit_style") == "h2_fit_nuisance"
            and fit_res_for_mu
            and fit_res_for_mu.get("fitted_nuisance_params")
        ):
            nuisance = fit_res_for_mu["fitted_nuisance_params"]
            M_B, alpha, beta = (
                nuisance["M_B"],
                nuisance["alpha_salt2"],
                nuisance["beta_salt2"],
            )
            sne_data_df["mu_obs"] = (
                sne_data_df["mb"]
                - M_B
                + alpha * sne_data_df["x1"]
                - beta * sne_data_df["c"]
            )
        else:
            logger.error(
                "Cannot plot Hubble Diagram: 'mu_obs' column missing and could not be calculated."
            )
            return

    mu_obs_data = sne_data_df["mu_obs"].values
    z_data = sne_data_df["zcmb"].values
    diag_errors_plot = sne_data_df.attrs.get(
        "diag_errors_for_plot", np.ones_like(z_data) * 0.2
    )
    z_plot_smooth = np.geomspace(
        max(np.min(z_data) * 0.9, 0.001), np.max(z_data) * 1.05, 200
    )

    left = 0.08
    right = 0.75
    top = 0.92
    box_height = 0.33
    info_x = 0.77
    info_gap = info_x - right

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(17, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1.5], "hspace": 0.05},
    )

    footer_lines = compose_footer(
        f"\u039bCDM vs {alt_model_plugin.MODEL_NAME} | Copernican Suite {COPERNICAN_VERSION} | {timestamp or get_timestamp()}",
        sne_data_df.attrs,
    )
    line_height = 0.015
    start_y = left + (len(footer_lines) - 1) * line_height
    footer_pad = 0.03
    plt.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=start_y + info_gap + footer_pad,
    )

    axs[0].errorbar(
        z_data,
        mu_obs_data,
        yerr=diag_errors_plot,
        fmt=".",
        color="darkgray",
        alpha=0.6,
        label=f"{dataset_name}",
        elinewidth=1,
        capsize=2,
        ms=5,
        ecolor="lightgray",
        zorder=1,
    )

    if lcdm_fit_results and lcdm_fit_results.get("success"):
        p_lcdm = list(lcdm_fit_results["fitted_cosmological_params"].values())
        mu_model_lcdm_smooth = lcdm_plugin.distance_modulus_model(
            z_plot_smooth, *p_lcdm
        )
        mu_model_lcdm_points = lcdm_plugin.distance_modulus_model(z_data, *p_lcdm)
        res_lcdm = mu_obs_data - mu_model_lcdm_points
        chi2_lcdm = f"{lcdm_fit_results.get('chi2_min', np.nan):.2f}"
        axs[0].plot(
            z_plot_smooth,
            mu_model_lcdm_smooth,
            color="red",
            ls="-",
            label=rf"$\Lambda$CDM ($\chi^2$={chi2_lcdm})",
            lw=2.5,
        )
        axs[1].errorbar(
            z_data,
            res_lcdm,
            yerr=diag_errors_plot,
            fmt=".",
            color="red",
            alpha=0.5,
            label=r"$\Lambda$CDM Res.",
            elinewidth=1,
            capsize=2,
            ms=4,
        )
        z_lcdm_avg, res_lcdm_avg = get_binned_average(z_data, res_lcdm)
        z_lcdm_avg, res_lcdm_avg = _smooth_line(z_lcdm_avg, res_lcdm_avg)
        axs[1].plot(
            z_lcdm_avg,
            res_lcdm_avg,
            color="darkred",
            ls="-",
            lw=2,
            zorder=10,
            label=r"Avg. $\Lambda$CDM Res.",
        )

    alt_name_raw = getattr(alt_model_plugin, "MODEL_NAME", "AltModel")
    alt_name_latex = alt_name_raw.replace("_", r"\_")
    if alt_model_fit_results and alt_model_fit_results.get("success"):
        p_alt = list(alt_model_fit_results["fitted_cosmological_params"].values())
        mu_model_alt_smooth = alt_model_plugin.distance_modulus_model(
            z_plot_smooth, *p_alt
        )
        mu_model_alt_points = alt_model_plugin.distance_modulus_model(z_data, *p_alt)
        res_alt = mu_obs_data - mu_model_alt_points
        chi2_alt = f"{alt_model_fit_results.get('chi2_min', np.nan):.2f}"
        axs[0].plot(
            z_plot_smooth,
            mu_model_alt_smooth,
            color="blue",
            ls="--",
            label=rf"{alt_name_latex} ($\chi^2$={chi2_alt})",
            lw=2.5,
        )
        axs[1].errorbar(
            z_data,
            res_alt,
            yerr=diag_errors_plot,
            fmt=".",
            mfc="none",
            mec="blue",
            ecolor="lightblue",
            alpha=0.5,
            label=rf"{alt_name_latex} Res.",
            elinewidth=1,
            capsize=2,
            ms=4,
        )
        z_alt_avg, res_alt_avg = get_binned_average(z_data, res_alt)
        z_alt_avg, res_alt_avg = _smooth_line(z_alt_avg, res_alt_avg)
        axs[1].plot(
            z_alt_avg,
            res_alt_avg,
            color="darkblue",
            ls="--",
            lw=2,
            zorder=11,
            label=f"Avg. {alt_name_latex} Res.",
        )

    axs[0].set_ylabel(r"Distance Modulus ($\mu$)", fontsize=font_sizes["label"])
    axs[0].legend(fontsize=font_sizes["legend"], loc="lower right")
    axs[0].set_title(f"Hubble Diagram: {dataset_name}", fontsize=font_sizes["title"])
    axs[0].minorticks_on()
    axs[0].tick_params(axis="both", which="major", labelsize=font_sizes["ticks"])
    axs[0].grid(True, which="both", color="#E0E0E0", linestyle="-", linewidth=0.5)

    axs[1].axhline(0, color="black", ls="--", lw=1)
    axs[1].set_xlabel("Redshift (z)", fontsize=font_sizes["label"])
    axs[1].set_ylabel(r"$\mu_{obs} - \mu_{model}$", fontsize=font_sizes["label"])
    axs[1].legend(fontsize=font_sizes["legend"], loc="lower right")
    axs[1].minorticks_on()
    axs[1].tick_params(axis="both", which="major", labelsize=font_sizes["ticks"])
    axs[1].grid(True, which="both", color="#E0E0E0", linestyle="-", linewidth=0.5)

    bbox_lcdm = dict(boxstyle="round,pad=0.5", fc="#FFEEEE", ec="darkred", alpha=0.8)
    bbox_alt = dict(boxstyle="round,pad=0.5", fc="#EEF2FF", ec="darkblue", alpha=0.8)
    red_y = top - info_gap
    blue_y = red_y - box_height - info_gap
    fig.text(
        info_x,
        red_y,
        format_model_summary_text(
            lcdm_plugin,
            "sne",
            lcdm_fit_results,
        ),
        fontsize=font_sizes["infobox"],
        va="top",
        ha="left",
        wrap=False,
        bbox=bbox_lcdm,
    )
    fig.text(
        info_x,
        blue_y,
        format_model_summary_text(
            alt_model_plugin,
            "sne",
            alt_model_fit_results,
        ),
        fontsize=font_sizes["infobox"],
        va="top",
        ha="left",
        wrap=False,
        bbox=bbox_alt,
    )

    y = start_y
    for idx, line in enumerate(footer_lines):
        fs = font_sizes["ticks"] if idx == 0 else font_sizes["ticks"] - 1
        fig.text(0.5, y, line, ha="center", fontsize=fs, wrap=True)
        y -= line_height

    model_comparison_name = f"{lcdm_plugin.MODEL_NAME}-vs-{alt_model_plugin.MODEL_NAME}"
    filename = generate_filename(
        "hubble-plot",
        dataset_name,
        "png",
        model_name=model_comparison_name,
        timestamp=timestamp,
    )
    try:
        plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        logger.info(f"Hubble diagram saved to {filename}")
    except Exception as exc:
        logger.error(f"Error saving Hubble diagram: {exc}")
    finally:
        plt.close(fig)


def plot_bao_observables(
    bao_data_df: Any,
    lcdm_full_results: Any,
    alt_model_full_results: Any,
    lcdm_plugin: Any,
    alt_model_plugin: Any,
    sne_data_df: Any,
    plot_dir: str = ".",
    timestamp: str | None = None,
) -> None:
    """Generate and save a plot of BAO observables versus redshift."""
    ensure_dir_exists(plot_dir)
    logger = get_logger()
    dataset_name = bao_data_df.attrs.get("dataset_name_attr", "BAO_data")
    logger.info(f"Generating BAO Plot for {dataset_name}...")

    _apply_common_style()

    font_sizes = {
        "title": 22,
        "label": 18,
        "legend": 14,
        "infobox": 12,
        "ticks": 12,
    }

    left = 0.08
    right = 0.75
    top = 0.90
    box_height = 0.33
    info_x = 0.77
    info_gap = info_x - right

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(17, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1.5], "hspace": 0.05},
    )
    ax = axs[0]
    res_ax = axs[1]

    footer_lines = compose_footer(
        f"\u039bCDM vs {alt_model_plugin.MODEL_NAME} | Copernican Suite {COPERNICAN_VERSION} | {timestamp or get_timestamp()}",
        bao_data_df.attrs,
    )
    line_height = 0.015
    start_y = left + (len(footer_lines) - 1) * line_height
    footer_pad = 0.03
    plt.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=start_y + info_gap + footer_pad,
    )

    obs_types = bao_data_df["observable_type"].unique()
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.8, len(obs_types)))
    for i, obs_type in enumerate(obs_types):
        subset = bao_data_df[bao_data_df["observable_type"] == obs_type]
        label = f"Data: {obs_type.replace('_', '/')}"
        ax.errorbar(
            subset["redshift"],
            subset["value"],
            yerr=subset["error"],
            fmt="o",
            label=label,
            capsize=3,
            color=colors[i],
            ms=8,
            zorder=5,
        )

    def plot_model_bao(
        results: Any,
        color: str,
        line_styles: list[str],
        label_prefix: str,
        alpha: float = 1.0,
    ) -> None:
        """Internal helper to plot a model's smooth BAO curves."""
        if not results or not results.get("smooth_predictions"):
            logger.warning(
                f"Skipping BAO plot for {label_prefix} as smooth predictions are missing."
            )
            return

        smooth_preds = results["smooth_predictions"]
        z = smooth_preds["z"]

        def robust_plot(z_vals: np.ndarray, y_vals: np.ndarray, **kwargs: Any) -> None:
            """Plot only valid data points to avoid runtime warnings."""
            valid_indices = np.isfinite(z_vals) & np.isfinite(y_vals)
            if np.any(valid_indices):
                ax.plot(z_vals[valid_indices], y_vals[valid_indices], **kwargs)
            else:
                logger.warning(
                    f"No valid data points to plot for {kwargs.get('label')}"
                )

        if "DM_over_rs" in obs_types:
            robust_plot(
                z,
                smooth_preds["dm_over_rs"],
                color=color,
                ls=line_styles[0],
                lw=2.5,
                label=rf"{label_prefix} ($D_M/r_s$)",
                alpha=alpha,
            )
        if "DH_over_rs" in obs_types:
            robust_plot(
                z,
                smooth_preds["dh_over_rs"],
                color=color,
                ls=line_styles[1],
                lw=2.5,
                label=rf"{label_prefix} ($D_H/r_s$)",
                alpha=alpha,
            )
        if "DV_over_rs" in obs_types:
            robust_plot(
                z,
                smooth_preds["dv_over_rs"],
                color=color,
                ls=line_styles[2],
                lw=2.5,
                label=rf"{label_prefix} ($D_V/r_s$)",
                alpha=alpha,
            )

    line_styles = ["-", "--", ":"]
    plot_model_bao(lcdm_full_results, "red", line_styles, r"$\Lambda$CDM")

    alt_name_raw = getattr(alt_model_plugin, "MODEL_NAME", "AltModel")
    alt_name_latex = alt_name_raw.replace("_", r"\_")
    plot_model_bao(
        alt_model_full_results, "blue", line_styles, alt_name_latex, alpha=0.25
    )

    # --- Residuals ---
    all_res = []
    lcdm_pred = lcdm_full_results.get("pred_df")
    if lcdm_pred is not None:
        res_lcdm = bao_data_df["value"].values - lcdm_pred["model_prediction"].values
        all_res.append(res_lcdm)
        res_ax.errorbar(
            bao_data_df["redshift"],
            res_lcdm,
            yerr=bao_data_df["error"],
            fmt=".",
            color="red",
            alpha=0.5,
            label=r"$\Lambda$CDM Res.",
            elinewidth=1,
            capsize=2,
            ms=4,
        )
        z_avg, r_avg = get_binned_average(bao_data_df["redshift"].values, res_lcdm)
        z_avg, r_avg = _smooth_line(z_avg, r_avg)
        res_ax.plot(
            z_avg,
            r_avg,
            color="darkred",
            ls="-",
            lw=2,
            zorder=10,
            label=r"Avg. $\Lambda$CDM Res.",
        )

    alt_pred = alt_model_full_results.get("pred_df")
    if alt_pred is not None:
        res_alt = bao_data_df["value"].values - alt_pred["model_prediction"].values
        all_res.append(res_alt)
        res_ax.errorbar(
            bao_data_df["redshift"],
            res_alt,
            yerr=bao_data_df["error"],
            fmt=".",
            mfc="none",
            mec="blue",
            ecolor="lightblue",
            alpha=0.5,
            label=rf"{alt_name_latex} Res.",
            elinewidth=1,
            capsize=2,
            ms=7,
        )
        z_avg, r_avg = get_binned_average(bao_data_df["redshift"].values, res_alt)
        z_avg, r_avg = _smooth_line(z_avg, r_avg)
        res_ax.plot(
            z_avg,
            r_avg,
            color="darkblue",
            ls="--",
            lw=2,
            zorder=11,
            label=f"Avg. {alt_name_latex} Res.",
        )

    if all_res:
        max_res = np.nanmax(np.abs(np.concatenate(all_res)))
        if np.isfinite(max_res) and max_res > 0:
            res_ax.set_ylim(-1.2 * max_res, 1.2 * max_res)

    ax.set_ylabel(r"$D_X/r_s$", fontsize=font_sizes["label"])
    ax.set_title(
        f"BAO Observables vs. Redshift: {dataset_name}", fontsize=font_sizes["title"]
    )
    ax.legend(fontsize=font_sizes["legend"], loc="best")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=font_sizes["ticks"])
    ax.grid(True, which="both", color="#E0E0E0", linestyle="-", linewidth=0.5)

    res_ax.axhline(0, color="black", ls="--", lw=1)
    res_ax.set_xlabel("Redshift (z)", fontsize=font_sizes["label"])
    res_ax.set_ylabel(r"$D_X/r_s^{obs} - D_X/r_s^{th}$", fontsize=font_sizes["label"])
    res_ax.legend(fontsize=font_sizes["legend"], loc="best")
    res_ax.minorticks_on()
    res_ax.tick_params(axis="both", which="major", labelsize=font_sizes["ticks"])
    res_ax.grid(True, which="both", color="#E0E0E0", linestyle="-", linewidth=0.5)

    bbox_lcdm = dict(boxstyle="round,pad=0.5", fc="#FFEEEE", ec="darkred", alpha=0.8)
    bbox_alt = dict(boxstyle="round,pad=0.5", fc="#EEF2FF", ec="darkblue", alpha=0.8)
    red_y = top - info_gap
    blue_y = red_y - box_height - info_gap
    fig.text(
        info_x,
        red_y,
        format_model_summary_text(
            lcdm_plugin,
            "bao",
            lcdm_full_results.get("sne_fit_results", {}),
            **lcdm_full_results,
        ),
        fontsize=font_sizes["infobox"],
        va="top",
        ha="left",
        wrap=False,
        bbox=bbox_lcdm,
    )
    fig.text(
        info_x,
        blue_y,
        format_model_summary_text(
            alt_model_plugin,
            "bao",
            alt_model_full_results.get("sne_fit_results", {}),
            **alt_model_full_results,
        ),
        fontsize=font_sizes["infobox"],
        va="top",
        ha="left",
        wrap=False,
        bbox=bbox_alt,
    )

    y = start_y
    for idx, line in enumerate(footer_lines):
        fs = font_sizes["ticks"] if idx == 0 else font_sizes["ticks"] - 1
        fig.text(0.5, y, line, ha="center", fontsize=fs, wrap=True)
        y -= line_height

    model_comparison_name = f"{lcdm_plugin.MODEL_NAME}-vs-{alt_model_plugin.MODEL_NAME}"
    filename = generate_filename(
        "bao-plot",
        dataset_name,
        "png",
        model_name=model_comparison_name,
        timestamp=timestamp,
    )
    try:
        plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        logger.info(f"BAO plot saved to {filename}")
    except Exception as exc:
        logger.error(f"Error saving BAO plot: {exc}")
    finally:
        plt.close(fig)


def plot_cmb_spectrum(
    cmb_data_df: Any,
    lcdm_cmb_results: Any,
    alt_cmb_results: Any,
    lcdm_sne_results: Any,
    alt_sne_results: Any,
    lcdm_plugin: Any,
    alt_model_plugin: Any,
    plot_dir: str = ".",
    timestamp: str | None = None,
) -> None:
    """Generate and save a CMB power spectrum plot with residuals."""
    ensure_dir_exists(plot_dir)
    logger = get_logger()
    dataset_name = cmb_data_df.attrs.get("dataset_name_attr", "CMB_data")
    logger.info(f"Generating CMB Spectrum Plot for {dataset_name}...")

    _apply_common_style()

    font_sizes = {
        "title": 22,
        "label": 18,
        "legend": 14,
        "infobox": 12,
        "ticks": 12,
    }

    ells = cmb_data_df["ell"].values
    dl_obs = cmb_data_df["Dl_obs"].values
    diag_errors_plot = None
    if "covariance_matrix_inv" in cmb_data_df.attrs:
        try:
            cov = np.linalg.inv(cmb_data_df.attrs["covariance_matrix_inv"])
            diag_errors_plot = np.sqrt(np.diag(cov))
        except Exception as exc:
            logger.warning(f"Could not derive CMB errors from covariance: {exc}")
            diag_errors_plot = np.full_like(dl_obs, 1.0)
    else:
        diag_errors_plot = np.full_like(dl_obs, 1.0)

    components = ["TT"]
    if "Dl_te_obs" in cmb_data_df.columns:
        components.append("TE")
    if "Dl_ee_obs" in cmb_data_df.columns:
        components.append("EE")

    left = 0.08
    right = 0.75
    top = 0.92
    box_height = 0.33
    info_x = 0.77
    info_gap = info_x - right

    fig, axs = plt.subplots(
        len(components) * 2,
        1,
        figsize=(17, 6 * len(components)),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1.5] * len(components), "hspace": 0.25},
    )

    footer_lines = compose_footer(
        f"\u039bCDM vs {alt_model_plugin.MODEL_NAME} | Copernican Suite {COPERNICAN_VERSION} | {timestamp or get_timestamp()}",
        cmb_data_df.attrs,
    )
    line_height = 0.015
    start_y = left + (len(footer_lines) - 1) * line_height
    footer_pad = 0.03
    plt.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=start_y + info_gap + footer_pad,
    )

    lcdm_theory = None
    alt_theory = None
    if lcdm_cmb_results:
        lcdm_theory = lcdm_cmb_results.get("theory_spectrum")
    if alt_cmb_results:
        alt_theory = alt_cmb_results.get("theory_spectrum")

    alt_name_raw = getattr(alt_model_plugin, "MODEL_NAME", "AltModel")
    alt_name_latex = alt_name_raw.replace("_", r"\_")

    for i, comp in enumerate(components):
        idx_main = i * 2
        idx_res = idx_main + 1
        obs_key = "Dl_obs" if comp == "TT" else f"Dl_{comp.lower()}_obs"
        obs = cmb_data_df[obs_key].values
        if comp == "TT":
            err = diag_errors_plot
        else:
            err = cmb_data_df.get(f"e_{comp.lower()}_obs", np.full_like(obs, 1.0))

        axs[idx_main].errorbar(
            ells,
            obs,
            yerr=err,
            fmt=".",
            color="darkgray",
            alpha=0.6,
            label=f"{dataset_name}",
            elinewidth=1,
            capsize=2,
            ms=5,
            ecolor="lightgray",
            zorder=1,
        )

        axs[idx_main].fill_between(
            ells,
            obs - err,
            obs + err,
            color="lightgray",
            alpha=0.3,
            label="Data ±1σ",
        )

        if lcdm_theory is not None:
            th = (
                lcdm_theory.get(comp)
                if isinstance(lcdm_theory, dict)
                else (lcdm_theory if comp == "TT" else None)
            )
            if th is not None:
                chi2_lcdm = (
                    f"{lcdm_cmb_results.get('chi2_cmb', np.nan):.2f}"
                    if comp == "TT"
                    else ""
                )
                label = r"$\Lambda$CDM" + (
                    rf" ($\chi^2$={chi2_lcdm})" if chi2_lcdm else ""
                )
                axs[idx_main].plot(ells, th, color="red", ls="-", lw=2.0, label=label)
                cv = np.sqrt(2.0 / (2 * ells + 1.0)) * th
                lower = np.clip(th - cv, 1e-8, None)
                axs[idx_main].fill_between(
                    ells,
                    lower,
                    th + cv,
                    color="red",
                    alpha=0.1,
                    label="Cosmic var.",
                    zorder=0,
                )
                res = obs - th
                axs[idx_res].errorbar(
                    ells,
                    res,
                    yerr=err,
                    fmt=".",
                    color="red",
                    alpha=0.5,
                    label=r"$\Lambda$CDM Res." if i == 0 else None,
                    elinewidth=1,
                    capsize=2,
                    ms=4,
                )
                z_avg, r_avg = get_binned_average(ells, res)
                z_avg, r_avg = _smooth_line(z_avg, r_avg)
                axs[idx_res].plot(
                    z_avg,
                    r_avg,
                    color="darkred",
                    ls="-",
                    lw=2,
                    zorder=10,
                    label=r"Avg. $\Lambda$CDM Res." if i == 0 else None,
                )

        if alt_theory is not None:
            th = (
                alt_theory.get(comp)
                if isinstance(alt_theory, dict)
                else (alt_theory if comp == "TT" else None)
            )
            if th is not None:
                chi2_alt = (
                    f"{alt_cmb_results.get('chi2_cmb', np.nan):.2f}"
                    if comp == "TT"
                    else ""
                )
                label = rf"{alt_name_latex}" + (
                    rf" ($\chi^2$={chi2_alt})" if chi2_alt else ""
                )
                axs[idx_main].plot(ells, th, color="blue", ls="--", lw=2.0, label=label)
                res = obs - th
                axs[idx_res].errorbar(
                    ells,
                    res,
                    yerr=err,
                    fmt=".",
                    mfc="none",
                    mec="blue",
                    ecolor="lightblue",
                    alpha=0.5,
                    label=rf"{alt_name_latex} Res." if i == 0 else None,
                    elinewidth=1,
                    capsize=2,
                    ms=4,
                )
                z_avg, r_avg = get_binned_average(ells, res)
                z_avg, r_avg = _smooth_line(z_avg, r_avg)
                axs[idx_res].plot(
                    z_avg,
                    r_avg,
                    color="darkblue",
                    ls="--",
                    lw=2,
                    zorder=11,
                    label=f"Avg. {alt_name_latex} Res." if i == 0 else None,
                )

        axs[idx_main].set_ylabel(r"$D_\ell\ (\mu K^2)$", fontsize=font_sizes["label"])
        if comp in ("TT", "EE"):
            axs[idx_main].set_yscale("log")
        if i == 0:
            axs[idx_main].legend(fontsize=font_sizes["legend"], loc="best")
        # Reduce padding so titles fit in the vertical gaps between
        # spectrum and residual panels without overlapping.
        title_pad = 6
        axs[idx_main].set_title(
            f"CMB {comp} Power Spectrum: {dataset_name}",
            fontsize=font_sizes["title"],
            pad=title_pad,
        )
        axs[idx_main].minorticks_on()
        axs[idx_main].tick_params(
            axis="both", which="major", labelsize=font_sizes["ticks"]
        )
        axs[idx_main].grid(
            True, which="both", color="#E0E0E0", linestyle="-", linewidth=0.5
        )

        axs[idx_res].axhline(0, color="black", ls="--", lw=1)
        if i == len(components) - 1:
            axs[idx_res].set_xlabel(r"Multipole $\ell$", fontsize=font_sizes["label"])
        axs[idx_res].set_ylabel(
            r"$D_\ell^{obs} - D_\ell^{th}$", fontsize=font_sizes["label"]
        )
        if i == 0:
            axs[idx_res].legend(fontsize=font_sizes["legend"], loc="best")
        axs[idx_res].minorticks_on()
        axs[idx_res].tick_params(
            axis="both", which="major", labelsize=font_sizes["ticks"]
        )
        axs[idx_res].grid(
            True, which="both", color="#E0E0E0", linestyle="-", linewidth=0.5
        )

    bbox_lcdm = dict(boxstyle="round,pad=0.5", fc="#FFEEEE", ec="darkred", alpha=0.8)
    bbox_alt = dict(boxstyle="round,pad=0.5", fc="#EEF2FF", ec="darkblue", alpha=0.8)
    red_y = top - info_gap
    blue_y = red_y - box_height - info_gap
    fig.text(
        info_x,
        red_y,
        format_model_summary_text(
            lcdm_plugin,
            "cmb",
            lcdm_sne_results,
            chi2_cmb=lcdm_cmb_results.get("chi2_cmb"),
            chi2_total=lcdm_sne_results.get("chi2_total"),
        ),
        fontsize=font_sizes["infobox"],
        va="top",
        ha="left",
        wrap=False,
        bbox=bbox_lcdm,
    )
    fig.text(
        info_x,
        blue_y,
        format_model_summary_text(
            alt_model_plugin,
            "cmb",
            alt_sne_results,
            chi2_cmb=alt_cmb_results.get("chi2_cmb"),
            chi2_total=alt_sne_results.get("chi2_total"),
        ),
        fontsize=font_sizes["infobox"],
        va="top",
        ha="left",
        wrap=False,
        bbox=bbox_alt,
    )

    y = start_y
    for idx, line in enumerate(footer_lines):
        fs = font_sizes["ticks"] if idx == 0 else font_sizes["ticks"] - 1
        fig.text(0.5, y, line, ha="center", fontsize=fs, wrap=True)
        y -= line_height

    model_comparison_name = f"{lcdm_plugin.MODEL_NAME}-vs-{alt_model_plugin.MODEL_NAME}"
    filename = generate_filename(
        "cmb-plot",
        dataset_name,
        "png",
        model_name=model_comparison_name,
        timestamp=timestamp,
    )
    try:
        plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        logger.info(f"CMB plot saved to {filename}")
    except Exception as exc:
        logger.error(f"Error saving CMB plot: {exc}")
    finally:
        plt.close(fig)
