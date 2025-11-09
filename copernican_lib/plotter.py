# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# Copernican Suite Plotter
"""Plotting utilities for the Copernican Suite."""
# All plotting code lives here so that engines only perform computations.
# Functions create Matplotlib figures summarising SNe, BAO and CMB results.

import os
import textwrap
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from . import latex_utils
from . import version as version_module
from .logger import get_logger
from .utils import ensure_dir_exists, generate_filename, get_timestamp

# ``MAX_CORNER_SAMPLES`` bounds the number of posterior draws processed by the
# corner plot helper. Stage 2 runs can easily accumulate millions of samples
# when walkers, chains and production steps multiply together. Rendering every
# draw would balloon histogram grids into tens of millions of bins, exhausting
# memory and stalling Stage 5 of the suite. Capping the processed samples keeps
# plotting predictable while still conveying the global posterior geometry.
MAX_CORNER_SAMPLES = 100_000


# NOTE: ``_prepare_corner_inputs`` used to be spelled
# ``_validate_corner_inputs``.  The new name emphasises that the helper both
# validates and flattens the raw sampler output.  A backwards-compatibility
# alias keeps Stage 5 imports functional for older automation.
def _prepare_corner_inputs(
    posterior_samples: np.ndarray,
    parameter_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, int | bool]]:
    """Return flattened samples, labels and downsampling statistics.

    The corner plot accepts sampler output either in raw ``(n_steps,
    n_walkers, n_params)`` form or as an ``(n_samples, n_params)`` array.
    This helper normalises both layouts while rejecting empty or
    degenerate inputs so downstream plotting logic never divides by zero
    or tries to index missing parameters.  Stage 2 runs can emit millions of
    draws, so the helper also thins dense chains deterministically to at most
    :data:`MAX_CORNER_SAMPLES` rows.  Returning basic statistics allows the
    caller to report how much data the figure contains. The
    ``legacy_validator`` flag surfaces whether compatibility mode derived those
    statistics.
    """

    samples = np.asarray(posterior_samples, dtype=float)
    if samples.size == 0:
        raise ValueError(
            "posterior_samples is empty; cannot render corner plot",
        )

    if samples.ndim == 3:
        n_steps, n_walkers, n_params = samples.shape
        samples = samples.reshape(n_steps * n_walkers, n_params)
    elif samples.ndim == 2:
        n_params = samples.shape[1]
    else:
        raise ValueError(
            "posterior_samples must have 2 or 3 dimensions for "
            "corner plotting",
        )

    initial_count = int(samples.shape[0])
    clean_samples = samples[~np.any(~np.isfinite(samples), axis=1)]
    if clean_samples.size == 0:
        raise ValueError("All posterior samples contain NaN or inf values")

    if len(parameter_names) < n_params:
        raise ValueError(
            "parameter_names must describe every sampled dimension",
        )

    finite_count = int(clean_samples.shape[0])
    stats: dict[str, int | bool] = {
        "original_count": initial_count,
        "finite_count": finite_count,
        "processed_count": finite_count,
        "stride": 1,
        "downsampled": False,
    }

    if clean_samples.shape[0] > MAX_CORNER_SAMPLES:
        stride = int(np.ceil(clean_samples.shape[0] / MAX_CORNER_SAMPLES))
        stride = max(stride, 1)
        stats["stride"] = stride
        clean_samples = clean_samples[::stride]
        if clean_samples.shape[0] > MAX_CORNER_SAMPLES:
            clean_samples = clean_samples[:MAX_CORNER_SAMPLES]
        stats["processed_count"] = int(clean_samples.shape[0])
    else:
        stats["processed_count"] = finite_count

    stats["downsampled"] = stats["processed_count"] < stats["finite_count"]
    return clean_samples, parameter_names[:n_params], stats


# Backwards compatibility --------------------------------------------------
#
# Stage 5 previously imported ``_validate_corner_inputs`` directly.  Retain the
# public spelling by delegating to :func:`_prepare_corner_inputs` so legacy
# imports keep working while linters still see a real function definition.
def _validate_corner_inputs(
    posterior_samples: np.ndarray,
    parameter_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, int | bool]]:
    """Compatibility wrapper for callers expecting the historic helper name.

    Older automation accessed :func:`_validate_corner_inputs` directly.  The
    renamed :func:`_prepare_corner_inputs` performs the actual work, so this
    wrapper simply forwards the call without altering the return signature.
    Keeping the function definition—rather than a plain assignment—avoids
    the ``flake8`` ``F811`` redefinition warning while preserving runtime
    behaviour.
    """

    return _prepare_corner_inputs(posterior_samples, parameter_names)


def _density_levels(
    histogram: np.ndarray,
    levels: tuple[float, ...],
) -> list[float]:
    """Return histogram heights for requested cumulative density levels."""

    flat = histogram.ravel()
    if flat.size == 0 or not np.isfinite(flat).any():
        return [0.0 for _ in levels]

    order = np.sort(flat)[::-1]
    cumulative = np.cumsum(order)
    if cumulative[-1] == 0:
        return [0.0 for _ in levels]
    cumulative /= cumulative[-1]

    level_values: list[float] = []
    for level in levels:
        idx = np.searchsorted(cumulative, level, side="left")
        idx = min(idx, order.size - 1)
        level_values.append(order[idx])
    level_values.sort()
    return level_values


def _copernican_version() -> str:
    """Return the suite version while tolerating missing helpers.

    The plotting layer executes in subprocesses launched by Matplotlib and by
    the optimisation engines.  Import errors bubbled up when
    ``copernican_lib.version.get_version`` was absent even though the module
    itself was present, preventing residual plots from rendering on macOS.
    Falling back to the "unknown" placeholder keeps Matplotlib usable while
    matching the final stage of :func:`copernican_lib.version.get_version`.
    """

    getter = getattr(version_module, "get_version", None)
    if callable(getter):
        return getter()
    return "0+unknown"


# Query package metadata once so every plot records the same version string.
COPERNICAN_VERSION = _copernican_version()


def _wrap_math(text: str) -> str:
    """Return ``text`` wrapped in dollar signs for MathText rendering."""
    # Delegate the heavy lifting to :mod:`latex_utils` so that the same
    # sanitisation rules are shared across plotting,
    # parsing and code generation.
    return latex_utils.wrap_math(text)


def _compute_corner_layout(
    n_params: int,
    footer_line_count: int,
) -> tuple[tuple[float, float], dict[str, float], float, dict[str, float]]:
    """Return responsive geometry and typography settings for corner plots.

    The Stage 5 report must adapt to wildly different parameter counts: a
    single-parameter posterior should not sprawl across a poster-sized
    canvas while a ten-parameter run still needs legible labels once
    exported.  The helper keeps the familiar "large panel" look for small
    systems, gradually shrinking each cell while clamping the overall
    figure to twelve inches per side so Matplotlib never allocates
    oversized canvases.  Font sizes and footer spacing scale with the
    resulting panel width which keeps axis labels readable when the figure
    is downscaled in documentation or embeds.
    """

    if n_params <= 0:
        raise ValueError("Corner plots require at least one parameter")

    # Maintain the previous four-inch baseline while constraining the rendered
    # canvas to a printable footprint.  The minimum prevents collapsed panels
    # when developers request a single-parameter diagnostic.
    base_panel_width = 3.6
    max_side_length = 12.0
    min_side_length = 3.6
    unclamped_side = float(base_panel_width * n_params)
    side_length = min(max(unclamped_side, min_side_length), max_side_length)
    panel_width = side_length / float(n_params)

    # Derive typography from the ratio between the current panel width and the
    # Stage 5 baseline.  Bounding the scaling factor prevents microscopic text
    # for high-dimensional posteriors while retaining the original sizes for
    # the typical three-parameter comparison plots.
    scale = max(panel_width / base_panel_width, 0.55)
    font_sizes = {
        "title": float(np.clip(26.0 * scale, 18.0, 30.0)),
        "label": float(np.clip(18.0 * scale, 12.0, 20.0)),
        "ticks": float(np.clip(12.0 * scale, 8.0, 14.0)),
        "footer": float(np.clip(12.0 * scale, 9.0, 14.0)),
    }

    # Footer line spacing mirrors the footer font so multi-line summaries stay
    # readable without colliding with the axes.  The range matches the legacy
    # default while allowing taller text to breathe.
    line_height = float(np.clip(0.018 + 0.004 * scale, 0.018, 0.03))
    bottom_margin = float(
        np.clip(0.05 + footer_line_count * line_height, 0.08, 0.32)
    )

    # Stretch horizontal margins slightly as the panels shrink so tick labels
    # do not overlap the figure edge.  The adjustments remain subtle to keep
    # the grid centred regardless of dimensionality.
    shrink_penalty = max(0.0, 1.0 - min(scale, 1.0))
    margins = {
        "left": 0.07 + 0.01 * shrink_penalty,
        "right": 0.95 - 0.01 * shrink_penalty,
        "top": 0.9 - 0.02 * shrink_penalty,
        "bottom": bottom_margin,
    }

    figsize = (side_length, side_length)
    return figsize, font_sizes, line_height, margins


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
        get_logger().warning(
            "Scipy not found, cannot plot binned residual averages.",
        )
        return np.array([]), np.array([])
    except Exception as exc:
        get_logger().warning(
            f"Could not calculate binned average due to an error: {exc}"
        )
        return np.array([]), np.array([])


def compose_footer(base_line: str, data_attrs: dict) -> list[tuple[str, bool]]:
    """Return formatted footer lines.

    Parameters
    ----------
    base_line:
        Preformatted first line containing the model comparison,
        Copernican Suite version and a timestamp. The line is already
        styled by the caller.
    data_attrs:
        Metadata dictionary attached to the parsed dataset. Expected
        keys include ``dataset_name``, ``description``, ``notes`` and
        ``citation``.

    Returns
    -------
    list[tuple[str, bool]]
        Iterable of ``(line, is_bold)`` pairs to draw at the bottom of
        each plot. The first element is ``base_line`` and subsequent
        lines describe the dataset and citation. Lines are wrapped at
        190 characters so wide figures remain readable without manual
        line breaks.
    """

    dataset_name = data_attrs.get("dataset_name", "")
    # Escape characters that could break TeX-style formatting used for
    # bold text while preserving spaces in the displayed name.
    safe_name = (
        dataset_name.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("_", "\\_")
        .replace(" ", "\\ ")  # Preserve spaces for MathText rendering
    )
    description = data_attrs.get("description", "")
    notes = data_attrs.get("notes", "")
    citation = data_attrs.get("citation", "")

    # ``mathtext`` does not support ``\textbf`` so ``\mathbf`` is used instead
    # to emphasise the dataset name while keeping the rest of the line in the
    # default font. Spaces are escaped above to preserve their appearance.
    second_line = (
        "Observational dataset and processing: "
        f"$\\mathbf{{{safe_name}}}$: {description} {notes}"
    ).strip()

    wrapped: list[tuple[str, bool]] = [(base_line, True)]
    if second_line:
        wrapped.extend(
            ((line, False) for line in textwrap.wrap(second_line, width=190)),
        )
    if citation:
        wrapped.extend(
            (
                (line, True)
                for line in textwrap.wrap(
                    citation.strip(),
                    width=190,
                )
            ),
        )
    return wrapped


def _format_corner_footer_stats(
    stats: dict[str, int | bool],
) -> list[tuple[str, bool]]:
    """Describe how posterior samples were prepared for the corner plot.

    Corner plots are most useful when operators immediately understand how
    many draws survived filtering and whether any automated thinning
    occurred.  The statistics returned by :func:`_prepare_corner_inputs`
    already contain that information, so this helper converts them into a set
    of footer lines for :func:`plot_corner`.  The text favours short, direct
    sentences so the details stay readable even on dense figures.
    """

    original = int(stats.get("original_count", 0))
    finite = int(stats.get("finite_count", original))
    processed = int(stats.get("processed_count", finite))
    stride = int(stats.get("stride", 1))
    downsampled = bool(stats.get("downsampled", False))
    legacy = bool(stats.get("legacy_validator", False))

    lines: list[tuple[str, bool]] = []
    lines.append(
        (
            "Corner plot generation: "
            f"{processed:,} samples used from {finite:,} finite draws.",
            False,
        ),
    )
    lines.append(
        (
            f"Original chain length {original:,}; stride {stride}.",
            False,
        ),
    )

    invalid = max(original - finite, 0)
    if invalid:
        lines.append(
            (
                f"Discarded {invalid:,} non-finite samples before plotting.",
                False,
            ),
        )

    if downsampled:
        lines.append(
            (
                "Automatic thinning applied to satisfy MAX_CORNER_SAMPLES.",
                False,
            ),
        )

    if legacy:
        lines.append(
            (
                "Legacy validator output detected; statistics inferred.",
                False,
            ),
        )

    return lines


def build_footer_lines(
    alt_model_plugin: Any,
    data_attrs: dict,
    timestamp: str | None = None,
    *,
    extra_lines: Iterable[tuple[str, bool]] | None = None,
) -> list[tuple[str, bool]]:
    """Return footer lines for a given dataset and model comparison.

    This helper centralises footer construction so each plotting
    function draws from the same template without duplicating string
    formatting logic. ``timestamp`` allows callers to supply a fixed
    time for reproducible tests; when omitted the current time is
    generated.
    """

    base_line = (
        f"\u039bCDM vs {alt_model_plugin.MODEL_NAME} | Copernican Suite "
        f"{COPERNICAN_VERSION} | {timestamp or get_timestamp()}"
    )
    composed = compose_footer(base_line, data_attrs)
    if not extra_lines:
        return composed

    enriched: list[tuple[str, bool]] = []
    if composed:
        enriched.append(composed[0])
        enriched.extend(extra_lines)
        enriched.extend(composed[1:])
        return enriched

    return list(extra_lines)


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

    def _format_numeric_line(
        label_tex: str, value: Any, *, unit: str | None = None
    ) -> str:
        """Return a readable line or ``N/A`` when ``value`` is non-finite.

        The helper prevents ``:.2f`` formatting from raising when optimisation
        metadata is incomplete. Cosmological fits occasionally leave global
        chi-squared totals undefined, so the summary must degrade gracefully
        instead of crashing the plotting workflow.
        """

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return rf"  {label_tex} = N/A"
        if not np.isfinite(numeric):
            return rf"  {label_tex} = N/A"
        suffix = f" {unit}" if unit else ""
        formatted = f"{numeric:.2f}"
        return rf"  {label_tex} = {formatted}{suffix}"

    lines.append("$\\mathbf{Mathematical\\ Form:}$")
    for eq_line in getattr(model_plugin, "MODEL_EQUATIONS_LATEX_SN", []):
        lines.append(f"  {_wrap_math(eq_line)}")
    if dataset_type == "bao":
        for eq_line in getattr(model_plugin, "MODEL_EQUATIONS_LATEX_BAO", []):
            lines.append(f"  {_wrap_math(eq_line)}")

    lines.append("$\\mathbf{Cosmological\\ Parameters:}$")
    param_names = getattr(model_plugin, "PARAMETER_NAMES", [])
    param_latex_names = getattr(
        model_plugin,
        "PARAMETER_LATEX_NAMES",
        [],
    )
    fitted_cosmo_params = fit_results.get("fitted_cosmological_params")

    if fitted_cosmo_params:
        for i, name in enumerate(param_names):
            val = fitted_cosmo_params.get(name)
            if i < len(param_latex_names):
                latex_name = param_latex_names[i]
            else:
                latex_name = name
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
        chi2_min = fit_results.get("chi2_min", np.nan)
        chi2_sne = fit_results.get("chi2_sne", chi2_min)
        lines.append(_format_numeric_line(r"$\chi^2_{SNe}$", chi2_sne))
        if "chi2_total" in fit_results:
            chi2_tot = fit_results.get("chi2_total", np.nan)
            lines.append(_format_numeric_line(r"$\chi^2_{tot}$", chi2_tot))
    elif dataset_type == "bao":
        lines.append("$\\mathbf{BAO\\ Fit\\ Results:}$")
        lines.append(
            _format_numeric_line(
                r"$r_s$", kwargs.get("rs_Mpc", np.nan), unit="Mpc"
            )
        )
        chi2_bao = kwargs.get("chi2_bao", np.nan)
        lines.append(_format_numeric_line(r"$\chi^2_{BAO}$", chi2_bao))
        if "chi2_total" in kwargs:
            chi2_tot = kwargs.get("chi2_total", np.nan)
            lines.append(_format_numeric_line(r"$\chi^2_{tot}$", chi2_tot))
    elif dataset_type == "cmb":
        lines.append("$\\mathbf{CMB\\ Fit\\ Statistics:}$")
        chi2_cmb = kwargs.get("chi2_cmb", np.nan)
        lines.append(_format_numeric_line(r"$\chi^2_{CMB}$", chi2_cmb))
        if "chi2_total" in kwargs:
            chi2_tot = kwargs.get("chi2_total", np.nan)
            lines.append(_format_numeric_line(r"$\chi^2_{tot}$", chi2_tot))

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
    dataset_name = sne_data_df.attrs.get("dataset_name", "SNe data")
    dataset_id = sne_data_df.attrs.get("dataset_id", "sne_data")
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
                "Cannot plot Hubble Diagram: 'mu_obs' column missing and "
                "could not be calculated."
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

    footer_lines = build_footer_lines(
        alt_model_plugin,
        sne_data_df.attrs,
        timestamp,
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
        mu_model_lcdm_points = lcdm_plugin.distance_modulus_model(
            z_data,
            *p_lcdm,
        )
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
        fitted_vals = alt_model_fit_results["fitted_cosmological_params"]
        p_alt = list(fitted_vals.values())
        mu_model_alt_smooth = alt_model_plugin.distance_modulus_model(
            z_plot_smooth,
            *p_alt,
        )
        mu_model_alt_points = alt_model_plugin.distance_modulus_model(
            z_data,
            *p_alt,
        )
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

    axs[0].set_ylabel(
        r"Distance Modulus ($\mu$)",
        fontsize=font_sizes["label"],
    )
    axs[0].legend(fontsize=font_sizes["legend"], loc="lower right")
    axs[0].set_title(
        f"Hubble Diagram: {dataset_name}",
        fontsize=font_sizes["title"],
    )
    axs[0].minorticks_on()
    axs[0].tick_params(
        axis="both",
        which="major",
        labelsize=font_sizes["ticks"],
    )
    axs[0].grid(
        True,
        which="both",
        color="#E0E0E0",
        linestyle="-",
        linewidth=0.5,
    )

    axs[1].axhline(0, color="black", ls="--", lw=1)
    axs[1].set_xlabel("Redshift (z)", fontsize=font_sizes["label"])
    axs[1].set_ylabel(
        r"$\mu_{obs} - \mu_{model}$",
        fontsize=font_sizes["label"],
    )
    axs[1].legend(fontsize=font_sizes["legend"], loc="lower right")
    axs[1].minorticks_on()
    axs[1].tick_params(
        axis="both",
        which="major",
        labelsize=font_sizes["ticks"],
    )
    axs[1].grid(
        True,
        which="both",
        color="#E0E0E0",
        linestyle="-",
        linewidth=0.5,
    )

    bbox_lcdm = dict(
        boxstyle="round,pad=0.5",
        fc="#FFEEEE",
        ec="darkred",
        alpha=0.8,
    )
    bbox_alt = dict(
        boxstyle="round,pad=0.5",
        fc="#EEF2FF",
        ec="darkblue",
        alpha=0.8,
    )
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
    for idx, (line, is_bold) in enumerate(footer_lines):
        fs = font_sizes["ticks"] if idx == 0 else font_sizes["ticks"] - 1
        weight = "bold" if is_bold else "normal"
        fig.text(
            0.5,
            y,
            line,
            ha="center",
            fontsize=fs,
            fontweight=weight,
            wrap=True,
        )
        y -= line_height

    alt_model_name = alt_model_plugin.MODEL_NAME.replace(" ", "_").replace(
        ".", ""
    )
    model_comparison_name = f"vs-{alt_model_name}"
    filename = generate_filename(
        "hubble-plot",
        dataset_id,
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
    dataset_name = bao_data_df.attrs.get("dataset_name", "BAO data")
    dataset_id = bao_data_df.attrs.get("dataset_id", "bao_data")
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

    footer_lines = build_footer_lines(
        alt_model_plugin,
        bao_data_df.attrs,
        timestamp,
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
                f"Skipping BAO plot for {label_prefix} as smooth predictions "
                "are missing."
            )
            return

        smooth_preds = results["smooth_predictions"]
        z = smooth_preds["z"]

        def robust_plot(
            z_vals: np.ndarray,
            y_vals: np.ndarray,
            **kwargs: Any,
        ) -> None:
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
    val_data = bao_data_df["value"].values
    lcdm_pred = lcdm_full_results.get("pred_df")
    if lcdm_pred is not None:
        res_lcdm = val_data - lcdm_pred["model_prediction"].values
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
        z_avg, r_avg = get_binned_average(
            bao_data_df["redshift"].values,
            res_lcdm,
        )
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
        res_alt = val_data - alt_pred["model_prediction"].values
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
        z_avg, r_avg = get_binned_average(
            bao_data_df["redshift"].values,
            res_alt,
        )
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
        f"BAO Observables vs. Redshift: {dataset_name}",
        fontsize=font_sizes["title"],
    )
    ax.legend(fontsize=font_sizes["legend"], loc="best")
    ax.minorticks_on()
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=font_sizes["ticks"],
    )
    ax.grid(
        True,
        which="both",
        color="#E0E0E0",
        linestyle="-",
        linewidth=0.5,
    )

    res_ax.axhline(0, color="black", ls="--", lw=1)
    res_ax.set_xlabel("Redshift (z)", fontsize=font_sizes["label"])
    res_ax.set_ylabel(
        r"$D_X/r_s^{obs} - D_X/r_s^{th}$",
        fontsize=font_sizes["label"],
    )
    res_ax.legend(fontsize=font_sizes["legend"], loc="best")
    res_ax.minorticks_on()
    res_ax.tick_params(
        axis="both",
        which="major",
        labelsize=font_sizes["ticks"],
    )
    res_ax.grid(
        True,
        which="both",
        color="#E0E0E0",
        linestyle="-",
        linewidth=0.5,
    )

    bbox_lcdm = dict(
        boxstyle="round,pad=0.5",
        fc="#FFEEEE",
        ec="darkred",
        alpha=0.8,
    )
    bbox_alt = dict(
        boxstyle="round,pad=0.5",
        fc="#EEF2FF",
        ec="darkblue",
        alpha=0.8,
    )
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
    for idx, (line, is_bold) in enumerate(footer_lines):
        fs = font_sizes["ticks"] if idx == 0 else font_sizes["ticks"] - 1
        weight = "bold" if is_bold else "normal"
        fig.text(
            0.5,
            y,
            line,
            ha="center",
            fontsize=fs,
            fontweight=weight,
            wrap=True,
        )
        y -= line_height

    alt_model_name = alt_model_plugin.MODEL_NAME.replace(" ", "_").replace(
        ".", ""
    )
    model_comparison_name = f"vs-{alt_model_name}"
    filename = generate_filename(
        "bao-plot",
        dataset_id,
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
    dataset_name = cmb_data_df.attrs.get("dataset_name", "CMB data")
    dataset_id = cmb_data_df.attrs.get("dataset_id", "cmb_data")
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
            logger.warning(
                f"Could not derive CMB errors from covariance: {exc}",
            )
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
        gridspec_kw={
            "height_ratios": [4, 1.5] * len(components),
            "hspace": 0.25,
        },
    )

    footer_lines = build_footer_lines(
        alt_model_plugin,
        cmb_data_df.attrs,
        timestamp,
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
            err = cmb_data_df.get(
                f"e_{comp.lower()}_obs",
                np.full_like(obs, 1.0),
            )

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
                axs[idx_main].plot(
                    ells,
                    th,
                    color="red",
                    ls="-",
                    lw=2.0,
                    label=label,
                )
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
                axs[idx_main].plot(
                    ells,
                    th,
                    color="blue",
                    ls="--",
                    lw=2.0,
                    label=label,
                )
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

        axs[idx_main].set_ylabel(
            r"$D_\ell\ (\mu K^2)$",
            fontsize=font_sizes["label"],
        )
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
            axs[idx_res].set_xlabel(
                r"Multipole $\ell$",
                fontsize=font_sizes["label"],
            )
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

    bbox_lcdm = dict(
        boxstyle="round,pad=0.5",
        fc="#FFEEEE",
        ec="darkred",
        alpha=0.8,
    )
    bbox_alt = dict(
        boxstyle="round,pad=0.5",
        fc="#EEF2FF",
        ec="darkblue",
        alpha=0.8,
    )
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
    for idx, (line, is_bold) in enumerate(footer_lines):
        fs = font_sizes["ticks"] if idx == 0 else font_sizes["ticks"] - 1
        weight = "bold" if is_bold else "normal"
        fig.text(
            0.5,
            y,
            line,
            ha="center",
            fontsize=fs,
            fontweight=weight,
            wrap=True,
        )
        y -= line_height

    alt_model_name = alt_model_plugin.MODEL_NAME.replace(" ", "_").replace(
        ".", ""
    )
    model_comparison_name = f"vs-{alt_model_name}"
    filename = generate_filename(
        "cmb-plot",
        dataset_id,
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


def plot_corner(
    posterior_samples: np.ndarray,
    alt_model_plugin: Any,
    data_attrs: dict[str, Any] | None,
    plot_dir: str = ".",
    parameter_names: list[str] | None = None,
    timestamp: str | None = None,
) -> None:
    """Generate a corner plot for the joint posterior samples.

    Parameters
    ----------
    posterior_samples:
        Sampler output arranged as ``(n_steps, n_walkers, n_params)`` or a
        two-dimensional ``(n_samples, n_params)`` array. Samples may contain
        NaNs or infinities; these are dropped before plotting to avoid
        distorting the marginal distributions.
    alt_model_plugin:
        Plugin describing the alternative model in the Stage 2 comparison.
        The helper inspects ``PARAMETER_NAMES`` and ``PARAMETER_LATEX_NAMES``
        when axis labels are not supplied explicitly. The model name also
        drives filename generation and footer text so plots tie back to the
        original run directory.
    data_attrs:
        Metadata dictionary associated with the combined dataset. The
        ``dataset_id`` and ``dataset_name`` entries are used both for the
        output filename and for the footer lines rendered below the plot. Pass
        ``None`` to fall back to generic identifiers when metadata is
        unavailable.
    plot_dir:
        Directory that will receive the rendered figure. Directories are
        created automatically so callers can supply fresh output folders.
    parameter_names:
        Optional overrides for the parameter labels. When omitted the
        plugin-provided names are reused.
    timestamp:
        Fixed timestamp passed from the caller so filenames and footer
        content remain deterministic during tests. The current time is used
        when the argument is ``None``.
    """

    ensure_dir_exists(plot_dir)
    logger = get_logger()
    logger.info("Generating corner plot for posterior samples...")
    attrs = data_attrs or {}

    default_names = list(getattr(alt_model_plugin, "PARAMETER_NAMES", []))
    # Store the LaTeX-friendly labels separately so that axis rendering falls
    # back to readable parameter names when the plugin omits a mapping.
    label_candidates = list(
        getattr(alt_model_plugin, "PARAMETER_LATEX_NAMES", []),
    )
    effective_names = parameter_names or default_names

    validated = _prepare_corner_inputs(
        posterior_samples,
        effective_names,
    )

    if not isinstance(validated, tuple):
        raise TypeError(
            "_prepare_corner_inputs must return a tuple of outputs",
        )

    if len(validated) == 3:
        samples, labels, stats = validated
    elif len(validated) == 2:
        samples, labels = validated
        stats = {
            "original_count": int(samples.shape[0]),
            "finite_count": int(samples.shape[0]),
            "processed_count": int(samples.shape[0]),
            "stride": 1,
            "downsampled": False,
            "legacy_validator": True,
        }
    else:
        raise ValueError(
            "_prepare_corner_inputs returned an unexpected number of values",
        )

    # The legacy flag differentiates modern validators from fallback paths so
    # Stage 5 logs can highlight when older helpers require migration.
    stats.setdefault("legacy_validator", False)
    n_params = samples.shape[1]

    if stats.get("legacy_validator", False):
        logger.info(
            "_prepare_corner_inputs returned the legacy two-value signature; "
            "derived summary statistics from the flattened samples",
        )

    if stats["downsampled"]:
        logger.info(
            "Corner plot downsampled to %s of %s finite samples "
            "using stride %s",
            stats["processed_count"],
            stats["finite_count"],
            stats["stride"],
        )
    if stats["finite_count"] < stats["original_count"]:
        logger.info(
            "Dropped %s invalid samples before rendering corner plot",
            stats["original_count"] - stats["finite_count"],
        )

    wrapped_labels: list[str] = []
    for idx in range(n_params):
        latex_name = None
        if idx < len(label_candidates):
            latex_name = label_candidates[idx]
        label = latex_name or labels[idx]
        wrapped_labels.append(_wrap_math(label))

    alt_name = getattr(alt_model_plugin, "MODEL_NAME", "AltModel")
    footer_lines = build_footer_lines(
        alt_model_plugin,
        attrs,
        timestamp,
        extra_lines=_format_corner_footer_stats(stats),
    )

    _apply_common_style()
    figsize, font_sizes, line_height, margins = _compute_corner_layout(
        n_params,
        len(footer_lines),
    )

    # Each dimension receives its own row and column, mirroring the familiar
    # triangle plot layout popularised by corner.py while letting us reuse the
    # Copernican Suite's styling helpers and footers.  The geometry is now
    # derived from ``_compute_corner_layout`` so the panels resize gracefully
    # as the dimensionality grows while remaining within a manageable figure.
    fig, axes = plt.subplots(
        n_params,
        n_params,
        figsize=figsize,
    )
    if n_params == 1:
        axes = np.array([[axes]])

    bins = max(25, int(np.sqrt(samples.shape[0]) // 2))
    percentile_lines = (16.0, 50.0, 84.0)
    contour_levels = (0.68, 0.95)

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if row < col:
                # Hide the upper-triangular panels so the plot reads as a
                # triangle, matching the Copernican documentation.
                ax.axis("off")
                continue

            if row == col:
                param_samples = samples[:, col]
                ax.hist(
                    param_samples,
                    bins=bins,
                    density=True,
                    color="#4e79a7",
                    alpha=0.7,
                    edgecolor="white",
                )
                quantiles = np.percentile(param_samples, percentile_lines)
                for quantile, style in zip(
                    quantiles,
                    ["dashed", "solid", "dashed"],
                ):
                    ax.axvline(
                        quantile,
                        color="#e15759",
                        linestyle=style,
                        linewidth=1.2,
                    )
                ax.set_ylabel("Density", fontsize=font_sizes["label"])
            else:
                x = samples[:, col]
                y = samples[:, row]
                hist, x_edges, y_edges = np.histogram2d(
                    x,
                    y,
                    bins=bins,
                    density=True,
                )
                if np.allclose(hist, 0.0):
                    ax.scatter(
                        x,
                        y,
                        s=6,
                        alpha=0.25,
                        color="#4e79a7",
                    )
                else:
                    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
                    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
                    levels = np.array(
                        _density_levels(hist, contour_levels),
                        dtype=float,
                    )
                    levels = levels[levels > 0]
                    levels = np.unique(levels)
                    high_level = float(hist.max())
                    if levels.size == 0:
                        levels = np.array([high_level])
                    filled_levels = [0.0]
                    filled_levels.extend(levels.tolist())
                    top_level = high_level
                    if top_level <= filled_levels[-1]:
                        top_level = filled_levels[-1] + np.finfo(float).eps
                    filled_levels.append(top_level)
                    ax.contourf(
                        x_centers,
                        y_centers,
                        hist.T,
                        levels=filled_levels,
                        colors=["#dbe9f6", "#afc5e5", "#7da0d4"],
                        alpha=0.9,
                    )
                    # Draw thin outlines so the contour levels remain legible
                    # when exported to PDF or scaled PNG assets.
                    ax.contour(
                        x_centers,
                        y_centers,
                        hist.T,
                        levels=levels.tolist(),
                        colors="#2a5176",
                        linewidths=1.0,
                    )
                ax.grid(True, alpha=0.3)

            if row == n_params - 1:
                ax.set_xlabel(
                    wrapped_labels[col],
                    fontsize=font_sizes["label"],
                )
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(
                    wrapped_labels[row],
                    fontsize=font_sizes["label"],
                )
            elif row != col:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", labelsize=font_sizes["ticks"])

    fig.suptitle(
        f"Posterior corner plot: {alt_name}",
        fontsize=font_sizes["title"],
    )

    plt.subplots_adjust(**margins)

    footer_bottom = margins["bottom"]

    y = footer_bottom - line_height
    for idx, (line, is_bold) in enumerate(footer_lines):
        weight = "bold" if is_bold else "normal"
        fig.text(
            0.5,
            y - idx * line_height,
            line,
            ha="center",
            fontsize=font_sizes["footer"],
            fontweight=weight,
            wrap=True,
        )

    dataset_id = attrs.get("dataset_id", "joint")
    alt_model_name = alt_name.replace(" ", "_").replace(".", "")
    filename = generate_filename(
        "corner-plot",
        dataset_id,
        "png",
        model_name=f"vs-{alt_model_name}",
        timestamp=timestamp,
    )

    try:
        plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        logger.info(f"Corner plot saved to {filename}")
    except Exception as exc:  # pragma: no cover - log path only
        logger.error(f"Error saving corner plot: {exc}")
    finally:
        plt.close(fig)
