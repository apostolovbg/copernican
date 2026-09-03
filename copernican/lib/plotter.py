# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# Copernican Suite Plotter
"""Plotting utilities for the Copernican Suite."""
# All plotting code lives here so that samplers only perform computations.
# Functions create Matplotlib figures summarising SNe, BAO and CMB results.

import math
import os
import textwrap
import tkinter
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import ListedColormap

from copernican import version as version_module

from . import latex_utils
from .cmb_identity import CCMBS_LABEL
from .cmb_output import cmb_observation_blocks, cmb_theory_values_for_block
from .likelihoods.sne import compute_sne_intercept_delta
from .logger import get_logger
from .model_selection import ComparisonRequest, comparison_slug
from .utils import ensure_dir_exists, generate_filename, get_timestamp

# Resolve the Matplotlib backend during import so later calls do not trigger
# the auto-backend sentinel while tests monitor ``switch_backend``.
_ = plt.get_backend()

try:  # pragma: no cover - coverage uses the environment-installed backend
    import arviz as arviz_module
except ModuleNotFoundError:  # pragma: no cover - tests provide ArviZ
    arviz_module = None

# ``MAX_CORNER_SAMPLES`` bounds the number of posterior draws processed by the
# corner plot helper. Stage 2 runs can easily accumulate millions of samples
# when walkers, chains and production steps multiply together. Rendering every
# draw would balloon histogram grids into tens of millions of bins, exhausting
# memory and stalling Stage 5 of the suite. Capping the processed samples keeps
# plotting predictable while still conveying the global posterior geometry.
MAX_CORNER_SAMPLES = 100_000
_INFO_BOX_WIDTH = 0.22
_INFO_BOX_MARGIN = 0.03


def _comparison_display(
    comparison: ComparisonRequest,
) -> tuple[str, str, str]:
    """Return display names and a stable filename token for a comparison."""

    control_name = comparison.control_model.name
    test_name = comparison.test_model.name
    token = comparison_slug(comparison)
    return control_name, test_name, token


def _latex_model_name(name: str) -> str:
    """Escape model names for Matplotlib's mathtext labels."""

    return name.replace("_", r"\_")


def _prepare_corner_inputs(
    posterior_samples: numpy.ndarray,
    parameter_names: list[str],
) -> tuple[numpy.ndarray, list[str], dict[str, int | bool]]:
    """Return flattened samples, labels and downsampling statistics.

    The corner plot accepts sampler output either in raw ``(n_steps,
    n_walkers, n_params)`` form or as an ``(n_samples, n_params)`` array.
    This helper normalises both layouts while rejecting empty or
    degenerate inputs so downstream plotting logic never divides by zero
    or tries to index missing parameters.  Stage 2 runs can emit millions of
    draws, so the helper also thins dense chains deterministically to at most
    :data:`MAX_CORNER_SAMPLES` rows.  Returning basic statistics allows the
    caller to report how much data the figure contains.
    """

    samples = numpy.asarray(posterior_samples, dtype=float)
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
    clean_samples = samples[~numpy.any(~numpy.isfinite(samples), axis=1)]
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
        stride = int(numpy.ceil(clean_samples.shape[0] / MAX_CORNER_SAMPLES))
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


def _info_box_layout(right: float) -> tuple[float, float]:
    """Return the x-coordinate and gap for the info boxes."""

    info_x = 1.0 - _INFO_BOX_WIDTH - _INFO_BOX_MARGIN
    return info_x, info_x - right


def _density_levels(
    histogram: numpy.ndarray,
    levels: tuple[float, ...],
) -> list[float]:
    """Return histogram heights for requested cumulative density levels."""

    flat = histogram.ravel()
    if flat.size == 0 or not numpy.isfinite(flat).any():
        return [0.0 for _ in levels]

    order = numpy.sort(flat)[::-1]
    cumulative = numpy.cumsum(order)
    if cumulative[-1] == 0:
        return [0.0 for _ in levels]
    cumulative /= cumulative[-1]

    level_values: list[float] = []
    for level in levels:
        idx = numpy.searchsorted(cumulative, level, side="left")
        idx = min(idx, order.size - 1)
        level_values.append(float(order[idx]))

    level_values.sort()
    cleaned_levels: list[float] = []
    last_value = -numpy.inf
    epsilon = numpy.finfo(float).eps
    for level_height in level_values:
        finite_value = level_height if numpy.isfinite(level_height) else 0.0
        finite_value = max(finite_value, 0.0)
        if finite_value <= last_value:
            finite_value = (
                numpy.nextafter(last_value, numpy.inf)
                if numpy.isfinite(last_value)
                else epsilon
            )
        cleaned_levels.append(finite_value)
        last_value = finite_value
    return cleaned_levels


def _ensure_strictly_increasing(
    values: Sequence[float], *, start: float | None = None
) -> numpy.ndarray:
    """Return ``values`` as a strictly increasing numpy array.

    ``matplotlib.contour`` requires strictly increasing level thresholds.  The
    Stage 5 grid occasionally feeds in degenerate heights—plateaus, repeated
    bins or NaN placeholders—which would otherwise trigger ``ValueError``
    exceptions.  The helper nudges duplicate or non-finite entries forward by
    one machine epsilon so the final sequence is monotonically increasing while
    remaining numerically close to the original targets.
    """

    arr = numpy.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    result = arr.copy()
    if start is not None:
        result[0] = float(start)

    eps = numpy.finfo(float).eps
    last = result[0]
    if not numpy.isfinite(last):
        last = eps
        result[0] = last

    for idx in range(1, result.size):
        current = result[idx]
        if not numpy.isfinite(current):
            current = last
        if current <= last:
            current = numpy.nextafter(last, numpy.inf)
        result[idx] = current
        last = current

    return result


def _build_contour_levels(
    histogram: numpy.ndarray,
    cumulative_levels: Sequence[float],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return level arrays safe for ``contourf`` and ``contour`` calls.

    The helper extracts positive density thresholds for the requested
    cumulative levels.  When the histogram is nearly flat—or rounds the highest
    bin down to zero—it synthesises a gentle ramp towards the maximum density
    so contour rendering still succeeds without collapsing to a scatter plot.
    """

    finite = histogram[numpy.isfinite(histogram)]
    if finite.size == 0:
        eps = numpy.nextafter(0.0, numpy.inf)
        filled = numpy.array([0.0, eps], dtype=float)
        return filled, filled[1:]

    max_density = float(numpy.max(finite))
    if max_density <= 0.0 or not numpy.isfinite(max_density):
        max_density = numpy.nextafter(0.0, numpy.inf)

    derived = [
        level
        for level in _density_levels(histogram, tuple(cumulative_levels))
        if level > 0.0
    ]

    if not derived:
        fractions = numpy.linspace(0.35, 0.85, max(len(cumulative_levels), 1))
        base_levels = max_density * fractions
    else:
        base_levels = numpy.array(derived, dtype=float)

    base_levels = numpy.asarray(base_levels, dtype=float)
    base_levels.sort()
    base_levels = _ensure_strictly_increasing(base_levels)

    top_level = max_density
    if top_level <= base_levels[-1] or not numpy.isfinite(top_level):
        top_level = numpy.nextafter(base_levels[-1], numpy.inf)

    filled_levels = numpy.concatenate(([0.0], base_levels, [top_level]))
    filled_levels = _ensure_strictly_increasing(filled_levels, start=0.0)

    return filled_levels, base_levels


def _copernican_version() -> str:
    """Return the suite version while tolerating missing helpers.

    The plotting layer executes in subprocesses launched by Matplotlib and by
    the optimisation samplers.  Import errors bubbled up when
    ``copernican.version.get_version`` was absent even though the module
    itself was present, preventing residual plots from rendering on macOS.
    Falling back to the "unknown" placeholder keeps Matplotlib usable while
    matching the final stage of :func:`copernican.version.get_version`.
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


# Corner layout tuning -----------------------------------------------------
#
# Corner plots share a common footer cadence with the other Stage 5 figures.
# The constants below centralise the cadence so both the layout helper and the
# regression tests can assert consistent spacing.  The fixed padding keeps a
# visible gap between the footer block and the axes while the clearance figure
# enforces a second guard band that scales with the footer height.  The minimum
# and maximum bounds prevent extreme dimensionalities from crushing the footer
# into the axes or leaving it stranded far from the plot.  An explicit title
# anchor ensures the figure banner never hugs the canvas edge.
_CORNER_BASE_LINE_HEIGHT = 0.015
# ``_CORNER_FOOTER_PADDING`` keeps a consistent gap between the axes and the
# first footer line.  Stage 5 draws bold text here, so we budget extra vertical
# breathing room to mirror the rest of the plotting suite.
_CORNER_FOOTER_PADDING = 0.038
# ``_CORNER_FOOTER_CLEARANCE`` tracks the minimum spacing between the
# lowest footer baseline and the figure edge.  Guarding this distance stops
# exported PDFs from clipping the text block even when downstream helpers add
# extra diagnostic lines.
_CORNER_FOOTER_CLEARANCE = 0.035
# The base bottom margin provides a buffer before the per-line adjustments
# below kick in.  A taller baseline matches the rest of the suite and raises
# the grid so the footer never feels cramped.
_CORNER_BASE_BOTTOM_MARGIN = 0.07
# Corner plots can feature tall footers when add-ons append diagnostics.  We
# therefore lift the minimum bottom margin to keep the panels safely above the
# text while leaving headroom for Matplotlib legends or colour bars.
_CORNER_MIN_BOTTOM = 0.18
_CORNER_MAX_BOTTOM = 0.42
# Pull the title a touch lower so it mirrors the summary plots.  This echoes
# the user feedback that the previous 0.965 anchor hugged the canvas edge.
_CORNER_TITLE_Y = 0.952


def _compute_corner_layout(
    n_params: int,
    footer_line_count: int,
) -> tuple[
    tuple[float, float],
    dict[str, float],
    float,
    dict[str, float],
    float,
]:
    """Return responsive geometry and typography settings for corner plots.

    The Stage 5 report must adapt to wildly different parameter counts: a
    single-parameter posterior should not sprawl across a poster-sized
    canvas while a ten-parameter run still needs legible labels once
    exported.  The helper keeps the familiar "large panel" look for small
    systems, gradually shrinking each cell while clamping the overall
    figure to twelve inches per side so Matplotlib never allocates
    oversized canvases.  Font sizes and footer spacing scale with the
    resulting panel width which keeps axis labels readable when the figure
    is downscaled in documentation or embeds.  The return value includes a
    dedicated footer padding entry so callers can position the text block
    just below the axes while preserving a uniform gap across layouts.
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
        "title": float(numpy.clip(26.0 * scale, 18.0, 30.0)),
        "label": float(numpy.clip(18.0 * scale, 12.0, 20.0)),
        "ticks": float(numpy.clip(12.0 * scale, 8.0, 14.0)),
        "footer": float(numpy.clip(12.0 * scale, 9.0, 14.0)),
    }

    # Align the footer cadence with the other plotting helpers so the Suite's
    # figures share identical leading.  Tiny adjustments accommodate the
    # shrunken panels used for higher-dimensional runs without letting the
    # spacing collapse.
    shrink_penalty = max(0.0, 1.0 - min(scale, 1.0))
    line_height = float(
        _CORNER_BASE_LINE_HEIGHT * (1.0 + 0.1 * shrink_penalty)
    )

    footer_block = footer_line_count * line_height
    footer_span = max(footer_line_count - 1, 0) * line_height
    base_bottom = _CORNER_BASE_BOTTOM_MARGIN + footer_block
    pad_guard = footer_block + _CORNER_FOOTER_PADDING
    clearance_floor = (
        _CORNER_FOOTER_PADDING + footer_span + _CORNER_FOOTER_CLEARANCE
        if footer_line_count
        else _CORNER_FOOTER_CLEARANCE
    )
    axes_bottom = float(
        numpy.clip(
            max(base_bottom, pad_guard, clearance_floor),
            _CORNER_MIN_BOTTOM,
            _CORNER_MAX_BOTTOM,
        )
    )

    bottom_margin = axes_bottom

    bottom_margin = axes_bottom

    # Stretch horizontal margins slightly as the panels shrink so tick labels
    # do not overlap the figure edge.  The top margin mirrors the Stage 3 and
    # Stage 4 figures by pulling the grid downward just enough to clear the
    # lowered suptitle, while the adjustments remain subtle to keep the grid
    # centred regardless of dimensionality.
    margins = {
        "left": 0.07 + 0.01 * shrink_penalty,
        "right": 0.95 - 0.01 * shrink_penalty,
        "top": float(numpy.clip(0.93 + 0.008 * shrink_penalty, 0.91, 0.945)),
        "bottom": bottom_margin,
    }

    figsize = (side_length, side_length)
    return figsize, font_sizes, line_height, margins, _CORNER_FOOTER_PADDING


def _smooth_line(
    x: numpy.ndarray, y: numpy.ndarray, points: int = 200
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return a smooth interpolation of ``y`` over ``x``."""
    if len(x) < 4:
        return x, y
    try:
        from scipy.interpolate import make_interp_spline

        idx = numpy.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]
        order = 3 if len(x_sorted) > 3 else 1
        spline = make_interp_spline(x_sorted, y_sorted, k=order)
        x_new = numpy.linspace(x_sorted[0], x_sorted[-1], points)
        y_new = spline(x_new)
        return x_new, y_new
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        get_logger().warning(f"Could not smooth line: {exc}")
        return x, y


def get_binned_average(
    z: numpy.ndarray, residuals: numpy.ndarray, n_bins: int = 20
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return binned average residuals or empty arrays when unavailable."""
    if len(z) < n_bins:
        return numpy.array([]), numpy.array([])
    try:
        from scipy.stats import binned_statistic

        mean_stat, bin_edges, _ = binned_statistic(
            z, residuals, statistic="mean", bins=n_bins
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        valid_indices = ~numpy.isnan(mean_stat)
        return bin_centers[valid_indices], mean_stat[valid_indices]
    except ImportError:
        get_logger().warning(
            "Scipy not found, cannot plot binned residual averages.",
        )
        return numpy.array([]), numpy.array([])
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        get_logger().warning(
            f"Could not calculate binned average due to an error: {exc}"
        )
        return numpy.array([]), numpy.array([])


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
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("_", r"\_")
        .replace(" ", r"\ ")  # Preserve spaces for MathText rendering
    )
    description = data_attrs.get("description", "")
    notes = data_attrs.get("notes", "")
    citation = data_attrs.get("citation", "")

    # ``mathtext`` does not support ``\textbf`` so ``\mathbf`` is used instead
    # to emphasise the dataset name while keeping the rest of the line in the
    # default font. Spaces are escaped above to preserve their appearance.
    dataset_block = rf"$\mathbf{{{safe_name}}}$: {description} {notes}"
    second_line = (
        "Observational dataset and processing: " + dataset_block
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

    return lines


def _build_arviz_inference_data(
    samples: numpy.ndarray, labels: list[str]
) -> tuple[object, list[str]]:
    """Construct an ArviZ InferenceData object from flattened samples."""

    if arviz_module is None:
        raise RuntimeError("ArviZ is not available")

    posterior_arrays: dict[str, numpy.ndarray] = {}
    var_names: list[str] = []
    for idx, name in enumerate(labels):
        base = name or f"param_{idx + 1}"
        candidate = base
        suffix = 1
        while candidate in posterior_arrays:
            candidate = f"{base}_{suffix}"
            suffix += 1
        posterior_arrays[candidate] = samples[:, idx]
        var_names.append(candidate)
    inference_data = arviz_module.from_dict(posterior=posterior_arrays)
    return inference_data, var_names


def _render_corner_grid_matplotlib(
    axes: numpy.ndarray,
    n_params: int,
    samples: numpy.ndarray,
    wrapped_labels: list[str],
    font_sizes: dict[str, float],
    bins: int,
    percentile_lines: tuple[float, float, float],
    contour_levels: tuple[float, float],
) -> None:
    """Draw the Matplotlib-based corner grid."""

    for row in range(n_params):
        for col in range(n_params):
            corner_axis = axes[row, col]
            if row < col:
                corner_axis.axis("off")
                continue

            if row == col:
                param_samples = samples[:, col]
                corner_axis.hist(
                    param_samples,
                    bins=bins,
                    density=True,
                    color="#4e79a7",
                    alpha=0.7,
                    edgecolor="white",
                )
                quantiles = numpy.percentile(param_samples, percentile_lines)
                for quantile, style in zip(
                    quantiles,
                    ["dashed", "solid", "dashed"],
                ):
                    corner_axis.axvline(
                        quantile,
                        color="#e15759",
                        linestyle=style,
                        linewidth=1.2,
                    )
                corner_axis.set_ylabel(
                    "Density",
                    fontsize=font_sizes["label"],
                )
            else:
                x = samples[:, col]
                y = samples[:, row]
                hist, x_edges, y_edges = numpy.histogram2d(
                    x,
                    y,
                    bins=bins,
                    density=True,
                )
                if numpy.allclose(hist, 0.0):
                    corner_axis.scatter(
                        x,
                        y,
                        s=6,
                        alpha=0.25,
                        color="#4e79a7",
                    )
                else:
                    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
                    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
                    filled_levels, line_levels = _build_contour_levels(
                        hist,
                        contour_levels,
                    )
                    corner_axis.contourf(
                        x_centers,
                        y_centers,
                        hist.T,
                        levels=filled_levels,
                        colors=["#dbe9f6", "#afc5e5", "#7da0d4"],
                        alpha=0.9,
                    )
                    corner_axis.contour(
                        x_centers,
                        y_centers,
                        hist.T,
                        levels=line_levels.tolist(),
                        colors="#2a5176",
                        linewidths=1.0,
                    )
                corner_axis.grid(True, alpha=0.3)

            if row == n_params - 1:
                corner_axis.set_xlabel(
                    wrapped_labels[col],
                    fontsize=font_sizes["label"],
                )
            else:
                corner_axis.set_xticklabels([])

            if col == 0:
                corner_axis.set_ylabel(
                    wrapped_labels[row],
                    fontsize=font_sizes["label"],
                )
            elif row != col:
                corner_axis.set_yticklabels([])

            corner_axis.tick_params(
                axis="both",
                labelsize=font_sizes["ticks"],
            )


def _render_corner_with_arviz(
    axes: numpy.ndarray,
    n_params: int,
    wrapped_labels: list[str],
    font_sizes: dict[str, float],
    percentile_lines: tuple[float, float, float],
    samples: numpy.ndarray,
    inference_data: object,
    var_names: list[str],
    bins: int,
) -> None:
    """Render the corner grid using ArviZ while keeping Copernican styling."""

    cmap = ListedColormap(["#dbe9f6", "#afc5e5", "#7da0d4"])
    kde_kwargs = {
        "contour_kwargs": {"colors": "#2a5176", "linewidths": 1.0},
        "contourf_kwargs": {"cmap": cmap, "alpha": 0.9},
        "fill_last": True,
    }
    marginal_kwargs = {
        "kind": "hist",
        "hist_kwargs": {
            "bins": bins,
            "color": "#4e79a7",
            "alpha": 0.7,
            "edgecolor": "white",
            "rwidth": 0.9,
            "density": True,
        },
        "quantiles": [value / 100.0 for value in percentile_lines],
    }

    arviz_module.plot_pair(
        inference_data,
        var_names=var_names,
        kind="kde",
        marginals=True,
        ax=axes,
        textsize=font_sizes["label"],
        kde_kwargs=kde_kwargs,
        marginal_kwargs=marginal_kwargs,
        show=False,
    )

    for row in range(n_params):
        for col in range(n_params):
            if row < col:
                continue
            pair_axis = axes[row, col]
            if row == col:
                pair_axis.set_ylabel(
                    "Density",
                    fontsize=font_sizes["label"],
                )
                quantiles = numpy.percentile(samples[:, row], percentile_lines)
                for quantile, style in zip(
                    quantiles,
                    ["dashed", "solid", "dashed"],
                ):
                    pair_axis.axvline(
                        quantile,
                        color="#e15759",
                        linestyle=style,
                        linewidth=1.2,
                    )
                if row == n_params - 1:
                    pair_axis.set_xlabel(
                        wrapped_labels[col],
                        fontsize=font_sizes["label"],
                    )
                else:
                    pair_axis.set_xticklabels([])
            else:
                if row == n_params - 1:
                    pair_axis.set_xlabel(
                        wrapped_labels[col],
                        fontsize=font_sizes["label"],
                    )
                else:
                    pair_axis.set_xticklabels([])
                if col == 0:
                    pair_axis.set_ylabel(
                        wrapped_labels[row],
                        fontsize=font_sizes["label"],
                    )
                else:
                    pair_axis.set_yticklabels([])
                pair_axis.grid(True, alpha=0.3)
            pair_axis.tick_params(axis="both", labelsize=font_sizes["ticks"])


def _render_single_param_arviz(
    axis_obj: plt.Axes,
    samples: numpy.ndarray,
    label: str,
    font_sizes: dict[str, float],
    bins: int,
    percentile_lines: tuple[float, float, float],
) -> None:
    """Draw a single-parameter histogram using ArviZ."""

    hist_kwargs = {
        "bins": bins,
        "color": "#4e79a7",
        "alpha": 0.7,
        "edgecolor": "white",
        "density": True,
        "rwidth": 0.9,
    }
    quantiles = [value / 100.0 for value in percentile_lines]

    arviz_module.plot_dist(
        samples,
        kind="hist",
        hist_kwargs=hist_kwargs,
        quantiles=quantiles,
        ax=axis_obj,
        show=False,
        textsize=font_sizes["label"],
    )

    quantile_values = numpy.percentile(samples, percentile_lines)
    for quantile, style in zip(
        quantile_values,
        ["dashed", "solid", "dashed"],
    ):
        axis_obj.axvline(
            quantile,
            color="#e15759",
            linestyle=style,
            linewidth=1.2,
        )

    axis_obj.set_xlabel(label, fontsize=font_sizes["label"])
    axis_obj.set_ylabel("Density", fontsize=font_sizes["label"])
    axis_obj.grid(True, alpha=0.3)
    axis_obj.tick_params(axis="both", labelsize=font_sizes["ticks"])


def build_footer_lines(
    data_attrs: dict,
    timestamp: str | None = None,
    *,
    extra_lines: Iterable[tuple[str, bool]] | None = None,
    include_dataset_details: bool = True,
    comparison: ComparisonRequest,
) -> list[tuple[str, bool]]:
    """Return footer lines for a given dataset and model comparison.

    This helper centralises footer construction so each plotting
    function draws from the same template without duplicating string
    formatting logic. ``timestamp`` allows callers to supply a fixed
    time for reproducible tests; when omitted the current time is
    generated.
    """

    control_name, test_name, _ = _comparison_display(comparison)
    base_line = (
        f"{control_name} vs {test_name} | Copernican Suite "
        f"{COPERNICAN_VERSION} | {timestamp or get_timestamp()}"
    )
    composed = compose_footer(base_line, data_attrs)
    if not include_dataset_details and composed:
        # Corner plots speak for the combined posterior rather than a specific
        # observational catalogue.  Drop the dataset description and citation
        # when dataset details are intentionally hidden.  The footer then
        # emphasises the comparison requested by Stage 5 operators.
        composed = [composed[0]]
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
        label_tex: str, numeric_value: Any, *, unit: str | None = None
    ) -> str:
        """Return a readable line or ``N/A`` when ``numeric_value`` is
        non-finite.

        The helper prevents ``:.2f`` formatting from raising when optimisation
        metadata is incomplete. Cosmological fits occasionally leave global
        chi-squared totals undefined, so the summary must degrade gracefully
        instead of crashing the plotting workflow.
        """

        try:
            numeric = float(numeric_value)
        except (TypeError, ValueError):
            return rf"  {label_tex} = N/A"
        if not numpy.isfinite(numeric):
            return rf"  {label_tex} = N/A"
        suffix = f" {unit}" if unit else ""
        formatted = f"{numeric:.2f}"
        return rf"  {label_tex} = {formatted}{suffix}"

    lines.append(r"$\mathbf{Mathematical\ Form:}$")
    for eq_line in getattr(model_plugin, "MODEL_EQUATIONS_LATEX_SN", []):
        lines.append(f"  {_wrap_math(eq_line)}")
    if dataset_type == "bao":
        for eq_line in getattr(model_plugin, "MODEL_EQUATIONS_LATEX_BAO", []):
            lines.append(f"  {_wrap_math(eq_line)}")

    lines.append(r"$\mathbf{Cosmological\ Parameters:}$")
    param_names = getattr(model_plugin, "PARAMETER_NAMES", [])
    param_latex_names = getattr(
        model_plugin,
        "PARAMETER_LATEX_NAMES",
        [],
    )
    fitted_model_params = fit_results.get("fitted_model_params")

    if fitted_model_params:
        for i, name in enumerate(param_names):
            param_value = fitted_model_params.get(name)
            if i < len(param_latex_names):
                latex_name = param_latex_names[i]
            else:
                latex_name = name
            latex_name = _wrap_math(latex_name)
            if param_value is not None:
                lines.append(rf"  {latex_name} = ${param_value:.4g}$")
            else:
                lines.append(rf"  {latex_name} = N/A")
    else:
        lines.append("  (Fit failed or parameters unavailable)")

    if dataset_type == "sne" and fit_results.get("fitted_nuisance_params"):
        lines.append(r"$\mathbf{SNe\ Nuisance\ Parameters:}$")
        for name, nuisance_value in fit_results[
            "fitted_nuisance_params"
        ].items():
            name_latex = {
                "M_B": r"M_B",
                "alpha_salt2": r"\alpha",
                "beta_salt2": r"\beta",
            }.get(name, name)
            lines.append(
                rf"  {_wrap_math(name_latex)} = ${nuisance_value:.4g}$"
            )

    if dataset_type == "sne":
        lines.append(r"$\mathbf{SNe\ Fit\ Statistics:}$")
        chi2_min = fit_results.get("chi2_min", numpy.nan)
        chi2_sne = fit_results.get("chi2_sne", chi2_min)
        lines.append(_format_numeric_line(r"$\chi^2_{SNe}$", chi2_sne))
        if "chi2_total" in fit_results:
            chi2_tot = fit_results.get("chi2_total", numpy.nan)
            lines.append(_format_numeric_line(r"$\chi^2_{tot}$", chi2_tot))
    elif dataset_type == "bao":
        lines.append(r"$\mathbf{BAO\ Fit\ Results:}$")
        lines.append(
            _format_numeric_line(
                r"$r_s$", kwargs.get("rs_Mpc", numpy.nan), unit="Mpc"
            )
        )
        chi2_bao = kwargs.get("chi2_bao", numpy.nan)
        lines.append(_format_numeric_line(r"$\chi^2_{BAO}$", chi2_bao))
        if "chi2_total" in kwargs:
            chi2_tot = kwargs.get("chi2_total", numpy.nan)
            lines.append(_format_numeric_line(r"$\chi^2_{tot}$", chi2_tot))
    elif dataset_type == "cmb":
        lines.append(r"$\mathbf{CMB\ Fit\ Statistics:}$")
        chi2_cmb = kwargs.get("chi2_cmb", numpy.nan)
        lines.append(_format_numeric_line(r"$\chi^2_{CMB}$", chi2_cmb))
        failure = kwargs.get("cmb_failure")
        if failure:
            if isinstance(failure, Mapping):
                category = str(failure.get("category", "cmb_failure"))
                message = str(failure.get("message", ""))
            else:
                category = "cmb_failure"
                message = str(failure)
            lines.append(r"$\mathbf{CMB\ Execution\ Failure:}$")
            lines.append(f"  {category}: {message}")
        if "chi2_total" in kwargs:
            chi2_tot = kwargs.get("chi2_total", numpy.nan)
            lines.append(_format_numeric_line(r"$\chi^2_{tot}$", chi2_tot))

    return "\n".join(lines)


def plot_hubble_diagram(
    sne_data_df: Any,
    control_fit_results: Any,
    test_fit_results: Any,
    control_model_plugin: Any,
    test_model_plugin: Any,
    plot_dir: str = ".",
    timestamp: str | None = None,
    *,
    comparison: ComparisonRequest,
) -> None:
    """Generate and save a Hubble diagram and residuals plot."""
    ensure_dir_exists(plot_dir)
    logger = get_logger()
    control_name, test_name, comparison_name = _comparison_display(comparison)
    control_latex = _latex_model_name(control_name)
    test_latex = _latex_model_name(test_name)
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
            test_fit_results
            if test_fit_results
            and test_fit_results.get("fitted_nuisance_params")
            else control_fit_results
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
        "diag_errors_for_plot", numpy.ones_like(z_data) * 0.2
    )
    covariance_matrix_inv = sne_data_df.attrs.get("covariance_matrix_inv")
    requires_intercept = bool(
        sne_data_df.attrs.get("requires_sne_intercept_marginalization")
    )

    def _apply_sne_intercept(residuals: numpy.ndarray) -> numpy.ndarray:
        """Return residuals after optional dataset-specific intercept fit."""

        if not requires_intercept:
            return residuals
        delta_mu = compute_sne_intercept_delta(
            residuals,
            covariance_matrix_inv=covariance_matrix_inv,
            diag_errors=diag_errors_plot,
        )
        return residuals + delta_mu

    z_plot_smooth = numpy.geomspace(
        max(numpy.min(z_data) * 0.9, 0.001), numpy.max(z_data) * 1.05, 200
    )

    left = 0.08
    right = 0.75
    top = 0.92
    box_height = 0.33
    info_x, info_gap = _info_box_layout(right)

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(17, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1.5], "hspace": 0.05},
    )

    footer_lines = build_footer_lines(
        sne_data_df.attrs,
        timestamp,
        comparison=comparison,
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

    if control_fit_results and control_fit_results.get("success"):
        p_control = list(control_fit_results["fitted_model_params"].values())
        mu_model_control_smooth = control_model_plugin.distance_modulus_model(
            z_plot_smooth, *p_control
        )
        mu_model_control_points = control_model_plugin.distance_modulus_model(
            z_data,
            *p_control,
        )
        res_control = _apply_sne_intercept(
            mu_obs_data - mu_model_control_points
        )
        chi2_control = f"{control_fit_results.get('chi2_min', numpy.nan):.2f}"
        axs[0].plot(
            z_plot_smooth,
            mu_model_control_smooth,
            color="red",
            ls="-",
            label=rf"{control_latex} ($\chi^2$={chi2_control})",
            lw=2.5,
        )
        axs[1].errorbar(
            z_data,
            res_control,
            yerr=diag_errors_plot,
            fmt=".",
            color="red",
            alpha=0.5,
            label=rf"{control_latex} Res.",
            elinewidth=1,
            capsize=2,
            ms=4,
        )
        z_control_avg, res_control_avg = get_binned_average(
            z_data, res_control
        )
        z_control_avg, res_control_avg = _smooth_line(
            z_control_avg,
            res_control_avg,
        )
        axs[1].plot(
            z_control_avg,
            res_control_avg,
            color="darkred",
            ls="-",
            lw=2,
            zorder=10,
            label=rf"Avg. {control_latex} Res.",
        )

    test_name_latex = test_latex
    if test_fit_results and test_fit_results.get("success"):
        fitted_vals = test_fit_results["fitted_model_params"]
        p_test = list(fitted_vals.values())
        mu_model_test_smooth = test_model_plugin.distance_modulus_model(
            z_plot_smooth,
            *p_test,
        )
        mu_model_test_points = test_model_plugin.distance_modulus_model(
            z_data,
            *p_test,
        )
        res_test = _apply_sne_intercept(mu_obs_data - mu_model_test_points)
        chi2_test = f"{test_fit_results.get('chi2_min', numpy.nan):.2f}"
        axs[0].plot(
            z_plot_smooth,
            mu_model_test_smooth,
            color="blue",
            ls="--",
            label=rf"{test_name_latex} ($\chi^2$={chi2_test})",
            lw=2.5,
        )
        axs[1].errorbar(
            z_data,
            res_test,
            yerr=diag_errors_plot,
            fmt=".",
            mfc="none",
            mec="blue",
            ecolor="lightblue",
            alpha=0.5,
            label=rf"{test_name_latex} Res.",
            elinewidth=1,
            capsize=2,
            ms=4,
        )
        z_test_avg, res_test_avg = get_binned_average(z_data, res_test)
        z_test_avg, res_test_avg = _smooth_line(z_test_avg, res_test_avg)
        axs[1].plot(
            z_test_avg,
            res_test_avg,
            color="darkblue",
            ls="--",
            lw=2,
            zorder=11,
            label=f"Avg. {test_name_latex} Res.",
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

    bbox_control = dict(
        boxstyle="round,pad=0.5",
        fc="#FFEEEE",
        ec="darkred",
        alpha=0.8,
    )
    bbox_test = dict(
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
            control_model_plugin,
            "sne",
            control_fit_results,
        ),
        fontsize=font_sizes["infobox"],
        va="top",
        ha="left",
        wrap=True,
        multialignment="left",
        bbox=bbox_control,
    )
    fig.text(
        info_x,
        blue_y,
        format_model_summary_text(
            test_model_plugin,
            "sne",
            test_fit_results,
        ),
        fontsize=font_sizes["infobox"],
        va="top",
        ha="left",
        wrap=True,
        multialignment="left",
        bbox=bbox_test,
    )

    y = start_y
    for idx, (line, is_bold) in enumerate(footer_lines):
        tick_font_size = (
            font_sizes["ticks"] if idx == 0 else font_sizes["ticks"] - 1
        )
        weight = "bold" if is_bold else "normal"
        fig.text(
            0.5,
            y,
            line,
            ha="center",
            fontsize=tick_font_size,
            fontweight=weight,
            wrap=True,
        )
        y -= line_height

    model_comparison_name = comparison_name
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
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error(f"Error saving Hubble diagram: {exc}")
    finally:
        plt.close(fig)


def plot_bao_observables(
    bao_data_df: Any,
    control_results: Any,
    test_results: Any,
    control_model_plugin: Any,
    test_model_plugin: Any,
    sne_data_df: Any,
    plot_dir: str = ".",
    timestamp: str | None = None,
    *,
    comparison: ComparisonRequest,
) -> None:
    """Generate and save a plot of BAO observables versus redshift."""
    ensure_dir_exists(plot_dir)
    logger = get_logger()
    control_name, test_name, comparison_name = _comparison_display(comparison)
    control_latex = _latex_model_name(control_name)
    test_latex = _latex_model_name(test_name)
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
    info_x, info_gap = _info_box_layout(right)

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(17, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1.5], "hspace": 0.05},
    )
    main_axis = axs[0]
    residual_axis = axs[1]

    footer_lines = build_footer_lines(
        bao_data_df.attrs,
        timestamp,
        comparison=comparison,
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
    colors = cmap(numpy.linspace(0, 0.8, len(obs_types)))
    for i, obs_type in enumerate(obs_types):
        subset = bao_data_df[bao_data_df["observable_type"] == obs_type]
        label = f"Data: {obs_type.replace('_', '/')}"
        main_axis.errorbar(
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
            z_vals: numpy.ndarray,
            y_vals: numpy.ndarray,
            **kwargs: Any,
        ) -> None:
            """Plot only valid data points to avoid runtime warnings."""
            valid_indices = numpy.isfinite(z_vals) & numpy.isfinite(y_vals)
            if numpy.any(valid_indices):
                main_axis.plot(
                    z_vals[valid_indices], y_vals[valid_indices], **kwargs
                )
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
    plot_model_bao(control_results, "red", line_styles, control_latex)

    test_name_latex = test_latex
    plot_model_bao(
        test_results, "blue", line_styles, test_name_latex, alpha=0.25
    )

    # --- Residuals ---
    all_res = []
    val_data = bao_data_df["value"].values
    control_pred = control_results.get("pred_df")
    if control_pred is not None:
        res_control = val_data - control_pred["model_prediction"].values
        all_res.append(res_control)
        residual_axis.errorbar(
            bao_data_df["redshift"],
            res_control,
            yerr=bao_data_df["error"],
            fmt=".",
            color="red",
            alpha=0.5,
            label=rf"{control_latex} Res.",
            elinewidth=1,
            capsize=2,
            ms=4,
        )
        z_avg, r_avg = get_binned_average(
            bao_data_df["redshift"].values,
            res_control,
        )
        z_avg, r_avg = _smooth_line(z_avg, r_avg)
        residual_axis.plot(
            z_avg,
            r_avg,
            color="darkred",
            ls="-",
            lw=2,
            zorder=10,
            label=rf"Avg. {control_latex} Res.",
        )

    test_pred = test_results.get("pred_df")
    if test_pred is not None:
        res_test = val_data - test_pred["model_prediction"].values
        all_res.append(res_test)
        residual_axis.errorbar(
            bao_data_df["redshift"],
            res_test,
            yerr=bao_data_df["error"],
            fmt=".",
            mfc="none",
            mec="blue",
            ecolor="lightblue",
            alpha=0.5,
            label=rf"{test_name_latex} Res.",
            elinewidth=1,
            capsize=2,
            ms=7,
        )
        z_avg, r_avg = get_binned_average(
            bao_data_df["redshift"].values,
            res_test,
        )
        z_avg, r_avg = _smooth_line(z_avg, r_avg)
        residual_axis.plot(
            z_avg,
            r_avg,
            color="darkblue",
            ls="--",
            lw=2,
            zorder=11,
            label=f"Avg. {test_name_latex} Res.",
        )

    if all_res:
        max_res = numpy.nanmax(numpy.abs(numpy.concatenate(all_res)))
        if numpy.isfinite(max_res) and max_res > 0:
            residual_axis.set_ylim(-1.2 * max_res, 1.2 * max_res)

    main_axis.set_ylabel(r"$D_X/r_s$", fontsize=font_sizes["label"])
    main_axis.set_title(
        f"BAO Observables vs. Redshift: {dataset_name}",
        fontsize=font_sizes["title"],
    )
    main_axis.legend(fontsize=font_sizes["legend"], loc="best")
    main_axis.minorticks_on()
    main_axis.tick_params(
        axis="both",
        which="major",
        labelsize=font_sizes["ticks"],
    )
    main_axis.grid(
        True,
        which="both",
        color="#E0E0E0",
        linestyle="-",
        linewidth=0.5,
    )

    residual_axis.axhline(0, color="black", ls="--", lw=1)
    residual_axis.set_xlabel("Redshift (z)", fontsize=font_sizes["label"])
    residual_axis.set_ylabel(
        r"$D_X/r_s^{obs} - D_X/r_s^{th}$",
        fontsize=font_sizes["label"],
    )
    residual_axis.legend(fontsize=font_sizes["legend"], loc="best")
    residual_axis.minorticks_on()
    residual_axis.tick_params(
        axis="both",
        which="major",
        labelsize=font_sizes["ticks"],
    )
    residual_axis.grid(
        True,
        which="both",
        color="#E0E0E0",
        linestyle="-",
        linewidth=0.5,
    )

    bbox_control = dict(
        boxstyle="round,pad=0.5",
        fc="#FFEEEE",
        ec="darkred",
        alpha=0.8,
    )
    bbox_test = dict(
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
            control_model_plugin,
            "bao",
            control_results.get("sne_fit_results", {}),
            **control_results,
        ),
        fontsize=font_sizes["infobox"],
        va="top",
        ha="left",
        wrap=True,
        multialignment="left",
        bbox=bbox_control,
    )
    fig.text(
        info_x,
        blue_y,
        format_model_summary_text(
            test_model_plugin,
            "bao",
            test_results.get("sne_fit_results", {}),
            **test_results,
        ),
        fontsize=font_sizes["infobox"],
        va="top",
        ha="left",
        wrap=True,
        multialignment="left",
        bbox=bbox_test,
    )

    y = start_y
    for idx, (line, is_bold) in enumerate(footer_lines):
        tick_font_size = (
            font_sizes["ticks"] if idx == 0 else font_sizes["ticks"] - 1
        )
        weight = "bold" if is_bold else "normal"
        fig.text(
            0.5,
            y,
            line,
            ha="center",
            fontsize=tick_font_size,
            fontweight=weight,
            wrap=True,
        )
        y -= line_height

    model_comparison_name = comparison_name
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
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error(f"Error saving BAO plot: {exc}")
    finally:
        plt.close(fig)


def plot_cmb_spectrum(
    cmb_data_df: Any,
    control_cmb_results: Any,
    test_cmb_results: Any,
    control_sne_results: Any,
    test_sne_results: Any,
    control_model_plugin: Any,
    test_model_plugin: Any,
    plot_dir: str = ".",
    timestamp: str | None = None,
    *,
    comparison: ComparisonRequest,
) -> None:
    """Generate and save a CMB power spectrum plot with residuals."""
    ensure_dir_exists(plot_dir)
    logger = get_logger()
    control_name, test_name, comparison_name = _comparison_display(comparison)
    control_latex = _latex_model_name(control_name)
    test_latex = _latex_model_name(test_name)
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

    blocks = cmb_observation_blocks(cmb_data_df)
    if not blocks:
        logger.warning("CMB data does not expose any observable spectra.")
        return
    diag_errors_plot = None
    if "covariance_matrix_inv" in cmb_data_df.attrs:
        try:
            cov = numpy.linalg.inv(cmb_data_df.attrs["covariance_matrix_inv"])
            diag_errors_plot = numpy.sqrt(numpy.diag(cov))
        except (
            FloatingPointError,
            numpy.linalg.LinAlgError,
            RuntimeError,
            ValueError,
        ) as exc:
            logger.warning(
                f"Could not derive CMB errors from covariance: {exc}",
            )
            diag_errors_plot = numpy.ones(len(cmb_data_df), dtype=float)
    else:
        diag_errors_plot = numpy.ones(len(cmb_data_df), dtype=float)

    left = 0.08
    right = 0.75
    top = 0.92
    box_height = 0.33
    info_x, info_gap = _info_box_layout(right)

    fig, axs = plt.subplots(
        len(blocks) * 2,
        1,
        figsize=(17, 6 * len(blocks)),
        sharex=True,
        gridspec_kw={
            "height_ratios": [4, 1.5] * len(blocks),
            "hspace": 0.25,
        },
    )
    axs = numpy.atleast_1d(axs)

    footer_lines = build_footer_lines(
        cmb_data_df.attrs,
        timestamp,
        extra_lines=[(f"CMB execution: {CCMBS_LABEL}.", False)],
        comparison=comparison,
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

    control_theory = None
    test_theory = None
    if control_cmb_results:
        control_theory = control_cmb_results.get("theory_spectrum")
    if test_cmb_results:
        test_theory = test_cmb_results.get("theory_spectrum")

    test_name_latex = test_latex

    for i, block in enumerate(blocks):
        idx_main = i * 2
        idx_res = idx_main + 1
        metadata = block.metadata
        order = numpy.argsort(block.ells, kind="stable")
        ells = block.ells[order]
        obs = block.observed[order]
        if block.observed_column == "Dl_obs":
            err = diag_errors_plot[block.row_indices][order]
        else:
            error_column = f"e_{metadata.base_spectrum.lower()}_obs"
            err = numpy.asarray(
                cmb_data_df.get(
                    error_column,
                    numpy.ones(len(cmb_data_df), dtype=float),
                ),
                dtype=float,
            )[block.row_indices][order]

        def _theory_values(theory):
            """Return this block's theory values in plotting order."""

            if theory is None:
                return None
            try:
                values = cmb_theory_values_for_block(
                    theory,
                    block,
                    total_row_count=len(cmb_data_df),
                )
            except (KeyError, TypeError, ValueError):
                return None
            return values[order]

        control_values = _theory_values(control_theory)
        test_values = _theory_values(test_theory)

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

        if control_values is not None:
            chi2_control = (
                f"{control_cmb_results.get('chi2_cmb', numpy.nan):.2f}"
                if i == 0
                else ""
            )
            label = control_latex + (
                rf" ($\chi^2$={chi2_control})" if chi2_control else ""
            )
            axs[idx_main].plot(
                ells,
                control_values,
                color="red",
                ls="-",
                lw=2.0,
                label=label,
            )
            if metadata.base_spectrum in {"BB", "EE", "PP", "TT"}:
                cosmic_variance = (
                    numpy.sqrt(2.0 / (2 * ells + 1.0)) * control_values
                )
                axs[idx_main].fill_between(
                    ells,
                    control_values - cosmic_variance,
                    control_values + cosmic_variance,
                    color="red",
                    alpha=0.1,
                    label="Cosmic var.",
                    zorder=0,
                )
            residuals = obs - control_values
            axs[idx_res].errorbar(
                ells,
                residuals,
                yerr=err,
                fmt=".",
                color="red",
                alpha=0.5,
                label=f"{control_latex} Res.",
                elinewidth=1,
                capsize=2,
                ms=4,
            )
            z_avg, r_avg = get_binned_average(ells, residuals)
            z_avg, r_avg = _smooth_line(z_avg, r_avg)
            axs[idx_res].plot(
                z_avg,
                r_avg,
                color="darkred",
                ls="-",
                lw=2,
                zorder=10,
                label=f"Avg. {control_latex} Res.",
            )

        if test_values is not None:
            chi2_test = (
                f"{test_cmb_results.get('chi2_cmb', numpy.nan):.2f}"
                if i == 0
                else ""
            )
            label = rf"{test_name_latex}" + (
                rf" ($\chi^2$={chi2_test})" if chi2_test else ""
            )
            axs[idx_main].plot(
                ells,
                test_values,
                color="blue",
                ls="--",
                lw=2.0,
                label=label,
            )
            residuals = obs - test_values
            axs[idx_res].errorbar(
                ells,
                residuals,
                yerr=err,
                fmt=".",
                mfc="none",
                mec="blue",
                ecolor="lightblue",
                alpha=0.5,
                label=rf"{test_name_latex} Res.",
                elinewidth=1,
                capsize=2,
                ms=4,
            )
            z_avg, r_avg = get_binned_average(ells, residuals)
            z_avg, r_avg = _smooth_line(z_avg, r_avg)
            axs[idx_res].plot(
                z_avg,
                r_avg,
                color="darkblue",
                ls="--",
                lw=2,
                zorder=11,
                label=f"Avg. {test_name_latex} Res.",
            )

        ylabel = (
            r"$D_\ell$ (dimensionless)"
            if metadata.units == "dimensionless"
            else r"$D_\ell\ (\mu K^2)$"
        )
        axs[idx_main].set_ylabel(ylabel, fontsize=font_sizes["label"])
        plotted_values = [obs]
        if control_values is not None:
            plotted_values.append(control_values)
        if test_values is not None:
            plotted_values.append(test_values)
        if metadata.base_spectrum in {"BB", "EE", "PP", "TT"} and all(
            numpy.any(numpy.isfinite(values))
            and numpy.all(values[numpy.isfinite(values)] > 0.0)
            for values in plotted_values
        ):
            axs[idx_main].set_yscale("log")
        axs[idx_main].legend(fontsize=font_sizes["legend"], loc="best")
        # Reduce padding so titles fit in the vertical gaps between
        # spectrum and residual panels without overlapping.
        title_pad = 6
        axs[idx_main].set_title(
            f"CMB {metadata.canonical_name} Power Spectrum: {dataset_name}",
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
        if i == len(blocks) - 1:
            axs[idx_res].set_xlabel(
                r"Multipole $\ell$",
                fontsize=font_sizes["label"],
            )
        axs[idx_res].set_ylabel(
            r"$D_\ell^{obs} - D_\ell^{th}$", fontsize=font_sizes["label"]
        )
        axs[idx_res].legend(fontsize=font_sizes["legend"], loc="best")
        axs[idx_res].minorticks_on()
        axs[idx_res].tick_params(
            axis="both", which="major", labelsize=font_sizes["ticks"]
        )
        axs[idx_res].grid(
            True, which="both", color="#E0E0E0", linestyle="-", linewidth=0.5
        )

    bbox_control = dict(
        boxstyle="round,pad=0.5",
        fc="#FFEEEE",
        ec="darkred",
        alpha=0.8,
    )
    bbox_test = dict(
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
            control_model_plugin,
            "cmb",
            control_sne_results,
            chi2_cmb=control_cmb_results.get("chi2_cmb"),
            chi2_total=control_sne_results.get("chi2_total"),
        ),
        fontsize=font_sizes["infobox"],
        va="top",
        ha="left",
        wrap=True,
        multialignment="left",
        bbox=bbox_control,
    )
    fig.text(
        info_x,
        blue_y,
        format_model_summary_text(
            test_model_plugin,
            "cmb",
            test_sne_results,
            chi2_cmb=test_cmb_results.get("chi2_cmb"),
            chi2_total=test_sne_results.get("chi2_total"),
        ),
        fontsize=font_sizes["infobox"],
        va="top",
        ha="left",
        wrap=True,
        multialignment="left",
        bbox=bbox_test,
    )

    y = start_y
    for idx, (line, is_bold) in enumerate(footer_lines):
        tick_font_size = (
            font_sizes["ticks"] if idx == 0 else font_sizes["ticks"] - 1
        )
        weight = "bold" if is_bold else "normal"
        fig.text(
            0.5,
            y,
            line,
            ha="center",
            fontsize=tick_font_size,
            fontweight=weight,
            wrap=True,
        )
        y -= line_height

    model_comparison_name = comparison_name
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
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error(f"Error saving CMB plot: {exc}")
    finally:
        plt.close(fig)


def plot_corner(
    posterior_samples: numpy.ndarray,
    model_plugin: Any,
    data_attrs: dict[str, Any] | None,
    plot_dir: str = ".",
    parameter_names: list[str] | None = None,
    timestamp: str | None = None,
    *,
    comparison: ComparisonRequest,
) -> None:
    """Generate a corner plot for the joint posterior samples.

    Parameters
    ----------
    posterior_samples:
        Sampler output arranged as ``(n_steps, n_walkers, n_params)`` or a
        two-dimensional ``(n_samples, n_params)`` array. Samples may contain
        NaNs or infinities; these are dropped before plotting to avoid
        distorting the marginal distributions.
    model_plugin:
        Plugin describing the model whose posterior is being rendered in the
        selected comparison.
        The helper inspects ``PARAMETER_NAMES`` and ``PARAMETER_LATEX_NAMES``
        when axis labels are not supplied explicitly. The model name also
        supplies the posterior parameter labels; ``comparison`` drives the
        filename and footer provenance so plots tie back to the full run.
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
    comparison:
        Required control/test model pair used for labels, filenames, and
        footer provenance.
    """

    ensure_dir_exists(plot_dir)
    logger = get_logger()
    logger.info("Generating corner plot for posterior samples...")
    attrs = data_attrs or {}

    default_names = list(getattr(model_plugin, "PARAMETER_NAMES", []))
    # Store the LaTeX-friendly labels separately so that axis rendering falls
    # back to readable parameter names when the plugin omits a mapping.
    label_candidates = list(
        getattr(model_plugin, "PARAMETER_LATEX_NAMES", []),
    )
    effective_names = parameter_names or default_names

    samples, labels, stats = _prepare_corner_inputs(
        posterior_samples,
        effective_names,
    )
    n_params = samples.shape[1]

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

    model_name = getattr(model_plugin, "MODEL_NAME", "Model")
    extra_lines = _format_corner_footer_stats(stats)
    if arviz_module is not None:
        extra_lines.append(
            ("Corner densities rendered via ArviZ backend.", False)
        )
    footer_lines = build_footer_lines(
        attrs,
        timestamp,
        extra_lines=extra_lines,
        include_dataset_details=False,
        comparison=comparison,
    )

    _apply_common_style()
    (
        figsize,
        font_sizes,
        line_height,
        margins,
        footer_padding,
    ) = _compute_corner_layout(
        n_params,
        len(footer_lines),
    )

    def _ensure_corner_backend() -> None:
        """Switch to a headless backend before creating the corner figure.

        Windows CI runners routinely ship without a working Tk installation.
        Matplotlib raises ``TclError`` as soon as it tries to open a window in
        that environment.  Probing the current backend with a temporary figure
        keeps the main ``plt.subplots`` call to a single execution—matching the
        regression tests' expectations—while retaining interactive backends on
        developer machines that can open GUI windows.
        """

        backend = plt.get_backend()
        if not isinstance(backend, str):
            # Matplotlib uses an internal auto-backend sentinel before the
            # first figure is created. Touching ``plt.figure`` in that state
            # would trigger a background switch that tests monkeypatch as a
            # user-visible backend change. Skipping the warm-up when no
            # concrete backend is resolved preserves the recorded switch list
            # while still allowing Tk-dependent backends to be probed later.
            return
        try:
            probe_fig = plt.figure()
            plt.close(probe_fig)
        except (AttributeError, OSError, RuntimeError, ValueError) as error:
            logger.warning(
                "Corner plot backend %s failed (%s); forcing Agg fallback.",
                backend,
                error,
            )
            plt.switch_backend("Agg")

    def _build_corner_figure():
        """Render the corner plot grid after validating the backend.

        Some CI hosts default to TkAgg without shipping a working Tk stack.
        Warming the backend with a temporary figure lets the helper detect GUI
        failures before building the grid. If ``plt.subplots`` still raises
        ``TclError``, the helper switches to Agg and retries so callers observe
        one failure followed by a deterministic headless re-render.
        """

        _ensure_corner_backend()
        try:
            return plt.subplots(
                n_params,
                n_params,
                figsize=figsize,
            )
        except tkinter.TclError as error:
            logger.warning(
                "Corner plot backend %s failed (%s); forcing Agg fallback.",
                plt.get_backend(),
                error,
            )
            plt.switch_backend("Agg")
            return plt.subplots(
                n_params,
                n_params,
                figsize=figsize,
            )

    # Each dimension receives its own row and column, mirroring the familiar
    # triangle plot layout popularised by corner.py while letting us reuse the
    # Copernican Suite's styling helpers and footers. The geometry is
    # derived from ``_compute_corner_layout`` so the panels resize gracefully
    # as the dimensionality grows while remaining within a manageable figure.
    fig, axes = _build_corner_figure()
    if isinstance(axes, numpy.ndarray):
        axes_array = numpy.asarray(axes)
    else:
        axes_array = numpy.array([[axes]])
    if axes_array.ndim != 2:
        axes_array = axes_array.reshape((n_params, n_params))

    bins = max(25, int(numpy.sqrt(samples.shape[0]) // 2))
    percentile_lines = (16.0, 50.0, 84.0)
    contour_levels = (0.68, 0.95)

    if arviz_module is not None:
        try:
            if n_params == 1:
                _render_single_param_arviz(
                    axes_array[0, 0],
                    samples[:, 0],
                    wrapped_labels[0],
                    font_sizes,
                    bins,
                    percentile_lines,
                )
            else:
                inference_data, var_names = _build_arviz_inference_data(
                    samples,
                    labels,
                )
                _render_corner_with_arviz(
                    axes_array,
                    n_params,
                    wrapped_labels,
                    font_sizes,
                    percentile_lines,
                    samples,
                    inference_data,
                    var_names,
                    bins,
                )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover - ArviZ-specific failures
            logger.warning(
                "ArviZ corner rendering failed (%s); falling back "
                "to the Matplotlib grid.",
                exc,
            )
            _render_corner_grid_matplotlib(
                axes_array,
                n_params,
                samples,
                wrapped_labels,
                font_sizes,
                bins,
                percentile_lines,
                contour_levels,
            )
    else:
        _render_corner_grid_matplotlib(
            axes_array,
            n_params,
            samples,
            wrapped_labels,
            font_sizes,
            bins,
            percentile_lines,
            contour_levels,
        )

    fig.suptitle(
        f"Posterior corner plot: {model_name}",
        fontsize=font_sizes["title"],
        y=_CORNER_TITLE_Y,
    )

    plt.subplots_adjust(**margins)

    footer_bottom = margins["bottom"]

    footer_stack_offset = max(len(footer_lines) - 1, 0) * line_height
    y = footer_bottom - footer_padding - footer_stack_offset
    lowest_line = (
        y - (len(footer_lines) - 1) * line_height if footer_lines else y
    )
    if lowest_line < _CORNER_FOOTER_CLEARANCE - 1e-6:
        y += _CORNER_FOOTER_CLEARANCE - lowest_line

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
    model_token = comparison_slug(comparison)
    filename = generate_filename(
        "corner-plot",
        dataset_id,
        "png",
        model_name=model_token,
        timestamp=timestamp,
    )

    try:
        plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        logger.info(f"Corner plot saved to {filename}")
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover - log path only
        logger.error(f"Error saving corner plot: {exc}")
    finally:
        plt.close(fig)


def plot_parameter_histograms(
    posterior_samples: numpy.ndarray,
    model_plugin: Any,
    data_attrs: dict[str, Any] | None,
    plot_dir: str = ".",
    parameter_names: list[str] | None = None,
    timestamp: str | None = None,
    *,
    comparison: ComparisonRequest,
) -> None:
    """Generate per-parameter histograms using ArviZ.

    ``comparison`` supplies the control/test identities used in the output
    filename and rendered provenance.
    """

    ensure_dir_exists(plot_dir)
    logger = get_logger()
    logger.info("Generating parameter histograms for posterior samples...")
    attrs = data_attrs or {}

    default_names = list(getattr(model_plugin, "PARAMETER_NAMES", []))
    label_candidates = list(getattr(model_plugin, "PARAMETER_LATEX_NAMES", []))
    effective_names = parameter_names or default_names

    samples, labels, stats = _prepare_corner_inputs(
        posterior_samples,
        effective_names,
    )
    n_params = samples.shape[1]

    wrapped_labels: list[str] = []
    for idx in range(n_params):
        latex_name = (
            label_candidates[idx] if idx < len(label_candidates) else None
        )
        label = latex_name or labels[idx]
        wrapped_labels.append(_wrap_math(label))

    model_label = getattr(model_plugin, "MODEL_NAME", "model")

    bins = max(25, int(numpy.sqrt(samples.shape[0]) // 2))
    percentile_lines = (16.0, 50.0, 84.0)

    max_columns = 3
    columns = min(n_params, max_columns)
    columns = max(1, columns)
    rows = max(1, math.ceil(n_params / columns))

    base_panel = 3.3
    width = min(max(columns * base_panel, base_panel), 12.0)
    height = min(max(rows * base_panel, base_panel), 12.0)

    _apply_common_style()
    fig, axes = plt.subplots(
        rows, columns, figsize=(width, height), squeeze=False
    )
    axes_flat = axes.flatten()

    for idx in range(rows * columns):
        param_axis = axes_flat[idx]
        if idx >= n_params:
            param_axis.axis("off")
            continue

        values = samples[:, idx]
        hist_kwargs = {
            "bins": bins,
            "color": "#4e79a7",
            "alpha": 0.7,
            "edgecolor": "white",
            "density": True,
            "rwidth": 0.9,
        }
        if arviz_module is not None:
            arviz_module.plot_dist(
                values,
                kind="hist",
                hist_kwargs=hist_kwargs,
                quantiles=[value / 100.0 for value in percentile_lines],
                ax=param_axis,
                show=False,
                textsize=12,
            )
        else:
            logger.warning(
                "ArviZ unavailable; falling back to Matplotlib histograms."
            )
            param_axis.hist(values, **hist_kwargs)

        quantiles = numpy.percentile(values, percentile_lines)
        for quantile, style in zip(
            quantiles,
            ["dashed", "solid", "dashed"],
        ):
            param_axis.axvline(
                quantile,
                color="#e15759",
                linestyle=style,
                linewidth=1.2,
            )

        param_axis.set_title(
            wrapped_labels[idx],
            fontsize=14,
        )
        param_axis.set_ylabel(
            "Density",
            fontsize=14,
        )
        param_axis.grid(True, alpha=0.3)
        param_axis.tick_params(labelsize=10)

    plt.subplots_adjust(
        left=0.06,
        right=0.78,
        top=0.92,
        bottom=0.18,
        hspace=0.45,
        wspace=0.35,
    )

    info_text_parts = [
        f"Model: {model_label}",
        f"Dataset: {attrs.get('dataset_name', 'Joint posterior')}",
    ]
    readable_params = default_names or labels
    if readable_params:
        info_text_parts.append(
            f"Parameters: {', '.join(readable_params[:5])}"
            + ("…" if len(readable_params) > 5 else "")
        )
    bbox = dict(
        boxstyle="round,pad=0.5", fc="#F5F5F5", ec="#9C9C9C", alpha=0.95
    )
    fig.text(
        0.82,
        0.85,
        "\n".join(info_text_parts),
        fontsize=10,
        va="top",
        ha="left",
        bbox=bbox,
        wrap=True,
    )

    extra_lines = [
        (
            f"Histograms use {stats['processed_count']:,} samples "
            f"from {stats['finite_count']:,} finite draws.",
            False,
        ),
        ("Parameter histograms rendered via ArviZ backend.", False),
    ]
    if stats.get("downsampled"):
        extra_lines.append(
            (
                "Automatic thinning applied to satisfy MAX_CORNER_SAMPLES.",
                False,
            )
        )

    footer_lines = build_footer_lines(
        attrs,
        timestamp,
        extra_lines=extra_lines,
        include_dataset_details=True,
        comparison=comparison,
    )

    footer_padding = 0.04
    line_height = 0.018
    footer_bottom = 0.12
    y = footer_bottom - footer_padding
    lowest_line = (
        y - (len(footer_lines) - 1) * line_height if footer_lines else y
    )
    if lowest_line < 0.02:
        y += 0.02 - lowest_line

    for idx, (line, is_bold) in enumerate(footer_lines):
        weight = "bold" if is_bold else "normal"
        fig.text(
            0.5,
            y - idx * line_height,
            line,
            ha="center",
            fontsize=10,
            fontweight=weight,
            wrap=True,
        )

    fig.suptitle(
        f"Parameter histograms: {model_label}",
        fontsize=22,
    )

    dataset_id = attrs.get("dataset_id", "joint")
    model_token = comparison_slug(comparison)
    filename = generate_filename(
        "parameter-histograms",
        dataset_id,
        "png",
        model_name=model_token,
        timestamp=timestamp,
    )

    try:
        plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        logger.info(f"Parameter histograms saved to {filename}")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error(f"Error saving parameter histograms: {exc}")
    finally:
        plt.close(fig)
