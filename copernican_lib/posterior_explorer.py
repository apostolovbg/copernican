"""Posterior exploration helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import analysis

try:
    import arviz as az
except ModuleNotFoundError:
    az = None


def find_posterior_files(run_dir: Path | str) -> list[Path]:
    """Return every posterior NetCDF file saved inside a run directory."""

    directory = Path(run_dir)
    return sorted(directory.glob("posterior-*.nc"))


def load_inference_data(path: Path | str) -> xr.Dataset:
    """Load the posterior data from disk via ArviZ or xarray."""

    path = Path(path)
    if az is not None:
        try:
            return az.from_netcdf(str(path)).posterior
        except Exception:
            pass
    return xr.open_dataset(path, engine="scipy")


def extract_posterior_arrays(dataset: xr.Dataset) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, da in dataset.data_vars.items():
        if da.values is None:
            continue
        arr = np.asarray(da.values).reshape(-1)
        if arr.size == 0:
            continue
        arrays[name] = arr
    return arrays


def flatten_posterior_arrays(
    dataset: xr.Dataset,
) -> tuple[np.ndarray, list[str]]:
    arrays = extract_posterior_arrays(dataset)
    if not arrays:
        return np.empty((0, 0)), []
    names = list(arrays.keys())
    stacked = np.column_stack([arrays[name] for name in names])
    return stacked, names


def _footer_lines(
    result: analysis.RunAnalysisResult, posterior_path: Path
) -> list[tuple[str, bool]]:
    lines: list[tuple[str, bool]] = []
    model_names = list(result.model_summaries.keys())
    if model_names:
        lines.append((f"{model_names[0]} posterior overview", True))
    name = next(
        iter(result.datasets.values()), {"name": "Joint posterior"}
    ).get("name", "Dataset")
    lines.append((f"{name}", False))
    lines.append((f"File: {posterior_path.name}", False))
    return lines


def _render_footer(fig: plt.Figure, lines: list[tuple[str, bool]]) -> None:
    y = 0.02
    for line, bold in lines:
        fig.text(
            0.5,
            y,
            line,
            ha="center",
            fontsize=10,
            fontweight="bold" if bold else "normal",
        )
        y -= 0.015


def create_posterior_overview_figure(
    result: analysis.RunAnalysisResult,
    posterior_path: Path,
    *,
    limit_parameters: int = 4,
) -> plt.Figure:
    """Build a compact posterior overview figure from NetCDF output."""

    dataset = load_inference_data(posterior_path)
    arrays = extract_posterior_arrays(dataset)
    if not arrays:
        raise RuntimeError("No posterior variables found in the file.")

    selected = list(arrays.keys())[:limit_parameters]
    fig = plt.Figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.35)
    trace_ax = fig.add_subplot(gs[0])
    hist_ax = fig.add_subplot(gs[1])

    for idx, name in enumerate(selected):
        values = arrays[name]
        trace_ax.plot(
            values,
            label=name,
            linewidth=1,
            alpha=0.9 if idx == 0 else 0.65,
        )
    trace_ax.set_title("Parameter traces")
    trace_ax.set_ylabel("Value")
    trace_ax.grid(True, linestyle=":", alpha=0.6)
    trace_ax.legend(loc="upper right", fontsize="small")

    first_param = selected[0]
    hist_ax.hist(
        arrays[first_param],
        bins=min(32, len(arrays[first_param]) // 2 + 1),
        color="#465775",
        alpha=0.8,
    )
    hist_ax.set_title(f"Marginal: {first_param}")
    hist_ax.set_ylabel("Frequency")
    hist_ax.set_xlabel("Value")
    hist_ax.grid(True, linestyle=":", alpha=0.5)

    footer = _footer_lines(result, posterior_path)
    _render_footer(fig, footer)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig


__all__ = [
    "find_posterior_files",
    "load_inference_data",
    "extract_posterior_arrays",
    "flatten_posterior_arrays",
    "create_posterior_overview_figure",
]
