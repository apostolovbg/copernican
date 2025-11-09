"""Unit tests for :mod:`copernican_lib.plotter`.

These tests ensure summary formatting tolerates incomplete optimisation
metadata. Plot generation uses the same helper, so catching regressions here
prevents GUI failures during post-processing.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

from copernican_lib import plotter
from copernican_lib import utils as plot_utils


class _DummyPlugin:
    """Lightweight stand-in exposing the attributes the plotter expects."""

    MODEL_NAME = "TestModel"
    MODEL_EQUATIONS_LATEX_SN: list[str] = []
    MODEL_EQUATIONS_LATEX_BAO: list[str] = []
    PARAMETER_NAMES: list[str] = []
    PARAMETER_LATEX_NAMES: list[str] = []


class _CornerPlugin(_DummyPlugin):
    """Extended dummy plugin carrying names for the corner plot."""

    PARAMETER_NAMES = ["alpha", "beta", "gamma"]
    PARAMETER_LATEX_NAMES = [r"\alpha", r"\beta", r"\gamma"]


def test_format_model_summary_text_handles_missing_chi2_total() -> None:
    """Ensure missing totals render as ``N/A`` instead of raising errors."""

    fit_results = {
        "fitted_cosmological_params": types.MappingProxyType({}),
        "chi2_min": None,
    }

    summary = plotter.format_model_summary_text(
        _DummyPlugin,
        "cmb",
        fit_results,
        chi2_cmb=None,
        chi2_total=None,
    )

    assert "$\\chi^2_{tot}$ = N/A" in summary
    assert "$\\chi^2_{CMB}$ = N/A" in summary


@pytest.mark.parametrize(
    "value,expected_fragment",
    [
        (42.0, "$\\chi^2_{SNe}$ = 42.00"),
        (float("nan"), "$\\chi^2_{SNe}$ = N/A"),
    ],
)
def test_format_model_summary_text_numeric_rendering(
    value: float, expected_fragment: str
) -> None:
    """Verify numeric chi-squared statistics include two decimal places."""

    fit_results = {
        "fitted_cosmological_params": types.MappingProxyType({}),
        "chi2_min": value,
        "chi2_sne": value,
        "chi2_total": value,
    }

    summary = plotter.format_model_summary_text(
        _DummyPlugin,
        "sne",
        fit_results,
    )

    assert expected_fragment in summary
    if expected_fragment.endswith("N/A"):
        assert "$\\chi^2_{tot}$ = N/A" in summary
    else:
        assert "$\\chi^2_{tot}$ = 42.00" in summary


def test_plot_corner_renders_expected_file(tmp_path) -> None:
    """Ensure the new corner plot helper writes a PNG with suite styling."""

    samples = np.zeros((10, 4, 3))
    samples[:, :, 0] = np.linspace(0.0, 1.0, 10)[:, None]
    samples[:, :, 1] = np.linspace(-1.0, 1.0, 10)[:, None]
    samples[:, :, 2] = 0.5

    attrs = {
        "dataset_id": "joint_posterior",
        "dataset_name": "Joint posterior",
        "description": "Synthetic check",
        "citation": "Corner validation stub",
    }

    timestamp = "20251108_000000"
    plotter.plot_corner(
        samples,
        _CornerPlugin,
        attrs,
        plot_dir=str(tmp_path),
        timestamp=timestamp,
    )

    expected_name = plot_utils.generate_filename(
        "corner-plot",
        "joint_posterior",
        "png",
        model_name="vs-TestModel",
        timestamp=timestamp,
    )
    assert (tmp_path / expected_name).exists()


def test_plot_corner_scales_layout_with_dimension(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify responsive geometry shrinks panels and fonts for larger grids."""

    rng = np.random.default_rng(4)
    small_samples = rng.normal(size=(6, 2, 2))
    large_samples = rng.normal(size=(6, 2, 6))

    class _SixParamPlugin(_CornerPlugin):
        PARAMETER_NAMES = [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
        ]
        PARAMETER_LATEX_NAMES = [
            r"\alpha",
            r"\beta",
            r"\gamma",
            r"\delta",
            r"\epsilon",
            r"\zeta",
        ]

    attrs = {
        "dataset_id": "joint_posterior",
        "dataset_name": "Joint posterior",
        "description": "Responsive layout check",
        "citation": "Corner validation stub",
    }

    recorded_figsizes: list[tuple[float, float] | None] = []
    layout_calls: list[
        tuple[
            int,
            int,
            tuple[
                tuple[float, float],
                dict[str, float],
                float,
                dict[str, float],
            ],
        ]
    ] = []

    original_subplots = plotter.plt.subplots

    def _recording_subplots(*args: Any, **kwargs: Any):
        if "figsize" in kwargs:
            recorded_figsizes.append(kwargs["figsize"])
        elif len(args) >= 3:
            recorded_figsizes.append(args[2])
        else:
            recorded_figsizes.append(None)
        return original_subplots(*args, **kwargs)

    monkeypatch.setattr(plotter.plt, "subplots", _recording_subplots)

    original_layout = plotter._compute_corner_layout

    def _recording_layout(n_params: int, footer_line_count: int):
        layout = original_layout(n_params, footer_line_count)
        layout_calls.append((n_params, footer_line_count, layout))
        return layout

    monkeypatch.setattr(plotter, "_compute_corner_layout", _recording_layout)

    plotter.plot_corner(
        small_samples,
        _SixParamPlugin,
        attrs,
        plot_dir=str(tmp_path),
        timestamp="20251108_000000",
    )

    plotter.plot_corner(
        large_samples,
        _SixParamPlugin,
        attrs,
        plot_dir=str(tmp_path),
        timestamp="20251108_000100",
    )

    assert len(layout_calls) == 2
    small_layout = layout_calls[0][2]
    large_layout = layout_calls[1][2]

    small_figsize = small_layout[0]
    large_figsize = large_layout[0]

    assert small_figsize[0] == pytest.approx(small_figsize[1])
    assert large_figsize[0] == pytest.approx(large_figsize[1])
    assert large_figsize[0] == pytest.approx(12.0)
    assert small_figsize[0] < large_figsize[0]

    assert small_layout[1]["label"] > large_layout[1]["label"]
    assert small_layout[1]["ticks"] > large_layout[1]["ticks"]
    assert large_layout[1]["ticks"] >= 8.0

    assert recorded_figsizes[0] == small_figsize
    assert recorded_figsizes[1] == large_figsize


def test_format_corner_footer_stats_reports_processing() -> None:
    """Summaries should mention sample counts, stride and thinning."""

    stats = {
        "original_count": 1000,
        "finite_count": 900,
        "processed_count": 300,
        "stride": 3,
        "downsampled": True,
        "legacy_validator": True,
    }

    lines = plotter._format_corner_footer_stats(stats)
    rendered = [line for line, _ in lines]

    assert any("300 samples used" in line for line in rendered)
    assert any("stride 3" in line for line in rendered)
    assert any("Automatic thinning" in line for line in rendered)
    assert any("Legacy validator" in line for line in rendered)


def test_plot_corner_downsamples_large_chains(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirm extremely long chains are thinned before plotting."""

    rng = np.random.default_rng(0)
    samples = rng.normal(size=(200, 10, 3))

    attrs = {
        "dataset_id": "joint_posterior",
        "dataset_name": "Joint posterior",
        "description": "Downsampling check",
        "citation": "Corner validation stub",
    }

    captured: dict[str, Any] = {}
    original_validator = plotter._prepare_corner_inputs

    def _recording_validator(
        posterior_samples: np.ndarray, parameter_names: list[str]
    ) -> tuple[np.ndarray, list[str], dict[str, int | bool]]:
        processed, labels, stats = original_validator(
            posterior_samples,
            parameter_names,
        )
        captured["stats"] = stats
        return processed, labels, stats

    monkeypatch.setattr(plotter, "MAX_CORNER_SAMPLES", 50)
    monkeypatch.setattr(
        plotter, "_prepare_corner_inputs", _recording_validator
    )
    monkeypatch.setattr(
        plotter, "_validate_corner_inputs", _recording_validator
    )

    original_build_footer_lines = plotter.build_footer_lines
    recorded: dict[str, Any] = {}

    def _recording_footer(*args: Any, **kwargs: Any) -> list[tuple[str, bool]]:
        recorded["extra"] = kwargs.get("extra_lines")
        return original_build_footer_lines(*args, **kwargs)

    monkeypatch.setattr(plotter, "build_footer_lines", _recording_footer)

    plotter.plot_corner(
        samples,
        _CornerPlugin,
        attrs,
        plot_dir=str(tmp_path),
        timestamp="20251108_000000",
    )

    stats = captured["stats"]
    assert stats["downsampled"] is True
    assert stats["processed_count"] <= plotter.MAX_CORNER_SAMPLES
    extra_lines = recorded["extra"]
    assert extra_lines is not None
    assert any("Corner plot generation" in line for line, _ in extra_lines)


def test_plot_corner_handles_legacy_validator_signature(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ensure Stage 5 tolerates two-value legacy validators."""

    rng = np.random.default_rng(42)
    samples = rng.normal(size=(5, 2, 2))

    attrs = {
        "dataset_id": "joint_posterior",
        "dataset_name": "Joint posterior",
        "description": "Legacy validator compatibility check",
        "citation": "Corner validation stub",
    }

    def _legacy_validator(
        posterior_samples: np.ndarray, parameter_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Mimic the pre-7.4 signature returning only flattened data."""

        flattened = np.asarray(posterior_samples).reshape(
            -1, posterior_samples.shape[-1]
        )
        return flattened, parameter_names[: flattened.shape[1]]

    monkeypatch.setattr(plotter, "_prepare_corner_inputs", _legacy_validator)
    monkeypatch.setattr(plotter, "_validate_corner_inputs", _legacy_validator)

    with caplog.at_level("INFO"):
        plotter.plot_corner(
            samples,
            _CornerPlugin,
            attrs,
            plot_dir=str(tmp_path),
            timestamp="20251108_000000",
        )

    expected_name = plot_utils.generate_filename(
        "corner-plot",
        "joint_posterior",
        "png",
        model_name="vs-TestModel",
        timestamp="20251108_000000",
    )
    assert (tmp_path / expected_name).exists()
    assert any(
        "legacy two-value signature" in message for message in caplog.messages
    )
