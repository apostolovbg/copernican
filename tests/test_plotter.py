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
                float,
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

    small_line_height = small_layout[2]
    large_line_height = large_layout[2]
    assert small_line_height == pytest.approx(
        plotter._CORNER_BASE_LINE_HEIGHT,
        rel=0.01,
    )
    assert large_line_height >= plotter._CORNER_BASE_LINE_HEIGHT

    small_footer_pad = small_layout[4]
    large_footer_pad = large_layout[4]
    assert small_footer_pad == pytest.approx(
        plotter._CORNER_FOOTER_PADDING,
        rel=1e-9,
    )
    assert large_footer_pad == pytest.approx(small_footer_pad, rel=1e-9)

    base_panel_width = 3.6
    for n_params, footer_lines, layout in layout_calls:
        margins = layout[3]
        line_height = layout[2]
        footer_pad = layout[4]
        footer_block = footer_lines * line_height
        axes_bottom = margins["bottom"]
        assert axes_bottom >= plotter._CORNER_MIN_BOTTOM - 1e-9
        assert axes_bottom <= plotter._CORNER_MAX_BOTTOM + 1e-9
        assert (
            axes_bottom - footer_block
            >= plotter._CORNER_FOOTER_CLEARANCE - 1e-9
        )
        assert footer_pad == pytest.approx(plotter._CORNER_FOOTER_PADDING)
        footer_start = axes_bottom - footer_pad
        lowest_line = footer_start - (footer_lines - 1) * line_height
        assert lowest_line >= plotter._CORNER_FOOTER_CLEARANCE - 1e-6
        assert axes_bottom - footer_start == pytest.approx(
            plotter._CORNER_FOOTER_PADDING
        )

        side_length = layout[0][0]
        panel_width = side_length / float(n_params)
        scale = max(panel_width / base_panel_width, 0.55)
        shrink_penalty = max(0.0, 1.0 - min(scale, 1.0))
        expected_top = np.clip(
            0.93 + 0.008 * shrink_penalty,
            0.91,
            0.945,
        )
        assert margins["top"] == pytest.approx(expected_top)


def test_plot_corner_positions_title_and_footer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The title and footer should respect the configured clearances."""

    rng = np.random.default_rng(2)
    samples = rng.normal(size=(12, 2, 3))

    attrs = {
        "dataset_id": "joint_posterior",
        "dataset_name": "Joint posterior",
        "description": "Spacing guard test",
        "citation": "Corner validation stub",
    }

    import matplotlib.figure as mpl_fig

    recorded_suptitles: list[dict[str, Any]] = []
    original_suptitle = mpl_fig.Figure.suptitle

    def _recording_suptitle(self, *args: Any, **kwargs: Any):
        recorded_suptitles.append(kwargs.copy())
        return original_suptitle(self, *args, **kwargs)

    monkeypatch.setattr(mpl_fig.Figure, "suptitle", _recording_suptitle)

    recorded_text_y: list[float] = []
    original_text = mpl_fig.Figure.text

    def _recording_text(self, *args: Any, **kwargs: Any):
        recorded_text_y.append(float(args[1]))
        return original_text(self, *args, **kwargs)

    monkeypatch.setattr(mpl_fig.Figure, "text", _recording_text)

    captured_layout: dict[str, Any] = {}
    original_layout = plotter._compute_corner_layout

    def _capturing_layout(n_params: int, footer_lines: int):
        layout = original_layout(n_params, footer_lines)
        captured_layout["value"] = (n_params, footer_lines, layout)
        return layout

    monkeypatch.setattr(plotter, "_compute_corner_layout", _capturing_layout)

    plotter.plot_corner(
        samples,
        _CornerPlugin,
        attrs,
        plot_dir=str(tmp_path),
        timestamp="20251108_000200",
    )

    assert recorded_suptitles
    suptitle_kwargs = recorded_suptitles[-1]
    assert suptitle_kwargs.get("y") == pytest.approx(plotter._CORNER_TITLE_Y)

    assert "value" in captured_layout
    _, footer_lines, layout = captured_layout["value"]
    margins = layout[3]
    line_height = layout[2]
    footer_pad = layout[4]

    footer_positions = [
        value for value in recorded_text_y if value <= margins["bottom"] + 1e-6
    ]
    assert footer_positions
    first_line = max(footer_positions)
    lowest_line = min(footer_positions)
    gap_to_axes = margins["bottom"] - first_line
    assert footer_pad == pytest.approx(plotter._CORNER_FOOTER_PADDING)
    assert gap_to_axes == pytest.approx(footer_pad)
    assert first_line == pytest.approx(margins["bottom"] - footer_pad)
    assert lowest_line >= plotter._CORNER_FOOTER_CLEARANCE - 1e-6
    expected_span = (footer_lines - 1) * line_height
    actual_span = first_line - lowest_line
    assert actual_span == pytest.approx(expected_span)


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


def test_density_levels_are_strictly_increasing() -> None:
    """Ensure contour thresholds never repeat when histogram bins coincide."""

    hist = np.full((2, 2), 0.25)
    levels = plotter._density_levels(hist, (0.5, 0.9))
    assert levels[0] < levels[1]


def test_build_contour_levels_produce_increasing_sequences() -> None:
    """Even plateaued histograms should yield strictly increasing levels."""

    hist = np.array([[0.4, 0.4], [0.4, 0.1]])
    filled, lines = plotter._build_contour_levels(hist, (0.68, 0.95))

    assert np.all(np.diff(filled) > 0.0)
    assert np.all(np.diff(lines) > 0.0)
    assert filled[0] == pytest.approx(0.0)


def test_plot_corner_omits_dataset_metadata_from_footer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corner plots should no longer repeat dataset descriptions in footers."""

    rng = np.random.default_rng(7)
    samples = rng.normal(size=(20, 3, 3))

    attrs = {
        "dataset_id": "joint_posterior",
        "dataset_name": "Joint posterior",
        "description": "Synthetic description",  # Should be stripped
        "citation": "Corner validation stub",
    }

    captured: dict[str, Any] = {}
    original_footer = plotter.build_footer_lines

    def _recording_footer(*args: Any, **kwargs: Any):
        captured["include_dataset_details"] = kwargs.get(
            "include_dataset_details"
        )
        lines = original_footer(*args, **kwargs)
        captured["lines"] = lines
        return lines

    monkeypatch.setattr(plotter, "build_footer_lines", _recording_footer)

    plotter.plot_corner(
        samples,
        _CornerPlugin,
        attrs,
        plot_dir=str(tmp_path),
        timestamp="20251109_120000",
    )

    assert captured["include_dataset_details"] is False
    footer_text = [line for line, _ in captured["lines"]]
    assert all("Observational dataset" not in line for line in footer_text)
    assert all("Corner validation stub" not in line for line in footer_text)
    assert any("Corner plot generation" in line for line in footer_text)


def test_build_footer_lines_preserves_citation_by_default() -> None:
    """Citation strings should remain when dataset details are shown."""

    attrs = {
        "dataset_name": "Joint posterior",
        "description": "Synthetic description",
        "citation": "Corner validation stub",
    }

    footer_lines = plotter.build_footer_lines(
        _CornerPlugin, attrs, "20250101_000000"
    )

    assert any("Corner validation stub" in line for line, _ in footer_lines)


def test_build_footer_lines_omits_citation_when_dataset_details_disabled() -> (
    None
):
    """Suppressing dataset details should also hide citation metadata."""

    attrs = {
        "dataset_name": "Joint posterior",
        "description": "Synthetic description",
        "citation": "Corner validation stub",
    }

    extra_lines = [("Corner plot generation: 12 samples used", False)]
    footer_lines = plotter.build_footer_lines(
        _CornerPlugin,
        attrs,
        "20250101_000000",
        extra_lines=extra_lines,
        include_dataset_details=False,
    )

    footer_text = [line for line, _ in footer_lines]
    assert footer_text[0].startswith("ΛCDM vs")
    assert "Corner validation stub" not in "\n".join(footer_text)
    generation_line = "Corner plot generation: 12 samples used"
    assert footer_text.count(generation_line) == 1
