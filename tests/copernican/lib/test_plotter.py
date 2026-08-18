"""Unit tests for :mod:`copernican.lib.plotter`.

These tests ensure summary formatting tolerates incomplete optimisation
metadata. Plot generation uses the same helper, so catching regressions here
prevents GUI failures during post-processing.
"""

from __future__ import annotations

import inspect
import tkinter
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy
import pandas
import pytest

from copernican.lib import plotter
from copernican.lib import utils as plot_utils
from copernican.lib.model_selection import build_comparison_request

_TEST_COMPARISON = build_comparison_request(
    "ReferenceModel",
    "CandidateModel",
)


class _DummyPlugin:
    """Lightweight stand-in exposing the attributes the plotter expects."""

    MODEL_NAME = "CandidateModel"
    MODEL_EQUATIONS_LATEX_SN: list[str] = []
    MODEL_EQUATIONS_LATEX_BAO: list[str] = []
    PARAMETER_NAMES: list[str] = []
    PARAMETER_LATEX_NAMES: list[str] = []


class _CornerPlugin(_DummyPlugin):
    """Extended dummy plugin carrying names for the corner plot."""

    PARAMETER_NAMES = ["alpha", "beta", "gamma"]
    PARAMETER_LATEX_NAMES = [r"\alpha", r"\beta", r"\gamma"]


class _ReferencePlugin(_DummyPlugin):
    """Control role used by comparison plot tests."""

    MODEL_NAME = "ReferenceModel"


def _tmp_path_or_default(tmp_path) -> Any:
    """Return a usable temporary directory path for unittest methods."""

    if tmp_path is None:
        import tempfile
        from pathlib import Path

        return Path(tempfile.mkdtemp())
    return tmp_path


class TestPlotter(unittest.TestCase):
    """Exercise plotter helpers and generated figures."""

    def test_format_model_summary_text_handles_missing_chi2_total(
        self,
    ) -> None:
        _case_format_model_summary_text_handles_missing_chi2_total(self)

    def test_format_model_summary_text_numeric_rendering(self) -> None:
        _case_format_model_summary_text_numeric_rendering(self)

    def test_plot_corner_renders_expected_file(
        self,
        tmp_path=None,
    ) -> None:
        _case_plot_corner_renders_expected_file(self, tmp_path)

    def test_plot_parameter_histograms_renders_expected_file(
        self,
        tmp_path=None,
    ) -> None:
        _case_plot_parameter_histograms_renders_expected_file(
            self,
            tmp_path,
        )

    def test_plot_corner_scales_layout_with_dimension(
        self,
        tmp_path=None,
        monkeypatch: pytest.MonkeyPatch | None = None,
    ) -> None:
        _case_plot_corner_scales_layout_with_dimension(
            self,
            tmp_path,
            monkeypatch,
        )

    def test_plot_corner_positions_title_and_footer(
        self,
        tmp_path=None,
        monkeypatch: pytest.MonkeyPatch | None = None,
    ) -> None:
        _case_plot_corner_positions_title_and_footer(
            self,
            tmp_path,
            monkeypatch,
        )

    def test_format_corner_footer_stats_reports_processing(self) -> None:
        _case_format_corner_footer_stats_reports_processing(self)

    def test_plot_corner_downsamples_large_chains(
        self,
        tmp_path=None,
        monkeypatch: pytest.MonkeyPatch | None = None,
    ) -> None:
        _case_plot_corner_downsamples_large_chains(
            self,
            tmp_path,
            monkeypatch,
        )

    def test_plot_corner_falls_back_to_agg_backend(
        self,
        tmp_path=None,
        monkeypatch: pytest.MonkeyPatch | None = None,
    ) -> None:
        _case_plot_corner_falls_back_to_agg_backend(
            self,
            tmp_path,
            monkeypatch,
        )

    def test_density_levels_are_strictly_increasing(self) -> None:
        _case_density_levels_are_strictly_increasing(self)

    def test_build_contour_levels_produce_increasing_sequences(self) -> None:
        _case_build_contour_levels_produce_increasing_sequences(self)

    def test_plot_corner_omits_dataset_metadata_from_footer(
        self,
        tmp_path=None,
        monkeypatch: pytest.MonkeyPatch | None = None,
    ) -> None:
        _case_plot_corner_omits_dataset_metadata_from_footer(
            self,
            tmp_path,
            monkeypatch,
        )

    def test_build_footer_lines_preserves_citation_by_default(self) -> None:
        _case_build_footer_lines_preserves_citation_by_default(self)

    def test_build_footer_lines_omits_citation_when_dataset_details_disabled(
        self,
    ) -> None:
        _case_build_footer_lines_omits_citation_when_dataset_details_disabled(
            self,
        )

    def test_plot_cmb_keeps_named_surfaces_in_separate_panels(
        self,
        tmp_path=None,
    ) -> None:
        """CMB plots must retain sector and lensing surface identities."""

        tmp_path = _tmp_path_or_default(tmp_path)
        observations = pandas.DataFrame(
            {
                "ell": [30, 20, 40, 30, 50, 40],
                "spectrum": [
                    "scalar_TT",
                    "tensor_TT",
                    "PP",
                    "scalar_TT",
                    "tensor_TT",
                    "PP",
                ],
                "Dl_obs": [10.0, 8.0, 0.1, 12.0, 9.0, 0.2],
            }
        )
        observations.attrs.update(
            {
                "dataset_id": "cmb_surfaces",
                "dataset_name": "CMB surfaces",
                "covariance_matrix_inv": numpy.eye(6),
            }
        )
        theory = {
            "scalar_TT": numpy.array([9.0, 0.0, 0.0, 11.0, 0.0, 0.0]),
            "tensor_TT": numpy.array([0.0, 7.5, 0.0, 0.0, 8.5, 0.0]),
            "PP": numpy.array([0.0, 0.0, 0.09, 0.0, 0.0, 0.18]),
        }
        cmb_results = {"theory_spectrum": theory, "chi2_cmb": 1.0}
        fit_results = {
            "fitted_model_params": {},
            "chi2_total": 1.0,
        }
        captured_titles: list[str] = []

        def _capture_savefig(path: str, **_kwargs: Any) -> None:
            captured_titles.extend(
                axis.get_title() for axis in plotter.plt.gcf().axes
            )
            Path(path).touch()

        timestamp = "20260812_000000"
        with mock.patch.object(
            plotter.plt,
            "savefig",
            side_effect=_capture_savefig,
        ):
            plotter.plot_cmb_spectrum(
                observations,
                cmb_results,
                cmb_results,
                fit_results,
                fit_results,
                _ReferencePlugin,
                _DummyPlugin,
                plot_dir=str(tmp_path),
                timestamp=timestamp,
                comparison=_TEST_COMPARISON,
            )

        expected_name = plot_utils.generate_filename(
            "cmb-plot",
            "cmb_surfaces",
            "png",
            model_name="ReferenceModel-vs-CandidateModel",
            timestamp=timestamp,
        )
        self.assertTrue((tmp_path / expected_name).exists())
        self.assertTrue(any("scalar_TT" in title for title in captured_titles))
        self.assertTrue(any("tensor_TT" in title for title in captured_titles))
        self.assertTrue(any("PP" in title for title in captured_titles))


def _case_format_model_summary_text_handles_missing_chi2_total(self) -> None:
    """Ensure missing totals render as ``N/A`` instead of raising errors."""

    fit_results = {
        "fitted_model_params": types.MappingProxyType({}),
        "chi2_min": None,
    }

    summary = plotter.format_model_summary_text(
        _DummyPlugin,
        "cmb",
        fit_results,
        chi2_cmb=None,
        chi2_total=None,
    )

    self.assertIn(r"$\chi^2_{tot}$ = N/A", summary)
    self.assertIn(r"$\chi^2_{CMB}$ = N/A", summary)


def _case_format_model_summary_text_numeric_rendering(self) -> None:
    """Verify numeric chi-squared statistics include two decimal places."""

    for value, expected_fragment in (
        (42.0, r"$\chi^2_{SNe}$ = 42.00"),
        (float("nan"), r"$\chi^2_{SNe}$ = N/A"),
    ):
        with self.subTest(value=value):
            fit_results = {
                "fitted_model_params": types.MappingProxyType({}),
                "chi2_min": value,
                "chi2_sne": value,
                "chi2_total": value,
            }

            summary = plotter.format_model_summary_text(
                _DummyPlugin,
                "sne",
                fit_results,
            )

            self.assertIn(expected_fragment, summary)
            if expected_fragment.endswith("N/A"):
                self.assertIn(r"$\chi^2_{tot}$ = N/A", summary)
            else:
                self.assertIn(r"$\chi^2_{tot}$ = 42.00", summary)


def _case_plot_corner_renders_expected_file(
    self,
    tmp_path=None,
) -> None:
    """Ensure the new corner plot helper writes a PNG with suite styling."""

    tmp_path = _tmp_path_or_default(tmp_path)
    samples = numpy.zeros((10, 4, 3))
    samples[:, :, 0] = numpy.linspace(0.0, 1.0, 10)[:, None]
    samples[:, :, 1] = numpy.linspace(-1.0, 1.0, 10)[:, None]
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
        comparison=_TEST_COMPARISON,
    )

    expected_name = plot_utils.generate_filename(
        "corner-plot",
        "joint_posterior",
        "png",
        model_name="ReferenceModel-vs-CandidateModel",
        timestamp=timestamp,
    )
    self.assertTrue((tmp_path / expected_name).exists())


def _case_plot_parameter_histograms_renders_expected_file(
    self,
    tmp_path=None,
) -> None:
    """Ensure the parameter histogram helper writes a PNG summary."""

    tmp_path = _tmp_path_or_default(tmp_path)
    rng = numpy.random.default_rng(8)
    samples = rng.normal(size=(10, 2, 3))

    attrs = {
        "dataset_id": "joint_posterior",
        "dataset_name": "Joint posterior",
        "description": "Synthetic description",
        "citation": "Histogram validation stub",
    }

    timestamp = "20251108_000500"
    plotter.plot_parameter_histograms(
        samples,
        _CornerPlugin,
        attrs,
        plot_dir=str(tmp_path),
        timestamp=timestamp,
        comparison=_TEST_COMPARISON,
    )

    expected_name = plot_utils.generate_filename(
        "parameter-histograms",
        "joint_posterior",
        "png",
        model_name="ReferenceModel-vs-CandidateModel",
        timestamp=timestamp,
    )
    self.assertTrue((tmp_path / expected_name).exists())


def _case_plot_corner_scales_layout_with_dimension(
    self,
    tmp_path=None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> None:
    """Verify responsive geometry shrinks panels and fonts for larger grids."""

    tmp_path = _tmp_path_or_default(tmp_path)
    if monkeypatch is None:
        monkeypatch = pytest.MonkeyPatch()
        self.addCleanup(monkeypatch.undo)

    rng = numpy.random.default_rng(4)
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
        comparison=_TEST_COMPARISON,
    )

    plotter.plot_corner(
        large_samples,
        _SixParamPlugin,
        attrs,
        plot_dir=str(tmp_path),
        timestamp="20251108_000100",
        comparison=_TEST_COMPARISON,
    )

    self.assertEqual(len(layout_calls), 2)
    small_layout = layout_calls[0][2]
    large_layout = layout_calls[1][2]

    small_figsize = small_layout[0]
    large_figsize = large_layout[0]

    self.assertTrue(small_figsize[0] == pytest.approx(small_figsize[1]))
    self.assertTrue(large_figsize[0] == pytest.approx(large_figsize[1]))
    self.assertTrue(large_figsize[0] == pytest.approx(12.0))
    self.assertLess(small_figsize[0], large_figsize[0])

    self.assertGreater(small_layout[1]["label"], large_layout[1]["label"])
    self.assertGreater(small_layout[1]["ticks"], large_layout[1]["ticks"])
    self.assertGreaterEqual(large_layout[1]["ticks"], 8.0)

    self.assertEqual(recorded_figsizes[0], small_figsize)
    self.assertEqual(recorded_figsizes[1], large_figsize)

    small_line_height = small_layout[2]
    large_line_height = large_layout[2]
    self.assertTrue(
        small_line_height
        == pytest.approx(
            plotter._CORNER_BASE_LINE_HEIGHT,
            rel=0.01,
        )
    )
    self.assertGreaterEqual(
        large_line_height, plotter._CORNER_BASE_LINE_HEIGHT
    )

    small_footer_pad = small_layout[4]
    large_footer_pad = large_layout[4]
    self.assertTrue(
        small_footer_pad
        == pytest.approx(
            plotter._CORNER_FOOTER_PADDING,
            rel=1e-9,
        )
    )
    self.assertTrue(
        large_footer_pad == pytest.approx(small_footer_pad, rel=1e-9)
    )

    base_panel_width = 3.6
    for n_params, footer_lines, layout in layout_calls:
        margins = layout[3]
        line_height = layout[2]
        footer_pad = layout[4]
        footer_block = footer_lines * line_height
        axes_bottom = margins["bottom"]
        self.assertGreaterEqual(axes_bottom, plotter._CORNER_MIN_BOTTOM - 1e-9)
        self.assertLessEqual(axes_bottom, plotter._CORNER_MAX_BOTTOM + 1e-9)
        self.assertTrue(
            axes_bottom - footer_block
            >= plotter._CORNER_FOOTER_CLEARANCE - 1e-9
        )
        self.assertTrue(
            footer_pad == pytest.approx(plotter._CORNER_FOOTER_PADDING)
        )
        footer_start = axes_bottom - footer_pad
        lowest_line = footer_start - (footer_lines - 1) * line_height
        self.assertGreaterEqual(
            lowest_line, plotter._CORNER_FOOTER_CLEARANCE - 1e-6
        )
        self.assertTrue(
            axes_bottom - footer_start
            == pytest.approx(plotter._CORNER_FOOTER_PADDING)
        )

        side_length = layout[0][0]
        panel_width = side_length / float(n_params)
        scale = max(panel_width / base_panel_width, 0.55)
        shrink_penalty = max(0.0, 1.0 - min(scale, 1.0))
        expected_top = numpy.clip(
            0.93 + 0.008 * shrink_penalty,
            0.91,
            0.945,
        )
        self.assertTrue(margins["top"] == pytest.approx(expected_top))


def _case_plot_corner_positions_title_and_footer(
    self,
    tmp_path=None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> None:
    """The title and footer should respect the configured clearances."""

    tmp_path = _tmp_path_or_default(tmp_path)
    if monkeypatch is None:
        monkeypatch = pytest.MonkeyPatch()
        self.addCleanup(monkeypatch.undo)

    rng = numpy.random.default_rng(2)
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
        comparison=_TEST_COMPARISON,
    )

    self.assertTrue(recorded_suptitles)
    suptitle_kwargs = recorded_suptitles[-1]
    self.assertTrue(
        suptitle_kwargs.get("y") == pytest.approx(plotter._CORNER_TITLE_Y)
    )

    self.assertIn("value", captured_layout)
    _, footer_lines, layout = captured_layout["value"]
    margins = layout[3]
    line_height = layout[2]
    footer_pad = layout[4]

    footer_positions = [
        value for value in recorded_text_y if value <= margins["bottom"] + 1e-6
    ]
    self.assertTrue(footer_positions)
    first_line = max(footer_positions)
    lowest_line = min(footer_positions)
    gap_to_axes = margins["bottom"] - first_line
    self.assertTrue(
        footer_pad == pytest.approx(plotter._CORNER_FOOTER_PADDING)
    )

    stack_offset = max(footer_lines - 1, 0) * line_height
    expected_first_line = margins["bottom"] - footer_pad - stack_offset
    expected_lowest_line = (
        expected_first_line - (footer_lines - 1) * line_height
        if footer_lines
        else expected_first_line
    )
    if (
        footer_lines
        and expected_lowest_line < plotter._CORNER_FOOTER_CLEARANCE - 1e-6
    ):
        delta = plotter._CORNER_FOOTER_CLEARANCE - expected_lowest_line
        expected_first_line += delta
        expected_lowest_line += delta

    expected_gap = margins["bottom"] - expected_first_line
    self.assertTrue(gap_to_axes == pytest.approx(expected_gap))
    self.assertTrue(first_line == pytest.approx(expected_first_line))
    self.assertTrue(lowest_line == pytest.approx(expected_lowest_line))
    self.assertGreaterEqual(
        lowest_line, plotter._CORNER_FOOTER_CLEARANCE - 1e-6
    )
    expected_span = (footer_lines - 1) * line_height
    actual_span = first_line - lowest_line
    self.assertTrue(actual_span == pytest.approx(expected_span))


def _case_format_corner_footer_stats_reports_processing(self) -> None:
    """Summaries should mention sample counts, stride and thinning."""

    stats = {
        "original_count": 1000,
        "finite_count": 900,
        "processed_count": 300,
        "stride": 3,
        "downsampled": True,
    }

    lines = plotter._format_corner_footer_stats(stats)
    rendered = [line for line, _ in lines]

    self.assertTrue(any("300 samples used" in line for line in rendered))
    self.assertTrue(any("stride 3" in line for line in rendered))
    self.assertTrue(any("Automatic thinning" in line for line in rendered))
    self.assertFalse(hasattr(plotter, "_validate_corner_inputs"))


def _case_plot_corner_downsamples_large_chains(
    self,
    tmp_path=None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> None:
    """Confirm extremely long chains are thinned before plotting."""

    tmp_path = _tmp_path_or_default(tmp_path)
    if monkeypatch is None:
        monkeypatch = pytest.MonkeyPatch()
        self.addCleanup(monkeypatch.undo)

    rng = numpy.random.default_rng(0)
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
        posterior_samples: numpy.ndarray, parameter_names: list[str]
    ) -> tuple[numpy.ndarray, list[str], dict[str, int | bool]]:
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
        comparison=_TEST_COMPARISON,
    )

    stats = captured["stats"]
    self.assertTrue(stats["downsampled"] is True)
    self.assertLessEqual(stats["processed_count"], plotter.MAX_CORNER_SAMPLES)
    extra_lines = recorded["extra"]
    self.assertIsNotNone(extra_lines)
    self.assertTrue(
        any("Corner plot generation" in line for line, _ in extra_lines)
    )


def _case_plot_corner_falls_back_to_agg_backend(
    self,
    tmp_path=None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> None:
    """Switch to the Agg backend when GUI rendering fails."""

    tmp_path = _tmp_path_or_default(tmp_path)
    if monkeypatch is None:
        monkeypatch = pytest.MonkeyPatch()
        self.addCleanup(monkeypatch.undo)

    rng = numpy.random.default_rng(3)
    samples = rng.normal(size=(10, 2, 2))

    attrs = {
        "dataset_id": "joint_posterior",
        "dataset_name": "Joint posterior",
        "description": "Backend resilience check",
        "citation": "Corner validation stub",
    }

    attempts = {"count": 0}
    original_subplots = plotter.plt.subplots
    original_switch = plotter.plt.switch_backend
    switched: list[str] = []

    def _flaky_subplots(*args: Any, **kwargs: Any):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise tkinter.TclError("missing tk")
        return original_subplots(*args, **kwargs)

    def _recording_switch(backend: str) -> None:
        switched.append(backend)
        original_switch(backend)

    monkeypatch.setattr(plotter.plt, "subplots", _flaky_subplots)
    monkeypatch.setattr(plotter.plt, "switch_backend", _recording_switch)

    plotter.plot_corner(
        samples,
        _CornerPlugin,
        attrs,
        plot_dir=str(tmp_path),
        timestamp="20251108_000000",
        comparison=_TEST_COMPARISON,
    )

    self.assertEqual(attempts["count"], 2)
    self.assertEqual(switched, ["Agg"])


def _case_density_levels_are_strictly_increasing(self) -> None:
    """Ensure contour thresholds never repeat when histogram bins coincide."""

    hist = numpy.full((2, 2), 0.25)
    levels = plotter._density_levels(hist, (0.5, 0.9))
    self.assertLess(levels[0], levels[1])


def _case_build_contour_levels_produce_increasing_sequences(self) -> None:
    """Even plateaued histograms should yield strictly increasing levels."""

    hist = numpy.array([[0.4, 0.4], [0.4, 0.1]])
    filled, lines = plotter._build_contour_levels(hist, (0.68, 0.95))

    self.assertTrue(numpy.all(numpy.diff(filled) > 0.0))
    self.assertTrue(numpy.all(numpy.diff(lines) > 0.0))
    self.assertTrue(filled[0] == pytest.approx(0.0))


def _case_plot_corner_omits_dataset_metadata_from_footer(
    self,
    tmp_path=None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> None:
    """Corner plots should no longer repeat dataset descriptions in footers."""

    tmp_path = _tmp_path_or_default(tmp_path)
    if monkeypatch is None:
        monkeypatch = pytest.MonkeyPatch()
        self.addCleanup(monkeypatch.undo)

    rng = numpy.random.default_rng(7)
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
        comparison=_TEST_COMPARISON,
    )

    self.assertTrue(captured["include_dataset_details"] is False)
    footer_text = [line for line, _ in captured["lines"]]
    self.assertTrue(
        all("Observational dataset" not in line for line in footer_text)
    )
    self.assertTrue(
        all("Corner validation stub" not in line for line in footer_text)
    )
    self.assertTrue(
        any("Corner plot generation" in line for line in footer_text)
    )


def _case_build_footer_lines_preserves_citation_by_default(self) -> None:
    """Citation strings should remain when dataset details are shown."""

    attrs = {
        "dataset_name": "Joint posterior",
        "description": "Synthetic description",
        "citation": "Corner validation stub",
    }

    footer_lines = plotter.build_footer_lines(
        attrs,
        "20250101_000000",
        comparison=_TEST_COMPARISON,
    )

    self.assertTrue(
        any("Corner validation stub" in line for line, _ in footer_lines)
    )


def _case_build_footer_lines_omits_citation_when_dataset_details_disabled(
    self,
) -> None:
    """Suppressing dataset details should also hide citation metadata."""

    attrs = {
        "dataset_name": "Joint posterior",
        "description": "Synthetic description",
        "citation": "Corner validation stub",
    }

    extra_lines = [("Corner plot generation: 12 samples used", False)]
    footer_lines = plotter.build_footer_lines(
        attrs,
        "20250101_000000",
        extra_lines=extra_lines,
        include_dataset_details=False,
        comparison=_TEST_COMPARISON,
    )

    footer_text = [line for line, _ in footer_lines]
    self.assertTrue(
        footer_text[0].startswith("ReferenceModel vs CandidateModel")
    )
    self.assertNotIn("Corner validation stub", "\n".join(footer_text))
    generation_line = "Corner plot generation: 12 samples used"
    self.assertEqual(footer_text.count(generation_line), 1)


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the plotter API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        source = inspect.getsource(plotter.plot_bao_observables)
        cmb_source = inspect.getsource(plotter.plot_cmb_spectrum)
        self.assertTrue(callable(plotter.build_footer_lines))
        self.assertTrue(callable(plotter.compose_footer))
        self.assertTrue(callable(plotter.format_model_summary_text))
        self.assertTrue(callable(plotter.get_binned_average))
        self.assertTrue(callable(plotter.plot_bao_observables))
        self.assertTrue(callable(plotter.plot_cmb_spectrum))
        self.assertTrue(callable(plotter.plot_corner))
        self.assertTrue(callable(plotter.plot_hubble_diagram))
        self.assertTrue(callable(plotter.plot_parameter_histograms))
        self.assertIn("def plot_model_bao(", source)
        self.assertIn("def robust_plot(", source)
        self.assertIn("CCMBS_LABEL", cmb_source)
        self.assertNotIn("lcdm", inspect.getsource(plotter).lower())
