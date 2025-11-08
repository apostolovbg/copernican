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
    original_validator = plotter._validate_corner_inputs

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
        plotter, "_validate_corner_inputs", _recording_validator
    )

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
