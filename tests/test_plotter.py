"""Unit tests for :mod:`copernican_lib.plotter`.

These tests ensure summary formatting tolerates incomplete optimisation
metadata. Plot generation uses the same helper, so catching regressions here
prevents GUI failures during post-processing.
"""

from __future__ import annotations

import types

import pytest

from copernican_lib import plotter


class _DummyPlugin:
    """Lightweight stand-in exposing the attributes the plotter expects."""

    MODEL_NAME = "TestModel"
    MODEL_EQUATIONS_LATEX_SN: list[str] = []
    MODEL_EQUATIONS_LATEX_BAO: list[str] = []
    PARAMETER_NAMES: list[str] = []
    PARAMETER_LATEX_NAMES: list[str] = []


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
