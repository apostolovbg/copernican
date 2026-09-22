"""Production CMB graph recovery evidence for Slice Seven."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

import numpy
import pandas

from copernican.lib import (
    model_adapter,
    model_coder,
    model_spec_validator,
    plotter,
)
from copernican.lib.likelihoods.cmb import cmb
from copernican.lib.likelihoods.cmb.diagnostics import (
    assess_acoustic_structure,
    assess_physical_spectrum_shape,
)
from copernican.lib.model_selection import build_comparison_request

_PRODUCTION_ELL_VALUES = (
    2,
    20,
    40,
    60,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    300,
)
_PRODUCTION_SPECTRA = ("TT", "TE", "EE")


class _SliceSevenTestPlugin:
    """Expose the model label required by the comparison renderer."""

    MODEL_NAME = "LambdaCDM"
    MODEL_EQUATIONS_LATEX_SN: list[str] = []
    MODEL_EQUATIONS_LATEX_BAO: list[str] = []
    PARAMETER_NAMES: list[str] = []
    PARAMETER_LATEX_NAMES: list[str] = []


def _build_lcdm_plugin():
    """Build LCDM without importing the broad model-adapter test module."""

    model_path = (
        Path(__file__).resolve().parents[3]
        / "copernican"
        / "models"
        / "model_lcdm.yml"
    )
    with tempfile.TemporaryDirectory() as cache_dir:
        cache_path = model_spec_validator.validate_and_cache_model(
            model_path,
            cache_dir,
        )
        functions, model_data = model_coder.generate_callables(cache_path)
    plugin = model_adapter.build_plugin(model_data, functions)
    plugin.MODEL_FILENAME = model_path.name
    return plugin


@lru_cache(maxsize=1)
def _build_production_evidence() -> dict[str, object]:
    """Run one public production request and retain graph evidence."""

    plugin = _build_lcdm_plugin()
    ell_values = numpy.asarray(_PRODUCTION_ELL_VALUES, dtype=int)
    spectra = cmb.compute_cmb_spectrum_cached(
        plugin,
        plugin.INITIAL_GUESSES,
        ell_values,
        spectra=_PRODUCTION_SPECTRA,
        workload="full_spectrum",
    )
    first_result = cmb._LAST_CMB_RESULT.get()
    if first_result is None:
        raise AssertionError("production CMB request retained no result")

    repeat = cmb.compute_cmb_spectrum_cached(
        plugin,
        plugin.INITIAL_GUESSES,
        ell_values,
        spectra=_PRODUCTION_SPECTRA,
        workload="full_spectrum",
    )
    repeat_result = cmb._LAST_CMB_RESULT.get()
    if repeat_result is None:
        raise AssertionError("exact-repeat CMB request retained no result")

    observations = pandas.DataFrame(
        {
            "ell": numpy.tile(ell_values, len(_PRODUCTION_SPECTRA)),
            "spectrum": numpy.repeat(
                _PRODUCTION_SPECTRA,
                ell_values.size,
            ),
            "Dl_obs": numpy.concatenate(
                [
                    numpy.asarray(spectra[name], dtype=float)
                    for name in _PRODUCTION_SPECTRA
                ]
            ),
        }
    )
    observations.attrs.update(
        {
            "dataset_id": "slice_seven_production",
            "dataset_name": "CCMBS production graph evidence",
            "covariance_matrix_inv": numpy.eye(len(observations)),
        }
    )
    graph_result = {
        "theory_spectrum": spectra,
        "chi2_cmb": 0.0,
    }
    fit_result = {
        "fitted_model_params": {},
        "chi2_total": 0.0,
    }
    comparison = build_comparison_request("LambdaCDM", "LambdaCDM")
    with tempfile.TemporaryDirectory() as plot_dir:
        plotter.plot_cmb_spectrum(
            observations,
            graph_result,
            graph_result,
            fit_result,
            fit_result,
            _SliceSevenTestPlugin,
            _SliceSevenTestPlugin,
            plot_dir=plot_dir,
            timestamp="20260918_000000",
            comparison=comparison,
        )
        artifacts = sorted(Path(plot_dir).glob("*.png"))
        if len(artifacts) != 1:
            raise AssertionError(
                "production graph path did not retain exactly one artifact"
            )
        artifact = artifacts[0]
        artifact_bytes = artifact.read_bytes()

    return {
        "ell_values": tuple(int(value) for value in ell_values),
        "spectra": {
            name: numpy.asarray(values, dtype=float).copy()
            for name, values in spectra.items()
        },
        "shape": assess_physical_spectrum_shape(ell_values, spectra),
        "acoustic": assess_acoustic_structure(ell_values, spectra),
        "graph_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "graph_size": len(artifact_bytes),
        "first_diagnostics": dict(first_result.diagnostics or {}),
        "repeat_diagnostics": dict(repeat_result.diagnostics or {}),
        "first_phase_timings": dict(first_result.phase_timings or {}),
        "repeat_phase_timings": dict(repeat_result.phase_timings or {}),
        "repeat_spectra": {
            name: numpy.asarray(values, dtype=float).copy()
            for name, values in repeat.items()
        },
    }


class SliceSevenProductionGraphTestCase(unittest.TestCase):
    """Require finite, wave-bearing output from the normal public route."""

    def test_production_request_retains_wave_graph_and_work_evidence(self):
        """One production solve feeds the graph and exact-repeat evidence."""

        evidence = _build_production_evidence()
        spectra = evidence["spectra"]
        shape = evidence["shape"]

        self.assertTrue(shape["finite"])
        self.assertTrue(shape["auto_spectra_nonnegative"])
        self.assertTrue(shape["smooth"])
        self.assertTrue(evidence["acoustic"]["te"]["finite"])
        self.assertGreaterEqual(
            int(evidence["acoustic"]["te"]["sign_change_count"]),
            1,
        )
        temperature_spectrum = numpy.asarray(spectra["TT"], dtype=float)
        acoustic_indices = numpy.flatnonzero(
            numpy.asarray(_PRODUCTION_ELL_VALUES, dtype=int) >= 20
        )
        trough_index = int(
            acoustic_indices[
                numpy.argmin(temperature_spectrum[acoustic_indices])
            ]
        )
        peak_index = (
            int(numpy.argmax(temperature_spectrum[trough_index:]))
            + trough_index
        )
        self.assertGreater(_PRODUCTION_ELL_VALUES[peak_index], 180)
        self.assertGreater(
            float(temperature_spectrum[peak_index]),
            1.5 * float(temperature_spectrum[trough_index]),
        )

        self.assertEqual(len(evidence["graph_sha256"]), 64)
        self.assertGreater(evidence["graph_size"], 0)
        first_work = evidence["first_diagnostics"]["performance_record"]
        repeat_work = evidence["repeat_diagnostics"]["performance_record"]
        self.assertEqual(first_work["cache_state"], "cold")
        self.assertEqual(repeat_work["cache_state"], "exact_cache_hit")
        self.assertGreater(first_work["work_units"]["total_work_units"], 0)
        self.assertEqual(
            first_work["context"]["runtime"]["accuracy_tier"],
            "final",
        )
        self.assertTrue(
            numpy.allclose(
                evidence["spectra"]["TT"],
                evidence["repeat_spectra"]["TT"],
                rtol=0.0,
                atol=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
