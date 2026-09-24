"""Production CMB graph recovery evidence for Slice Seven."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
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
_PRODUCTION_NUMERICAL_OVERRIDES = {
    "ell_max": 300,
    "k_sample_count": 64,
    "eta_sample_count": 192,
    "evolution_eta_sample_count": 128,
}


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


def _read_source_revision() -> str:
    """Read the current commit identity without launching a shell command."""

    git_root = Path(__file__).resolve().parents[3] / ".git"
    head = (git_root / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        return (git_root / head[5:]).read_text(encoding="utf-8").strip()
    return head


def _verify_artifact_manifest(artifact_root: str) -> None:
    """Reload retained graph evidence and verify every recorded digest."""

    root = Path(artifact_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for field in (
        "source_revision",
        "declaration_identity",
        "resolved_physical_inputs",
        "numerical_settings",
        "request_identity",
    ):
        if not manifest[field]:
            raise AssertionError(f"manifest field is empty: {field}")
    for item in manifest["artifacts"].values():
        artifact_bytes = (root / item["path"]).read_bytes()
        artifact_digest = hashlib.sha256(artifact_bytes).hexdigest()
        if artifact_digest != item["sha256"]:
            raise AssertionError("retained artifact digest mismatch")
    json.loads((root / "raw_arrays.json").read_text(encoding="utf-8"))
    json.loads((root / "refinement.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _build_production_evidence() -> dict[str, object]:
    """Run one bounded public request and retain durable graph evidence."""

    plugin = _build_lcdm_plugin()
    ell_values = numpy.asarray(_PRODUCTION_ELL_VALUES, dtype=int)
    spectra = cmb.compute_cmb_spectrum_cached(
        plugin,
        plugin.INITIAL_GUESSES,
        ell_values,
        spectra=_PRODUCTION_SPECTRA,
        workload="full_spectrum",
        numerical_overrides=_PRODUCTION_NUMERICAL_OVERRIDES,
        diagnostic_matrix_fast_path=True,
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
        numerical_overrides=_PRODUCTION_NUMERICAL_OVERRIDES,
        diagnostic_matrix_fast_path=True,
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
    with tempfile.TemporaryDirectory() as artifact_dir:
        artifact_root = Path(artifact_dir)
        graph_path = artifact_root / "graph.png"
        raw_path = artifact_root / "raw_arrays.json"
        refinement_path = artifact_root / "refinement.json"
        manifest_path = artifact_root / "manifest.json"
        with tempfile.TemporaryDirectory() as plot_dir:
            plot_path = Path(plot_dir)
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
            artifacts = sorted(plot_path.glob("*.png"))
            if len(artifacts) != 1:
                raise AssertionError(
                    "production graph path did not retain exactly one artifact"
                )
            graph_path.write_bytes(artifacts[0].read_bytes())

        raw_payload = {
            name: numpy.asarray(values, dtype=float).tolist()
            for name, values in (first_result.raw_spectra or {}).items()
        }
        raw_path.write_text(
            json.dumps(raw_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        postprocessing = dict(
            first_result.diagnostics.get("postprocessing_evidence", {})
        )
        refinement_payload = {
            "accepted": bool(postprocessing.get("accepted", False)),
            "intermediates": dict(postprocessing.get("intermediates", {})),
            "issues": list(postprocessing.get("issues", [])),
        }
        refinement_path.write_text(
            json.dumps(refinement_payload, sort_keys=True, default=list)
            + "\n",
            encoding="utf-8",
        )
        source_revision = _read_source_revision()
        manifest = {
            "schema_version": 1,
            "source_revision": source_revision,
            "declaration_identity": {
                "model_filename": plugin.MODEL_FILENAME,
                "model_name": _SliceSevenTestPlugin.MODEL_NAME,
            },
            "resolved_physical_inputs": {
                "model_parameters": [
                    float(value) for value in plugin.INITIAL_GUESSES
                ],
            },
            "numerical_settings": {
                "requested": dict(_PRODUCTION_NUMERICAL_OVERRIDES),
                "resolved": dict(
                    first_result.diagnostics["performance_record"]["context"][
                        "runtime"
                    ]
                ),
            },
            "request_identity": {
                "ells": [int(value) for value in ell_values],
                "spectra": list(_PRODUCTION_SPECTRA),
                "workload": "full_spectrum",
            },
            "reference_identity": "not_applicable",
            "artifacts": {},
        }
        for name, path in (
            ("graph", graph_path),
            ("raw_arrays", raw_path),
            ("refinement", refinement_path),
        ):
            manifest["artifacts"][name] = {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=_verify_artifact_manifest,
            args=(str(artifact_root),),
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise AssertionError(
                "retained graph evidence failed a fresh-process reload"
            )
        artifact_bytes = graph_path.read_bytes()

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
    """Require finite, wave-bearing output from the bounded public route."""

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
            None,
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
