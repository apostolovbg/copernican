"""Graph rendering/persistence checks without numerical production solves."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy

from tests.project.lib import ccmbs_production_graph as graph


class ProductionGraphArtifactTestCase(unittest.TestCase):
    """Manufactured products test the renderer, not scientific acceptance."""

    def test_supplied_wave_products_render_and_reload_in_a_fresh_process(self):
        ells = numpy.arange(2, 101)
        products = {
            "TT": 2 + numpy.sin(ells / 10) ** 2,
            "TE": numpy.sin(ells / 10),
            "EE": 1 + numpy.cos(ells / 10) ** 2,
        }
        plugin = SimpleNamespace(
            MODEL_FILENAME="manufactured", INITIAL_GUESSES=(1.0,)
        )
        result = SimpleNamespace(
            raw_spectra=products,
            phase_timings={},
            diagnostics={
                "performance_record": {
                    "context": {"runtime": {"fixture": True}}
                },
                "postprocessing_evidence": {"accepted": True},
            },
        )
        with mock.patch.object(
            graph.cmb,
            "compute_cmb_spectrum_cached",
            side_effect=AssertionError(
                "Artifact tests must not solve physics"
            ),
        ):
            evidence = graph.retain_graph_evidence(
                plugin, ells, products, result, products, result
            )
        self.assertGreater(evidence["graph_size"], 0)
        self.assertEqual(len(evidence["graph_sha256"]), 64)
        self.assertTrue(evidence["shape"]["finite"])
        for name in products:
            numpy.testing.assert_array_equal(
                evidence["spectra"][name], products[name]
            )

    def test_manifest_reader_rejects_a_changed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw_arrays.json").write_text("{}")
            (root / "refinement.json").write_text("{}")
            manifest = dict.fromkeys(
                (
                    "source_revision",
                    "declaration_identity",
                    "resolved_physical_inputs",
                    "numerical_settings",
                    "request_identity",
                ),
                "fixture",
            )
            manifest["artifacts"] = {
                "raw": {
                    "path": "raw_arrays.json",
                    "sha256": hashlib.sha256(b"{}").hexdigest(),
                }
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            graph._verify_artifact_manifest(str(root))
            (root / "raw_arrays.json").write_text('{"changed": true}')
            with self.assertRaisesRegex(AssertionError, "digest mismatch"):
                graph._verify_artifact_manifest(str(root))

    def test_discovery_does_not_import_the_explicit_physics_case(self):
        suite = unittest.defaultTestLoader.loadTestsFromModule(
            __import__(__name__, fromlist=["unused"])
        )

        def ids(suite):
            for item in suite:
                if isinstance(item, unittest.TestSuite):
                    yield from ids(item)
                else:
                    yield item.id()

        collected = tuple(ids(suite))
        self.assertTrue(collected)
        self.assertFalse(
            any("ProductionGraphTestCase" in name for name in collected)
        )
        self.assertFalse(Path(graph.__file__).name.startswith("test_"))


if __name__ == "__main__":
    unittest.main()
