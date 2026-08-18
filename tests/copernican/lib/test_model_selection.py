"""Tests for shared control/test model selection semantics."""

import unittest
from types import SimpleNamespace

from copernican.lib import run_manifest
from copernican.lib.model_selection import (
    ComparisonRequest,
    ModelRole,
    build_comparison_request,
    comparison_from_manifest,
    comparison_slug,
    model_role_from_value,
    validate_comparison_compatibility,
)


class ModelSelectionTestCase(unittest.TestCase):
    """Exercise role selection, manifest round-trips, and validation."""

    def test_manifest_round_trip_preserves_both_roles(self) -> None:
        request = build_comparison_request(
            "Reference Model",
            "Modified Model",
            control_filename="reference.yml",
            test_filename="modified.yml",
        )
        self.assertIsInstance(request, ComparisonRequest)
        self.assertEqual(
            request.as_manifest()["control"]["filename"], "reference.yml"
        )
        self.assertEqual(
            request.model_names, ("Reference Model", "Modified Model")
        )
        manifest = {
            "selection": {"comparison": request.as_manifest()},
        }
        restored = comparison_from_manifest(manifest)
        self.assertEqual(restored, request)
        self.assertEqual(
            comparison_slug(restored), "Reference_Model-vs-Modified_Model"
        )

    def test_model_role_normalization_preserves_identity(self) -> None:
        role = model_role_from_value(
            {"name": "Reference Model", "filename": "reference.yml"}
        )
        self.assertIsInstance(role, ModelRole)
        self.assertEqual(role.as_manifest()["name"], "Reference Model")
        self.assertEqual(
            model_role_from_value("Reference Model").name,
            "Reference Model",
        )

    def test_manifest_without_both_roles_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "control and test"):
            comparison_from_manifest(
                {"selection": {"models": ["Modified Model"]}}
            )

    def test_declared_surface_mismatch_is_rejected(self) -> None:
        request = build_comparison_request("A", "B")
        with self.assertRaisesRegex(ValueError, "ell_grids"):
            validate_comparison_compatibility(
                request,
                control_metadata={"ell_grid": [2, 3, 4]},
                test_metadata={"ell_grid": [2, 3, 5]},
            )

    def test_manifest_builder_records_comparison(self) -> None:
        control = SimpleNamespace(
            MODEL_NAME="Reference",
            MODEL_FILENAME="reference.yml",
            PARAMETER_NAMES=[],
            PARAMETER_PRIORS=[],
            CMB_CONTRACT={},
        )
        test = SimpleNamespace(
            MODEL_NAME="Test",
            MODEL_FILENAME="test.yml",
            PARAMETER_NAMES=[],
            PARAMETER_PRIORS=[],
            CMB_CONTRACT={},
        )
        manifest = run_manifest.build_manifest(
            models=[(control, "1"), (test, "1")],
            sampler_module=SimpleNamespace(__name__="sampler"),
            datasets=[],
        )
        comparison = manifest["selection"]["comparison"]
        self.assertEqual(comparison["control"]["name"], "Reference")
        self.assertEqual(comparison["test"]["name"], "Test")


if __name__ == "__main__":
    unittest.main()
