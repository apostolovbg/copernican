"""Smoke tests for copernican.validation.runner."""

import unittest
from pathlib import Path

import yaml

from copernican.lib import utils
from copernican.validation import runner as module
from copernican.version import get_version


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.validation.runner")

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "discover_manifests"))
        self.assertTrue(hasattr(module, "run_validation_suite"))

    def test_reference_manifest_resolves_current_assets(self) -> None:
        """The validation fixture must name the canonical model pair."""

        repo_root = Path(__file__).resolve().parents[3]
        manifest_path = (
            repo_root
            / "copernican"
            / "validation"
            / "manifests"
            / "reference_planck2018.yml"
        )
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["copernican"]["version"], get_version())
        comparison = manifest["selection"]["comparison"]
        self.assertEqual(comparison["control"]["filename"], "model_lcdm.yml")
        self.assertEqual(
            comparison["test"]["filename"],
            "model_ref_planck2018.yml",
        )
        self.assertEqual(len(manifest["models"]), 2)

        for dataset in manifest["datasets"].values():
            dataset_root = repo_root / dataset["path"]
            for relative_path, expected_hash in dataset["hashes"].items():
                asset_path = dataset_root / relative_path
                self.assertTrue(asset_path.is_file())
                self.assertEqual(
                    utils.compute_sha256(str(asset_path)),
                    expected_hash,
                )


if __name__ == "__main__":
    unittest.main()
