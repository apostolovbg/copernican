"""Tests for ``copernican.lib.model_spec_validator``."""

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from copernican.lib import model_spec_validator as module
from copernican.lib.likelihoods.cmb.errors import ModelDeclarationError


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__, "copernican.lib.model_spec_validator"
        )


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the model-spec validator surface to the coverage policy."""

    def test_public_symbols_are_present(self) -> None:
        self.assertTrue(hasattr(module, "validate_and_cache_model"))


class CMBDeclarationFirewallTestCase(unittest.TestCase):
    """Ensure solver controls cannot enter through the YAML boundary."""

    def test_solver_controls_are_rejected_in_yaml_declarations(self) -> None:
        """Reject top-level and nested legacy numerical controls."""

        source_path = (
            Path(__file__).resolve().parents[3]
            / "copernican"
            / "models"
            / "model_lcdm.yml"
        )
        source_model = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        mutations = (
            ("top-level numerical", ("numerical",)),
            ("nested numerics", ("perturbations", "numerics")),
            (
                "nested accuracy controls",
                ("perturbations", "accuracy_controls"),
            ),
        )
        for label, path in mutations:
            with self.subTest(control=label):
                model_data = copy.deepcopy(source_model)
                target = model_data["cmb"]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = {}
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = Path(temp_dir) / source_path.name
                    model_path.write_text(
                        yaml.safe_dump(model_data, sort_keys=False),
                        encoding="utf-8",
                    )
                    with self.assertRaises(ModelDeclarationError):
                        module.validate_and_cache_model(
                            model_path,
                            Path(temp_dir) / "cache",
                        )


if __name__ == "__main__":
    unittest.main()
