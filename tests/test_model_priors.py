"""Tests for parameter prior parsing and exposure."""

import tempfile
import unittest
from pathlib import Path

import yaml

import copernican_lib.engine_interface as engine_interface
import copernican_lib.model_coder as model_coder
import copernican_lib.model_parser as model_parser


class PriorParsingTestCase(unittest.TestCase):
    """Ensure priors are parsed and exposed on the plugin object."""

    def setUp(self):
        base = Path(__file__).resolve().parents[1]
        models_dir = base / "models"
        yaml_path = models_dir / "cosmo_model_lcdm.yml"
        cache_dir = models_dir / "cache"
        cache_path = model_parser.parse_model(yaml_path, cache_dir)
        funcs, parsed = model_coder.generate_callables(cache_path)
        self.plugin = engine_interface.build_plugin(parsed, funcs)

    def test_priors_exposed(self):
        """PARAMETER_PRIORS should mirror YAML prior blocks."""
        priors = self.plugin.PARAMETER_PRIORS
        self.assertEqual(priors[0]["type"], "uniform")
        self.assertEqual(priors[0]["lower"], 50.0)
        self.assertEqual(priors[0]["upper"], 100.0)


class PriorValidationTestCase(unittest.TestCase):
    """Invalid prior definitions must raise ValueError."""

    def test_missing_sigma(self):
        """Gaussian prior without sigma should be rejected."""
        model = {
            "model_name": "BadPrior",
            "version": "1.0",
            "parameters": [
                {
                    "name": "a",
                    "bounds": [0.0, 1.0],
                    "latex_name": "a",
                    "prior": {"type": "gaussian", "mean": 0.0},
                }
            ],
            "equations": {},
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yml", delete=False
        ) as tmp:
            yaml.safe_dump(model, tmp, sort_keys=False)
            tmp_path = tmp.name
        cache_dir = Path(tmp_path).parent
        with self.assertRaises(ValueError):
            model_parser.parse_model(tmp_path, cache_dir)


if __name__ == "__main__":
    unittest.main()
