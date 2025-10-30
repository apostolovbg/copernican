"""Tests for parameter prior parsing and exposure."""

import math
import tempfile
import unittest
from pathlib import Path

import yaml

import copernican_lib.engine_interface as engine_interface
import copernican_lib.model_coder as model_coder
import copernican_lib.model_parser as model_parser
from copernican_lib import priors as prior_mod


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
        prior_meta = self.plugin.PARAMETER_PRIORS
        self.assertEqual(prior_meta[0]["type"], "uniform")
        self.assertEqual(prior_meta[0]["lower"], 50.0)
        self.assertEqual(prior_meta[0]["upper"], 100.0)
        prior_objects = self.plugin.PARAMETER_PRIOR_OBJECTS
        self.assertIsInstance(prior_objects[0], prior_mod.UniformPrior)
        transforms = getattr(self.plugin, "PARAMETER_TRANSFORMS", [])
        if transforms:
            self.assertIsNone(transforms[0])

    def test_loguniform_transform(self):
        """Log-uniform priors should register transforms and objects."""
        model = {
            "model_name": "LogUniformModel",
            "version": "1.0",
            "parameters": [
                {
                    "name": "alpha",
                    "bounds": [1e-4, 1.0],
                    "latex_name": "\\alpha",
                    "prior": {
                        "type": "loguniform",
                        "lower": 1e-4,
                        "upper": 1.0,
                    },
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
        cache_path = model_parser.parse_model(tmp_path, cache_dir)
        funcs, parsed = model_coder.generate_callables(cache_path)
        plugin = engine_interface.build_plugin(parsed, funcs)
        prior_obj = plugin.PARAMETER_PRIOR_OBJECTS[0]
        self.assertIsInstance(prior_obj, prior_mod.LogUniformPrior)
        mapping = plugin.PARAMETER_PRIORS[0]
        self.assertEqual(mapping["transform"], "log")
        transforms = plugin.PARAMETER_TRANSFORMS
        transform = transforms[0]
        new_val, jac = transform(0.1)
        self.assertAlmostEqual(new_val, 0.1)
        self.assertAlmostEqual(jac, -math.log(0.1))
        log_density = prior_obj.log_density(0.1)
        self.assertTrue(math.isfinite(log_density))
        posterior = engine_interface.make_logposterior(
            lambda vals: 0.0, plugin.PARAMETER_PRIOR_OBJECTS
        )
        self.assertEqual(float(posterior([0.1])), log_density)


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
        try:
            with self.assertRaises(ValueError):
                model_parser.parse_model(tmp_path, cache_dir)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_loguniform_requires_positive_bounds(self):
        """Log-uniform priors reject non-positive bounds."""
        model = {
            "model_name": "BadLogUniform",
            "version": "1.0",
            "parameters": [
                {
                    "name": "beta",
                    "bounds": [0.0, 1.0],
                    "latex_name": "\\beta",
                    "prior": {
                        "type": "loguniform",
                        "lower": -1.0,
                        "upper": 1.0,
                    },
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
        try:
            with self.assertRaises(ValueError):
                model_parser.parse_model(tmp_path, cache_dir)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_prior_definition_requires_mapping(self):
        """Non-mapping prior declarations should fail validation."""
        model = {
            "model_name": "BadType",
            "version": "1.0",
            "parameters": [
                {
                    "name": "gamma",
                    "bounds": [0.0, 1.0],
                    "latex_name": "\\gamma",
                    "prior": ["not", "a", "mapping"],
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
        try:
            with self.assertRaises(ValueError):
                model_parser.parse_model(tmp_path, cache_dir)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_parser_normalises_prior_mappings(self):
        """Canonical prior mappings should be written to the cache file."""
        model = {
            "model_name": "CanonicalPriorModel",
            "version": "1.0",
            "parameters": [
                {
                    "name": "delta",
                    "bounds": [-5.0, 5.0],
                    "latex_name": "\\delta",
                    "prior": {
                        "distribution": "normal",
                        "mean": 0.0,
                        "sigma": 2.0,
                    },
                },
                {
                    "name": "epsilon",
                    "bounds": [1e-5, 10.0],
                    "latex_name": "\\epsilon",
                    "prior": {
                        "distribution": "log-uniform",
                        "lower": 1e-5,
                        "upper": 10.0,
                        "transform": "identity",
                    },
                },
            ],
            "equations": {},
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yml", delete=False
        ) as tmp:
            yaml.safe_dump(model, tmp, sort_keys=False)
            tmp_path = tmp.name
        cache_dir = Path(tmp_path).parent
        cache_path = None
        try:
            cache_path = Path(model_parser.parse_model(tmp_path, cache_dir))
            with cache_path.open("r") as handle:
                cached = yaml.safe_load(handle)
            first_param = cached["parameters"][0]
            self.assertEqual(
                first_param["prior"],
                {"type": "gaussian", "mean": 0.0, "sigma": 2.0},
            )
            self.assertNotIn("transform", first_param)
            second_param = cached["parameters"][1]
            self.assertEqual(
                second_param["prior"],
                {
                    "type": "loguniform",
                    "lower": 1e-5,
                    "upper": 10.0,
                    "transform": "log",
                },
            )
            self.assertEqual(second_param.get("transform"), "log")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            if cache_path is not None:
                cache_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
