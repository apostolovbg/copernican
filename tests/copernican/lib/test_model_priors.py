"""Tests for parameter prior parsing and exposure."""

import math
import tempfile
import unittest
from pathlib import Path

import yaml

import copernican.lib.engine_adapter as engine_plugin_validation
import copernican.lib.model_coder as model_coder
import copernican.lib.model_spec_validator as model_spec_validator
from copernican.lib import priors as prior_mod


class PriorParsingTestCase(unittest.TestCase):
    """Ensure priors are parsed and exposed on the plugin object."""

    def setUp(self):
        base = Path(__file__).resolve().parents[3]
        models_dir = base / "copernican" / "models"
        yaml_path = models_dir / "cosmo_model_lcdm.yml"
        cache_dir = models_dir / "cache"
        cache_path = model_spec_validator.validate_and_cache_model(
            yaml_path, cache_dir
        )
        funcs, parsed = model_coder.generate_callables(cache_path)
        self.plugin = engine_plugin_validation.build_plugin(parsed, funcs)

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

    def test_public_surface(self):
        """Public prior helpers should stay importable and usable."""
        self.assertTrue(issubclass(prior_mod.PriorError, ValueError))
        self.assertTrue(
            issubclass(prior_mod.UniformPrior, prior_mod.BasePrior)
        )
        self.assertTrue(issubclass(prior_mod.NormalPrior, prior_mod.BasePrior))
        self.assertTrue(
            issubclass(prior_mod.LogUniformPrior, prior_mod.BasePrior)
        )
        self.assertTrue(issubclass(prior_mod.FixedPrior, prior_mod.BasePrior))
        self.assertTrue(hasattr(prior_mod.BasePrior, "to_mapping"))
        self.assertTrue(hasattr(prior_mod.BasePrior, "create_transform"))

        normal_prior = prior_mod.NormalPrior("gaussian", 0.0, 2.0)
        self.assertEqual(
            normal_prior.to_mapping(),
            {"type": "gaussian", "mean": 0.0, "sigma": 2.0},
        )
        self.assertIsNone(normal_prior.create_transform())

        log_prior = prior_mod.LogUniformPrior("loguniform", 1e-4, 1.0)
        self.assertEqual(
            log_prior.to_mapping(),
            {
                "type": "loguniform",
                "lower": 1e-4,
                "upper": 1.0,
                "transform": "log",
            },
        )
        self.assertIsInstance(
            log_prior.create_transform(), prior_mod.LogUniformTransform
        )

        mapping = {
            "type": "gaussian",
            "mean": 0.0,
            "sigma": 2.0,
            "transform": "identity",
        }
        prior_mod.normalise_prior_mapping(mapping)
        self.assertEqual(
            mapping,
            {"type": "gaussian", "mean": 0.0, "sigma": 2.0},
        )
        self.assertIsInstance(
            prior_mod.prior_from_mapping(
                {
                    "type": "loguniform",
                    "lower": 1e-4,
                    "upper": 1.0,
                }
            ),
            prior_mod.LogUniformPrior,
        )
        self.assertIsInstance(
            prior_mod.transform_from_mapping(
                {
                    "type": "loguniform",
                    "lower": 1e-4,
                    "upper": 1.0,
                }
            ),
            prior_mod.LogUniformTransform,
        )

    def test_loguniform_transform(self):
        """Log-uniform priors should register transforms and objects."""
        model = {
            "model_name": "LogUniformModel",
            "version": "1.0",
            "valid_for_cmb": False,
            "parameters": [
                {
                    "name": "alpha",
                    "bounds": [1e-4, 1.0],
                    "latex_name": r"\alpha",
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
        ) as temporary_file:
            yaml.safe_dump(model, temporary_file, sort_keys=False)
            temporary_path = temporary_file.name
        cache_dir = Path(temporary_path).parent
        cache_path = model_spec_validator.validate_and_cache_model(
            temporary_path, cache_dir
        )
        funcs, parsed = model_coder.generate_callables(cache_path)
        for name in engine_plugin_validation.REQUIRED_FUNCTIONS:
            funcs.setdefault(name, lambda *args, **kwargs: 0.0)
        plugin = engine_plugin_validation.build_plugin(parsed, funcs)
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
        posterior = engine_plugin_validation.make_logposterior(
            lambda vals: 0.0, plugin.PARAMETER_PRIOR_OBJECTS
        )
        self.assertEqual(float(posterior([0.1])), log_density)

    def test_fixed_prior_from_bounds(self):
        """Identical bounds promote a parameter to a fixed prior."""
        model = {
            "model_name": "FixedModel",
            "version": "1.0",
            "valid_for_cmb": False,
            "parameters": [
                {
                    "name": "c_light",
                    "bounds": [299792.458, 299792.458],
                    "latex_name": "c",
                }
            ],
            "equations": {},
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yml", delete=False
        ) as temporary_file:
            yaml.safe_dump(model, temporary_file, sort_keys=False)
            temporary_path = temporary_file.name
        cache_dir = Path(temporary_path).parent
        cache_path = None
        try:
            cache_path = model_spec_validator.validate_and_cache_model(
                temporary_path, cache_dir
            )
            funcs, parsed = model_coder.generate_callables(cache_path)
            for name in engine_plugin_validation.REQUIRED_FUNCTIONS:
                funcs.setdefault(name, lambda *args, **kwargs: 0.0)
            plugin = engine_plugin_validation.build_plugin(parsed, funcs)
            prior_obj = plugin.PARAMETER_PRIOR_OBJECTS[0]
            self.assertIsInstance(prior_obj, prior_mod.FixedPrior)
            mapping = plugin.PARAMETER_PRIORS[0]
            self.assertEqual(mapping, {"type": "fixed", "value": 299792.458})
            self.assertIn("c", plugin.FIXED_PARAMS)
            self.assertAlmostEqual(plugin.FIXED_PARAMS["c"], 299792.458)
            self.assertIn("C", plugin.FIXED_PARAMS)
            self.assertAlmostEqual(plugin.FIXED_PARAMS["C"], 299792.458)
        finally:
            Path(temporary_path).unlink(missing_ok=True)
            if cache_path is not None:
                cache_path = Path(cache_path)
                cache_path.unlink(missing_ok=True)


class PriorValidationTestCase(unittest.TestCase):
    """Invalid prior definitions must raise ValueError."""

    def test_missing_sigma(self):
        """Gaussian prior without sigma should be rejected."""
        model = {
            "model_name": "BadPrior",
            "version": "1.0",
            "valid_for_cmb": False,
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
        ) as temporary_file:
            yaml.safe_dump(model, temporary_file, sort_keys=False)
            temporary_path = temporary_file.name
        cache_dir = Path(temporary_path).parent
        try:
            with self.assertRaises(ValueError):
                model_spec_validator.validate_and_cache_model(
                    temporary_path, cache_dir
                )
        finally:
            Path(temporary_path).unlink(missing_ok=True)

    def test_loguniform_requires_positive_bounds(self):
        """Log-uniform priors reject non-positive bounds."""
        model = {
            "model_name": "BadLogUniform",
            "version": "1.0",
            "valid_for_cmb": False,
            "parameters": [
                {
                    "name": "beta",
                    "bounds": [0.0, 1.0],
                    "latex_name": r"\beta",
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
        ) as temporary_file:
            yaml.safe_dump(model, temporary_file, sort_keys=False)
            temporary_path = temporary_file.name
        cache_dir = Path(temporary_path).parent
        try:
            with self.assertRaises(ValueError):
                model_spec_validator.validate_and_cache_model(
                    temporary_path, cache_dir
                )
        finally:
            Path(temporary_path).unlink(missing_ok=True)

    def test_prior_definition_requires_mapping(self):
        """Non-mapping prior declarations should fail validation."""
        model = {
            "model_name": "BadType",
            "version": "1.0",
            "valid_for_cmb": False,
            "parameters": [
                {
                    "name": "gamma",
                    "bounds": [0.0, 1.0],
                    "latex_name": r"\gamma",
                    "prior": ["not", "a", "mapping"],
                }
            ],
            "equations": {},
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yml", delete=False
        ) as temporary_file:
            yaml.safe_dump(model, temporary_file, sort_keys=False)
            temporary_path = temporary_file.name
        cache_dir = Path(temporary_path).parent
        try:
            with self.assertRaises(ValueError):
                model_spec_validator.validate_and_cache_model(
                    temporary_path, cache_dir
                )
        finally:
            Path(temporary_path).unlink(missing_ok=True)

    def test_uniform_with_identical_bounds_rejected(self):
        """Uniform priors cannot pin a parameter to a fixed value."""
        model = {
            "model_name": "BadUniform",
            "version": "1.0",
            "valid_for_cmb": False,
            "parameters": [
                {
                    "name": "gamma",
                    "bounds": [1.0, 1.0],
                    "latex_name": r"\gamma",
                    "prior": {
                        "type": "uniform",
                        "lower": 1.0,
                        "upper": 1.0,
                    },
                }
            ],
            "equations": {},
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yml", delete=False
        ) as temporary_file:
            yaml.safe_dump(model, temporary_file, sort_keys=False)
            temporary_path = temporary_file.name
        cache_dir = Path(temporary_path).parent
        try:
            with self.assertRaises(ValueError):
                model_spec_validator.validate_and_cache_model(
                    temporary_path, cache_dir
                )
        finally:
            Path(temporary_path).unlink(missing_ok=True)

    def test_fixed_prior_must_match_bounds(self):
        """Fixed priors must agree with the declared bounds."""
        model = {
            "model_name": "BadFixed",
            "version": "1.0",
            "valid_for_cmb": False,
            "parameters": [
                {
                    "name": "delta",
                    "bounds": [0.0, 0.0],
                    "latex_name": r"\delta",
                    "prior": {"type": "fixed", "value": 1.0},
                }
            ],
            "equations": {},
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yml", delete=False
        ) as temporary_file:
            yaml.safe_dump(model, temporary_file, sort_keys=False)
            temporary_path = temporary_file.name
        cache_dir = Path(temporary_path).parent
        try:
            with self.assertRaises(ValueError):
                model_spec_validator.validate_and_cache_model(
                    temporary_path, cache_dir
                )
        finally:
            Path(temporary_path).unlink(missing_ok=True)

    def test_parser_normalises_prior_mappings(self):
        """Canonical prior mappings should be written to the cache file."""
        model = {
            "model_name": "CanonicalPriorModel",
            "version": "1.0",
            "valid_for_cmb": False,
            "parameters": [
                {
                    "name": "delta",
                    "bounds": [-5.0, 5.0],
                    "latex_name": r"\delta",
                    "prior": {
                        "type": "gaussian",
                        "mean": 0.0,
                        "sigma": 2.0,
                    },
                },
                {
                    "name": "epsilon",
                    "bounds": [1e-5, 10.0],
                    "latex_name": r"\epsilon",
                    "prior": {
                        "type": "loguniform",
                        "lower": 1e-5,
                        "upper": 10.0,
                        "transform": "identity",
                    },
                },
                {
                    "name": "zeta",
                    "bounds": [42.0, 42.0],
                    "latex_name": r"\zeta",
                },
            ],
            "equations": {},
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yml", delete=False
        ) as temporary_file:
            yaml.safe_dump(model, temporary_file, sort_keys=False)
            temporary_path = temporary_file.name
        cache_dir = Path(temporary_path).parent
        cache_path = None
        try:
            cache_path = Path(
                model_spec_validator.validate_and_cache_model(
                    temporary_path, cache_dir
                )
            )
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
            third_param = cached["parameters"][2]
            self.assertEqual(
                third_param["prior"],
                {"type": "fixed", "value": 42.0},
            )
            self.assertNotIn("transform", third_param)
        finally:
            Path(temporary_path).unlink(missing_ok=True)
            if cache_path is not None:
                cache_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
