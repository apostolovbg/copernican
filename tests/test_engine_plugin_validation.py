# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Tests for ``copernican_lib.engine_plugin_validation`` helpers."""

import math
import pickle
import unittest
from types import SimpleNamespace

from copernican_lib import engine_plugin_validation
from copernican_lib.plugins import PluginValidationError

MAKE_POSTERIOR = engine_plugin_validation.make_logposterior

def _dummy_func(*_args, **_kwargs):
    """Return a placeholder numerical value."""
    return 0.0

def _linear_like(params):
    """Simple log-likelihood used to test pickling."""

    return -sum(params)

class EngineInterfaceTestCase(unittest.TestCase):
    """Validate plugin construction and associated helpers."""

    def setUp(self):
        """Build a minimal plugin for reuse across tests."""
        self.model_data = {
            "model_name": "Dummy",
            "description": "desc",
            "abstract": "abs",
            "parameters": [
                {"python_var": "h0", "latex_name": "H_0", "bounds": [60, 80]}
            ],
            "equations": {"sne": ["$$E=mc^2$$"], "bao": []},
            "cmb": {
                "param_map": {
                    "H0": "H_0",
                    "ombh2": 0.022,
                    "omch2": 0.12,
                    "Neff": 3.044,
                    "num_massive_neutrinos": 3,
                    "sum_mnu": 0.06,
                }
            },
        }
        req = engine_plugin_validation.REQUIRED_FUNCTIONS
        funcs = {name: _dummy_func for name in req}
        self.plugin = engine_plugin_validation.build_plugin(
            self.model_data, funcs
        )

    def test_plugin_validation(self):
        """Plugin built from minimal data should validate."""
        self.assertTrue(engine_plugin_validation.validate_plugin(self.plugin))

    def test_missing_attribute_fails_validation(self):
        """A plugin lacking required attributes is rejected."""
        bad = SimpleNamespace()
        with self.assertLogs(level="ERROR") as cm:
            with self.assertRaises(PluginValidationError):
                engine_plugin_validation.validate_plugin(bad)
        self.assertIn("Plugin validation issue", "".join(cm.output))

    def test_get_camb_params_expression(self):
        """LaTeX expressions in ``cmb.param_map`` evaluate correctly."""
        camb = self.plugin.get_camb_params([70.0])
        self.assertEqual(camb["H0"], 70.0)
        self.assertAlmostEqual(camb["ombh2"], 0.022)

    def test_get_camb_params_rejects_malicious_expression(self):
        """Expressions attempting attribute access raise ``ValueError``."""
        self.plugin.CMB_PARAM_MAP["bad"] = (
            "np.__class__.__mro__[2].__subclasses__()"
        )
        with self.assertRaises(ValueError):
            self.plugin.get_camb_params([70.0])

    def test_get_camb_params_rejects_recursion_depth(self):
        """Deeply nested calls exceed the evaluator's recursion limit."""
        expr = "exp(" * 30 + "1" + ")" * 30
        self.plugin.CMB_PARAM_MAP["deep"] = expr
        with self.assertRaises(ValueError):
            self.plugin.get_camb_params([70.0])

    def test_get_camb_params_rejects_node_blowup(self):
        """Expressions with too many nodes trigger a ``ValueError``."""
        expr = "+".join(["1"] * 200)
        self.plugin.CMB_PARAM_MAP["wide"] = expr
        with self.assertRaises(ValueError):
            self.plugin.get_camb_params([70.0])

    def test_cmb_param_map_rejects_invalid_keys(self):
        """Engine interface should reject unsupported CAMB parameters."""

        bad_model = dict(self.model_data)
        bad_model["cmb"] = {"param_map": {"H0": "H_0", "bad_key": 1}}
        funcs = {
            name: _dummy_func
            for name in engine_plugin_validation.REQUIRED_FUNCTIONS
        }
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(bad_model, funcs)

    def test_cmb_param_map_rejects_conflicting_neutrino_specs(self):
        """Sum and individual neutrino masses cannot be combined."""

        clash = dict(self.model_data)
        clash["cmb"] = {
            "param_map": {
                "H0": "H_0",
                "ombh2": 0.022,
                "omch2": 0.12,
                "sum_mnu": 0.06,
                "mnu1": 0.01,
            }
        }
        funcs = {
            name: _dummy_func
            for name in engine_plugin_validation.REQUIRED_FUNCTIONS
        }
        with self.assertRaises(ValueError):
            engine_plugin_validation.build_plugin(clash, funcs)

    def test_equation_sanitization(self):
        """Equations are sanitized into Matplotlib-friendly form."""
        self.assertEqual(self.plugin.MODEL_EQUATIONS_LATEX_SN[0], "$E=mc^2$")

    def test_make_logposterior_applies_priors(self):
        """Combined posterior should include prior contributions and bounds."""

        calls: list[tuple[float, ...]] = []

        def like(params):
            calls.append(tuple(params))
            return -0.5 * sum(val * val for val in params)

        like.parameter_bounds = [(0.0, 1.0), (None, None)]
        priors = [
            {"type": "uniform", "lower": 0.0, "upper": 1.0},
            {"type": "gaussian", "mean": 0.0, "sigma": 0.5},
        ]
        posterior = engine_plugin_validation.make_logposterior(like, priors)

        rejected = posterior((-0.1, 0.0))
        self.assertTrue(math.isinf(rejected) and rejected < 0)
        value = posterior((0.5, 0.0))
        base = like((0.5, 0.0))
        gaussian = -math.log(0.5 * math.sqrt(2.0 * math.pi))
        self.assertAlmostEqual(value, base + gaussian, places=7)
        self.assertIn((0.5, 0.0), calls)

    def test_make_logposterior_respects_transforms(self):
        """Transforms should apply Jacobian corrections automatically."""

        def like(params):
            return -0.5 * sum(val * val for val in params)

        def exp_transform(raw):
            transformed = math.exp(raw)
            return transformed, raw

        like.parameter_bounds = [(None, None), (0.5, 2.0)]
        like.parameter_transforms = [None, exp_transform]
        priors = [
            {"type": "uniform"},
            {"type": "uniform", "lower": 0.5, "upper": 2.0},
        ]
        posterior = engine_plugin_validation.make_logposterior(like, priors)

        # Raw value of 0.0 transforms to 1.0 and remains inside bounds.
        val = posterior((0.0, 0.0))
        expected = like((0.0, math.exp(0.0))) - math.log(2.0 - 0.5) + 0.0
        self.assertAlmostEqual(val, expected)
        # Raw value of log(3) transforms outside the allowed range and is
        # rejected.
        transformed = posterior((0.0, math.log(3.0)))
        self.assertTrue(math.isinf(transformed) and transformed < 0)

    def test_make_logposterior_is_picklable(self):
        """Posterior evaluator pickles with log-uniform transforms."""

        priors = [
            {"type": "loguniform", "lower": 1e-3, "upper": 1.0},
        ]
        posterior = MAKE_POSTERIOR(_linear_like, priors)
        payload = pickle.dumps(posterior)
        restored = pickle.loads(payload)
        self.assertAlmostEqual(restored([0.1]), posterior([0.1]))

if __name__ == "__main__":
    unittest.main()
