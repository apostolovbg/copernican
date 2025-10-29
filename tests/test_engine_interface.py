# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Tests for ``copernican_lib.engine_interface`` helpers."""

import unittest
from types import SimpleNamespace

from copernican_lib import engine_interface


def _dummy_func(*_args, **_kwargs):
    """Return a placeholder numerical value."""
    return 0.0


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
            "cmb": {"param_map": {"H0": "H_0", "ombh2": 0.022}},
        }
        req = engine_interface.REQUIRED_FUNCTIONS
        funcs = {name: _dummy_func for name in req}
        self.plugin = engine_interface.build_plugin(self.model_data, funcs)

    def test_plugin_validation(self):
        """Plugin built from minimal data should validate."""
        self.assertTrue(engine_interface.validate_plugin(self.plugin))

    def test_missing_attribute_fails_validation(self):
        """A plugin lacking required attributes is rejected."""
        bad = SimpleNamespace()
        with self.assertLogs(level="ERROR") as cm:
            self.assertFalse(engine_interface.validate_plugin(bad))
        self.assertIn("Plugin validation failed", "".join(cm.output))

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

    def test_equation_sanitization(self):
        """Equations are sanitized into Matplotlib-friendly form."""
        self.assertEqual(self.plugin.MODEL_EQUATIONS_LATEX_SN[0], "$E=mc^2$")


if __name__ == "__main__":
    unittest.main()
