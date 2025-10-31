# Copyright (c) 2025 Copernican Suite developers.
# Last Updated: 2025-10-31
# See LICENSE.md in the repository root for details.

"""Security tests for ``model_coder`` expression handling."""

import math
import pickle
import tempfile
import unittest
from pathlib import Path

import sympy as sp
import yaml
from scipy.integrate import quad

from copernican_lib import model_coder


class TestModelCoderSecurity(unittest.TestCase):
    """Ensure potentially dangerous expressions are not executed."""

    def test_compile_sympy_expr_blocks_import(self):
        """``_compile_sympy_expr`` should deny access to ``__import__``."""
        z = sp.symbols("z")
        malicious = sp.Function("__import__")(sp.Symbol("os"))
        fn = model_coder._compile_sympy_expr(malicious, (z,))
        with self.assertRaises(NameError):
            fn(0)

    def test_safe_parse_expr_rejects_dunder(self):
        """Expressions containing ``__`` should be rejected outright."""
        with self.assertRaises(ValueError):
            model_coder._safe_parse_expr("__import__('os')", {})

    def test_compile_sympy_expr_returns_picklable_callable(self):
        """Generated helpers should pickle under the spawn start method."""
        z = sp.symbols("z")
        expr = z + 1
        fn = model_coder._compile_sympy_expr(expr, (z,), name_hint="picklable")
        payload = pickle.dumps(fn)
        self.assertIsInstance(payload, bytes)
        restored = pickle.loads(payload)
        self.assertIsInstance(
            restored,
            model_coder._GeneratedCallable,
        )
        self.assertEqual(restored(1), 2)
        self.assertIsInstance(
            fn,
            model_coder._GeneratedCallable,
        )
        self.assertEqual(
            fn.python_function.__module__,
            "copernican_lib.model_coder",
        )
        self.assertTrue(hasattr(model_coder, fn.python_function.__name__))

    def test_compile_sympy_expr_integral_execution(self):
        """``_compile_sympy_expr`` should handle integrals safely."""
        z = sp.symbols("z")
        expr = sp.Integral(z, (z, 0, 1))  # integral of z from 0 to 1 = 0.5
        fn = model_coder._compile_sympy_expr(expr, (z,))
        self.assertAlmostEqual(fn(0), 0.5)
        self.assertEqual(
            fn.python_function.__globals__.get("__builtins__"),
            {},
        )


class TestSoundHorizonRigour(unittest.TestCase):
    """Validate the stricter sound-horizon requirements."""

    def _write_model(self, tmpdir: Path, payload: dict) -> Path:
        path = tmpdir / "model.yml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                payload, handle, sort_keys=False, allow_unicode=True
            )
        return path

    def test_generate_callables_requires_rs_expression_for_bao(self):
        """BAO-capable models must provide an explicit ``rs_expression``."""

        payload = {
            "parameters": [
                {
                    "name": "Hubble",
                    "python_var": "H0",
                    "bounds": [60.0, 80.0],
                },
                {
                    "name": "Omega_b",
                    "python_var": "Omega_b",
                    "bounds": [0.04, 0.06],
                },
                {
                    "name": "Omega_gamma",
                    "python_var": "Omega_gamma",
                    "bounds": [4e-5, 6e-5],
                },
                {
                    "name": "z_rec",
                    "python_var": "z_rec",
                    "bounds": [900.0, 1200.0],
                },
            ],
            "Hz_expression": "H(z) = H0 * sqrt(1 + z)",
            "predicts_bao": True,
            "valid_for_bao": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache = self._write_model(tmp_path, payload)
            with self.assertRaises(ValueError):
                model_coder.generate_callables(cache)

    def test_sound_horizon_uses_supplied_hubble_curve(self):
        """``rs_expression`` integrals must rely on the model's ``H(z)``."""

        payload = {
            "parameters": [
                {
                    "name": "Hubble",
                    "python_var": "H0",
                    "bounds": [70.0, 70.0],
                },
                {
                    "name": "Matter",
                    "python_var": "Omega_m0",
                    "bounds": [0.3, 0.3],
                },
                {
                    "name": "Baryon",
                    "python_var": "Omega_b",
                    "bounds": [0.05, 0.05],
                },
                {
                    "name": "Photon",
                    "python_var": "Omega_gamma",
                    "bounds": [5e-5, 5e-5],
                },
                {
                    "name": "Recombination",
                    "python_var": "z_rec",
                    "bounds": [1100.0, 1100.0],
                },
            ],
            "Hz_expression": (
                "H(z) = H0 * sqrt(Omega_m0*(1 + z)**3 + (1 - Omega_m0))"
            ),
            "rs_expression": (
                "r_s = Integral("
                "299792.458 / sqrt("
                "3 * (1 + 3 * Omega_b / (4 * Omega_gamma) / (1 + z))"
                ") / ("
                "H0 * sqrt(Omega_m0 * (1 + z)**3 + (1 - Omega_m0))"
                "), (z, z_rec, oo))"
            ),
            "predicts_bao": True,
            "valid_for_bao": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache = self._write_model(tmp_path, payload)
            funcs, data = model_coder.generate_callables(cache)

        hubble = 70.0
        omega_m0 = 0.3
        omega_b = 0.05
        omega_gamma = 5e-5
        z_rec = 1100.0
        rs_model = funcs["get_sound_horizon_rs_Mpc"](
            hubble, omega_m0, omega_b, omega_gamma, z_rec
        )

        def integrand(z_val: float) -> float:
            baryon_ratio = 3.0 * omega_b / (4.0 * omega_gamma) / (1.0 + z_val)
            sound_speed = 299792.458 / math.sqrt(3.0 * (1.0 + baryon_ratio))
            hubble_curve = hubble * math.sqrt(
                omega_m0 * (1.0 + z_val) ** 3 + (1.0 - omega_m0)
            )
            return sound_speed / hubble_curve

        rs_expected = quad(integrand, z_rec, math.inf, limit=200)[0]
        self.assertAlmostEqual(rs_model, rs_expected, places=6)
        self.assertTrue(data["valid_for_bao"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
