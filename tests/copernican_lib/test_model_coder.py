# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Security tests for ``model_coder`` expression handling."""

import math
import multiprocessing as mp
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import sympy as sp
import yaml
from scipy.integrate import IntegrationWarning, quad

from copernican_lib import model_coder


def _evaluate_generated_callable(fn):
    """Return the generated helper evaluation from a worker process."""

    return fn(1)


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
        with mp.get_context("spawn").Pool(1) as pool:
            restored_value = pool.apply(_evaluate_generated_callable, (fn,))
        self.assertEqual(restored_value, 2)
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
            "skip_bao": False,
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
            "skip_bao": False,
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

    def test_sound_horizon_divergence_raises_signal(self):
        """Divergent ``rs_expression`` integrals must raise a clear error."""

        payload = {
            "parameters": [
                {
                    "name": "Hubble",
                    "python_var": "H0",
                    "bounds": [70.0, 70.0],
                },
                {
                    "name": "Recombination",
                    "python_var": "z_rec",
                    "bounds": [1100.0, 1100.0],
                },
            ],
            "Hz_expression": "H(z) = H0",
            "rs_expression": ("r_s = Integral(1 / (1 + z), (z, z_rec, oo))"),
            "skip_bao": False,
            "valid_for_bao": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache = self._write_model(tmp_path, payload)
            funcs, _ = model_coder.generate_callables(cache)

        rs_helper = funcs["get_sound_horizon_rs_Mpc"]
        with self.assertRaises(model_coder.SoundHorizonComputationError):
            rs_helper(70.0, 1100.0)


class TestRobustQuad(unittest.TestCase):
    """Validate the resilient quadrature wrapper used by generated models."""

    def test_robust_quad_escalates_limit(self):
        """The helper should retry with larger limits when warnings occur."""

        call_limits = []

        def fake_quad(*args, **kwargs):
            limit = kwargs.get("limit")
            call_limits.append(limit)
            if limit is None or limit < 400:
                raise IntegrationWarning("Maximum subdivisions reached")
            a_val, b_val = args[1], args[2]
            # Integral of x from a to b equals (b^2 - a^2)/2.
            return ((b_val**2 - a_val**2) / 2.0, 1e-12)

        with mock.patch.object(
            model_coder, "_SCIPY_QUAD", side_effect=fake_quad
        ):
            result, err = model_coder.robust_quad(lambda x: x, 0.0, 1.0)

        self.assertAlmostEqual(result, 0.5, places=12)
        self.assertLess(err, 1e-6)
        self.assertGreaterEqual(len(call_limits), 2)
        self.assertIn(400, call_limits)

    def test_robust_quad_splits_interval(self):
        """Finite integrals split into sub-intervals when retries fail."""

        calls = []

        def fake_quad(func, a_val, b_val, *args, **kwargs):
            limit = kwargs.get("limit")
            calls.append((a_val, b_val, limit))
            if abs(b_val - a_val) > 0.3:
                raise IntegrationWarning("Interval too wide")
            # Analytic integral of x^2 from a to b.
            return ((b_val**3 - a_val**3) / 3.0, 1e-12)

        with mock.patch.object(
            model_coder, "_SCIPY_QUAD", side_effect=fake_quad
        ):
            result, err = model_coder.robust_quad(
                lambda x: x**2, 0.0, 1.0, max_attempts=1
            )

        self.assertAlmostEqual(result, 1.0 / 3.0, places=6)
        self.assertLess(err, 1e-6)
        # Ensure at least one fallback call used a narrower segment.
        self.assertTrue(
            any(abs(end - start) <= 0.3 for start, end, _ in calls[1:])
        )

    def test_robust_quad_transforms_positive_infinity(self):
        """Semi-infinite integrals map onto a finite logistic domain."""

        calls = []
        original_quad = model_coder._SCIPY_QUAD

        def fake_quad(func, a_val, b_val, *args, **kwargs):
            calls.append((a_val, b_val))
            if math.isinf(a_val) or math.isinf(b_val):
                raise IntegrationWarning("force logistic fallback")
            return original_quad(func, a_val, b_val, *args, **kwargs)

        with (
            mock.patch.object(
                model_coder, "_SCIPY_QUAD", side_effect=fake_quad
            ),
            mock.patch.object(model_coder.LOGGER, "warning") as mock_warning,
        ):
            result, err = model_coder.robust_quad(
                lambda x: 1.0 / (1.0 + x**2),
                0.0,
                math.inf,
            )

        self.assertAlmostEqual(result, math.pi / 2.0, places=6)
        self.assertLess(err, 1e-6)
        self.assertTrue(
            any(
                not math.isinf(start)
                and not math.isinf(end)
                and math.isclose(end, 1.0, rel_tol=1e-12, abs_tol=1e-12)
                for start, end in calls
            )
        )
        mock_warning.assert_not_called()

    def test_robust_quad_handles_two_sided_infinity(self):
        """Two-sided infinite integrals split into manageable segments."""

        calls = []
        original_quad = model_coder._SCIPY_QUAD

        def fake_quad(func, a_val, b_val, *args, **kwargs):
            calls.append((a_val, b_val))
            if math.isinf(a_val) or math.isinf(b_val):
                raise IntegrationWarning("force logistic fallback")
            return original_quad(func, a_val, b_val, *args, **kwargs)

        with (
            mock.patch.object(
                model_coder, "_SCIPY_QUAD", side_effect=fake_quad
            ),
            mock.patch.object(model_coder.LOGGER, "warning") as mock_warning,
        ):
            result, err = model_coder.robust_quad(
                lambda x: 1.0 / (1.0 + x**2),
                -math.inf,
                math.inf,
                points=(0.0,),
            )

        self.assertAlmostEqual(result, math.pi, places=6)
        self.assertLess(err, 1e-6)
        # Confirm the helper evaluated at least two logistic segments.
        finite_segments = [
            pair for pair in calls if all(map(math.isfinite, pair))
        ]
        self.assertGreaterEqual(len(finite_segments), 2)
        mock_warning.assert_not_called()

    def test_robust_quad_raises_when_warnings_persist(self):
        """Persistent ``IntegrationWarning`` emissions must raise errors."""

        def failing_quad(*_args, **_kwargs):
            warnings.warn(
                "forced failure for test coverage",
                IntegrationWarning,
            )

        with mock.patch.object(
            model_coder, "_SCIPY_QUAD", side_effect=failing_quad
        ):
            with self.assertRaises(model_coder.RobustQuadFailure) as context:
                model_coder.robust_quad(lambda x: x, 0.0, 1.0, max_attempts=1)

        self.assertIn("exhausted retries", str(context.exception))

    def test_robust_quad_flags_logistic_divergence(self):
        """Logistic divergence should trigger a ``RobustQuadFailure``."""

        with self.assertRaises(model_coder.RobustQuadFailure):
            model_coder.robust_quad(
                lambda z_val: 1.0 / (1.0 + z_val),
                1100.0,
                math.inf,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
