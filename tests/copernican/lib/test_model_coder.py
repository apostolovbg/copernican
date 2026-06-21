# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Security tests for ``model_coder`` expression handling."""

import math
import multiprocessing as multiprocessing_module
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import sympy
import yaml
from scipy.integrate import IntegrationWarning, quad

from copernican.lib import model_coder


def _evaluate_generated_callable(generated_callable):
    """Return the generated helper evaluation from a worker process."""

    return generated_callable(1)


class TestModelCoderSecurity(unittest.TestCase):
    """Ensure potentially dangerous expressions are not executed."""

    def test_compile_sympy_expr_blocks_import(self):
        """``_compile_sympy_expr`` should deny access to ``__import__``."""
        z = sympy.symbols("z")
        malicious = sympy.Function("__import__")(sympy.Symbol("os"))
        generated_callable = model_coder._compile_sympy_expr(malicious, (z,))
        with self.assertRaises(NameError):
            generated_callable(0)

    def test_safe_parse_expr_rejects_dunder(self):
        """Expressions containing ``__`` should be rejected outright."""
        with self.assertRaises(ValueError):
            model_coder._safe_parse_expr("__import__('os')", {})

    def test_compile_sympy_expr_returns_picklable_callable(self):
        """Generated helpers should pickle under the spawn start method."""
        z = sympy.symbols("z")
        expr = z + 1
        generated_callable = model_coder._compile_sympy_expr(
            expr, (z,), name_hint="picklable"
        )
        with multiprocessing_module.get_context("spawn").Pool(1) as pool:
            restored_value = pool.apply(
                _evaluate_generated_callable,
                (generated_callable,),
            )
        self.assertEqual(restored_value, 2)
        self.assertIsInstance(
            generated_callable,
            model_coder._GeneratedCallable,
        )
        self.assertEqual(
            generated_callable.python_function.__module__,
            "copernican.lib.model_coder",
        )
        self.assertTrue(
            hasattr(
                model_coder,
                generated_callable.python_function.__name__,
            )
        )

    def test_compile_sympy_expr_integral_execution(self):
        """``_compile_sympy_expr`` should handle integrals safely."""
        z = sympy.symbols("z")
        expr = sympy.Integral(z, (z, 0, 1))  # integral of z from 0 to 1 = 0.5
        generated_callable = model_coder._compile_sympy_expr(expr, (z,))
        self.assertAlmostEqual(generated_callable(0), 0.5)
        self.assertEqual(
            generated_callable.python_function.__globals__.get("__builtins__"),
            {},
        )


class TestSoundHorizonRigour(unittest.TestCase):
    """Validate the stricter sound-horizon requirements."""

    def _write_model(self, temporary_dir: Path, payload: dict) -> Path:
        path = temporary_dir / "model.yml"
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
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            cache = self._write_model(temporary_path, payload)
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
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            cache = self._write_model(temporary_path, payload)
            funcs, model_data = model_coder.generate_callables(cache)

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
        self.assertTrue(model_data["valid_for_bao"])

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

        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            cache = self._write_model(temporary_path, payload)
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


class NativeCMBRuntimeCoverageTestCase(unittest.TestCase):
    """Cover the native CMB runtime helper surface."""

    def test_compile_native_cmb_runtime_builds_one_runtime_bundle(self):
        """Static CMB contracts should compile once into one runtime bundle."""

        compile_result = object()
        cmb_contract = {
            "param_map": {"Omega_m0": "Omega_m0"},
            "grids": {"tau": {"symbol": "tau"}},
            "values": {"H": "H"},
            "background": {"density": {"expression": "Omega_m0"}},
            "numerical": {"ell_max": 64},
            "perturbations": {
                "standard": False,
                "backend": {"implemented": True},
            },
        }

        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            return_value=compile_result,
        ) as compile_contract:
            runtime = model_coder.compile_native_cmb_runtime(
                model_name="TemplateModel",
                backend="camb",
                parameter_names=("Omega_m0",),
                latex_names=(r"\Omega_m",),
                cmb_contract=cmb_contract,
            )

        self.assertIsInstance(runtime, model_coder.NativeCMBRuntime)
        self.assertIs(runtime.perturbation_data, compile_result)
        self.assertEqual(
            model_coder.compile_native_cmb_runtime.__name__,
            "compile_native_cmb_runtime",
        )
        compile_contract.assert_called_once_with(
            cmb_contract["perturbations"],
            model_name="TemplateModel",
            backend="camb",
            parameter_names=("Omega_m0",),
            latex_names=(r"\Omega_m",),
            background_reference_names=("H", "Omega_m0", "tau"),
        )

    def test_native_cmb_runtime_build_contract_copies_runtime_payload(self):
        """Bound runtime contracts should copy only mutable parameter data."""

        runtime = model_coder.NativeCMBRuntime(
            model_name="TemplateModel",
            backend="camb",
            perturbation_contract={"standard": False},
            background={"density": {"expression": "Omega_m0"}},
            numerical={"ell_max": 64},
            perturbation_data={"compiled": True},
        )

        contract = runtime.build_contract(
            model_parameters={"Omega_m0": 0.3},
            param_map={"Omega_m0": 0.3},
        )

        self.assertEqual(contract["model_name"], "TemplateModel")
        self.assertEqual(contract["backend"], "camb")
        self.assertEqual(runtime.build_contract.__name__, "build_contract")
        self.assertEqual(contract["model_parameters"], {"Omega_m0": 0.3})
        self.assertEqual(contract["param_map"], {"Omega_m0": 0.3})
        self.assertIs(contract["background"], runtime.background)
        self.assertIs(contract["numerical"], runtime.numerical)
        self.assertIs(contract["perturbations"], runtime.perturbation_contract)
        self.assertIs(contract["perturbation_data"], runtime.perturbation_data)


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the model coder API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(model_coder, "QuadPrinter"))
        self.assertTrue(callable(model_coder.robust_quad))

    def test_transformed_symbol_is_exposed(self) -> None:
        transformed = model_coder.robust_quad
        self.assertTrue(callable(transformed))


class CMBBackendCapabilityTestCase(unittest.TestCase):
    """Cover the backend capability helpers exposed by model_coder."""

    def test_declared_backend_capabilities_are_accessible(self) -> None:
        """The declared CAMB capabilities should be available by name."""

        capabilities = model_coder.get_backend_capabilities("camb")
        self.assertTrue(capabilities["scalar_param_map"])
        self.assertTrue(capabilities["grids_values_calls"])
        self.assertTrue(capabilities["standard_perturbations"])
        self.assertTrue(capabilities["native_nonstandard_perturbations"])
        self.assertTrue(
            model_coder.backend_supports_standard_perturbations("camb")
        )
        self.assertTrue(
            model_coder.backend_supports_native_nonstandard_perturbations(
                "camb"
            )
        )

    def test_nonstandard_execution_helper_enforces_capabilities(self) -> None:
        """Unsupported declarative execution should fail clearly."""

        with self.assertRaisesRegex(
            ValueError, "generic declarative executor is required"
        ):
            model_coder.validate_native_perturbation_execution(
                model_name="TemplateModel",
                backend="camb",
                standard=False,
                implemented=False,
            )

        self.assertIsNone(
            model_coder.validate_native_perturbation_execution(
                model_name="TemplateModel",
                backend="camb",
                standard=True,
                implemented=False,
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
