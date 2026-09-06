# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Security tests for ``model_coder`` expression handling."""

import copy
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

    def test_bao_sound_horizon_uses_drag_epoch_helper(self):
        """Generated BAO helpers expose drag and recombination horizons."""

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

        params = (70.0, 0.3, 0.05, 5e-5, 1100.0)
        recombination = funcs["get_sound_horizon_rs_rec_Mpc"](*params)
        drag = funcs["get_sound_horizon_rs_drag_Mpc"](*params)
        self.assertGreater(funcs["get_bao_drag_redshift"](*params), 0.0)
        self.assertTrue(model_data["bao_sound_horizon_epoch"] == "drag")
        self.assertGreater(drag, 0.0)
        self.assertNotAlmostEqual(drag, recombination)

    def test_declared_drag_redshift_overrides_standard_fit(self):
        """A theory may declare its own drag epoch without LCDM defaults."""

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
                "), (z, z_rec + 10, oo))"
            ),
            "bao_drag_redshift_expression": "z_drag = z_rec - 50",
            "skip_bao": False,
            "valid_for_bao": True,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            cache = self._write_model(temporary_path, payload)
            funcs, model_data = model_coder.generate_callables(cache)

        params = (70.0, 0.3, 0.05, 5e-5, 1100.0)
        self.assertEqual(funcs["get_bao_drag_redshift"](*params), 1050.0)
        self.assertEqual(
            model_data["bao_drag_redshift_expression"], "z_rec - 50"
        )
        self.assertGreater(
            funcs["get_sound_horizon_rs_drag_Mpc"](*params), 0.0
        )

    def test_drag_replacement_targets_outer_sound_horizon_integral(self):
        """Nested integrals retain their own limits during drag conversion."""

        x, y, z_rec, z_drag = sympy.symbols("x y z_rec z_drag")
        inner = sympy.Integral(y, (y, 0, 1))
        expression = sympy.Integral(inner + x, (x, z_rec, sympy.oo))

        replaced = model_coder._replace_sound_horizon_lower_limit(
            expression,
            z_drag,
        )

        self.assertIsNotNone(replaced)
        self.assertEqual(replaced.limits[0][1], z_drag)
        self.assertIn(inner, replaced.function.args)

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


class DeclaredCMBRuntimeCoverageTestCase(unittest.TestCase):
    """Cover the declared CMB runtime helper surface."""

    def test_compile_declared_cmb_runtime_builds_one_runtime_bundle(self):
        """Static CMB contracts should compile once into one runtime bundle."""

        compile_result = object()
        cmb_contract = {
            "param_map": {"Omega_m0": "Omega_m0"},
            "grids": {"tau": {"symbol": "tau"}},
            "values": {"H": "H"},
            "background": {
                "derived": {"density": "Omega_m0"},
                "recombination": {
                    "quantities": {
                        "hydrogen_temperature_K": "3000.0",
                        "hydrogen_alpha_B": "1.0e-19",
                        "beta_continuum": "5.0e-20",
                        "peebles_c": "0.75",
                    },
                },
                "reionization": {
                    "calibration": {
                        "symbol": "z_reio",
                        "target_optical_depth": "0.05 + 0.01",
                    },
                    "quantities": {
                        "hydrogen_ionization_rate": "density",
                    },
                },
            },
            "numerical": {"ell_max": 64},
            "perturbations": {},
        }

        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            return_value=compile_result,
        ) as compile_contract:
            runtime = model_coder.compile_declared_cmb_runtime(
                model_name="TemplateModel",
                parameter_names=("Omega_m0",),
                latex_names=(r"\Omega_m",),
                cmb_contract=cmb_contract,
            )

        self.assertIsInstance(runtime, model_coder.DeclaredCMBRuntime)
        self.assertIs(runtime.perturbation_data, compile_result)
        self.assertTrue(
            runtime.runtime_signature.startswith("declared-cmb-runtime:")
        )
        self.assertIsNotNone(runtime.compile_diagnostics)
        self.assertTrue(runtime.compile_diagnostics.compiled_upstream)
        self.assertFalse(
            runtime.compile_diagnostics.hot_path_recompilation_allowed
        )
        self.assertEqual(len(runtime.background_runtime.derived_plan), 1)
        self.assertEqual(
            len(runtime.background_runtime.recombination_quantity_plan),
            4,
        )
        self.assertEqual(
            runtime.background_runtime.reionization_calibration_symbol,
            "z_reio",
        )
        self.assertIsNotNone(
            runtime.background_runtime.reionization_target_tau
        )
        self.assertEqual(
            model_coder.compile_declared_cmb_runtime.__name__,
            "compile_declared_cmb_runtime",
        )
        compile_contract.assert_called_once_with(
            cmb_contract["perturbations"],
            model_name="TemplateModel",
            parameter_names=("Omega_m0",),
            latex_names=(r"\Omega_m",),
            background_reference_names=(
                "H",
                "Omega_m0",
                "beta_continuum",
                "density",
                "hydrogen_alpha_B",
                "hydrogen_ionization_rate",
                "hydrogen_temperature_K",
                "peebles_c",
                "tau",
            ),
        )

    def test_compile_declared_cmb_runtime_compiles_generic_recombination_roles(
        self,
    ):
        """Generic recombination roles compile without Peebles names."""

        compile_result = object()
        cmb_contract = {
            "param_map": {"Omega_m0": "Omega_m0"},
            "grids": {},
            "values": {},
            "background": {
                "derived": {"density": "Omega_m0"},
                "recombination": {
                    "quantities": {
                        "state": "1.0 / (1.0 + exp(z))",
                        "xe": "0.1 + 0.8 * state",
                        "opacity_rate": "1.0e-4 * xe",
                        "visibility_kernel": "exp(-z ** 2)",
                    },
                    "roles": {
                        "state": "state",
                        "rate": "opacity_rate",
                        "electron_fraction": "xe",
                        "opacity": "opacity_rate",
                        "visibility": "visibility_kernel",
                    },
                },
            },
            "numerical": {},
            "calls": [],
            "perturbations": {},
        }

        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            return_value=compile_result,
        ):
            runtime = model_coder.compile_declared_cmb_runtime(
                model_name="GenericRecombinationModel",
                parameter_names=("Omega_m0",),
                latex_names=(r"\Omega_m",),
                cmb_contract=cmb_contract,
            )

        self.assertEqual(
            dict(runtime.background_runtime.recombination_roles),
            {
                "electron_fraction": "xe",
                "opacity": "opacity_rate",
                "rate": "opacity_rate",
                "state": "state",
                "visibility": "visibility_kernel",
            },
        )

    def test_compile_declared_cmb_runtime_rejects_incomplete_generic_roles(
        self,
    ):
        """Generic role graphs must declare all required physical outputs."""

        cmb_contract = {
            "param_map": {"Omega_m0": "Omega_m0"},
            "grids": {},
            "values": {},
            "background": {
                "recombination": {
                    "quantities": {"xe": "0.5"},
                    "roles": {"electron_fraction": "xe"},
                }
            },
            "numerical": {},
            "calls": [],
            "perturbations": {},
        }

        with self.assertRaises(ValueError) as context:
            model_coder.compile_declared_cmb_runtime(
                model_name="IncompleteGenericRecombinationModel",
                parameter_names=("Omega_m0",),
                latex_names=(r"\Omega_m",),
                cmb_contract=cmb_contract,
            )
        self.assertIn("opacity", str(context.exception))
        self.assertIn("visibility", str(context.exception))

    def test_compile_declared_cmb_runtime_reuses_cached_runtime_bundle(self):
        """Repeated compilation requests should reuse one cached runtime."""

        compile_result = object()
        cmb_contract = {
            "param_map": {"Omega_m0": "Omega_m0"},
            "grids": {},
            "values": {},
            "background": {},
            "numerical": {},
            "calls": [],
            "perturbations": {},
        }

        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            return_value=compile_result,
        ) as compile_contract:
            first = model_coder.compile_declared_cmb_runtime(
                model_name="CachedTemplateModel",
                parameter_names=("Omega_m0",),
                latex_names=(r"\Omega_m",),
                cmb_contract=cmb_contract,
            )
            second = model_coder.compile_declared_cmb_runtime(
                model_name="CachedTemplateModel",
                parameter_names=("Omega_m0",),
                latex_names=(r"\Omega_m",),
                cmb_contract=cmb_contract,
            )

        self.assertIs(first, second)
        compile_contract.assert_called_once()

    def test_compile_declared_cmb_runtime_ignores_bound_parameter_values(self):
        """Runtime compilation should key on structure, not bound values."""

        compile_result = object()
        first_contract = {
            "model_parameters": {"Tcmb_K": 2.7255},
            "param_map": {"H0": 67.4, "ombh2": 0.02237},
            "grids": {},
            "values": {},
            "background": {
                "derived": {"Omega_b0": "ombh2 / ((H0 / 100.0) ** 2)"},
            },
            "numerical": {"ell_max": 64},
            "calls": [],
            "perturbations": {
                "contract_version": 2,
                "gauge": "conformal_newtonian",
                "variables": {
                    "delta_x": {"kind": "density_contrast"},
                },
                "derived": {},
                "equations": {
                    "evolve_delta_x": {
                        "lhs": {
                            "kind": "derivative",
                            "variable": "delta_x",
                            "wrt": "tau",
                            "order": 1,
                        },
                        "rhs": "-0.01 * delta_x",
                        "role": "continuity",
                    }
                },
                "constraints": {},
                "closures": {},
                "sources": {
                    "signal": {
                        "expression": "delta_x",
                        "role": "signal",
                    }
                },
                "observables": {
                    "transfer": {
                        "kind": "transfer_component",
                        "projection": "line_of_sight_signal",
                        "source_terms": {"signal": "signal"},
                    },
                    "TT": {
                        "kind": "angular_power_spectrum",
                        "primary": "transfer",
                        "secondary": "transfer",
                    },
                },
                "initial_conditions": {
                    "delta_seed": {
                        "target": {
                            "variable": "delta_x",
                            "wrt": "tau",
                            "order": 0,
                        },
                        "expression": "seed",
                    }
                },
                "boundary_conditions": {},
                "validity": {"regimes": ["synthetic"]},
            },
        }
        second_contract = copy.deepcopy(first_contract)
        second_contract["param_map"]["H0"] = 70.0
        second_contract["param_map"]["ombh2"] = 0.024
        second_contract["model_parameters"]["Tcmb_K"] = 2.73

        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            return_value=compile_result,
        ) as compile_contract:
            first = model_coder.compile_declared_cmb_runtime(
                model_name="StructuralCacheModel",
                parameter_names=("H0", "ombh2", "Tcmb_K"),
                latex_names=("H_0", "\\omega_b", "T_{cmb}"),
                cmb_contract=first_contract,
            )
            second = model_coder.compile_declared_cmb_runtime(
                model_name="StructuralCacheModel",
                parameter_names=("H0", "ombh2", "Tcmb_K"),
                latex_names=("H_0", "\\omega_b", "T_{cmb}"),
                cmb_contract=second_contract,
            )

        self.assertIs(first, second)
        compile_contract.assert_called_once()

    def test_declared_cmb_runtime_build_contract_reuses_frozen_payload(self):
        """Bound contracts should reuse immutable structural runtime assets."""

        runtime = model_coder.DeclaredCMBRuntime(
            model_name="TemplateModel",
            perturbation_contract={"gauge": "conformal_newtonian"},
            background={"density": {"expression": "Omega_m0"}},
            numerical={"ell_max": 64},
            perturbation_data={"compiled": True},
            grids={"tau": {"symbol": "tau"}},
            values={"H": "H0"},
            calls=({"method": "set_cosmology"},),
            background_runtime=model_coder.DeclaredCMBBackgroundRuntime(
                derived_plan=(),
                recombination_quantity_plan=(),
                reionization_quantity_plan=(),
                reionization_target_tau=None,
                reionization_calibration_symbol=None,
            ),
            runtime_signature="declared-cmb-runtime:test",
            compile_diagnostics=model_coder.DeclaredCMBCompileDiagnostics(
                runtime_signature="declared-cmb-runtime:test",
                compiler="compiler",
                compiled_upstream=True,
                hot_path_recompilation_allowed=False,
                parameter_names=("Omega_m0",),
                background_reference_names=("Omega_m0",),
            ),
        )

        contract = runtime.build_contract(
            model_parameters={"Omega_m0": 0.3},
            param_map={"Omega_m0": 0.3},
        )

        self.assertEqual(contract["model_name"], "TemplateModel")
        self.assertNotIn("backend", contract)
        self.assertEqual(runtime.build_contract.__name__, "build_contract")
        self.assertEqual(contract["model_parameters"], {"Omega_m0": 0.3})
        self.assertEqual(contract["param_map"], {"Omega_m0": 0.3})
        self.assertIsInstance(
            runtime.background,
            model_coder.DeclaredFrozenMapping,
        )
        self.assertIs(contract["background"], runtime.background)
        self.assertIs(
            contract["background_runtime"],
            runtime.background_runtime,
        )
        self.assertIs(contract["grids"], runtime.grids)
        self.assertIs(contract["values"], runtime.values)
        self.assertIs(contract["calls"], runtime.calls)
        self.assertIs(contract["numerical"], runtime.numerical)
        self.assertIs(
            contract["perturbations"],
            runtime.perturbation_contract,
        )
        with self.assertRaises(TypeError):
            contract["background"]["density"] = {}
        with self.assertRaises(TypeError):
            contract["background"]["density"]["expression"] = "0.0"
        self.assertIs(contract["perturbation_data"], runtime.perturbation_data)
        self.assertEqual(
            contract["runtime_signature"],
            "declared-cmb-runtime:test",
        )
        self.assertIs(
            contract["compile_diagnostics"],
            runtime.compile_diagnostics,
        )
        self.assertEqual(
            runtime.background["density"]["expression"],
            "Omega_m0",
        )
        self.assertEqual(runtime.numerical["ell_max"], 64)
        self.assertEqual(runtime.calls[0]["method"], "set_cosmology")
        self.assertEqual(
            runtime.perturbation_contract["gauge"],
            "conformal_newtonian",
        )

    def test_prepare_declared_cmb_execution_contract_binds_precompiled_data(
        self,
    ) -> None:
        """Direct declared contracts should be prepared before execution."""

        compile_result = object()
        cmb_contract = {
            "model_name": "PreparedModel",
            "param_map": {"H0": 67.4, "ombh2": 0.02237},
            "model_parameters": {"Tcmb_K": 2.7255},
            "background": {
                "derived": {"H": "H0"},
            },
            "grids": {},
            "values": {},
            "calls": [],
            "numerical": {},
            "perturbations": {
                "contract_version": 2,
                "gauge": "conformal_newtonian",
                "variables": {"delta_x": {"kind": "density_contrast"}},
                "derived": {},
                "equations": {
                    "evolve_delta_x": {
                        "lhs": {
                            "kind": "derivative",
                            "variable": "delta_x",
                            "wrt": "tau",
                            "order": 1,
                        },
                        "rhs": "-delta_x",
                        "role": "continuity",
                    }
                },
                "constraints": {},
                "closures": {},
                "collision_operators": {},
                "sources": {
                    "temperature_source": {
                        "expression": "visibility * delta_x",
                        "role": "monopole",
                    }
                },
                "observables": {
                    "temperature": {
                        "kind": "transfer_component",
                        "projection": "line_of_sight_temperature",
                        "source_terms": {
                            "monopole": "temperature_source",
                        },
                    }
                },
                "initial_conditions": {
                    "delta_seed": {
                        "target": {
                            "variable": "delta_x",
                            "wrt": "tau",
                            "order": 0,
                        },
                        "expression": "1.0",
                    }
                },
                "initial_condition_families": {},
                "boundary_conditions": {},
                "sectors": {},
                "species": {},
                "hierarchy_families": {},
                "projection_typing": {},
                "accuracy_controls": {},
                "validity": {"regimes": ["synthetic"]},
            },
        }

        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            return_value=compile_result,
        ):
            prepared = model_coder.prepare_declared_cmb_execution_contract(
                cmb_contract
            )

        self.assertIs(prepared["perturbation_data"], compile_result)
        self.assertIn("background_runtime", prepared)
        self.assertTrue(
            prepared["runtime_signature"].startswith("declared-cmb-runtime:")
        )
        self.assertIsNotNone(prepared["compile_diagnostics"])

    def test_prepare_declared_contract_needs_no_solver_route_metadata(
        self,
    ) -> None:
        """Declared preparation should compile a route-neutral CMB contract."""

        compile_result = object()
        cmb_contract = {
            "model_name": "RouteNeutralModel",
            "param_map": {"H0": 67.4},
            "model_parameters": {},
            "background": {},
            "grids": {},
            "values": {},
            "calls": [],
            "numerical": {},
            "perturbations": {
                "contract_version": 2,
                "gauge": "conformal_newtonian",
            },
        }

        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            return_value=compile_result,
        ):
            prepared = model_coder.prepare_declared_cmb_execution_contract(
                cmb_contract
            )

        self.assertIs(prepared["perturbation_data"], compile_result)
        self.assertNotIn("backend", prepared)
        self.assertNotIn("standard", prepared["perturbations"])
        self.assertNotIn("backend_mapping", prepared["perturbations"])

    def test_prepare_declared_contract_rejects_removed_route_keys(
        self,
    ) -> None:
        """Removed solver selectors must fail before graph compilation."""

        base_contract = {
            "model_name": "RouteNeutralModel",
            "param_map": {},
            "model_parameters": {},
            "background": {},
            "grids": {},
            "values": {},
            "calls": [],
            "numerical": {},
            "perturbations": {
                "contract_version": 2,
                "gauge": "conformal_newtonian",
            },
        }
        removed_entries = (
            ("backend", "camb"),
            ("standard", False),
            ("backend_mapping", {"camb": {"implemented": True}}),
        )

        for key, value in removed_entries:
            with self.subTest(key=key):
                contract = copy.deepcopy(base_contract)
                target = (
                    contract if key == "backend" else contract["perturbations"]
                )
                target[key] = value
                with self.assertRaisesRegex(ValueError, "removed route key"):
                    model_coder.prepare_declared_cmb_execution_contract(
                        contract
                    )
                with self.assertRaisesRegex(ValueError, "removed route key"):
                    model_coder.compile_declared_cmb_runtime(
                        model_name="RouteNeutralModel",
                        parameter_names=(),
                        latex_names=(),
                        cmb_contract=contract,
                    )

    def test_prepare_declared_strips_model_metadata_from_perturbations(
        self,
    ) -> None:
        """Outer model metadata must not enter the graph compiler."""

        compile_result = object()
        cmb_contract = {
            "model_name": "MetadataModel",
            "param_map": {},
            "model_parameters": {},
            "background": {},
            "grids": {},
            "values": {},
            "calls": [],
            "numerical": {},
            "perturbations": {
                "contract_version": 2,
                "model_name": "MetadataModel",
            },
        }

        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            return_value=compile_result,
        ) as compile_contract:
            model_coder.prepare_declared_cmb_execution_contract(cmb_contract)

        compiled_contract = compile_contract.call_args.args[0]
        self.assertNotIn("model_name", compiled_contract)


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the model coder API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(model_coder, "QuadPrinter"))
        self.assertTrue(callable(model_coder.robust_quad))
        self.assertTrue(hasattr(model_coder, "DeclaredCMBBackgroundRuntime"))
        self.assertTrue(hasattr(model_coder, "DeclaredCMBCompileDiagnostics"))
        self.assertTrue(
            callable(model_coder.prepare_declared_cmb_execution_contract)
        )

    def test_transformed_symbol_is_exposed(self) -> None:
        transformed = model_coder.robust_quad
        self.assertTrue(callable(transformed))


class DeclaredCMBRouteTestCase(unittest.TestCase):
    """Cover the single declared route exposed by model_coder."""

    def test_removed_backend_capability_api_is_absent(self) -> None:
        """Core code must not expose a second solver capability surface."""

        removed_names = (
            "CMB_BACKEND_CAPABILITIES",
            "get_backend_capabilities",
            "backend_supports_standard_perturbations",
            "backend_supports_declared_nonstandard_perturbations",
            "validate_declared_perturbation_execution",
        )
        for name in removed_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(model_coder, name))

    def test_compile_declared_runtime_accepts_declared_background_symbols(
        self,
    ) -> None:
        """Precompiled declared runtimes should accept background symbols."""

        cmb_contract = {
            "param_map": {"expansion_rate_today": 67.4},
            "grids": {},
            "values": {},
            "background": {
                "derived": {
                    "matter_budget_today": "0.3",
                    "H": (
                        "expansion_rate_today * sqrt("
                        "matter_budget_today / (a ** 3) + "
                        "(1.0 - matter_budget_today)"
                        ")"
                    ),
                },
                "reionization": {"calibration": {}, "quantities": {}},
            },
            "numerical": {},
            "perturbations": {
                "contract_version": 2,
                "gauge": "conformal_newtonian",
                "variables": {
                    "delta_x": {"kind": "density_contrast"},
                    "theta_x": {"kind": "velocity_divergence"},
                    "phi_aux": {"kind": "metric_potential_phi"},
                    "psi_aux": {"kind": "metric_potential_psi"},
                },
                "derived": {
                    "density_drive": {
                        "expression": "matter_budget_today * delta_x + phi_aux"
                    }
                },
                "equations": {
                    "continuity_x": {
                        "lhs": {
                            "kind": "derivative",
                            "variable": "delta_x",
                            "wrt": "tau",
                            "order": 1,
                        },
                        "rhs": "-theta_x + phi_aux",
                        "role": "continuity",
                    },
                    "euler_x": {
                        "lhs": {
                            "kind": "derivative",
                            "variable": "theta_x",
                            "wrt": "tau",
                            "order": 1,
                        },
                        "rhs": "-Hconf * theta_x + k * psi_aux",
                        "role": "euler",
                    },
                },
                "constraints": {
                    "poisson_phi": {
                        "target": "phi_aux",
                        "expression": "0.25 * delta_x",
                        "role": "constraint",
                    }
                },
                "closures": {
                    "psi_equals_phi": {
                        "target": "psi_aux",
                        "expression": "phi_aux",
                        "role": "closure",
                    }
                },
                "sources": {
                    "monopole_source": {
                        "expression": "visibility * density_drive",
                        "role": "monopole",
                    }
                },
                "observables": {
                    "temperature": {
                        "kind": "transfer_component",
                        "projection": "line_of_sight_temperature",
                        "source_terms": {"monopole": "monopole_source"},
                    },
                    "TT": {
                        "kind": "angular_power_spectrum",
                        "primary": "temperature",
                        "secondary": "temperature",
                    },
                },
                "initial_conditions": {
                    "delta_seed": {
                        "target": {
                            "variable": "delta_x",
                            "wrt": "tau",
                            "order": 0,
                        },
                        "expression": "seed",
                    },
                    "theta_seed": {
                        "target": {
                            "variable": "theta_x",
                            "wrt": "tau",
                            "order": 0,
                        },
                        "expression": "0.1 * seed",
                    },
                },
                "boundary_conditions": {},
                "validity": {"regimes": ["linear"]},
                "numerics": {},
            },
        }

        runtime = model_coder.compile_declared_cmb_runtime(
            model_name="TemplateModel",
            parameter_names=("expansion_rate_today",),
            latex_names=("H_0",),
            cmb_contract=cmb_contract,
        )
        dependency_summary = runtime.perturbation_data.dependency_graph_summary
        background_references = dependency_summary.background_references_used

        self.assertIn("matter_budget_today", background_references)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
