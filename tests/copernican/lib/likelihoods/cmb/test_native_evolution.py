"""Focused tests for the native CMB evolution module."""

import ast
import unittest
from pathlib import Path
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb import native_background, native_evolution
from copernican.lib.perturbation_contract import compile_perturbation_contract


def _compiled_graph_fixture():
    """Return one small declared graph with compiled expression entries."""

    contract = {
        "contract_version": 2,
        "gauge": "conformal_newtonian",
        "variables": {
            "delta_x": {"kind": "density_contrast"},
            "theta_x": {"kind": "velocity_divergence"},
            "phi_aux": {"kind": "metric_potential_phi"},
            "psi_aux": {"kind": "metric_potential_psi"},
        },
        "derived": {
            "density_drive": {"expression": "delta_x + phi_aux"},
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
        "numerics": {},
        "validity": {"regimes": ["linear"]},
    }
    return compile_perturbation_contract(
        contract,
        model_name="TemplateModel",
        parameter_names=("seed",),
        latex_names=("seed",),
        background_reference_names=("Hconf", "k", "seed", "visibility"),
    )


class NativeEvolutionModuleTestCase(unittest.TestCase):
    """Exercise native evolution helpers directly."""

    def test_context_name_rewriter_maps_runtime_names(self):
        """The generated context program must rewrite declared names."""

        rewriter = native_evolution._ContextNameRewriter()
        runtime_name = rewriter.visit_Name(
            ast.Name(id="delta_x", ctx=ast.Load())
        )
        allowed_function = rewriter.visit_Name(
            ast.Name(id="sqrt", ctx=ast.Load())
        )

        self.assertEqual(rewriter.visit_Name.__name__, "visit_Name")
        self.assertEqual(ast.unparse(runtime_name), "context['delta_x']")
        self.assertEqual(ast.unparse(allowed_function), "sqrt")

    def test_precompiled_perturbation_payload_is_reused(self):
        """Existing compiled perturbation data should bypass recompilation."""

        payload = object()
        contract = {"perturbation_data": payload}

        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            side_effect=AssertionError("recompilation should not run"),
        ):
            compiled = (
                native_evolution._compile_declared_perturbation_contract(
                    contract
                )
            )

        self.assertIs(compiled, payload)

    def test_missing_precompiled_payload_fails_loudly(self):
        """Native execution should reject raw contracts without payloads."""

        with self.assertRaisesRegex(
            ValueError,
            "precompiled perturbation_data",
        ):
            native_evolution._compile_declared_perturbation_contract({})

    def test_declared_graph_context_uses_compiled_expression_plans(self):
        """Declared graph resolution should avoid AST re-interpretation."""

        perturbation_data = _compiled_graph_fixture()
        execution_plan = (
            native_evolution._compile_declared_graph_execution_plan(
                perturbation_data
            )
        )
        context = {"delta_x": 2.0}

        with mock.patch.object(
            native_background,
            "_evaluate_safe_expression",
            side_effect=AssertionError("compiled graph plan should run"),
        ):
            resolved = native_evolution._resolve_declared_graph_context(
                context,
                perturbation_data,
                allow_partial=False,
                eta_grid=None,
                execution_plan=execution_plan,
            )

        self.assertAlmostEqual(resolved["phi_aux"], 0.5)
        self.assertAlmostEqual(resolved["psi_aux"], 0.5)
        self.assertAlmostEqual(resolved["density_drive"], 2.5)

    def test_native_evolution_source_does_not_import_camb(self):
        """The native evolution module should remain CAMB-free."""

        source_text = Path(native_evolution.__file__).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("import camb", source_text)

    def test_batched_rk4_uses_one_shared_schedule(self):
        """Batched rows share stages and retain deterministic histories."""

        eta_grid = numpy.asarray((0.0, 0.25, 0.5), dtype=float)
        initial = numpy.asarray(((1.0,), (2.0,)), dtype=float)
        required_substeps = numpy.asarray(((1, 2), (4, 1)), dtype=int)
        active = numpy.zeros((2, 2), dtype=bool)
        calls = []

        def rhs(state, *, step_index, blend, active):
            """Return a row-independent test derivative."""

            calls.append((int(step_index), float(blend), state.shape))
            return state

        first = native_evolution._integrate_batched_rk4(
            initial,
            eta_grid,
            required_substeps=required_substeps,
            active_intervals=active,
            rhs=rhs,
        )
        second = native_evolution._integrate_batched_rk4(
            initial,
            eta_grid,
            required_substeps=required_substeps,
            active_intervals=active,
            rhs=lambda state, **kwargs: state,
        )

        histories, final_states, stats = first
        self.assertIsInstance(
            stats,
            native_evolution.NativeBatchedEvolutionStats,
        )
        self.assertEqual(stats.mode_count, 2)
        self.assertEqual(stats.interval_count, 2)
        self.assertEqual(stats.maximum_substeps, 4)
        self.assertEqual(stats.rk_stage_count, 4 * (4 + 2))
        self.assertEqual(len(calls), stats.rk_stage_count)
        numpy.testing.assert_allclose(histories, second[0])
        numpy.testing.assert_allclose(final_states, second[1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
