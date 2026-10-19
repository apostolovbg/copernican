"""Focused tests for the declared CMB evolution module."""

import ast
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy
from scipy.linalg import expm

from copernican.lib.likelihoods.cmb.runtime import background, cache, evolution
from copernican.lib.perturbation_contract import compile_perturbation_contract
from tests.copernican.lib.test_perturbation_contract import (
    _scalar_metadata_only_contract,
)


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


class EvolutionModuleTestCase(unittest.TestCase):
    """Exercise declared evolution helpers directly."""

    def test_split_collisions_refine_against_the_coupled_solution(self):
        """Exact damping alone cannot certify a split oscillator's accuracy."""

        streaming = numpy.asarray(((0.0, -1.0), (1.0, 0.0)))
        collision = numpy.diag((0.0, -1.0))
        initial = numpy.asarray(((1.0, 0.0),))
        expected = expm(4 * (streaming + collision)) @ initial[0]
        errors = []
        for count in (9, 41, 81):

            def rhs(states, **kwargs):
                return states @ streaming.T

            def damp(states, *, dt, **kwargs):
                return states * numpy.exp(numpy.diag(collision) * dt)

            _, final, _ = evolution._integrate_batched_rk4(
                initial,
                numpy.linspace(0, 4, count),
                required_substeps=1,
                active_intervals=numpy.zeros((1, count - 1), dtype=bool),
                rhs=rhs,
                pre_step=damp,
                post_step=damp,
            )
            errors.append(float(numpy.max(numpy.abs(final[0] - expected))))
        self.assertGreater(errors[0], 1e-3)
        self.assertLess(errors[1], 5e-4)
        self.assertGreater(errors[1] / errors[2], 3.5)

    def test_manufactured_entropy_evolution_detects_a_wrong_metric_sign(self):
        """Integrated continuity must match a prescribed changing metric.

        Set Phi=eta^2/10, Theta_1=sin(eta), k=0.2 and theta_b=3kTheta_1.
        Independently integrating continuity gives Theta_0=Phi+k cos(eta)
        and delta_b=3Theta_0+0.7. A flipped metric drive breaks both.
        """

        compiled = compile_perturbation_contract(
            _scalar_metadata_only_contract(),
            model_name="ManufacturedEntropy",
            parameter_names=("H0",),
            latex_names=("H_0",),
            background_reference_names=("H0",),
        )
        eta = numpy.linspace(0.0, 2.0, 65)
        expected_photon = eta**2 / 10 + 0.2 * numpy.cos(eta)
        expected = numpy.stack(
            (expected_photon, 3 * expected_photon + 0.7), axis=-1
        )
        for metric_sign in (1, -1):

            def rhs(states, *, step_index, blend, active):
                time = eta[step_index] + blend * (
                    eta[step_index + 1] - eta[step_index]
                )
                context = {
                    "acoustic_k": 0.2,
                    "theta_gamma1": numpy.sin(time),
                    "theta_b": 0.6 * numpy.sin(time),
                    "Phi_tau": time / 5,
                }
                photon = evolution._evaluate_compiled_expression_noerr(
                    compiled.equations["evolve_theta_gamma0"].compiled_rhs,
                    context,
                )
                baryon = evolution._evaluate_compiled_expression_noerr(
                    compiled.equations["evolve_delta_b"].compiled_rhs, context
                )
                if metric_sign < 0:
                    photon -= 2 * context["Phi_tau"]
                return numpy.broadcast_to((photon, baryon), states.shape)

            histories, _, stats = evolution._integrate_batched_rk4(
                expected[:1],
                eta,
                required_substeps=1,
                active_intervals=numpy.zeros((1, eta.size - 1), dtype=bool),
                rhs=rhs,
            )
            error = float(numpy.max(numpy.abs(histories[0] - expected)))
            if metric_sign > 0:
                self.assertLess(error, 1.0e-8)
                numpy.testing.assert_allclose(
                    histories[0, :, 1] - 3 * histories[0, :, 0],
                    0.7,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            else:
                self.assertGreater(error, 0.5)
            self.assertEqual(stats.rk_stage_count, 4 * (eta.size - 1))

    def test_acoustic_oscillator_refines_against_an_independent_solution(self):
        """Actual RK evolution must converge without a phase/amplitude fit."""

        frequencies = numpy.asarray((0.5, 1.5))[:, numpy.newaxis]
        errors = []
        for count in (33, 65):
            eta = numpy.linspace(0.0, 4.0, count)

            def rhs(states, **kwargs):
                return frequencies * numpy.stack(
                    (states[:, 1], -states[:, 0]), axis=-1
                )

            histories, _, _ = evolution._integrate_batched_rk4(
                numpy.asarray(((1.0, 0.0), (1.0, 0.0))),
                eta,
                required_substeps=1,
                active_intervals=numpy.zeros((2, count - 1), dtype=bool),
                rhs=rhs,
            )
            phase = frequencies * eta[numpy.newaxis, :]
            expected = numpy.stack(
                (numpy.cos(phase), -numpy.sin(phase)), axis=-1
            )
            errors.append(float(numpy.max(numpy.abs(histories - expected))))
        self.assertLess(errors[-1], 5.0e-6)
        self.assertGreater(errors[0] / errors[1], 12.0)
        for corrupted in (-histories, numpy.roll(histories, 4, axis=1)):
            self.assertGreater(
                float(numpy.max(numpy.abs(corrupted - expected))), 0.1
            )

    def test_context_name_rewriter_maps_runtime_names(self):
        """The generated context program must rewrite declared names."""

        rewriter = evolution._ContextNameRewriter()
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
            compiled = evolution._compile_declared_perturbation_contract(
                contract
            )

        self.assertIs(compiled, payload)

    def test_missing_precompiled_payload_fails_loudly(self):
        """Declared execution should reject raw contracts without payloads."""

        with self.assertRaisesRegex(
            ValueError,
            "precompiled perturbation_data",
        ):
            evolution._compile_declared_perturbation_contract({})

    def test_declared_graph_context_uses_compiled_expression_plans(self):
        """Declared graph resolution should avoid AST re-interpretation."""

        perturbation_data = _compiled_graph_fixture()
        execution_plan = evolution._compile_declared_graph_execution_plan(
            perturbation_data
        )
        context = {"delta_x": 2.0}

        with mock.patch.object(
            background,
            "_evaluate_safe_expression",
            side_effect=AssertionError("compiled graph plan should run"),
        ):
            resolved = evolution._resolve_declared_graph_context(
                context,
                perturbation_data,
                allow_partial=False,
                eta_grid=None,
                execution_plan=execution_plan,
            )

        self.assertAlmostEqual(resolved["phi_aux"], 0.5)
        self.assertAlmostEqual(resolved["psi_aux"], 0.5)
        self.assertAlmostEqual(resolved["density_drive"], 2.5)

    def test_runtime_assets_compile_once_per_process_signature(self):
        """Repeated worker preparation should reuse one structural bundle."""

        cache.clear_cmb_caches()
        perturbation_data = _compiled_graph_fixture()
        compiler = evolution._compile_declared_graph_execution_plan
        with mock.patch.object(
            evolution,
            "_compile_declared_graph_execution_plan",
            wraps=compiler,
        ) as compile_plan:
            first = evolution.prepare_runtime_assets(
                "runtime:test",
                perturbation_data,
            )
            second = evolution.prepare_runtime_assets(
                "runtime:test",
                perturbation_data,
            )

        self.assertIs(first, second)
        compile_plan.assert_called_once_with(perturbation_data)
        third = evolution.prepare_runtime_assets(
            "runtime:test", perturbation_data
        )
        self.assertIs(third, first)
        self.assertGreater(first.owner_pid, 0)
        stats = cache.cmb_cache_stats()["runtime_assets"]
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["hits"], 2)

    def test_execution_schedule_identity_ignores_model_label(self):
        """Renamed declarations share one compiled execution schedule."""

        first = _compiled_graph_fixture()
        second = replace(first, model_name="UnrelatedTheory")

        self.assertEqual(
            evolution._declared_graph_execution_plan_cache_token(first),
            evolution._declared_graph_execution_plan_cache_token(second),
        )

    def test_execution_schedule_descriptor_is_structural_and_deterministic(
        self,
    ):
        """Expose stable structural evidence for the compiled schedule."""

        perturbation_data = _compiled_graph_fixture()
        self.assertTrue(
            callable(evolution.describe_declared_execution_schedule)
        )
        execution_plan = evolution._compile_declared_graph_execution_plan(
            perturbation_data
        )
        first = evolution.describe_declared_execution_schedule(
            perturbation_data,
            execution_plan,
        )
        second = evolution.describe_declared_execution_schedule(
            perturbation_data,
            execution_plan,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["evolution_variable"], "eta")
        self.assertIsInstance(first["mode_partitions"], dict)
        self.assertGreaterEqual(first["state_slot_count"], 2)
        self.assertTrue(first["signature"])

    def test_runtime_assets_are_isolated_by_process_owner(self):
        """A process identity change must materialize a distinct bundle."""

        cache.clear_cmb_caches()
        perturbation_data = _compiled_graph_fixture()
        with mock.patch.object(evolution.os, "getpid", return_value=101):
            first = evolution.prepare_runtime_assets(
                "runtime:test",
                perturbation_data,
            )
        with mock.patch.object(evolution.os, "getpid", return_value=202):
            second = evolution.prepare_runtime_assets(
                "runtime:test",
                perturbation_data,
            )

        self.assertIsNot(first, second)
        self.assertEqual(first.owner_pid, 101)
        self.assertEqual(second.owner_pid, 202)

    def test_evolution_source_does_not_import_camb(self):
        """The declared evolution module should remain CAMB-free."""

        source_text = Path(evolution.__file__).read_text(encoding="utf-8")
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

        first = evolution._integrate_batched_rk4(
            initial,
            eta_grid,
            required_substeps=required_substeps,
            active_intervals=active,
            rhs=rhs,
        )
        second = evolution._integrate_batched_rk4(
            initial,
            eta_grid,
            required_substeps=required_substeps,
            active_intervals=active,
            rhs=lambda state, **kwargs: state,
        )

        histories, final_states, stats = first
        self.assertIsInstance(
            stats,
            evolution.BatchedEvolutionStats,
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
