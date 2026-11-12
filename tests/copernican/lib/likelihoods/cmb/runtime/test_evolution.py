"""Focused tests for the declared CMB evolution module."""

import ast
import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy
from scipy.integrate import quad
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

    def test_exponential_rk4_resolves_coupled_variable_stiffness(self):
        """Compare both mode rows with independently integrated stiff ODEs."""
        from scipy.integrate import solve_ivp

        eta = numpy.linspace(0.0, 2.0, 81)
        strengths = numpy.asarray((100.0, 1e6))
        initial = numpy.asarray(((1.0, 0.0), (1.0, 0.0)))

        def rate(states, *, step_index, blend, active):
            del states, active
            time = eta[step_index] + blend * (
                eta[step_index + 1] - eta[step_index]
            )
            return strengths * (1.0 + 0.8 * numpy.sin(time))

        def rhs(states, *, step_index, blend, active):
            stiffness = rate(
                states, step_index=step_index, blend=blend, active=active
            )
            return numpy.column_stack(
                (
                    -states[:, 0] + states[:, 1],
                    -stiffness * states[:, 1] + states[:, 0],
                )
            )

        _, actual, _ = evolution._integrate_batched_rk4(
            initial,
            eta,
            required_substeps=1,
            active_intervals=numpy.ones((2, len(eta) - 1), dtype=bool),
            rhs=rhs,
            relaxation_slot=1,
            relaxation_rate=rate,
        )
        for index, strength in enumerate(strengths):
            expected = solve_ivp(
                lambda time, values: (
                    -values[0] + values[1],
                    -strength * (1 + 0.8 * numpy.sin(time)) * values[1]
                    + values[0],
                ),
                (eta[0], eta[-1]),
                initial[index],
                method="Radau",
                rtol=1e-11,
                atol=1e-14,
            ).y[:, -1]
            with self.subTest(strength=strength):
                numpy.testing.assert_allclose(
                    actual[index], expected, rtol=1e-5
                )

        def zero_rate(states, **unused):
            return numpy.zeros(len(states))

        def plain_rhs(states, **unused):
            return numpy.column_stack((states[:, 1], -states[:, 0]))

        options = {
            "required_substeps": 1,
            "active_intervals": numpy.ones((2, len(eta) - 1), dtype=bool),
            "rhs": plain_rhs,
        }
        ordinary = evolution._integrate_batched_rk4(initial, eta, **options)[0]
        zero = evolution._integrate_batched_rk4(
            initial,
            eta,
            relaxation_slot=1,
            relaxation_rate=zero_rate,
            **options,
        )[0]
        numpy.testing.assert_allclose(zero, ordinary, rtol=1e-13, atol=1e-14)

    def test_relativistic_kinetic_density_has_smooth_mass_limit(self):
        """Independent moments fix normalization at every mass and epoch."""
        family = SimpleNamespace(
            momentum_grid="species_q",
            physical_role="generic",
            phase_space={
                "distribution": "maxwell_boltzmann",
                "mass_parameter": "mass",
                "temperature_ratio": "ratio",
                "relativistic_density": "early_density",
            },
        )
        perturbation_data = SimpleNamespace(
            hierarchy_families={"unrelated_species": family},
            numerics={
                "momentum_grids": {
                    "species_q": {"count": 256, "q_min": 0.0001, "q_max": 40.0}
                }
            },
            accuracy_controls={},
        )
        physical = SimpleNamespace(hubble_ratio=0.7, Omega_nu0=0.0, Tcmb_K=2.7)
        parameters = {"mass": 0.0, "ratio": 0.5, "early_density": 1e-5}
        epochs = numpy.asarray((0.001, 0.1, 1.0))
        results = []
        for mass in (0.0, 1e-10, 0.01):
            parameters["mass"] = mass
            result = evolution._declared_momentum_grid_context(
                perturbation_data,
                model_parameters=parameters,
                physical_params=physical,
                scale_factor=epochs,
            )
            density = result["momentum_grid_species_q_density_fraction"]
            temperature = 0.5 * 2.7 * 8.617333262145e-5
            expected = numpy.asarray(
                [
                    1e-5
                    * epoch**-4
                    / 6.0
                    * quad(
                        lambda momentum: momentum**2
                        * numpy.hypot(momentum, epoch * mass / temperature)
                        * numpy.exp(-momentum),
                        0.0,
                        numpy.inf,
                    )[0]
                    for epoch in epochs
                ]
            )
            numpy.testing.assert_allclose(density, expected, rtol=2e-4)
            results.append(density)
        numpy.testing.assert_allclose(results[0], results[1], rtol=1e-10)
        parameters["mass"] = 0.0
        parameters["early_density"] = 2e-5
        doubled = evolution._declared_momentum_grid_context(
            perturbation_data,
            model_parameters=parameters,
            physical_params=physical,
            scale_factor=epochs,
        )["momentum_grid_species_q_density_fraction"]
        numpy.testing.assert_allclose(doubled, 2 * results[0], rtol=1e-12)
        parameters["early_density"] = -1.0
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            evolution._declared_momentum_grid_context(
                perturbation_data,
                model_parameters=parameters,
                physical_params=physical,
                scale_factor=epochs,
            )

    def test_kinetic_law_rebinding_invalidates_only_bound_distribution(self):
        law = {
            "distribution": "maxwell_boltzmann",
            "mass_parameter": "mass",
            "temperature_ratio": "ratio",
            "density_parameter": "density",
        }
        family = SimpleNamespace(
            momentum_grid="physical_q",
            physical_role="generic",
            phase_space=law,
        )
        perturbation_data = SimpleNamespace(
            hierarchy_families={"arbitrary_label": family},
            numerics={
                "momentum_grids": {
                    "physical_q": {"count": 16, "q_min": 0.02, "q_max": 24.0}
                }
            },
            accuracy_controls={},
        )
        physical = SimpleNamespace(hubble_ratio=0.7, Omega_nu0=0.0, Tcmb_K=2.7)
        parameters = {"mass": 0.1, "ratio": 0.5, "density": 0.01}
        cache.clear_cmb_caches()
        self.addCleanup(cache.clear_cmb_caches)

        def resolve():
            return evolution._resolve_declared_momentum_grid_runtimes(
                perturbation_data,
                model_parameters=parameters,
                physical_params=physical,
            )

        first = resolve()[0]
        self.assertIs(resolve()[0], first)
        parameters["unrelated_amplitude"] = 2e-9
        self.assertIs(resolve()[0], first)
        family.phase_space = {**law, "distribution": "fermi_dirac"}
        changed = resolve()[0]
        self.assertIsNot(changed, first)
        self.assertIs(changed.points, first.points)
        q_values = first.points
        numpy.testing.assert_allclose(first.occupation, numpy.exp(-q_values))
        numpy.testing.assert_allclose(
            changed.occupation, 1 / (1 + numpy.exp(q_values))
        )
        parameters["ratio"] = 0.75
        warmer = resolve()[0]
        self.assertAlmostEqual(
            warmer.temperature_eV / changed.temperature_eV, 1.5
        )

    def test_native_history_interpolation_is_smooth_and_shape_preserving(self):
        """Native states retain signs, tiny scales and smooth derivatives."""

        native = numpy.linspace(0.0, 2.0, 65)
        query = numpy.linspace(0.0, 2.0, 2001)
        histories = {
            "potential": numpy.exp(-native),
            "tiny": 1e-310 * numpy.exp(-native),
            "zero": numpy.zeros(native.size),
        }
        with numpy.errstate(over="raise", invalid="raise", divide="raise"):
            sampled = evolution._interpolate_declared_histories(
                native,
                histories,
                query,
            )
        self.assertTrue(numpy.all(numpy.diff(sampled["potential"]) <= 0))
        numpy.testing.assert_allclose(
            sampled["potential"],
            numpy.exp(-query),
            rtol=1e-5,
        )
        numpy.testing.assert_allclose(
            sampled["tiny"] / 1e-310,
            sampled["potential"],
            rtol=1e-12,
        )
        self.assertTrue(numpy.all(sampled["zero"] == 0))
        slope = numpy.gradient(sampled["potential"], query)
        self.assertLess(numpy.max(abs(slope + numpy.exp(-query))), 0.001)
        with self.assertRaises(ValueError):
            evolution._interpolate_declared_histories(
                native, histories, [-1.0]
            )

    def test_inactive_split_resolves_driven_collision_equilibrium(self):
        """Stable exact damping alone misses the analytic forced solution."""

        rate = 10.0
        duration = 2.0
        expected = (1.0 - numpy.exp(-rate * duration)) / rate

        def rhs(states, **kwargs):
            return numpy.ones_like(states)

        def damp(states, *, dt, **kwargs):
            return states * numpy.exp(-rate * dt)

        errors = []
        for active in (True, False):
            scale = float(
                evolution._split_collision_phase_scale(
                    0.02,
                    0.01,
                    rate,
                    active,
                )
            )
            _, final, _ = evolution._integrate_batched_rk4(
                numpy.zeros((1, 1)),
                numpy.asarray((0.0, duration)),
                required_substeps=int(numpy.ceil(duration * scale / 0.1)),
                active_intervals=numpy.zeros((1, 1), dtype=bool),
                rhs=rhs,
                pre_step=damp,
                post_step=damp,
            )
            errors.append(abs(float(final[0, 0]) - expected) / expected)
        self.assertGreater(errors[0], 0.9)
        self.assertLess(errors[1], 0.001)
        with self.assertRaises(ValueError):
            evolution._split_collision_phase_scale(
                0.02, 0.01, numpy.inf, False
            )

    def test_progress_uses_actual_rounded_schedule_and_preserves_states(self):
        """Nonuniform interval costs are counted once for each mode row."""

        observed = []

        def progress(completed, total, **details):
            observed.append((completed, total, details))

        def rhs(states, **kwargs):
            return -states

        arguments = {
            "initial_states": numpy.ones((2, 1)),
            "eta_grid": numpy.asarray((0.0, 0.1, 0.2)),
            "required_substeps": numpy.asarray(((1, 129), (3, 65))),
            "active_intervals": numpy.zeros((2, 2), dtype=bool),
            "rhs": rhs,
        }
        authority = evolution._integrate_batched_rk4(**arguments)
        measured = evolution._integrate_batched_rk4(
            **arguments, progress=progress
        )
        numpy.testing.assert_array_equal(authority[0], measured[0])
        numpy.testing.assert_array_equal(authority[1], measured[1])
        self.assertEqual(authority[2], measured[2])
        self.assertEqual(observed[0][:2], (0, 520))
        self.assertEqual(observed[-1][:2], (520, 520))
        self.assertEqual(observed[-1][2]["interval"], 2)
        self.assertTrue(all(total == 520 for _, total, _ in observed))
        self.assertEqual(
            [value[0] for value in observed],
            sorted(set(value[0] for value in observed)),
        )
        self.assertGreater(len(observed), 4)

    def test_generated_scalar_seeds_resolve_without_provisional_states(self):
        """All scalar families resolve before any state values exist."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb

        gauges = ("conformal_newtonian", "synchronous", "gauge_invariant")
        modes = (
            "adiabatic_scalar",
            "baryon_isocurvature",
            "cdm_isocurvature",
            "neutrino_density_isocurvature",
            "neutrino_velocity_isocurvature",
        )
        for gauge, massive, mode in product(gauges, (False, True), modes):
            with self.subTest(gauge=gauge, massive=massive, mode=mode):
                contract = test_cmb._prepare_declared_contract(
                    test_cmb._declared_scalar_hierarchy_contract(
                        gauge=gauge,
                        initial_mode=mode,
                        include_massive_neutrino=massive,
                    )
                )
                perturbation_data = contract["perturbation_data"]
                plan = evolution._compile_declared_graph_execution_plan(
                    perturbation_data
                )
                physical = background._resolve_custom_cmb_physical_parameters(
                    contract
                )
                numerics = background._resolve_custom_cmb_numerics(contract)
                context = evolution._build_declared_base_context(
                    perturbation_data=perturbation_data,
                    model_parameters=contract["param_map"],
                    physical_params=physical,
                    numerics=numerics,
                    k_value=1.0e-4,
                    eta_value=1.0,
                    background_scalars={
                        "a": 1.0e-5,
                        "eta": 1.0,
                        "Hconf": 1.0,
                        "Hconf_tau": -1.0,
                        "collision_rate": 100.0,
                        "sound_horizon": 1.0,
                        "sound_speed_sq": 1.0 / 3.0,
                        "free_streaming": 1.0,
                    },
                )
                state_names = {
                    slot.variable for slot in plan.runtime_spec.state_slots
                }
                self.assertFalse(state_names.intersection(context))
                state, targets = evolution._evaluate_declared_initial_state(
                    perturbation_data=perturbation_data,
                    execution_plan=plan,
                    base_context=context,
                )
                self.assertTrue(numpy.all(numpy.isfinite(state)))
                self.assertEqual(
                    set(targets),
                    set(plan.runtime_spec.state_index_by_key),
                )

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

    def test_tight_coupling_requires_fast_relaxation_relative_to_expansion(
        self,
    ):
        """Small k cannot keep slow collisions in equilibrium."""

        for k in (1e-4, 1e-8):
            with self.subTest(k=k):
                self.assertFalse(
                    evolution._tight_coupling_is_active(
                        active=True,
                        collision_rate=0.03,
                        k_value=k,
                        expansion_rate=0.01,
                        tight_coupling_ratio=50.0,
                    )
                )
                self.assertTrue(
                    evolution._tight_coupling_is_active(
                        active=False,
                        collision_rate=1.0,
                        k_value=k,
                        expansion_rate=0.01,
                        tight_coupling_ratio=50.0,
                    )
                )
        self.assertTrue(
            evolution._tight_coupling_is_active(
                active=True,
                collision_rate=0.03,
                k_value=1e-4,
                expansion_rate=0.0,
                tight_coupling_ratio=50.0,
            )
        )
        with self.assertRaisesRegex(ValueError, "scales must be finite"):
            evolution._tight_coupling_is_active(
                active=True,
                collision_rate=0.03,
                k_value=1e-4,
                expansion_rate=numpy.nan,
                tight_coupling_ratio=50.0,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
