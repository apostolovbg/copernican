"""Direct tests for declared hierarchy-evolution ownership."""

import math
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy
from scipy.interpolate import PchipInterpolator

from copernican.lib.likelihoods.cmb.runtime import mode_evolution


class _EarlyHistoryCaptured(BaseException):
    """Stop a real solver after its bounded physical prefix is measured."""

    def __init__(self, phi):
        self.phi = float(phi)
        super().__init__(self.phi)


class _ModeHistoryCaptured(BaseException):
    """Stop after real mode histories, before angular projection."""


class ModeEvolutionTestCase(unittest.TestCase):
    """Exercise hierarchy scheduling apart from public orchestration."""

    def test_vector_jacobian_columns_match_actual_scalar_declared_rhs(self):
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        observed = []

        def compare(rhs, initial, eta, **kwargs):
            self.assertTrue(kwargs["affine_system"])
            vectorized = kwargs["vectorized_rhs"]
            self.assertIsNotNone(vectorized)
            scale = numpy.maximum(abs(initial), 1e-12)
            probes = numpy.column_stack(
                (initial, numpy.diag(scale), -numpy.diag(scale))
            )
            expected = numpy.column_stack(
                [rhs(float(eta[0]), state) for state in probes.T]
            )
            actual = vectorized(float(eta[0]), probes)
            numpy.testing.assert_allclose(
                actual, expected, rtol=2e-13, atol=1e-30
            )
            observed.append(actual)
            raise _ModeHistoryCaptured()

        raw = test_cmb._speedup_contract(
            test_cmb._declared_vector_hierarchy_contract()
        )
        contract = test_cmb._prepare_declared_contract(raw)
        self.addCleanup(test_cmb.cache.clear_cmb_caches)
        with (
            mock.patch.object(
                mode_evolution,
                "_integrate_scaled_continuous_history",
                side_effect=compare,
            ),
            self.assertRaises(_ModeHistoryCaptured),
        ):
            test_cmb.cmb.compute_cmb_spectrum_from_contract(
                contract,
                numpy.asarray((20, 30, 40)),
                spectra=("BB",),
            )
        self.assertTrue(observed)

    def test_affine_initial_scale_probes_preserve_zero_and_tiny_chains(self):
        from copernican.lib.likelihoods.cmb.runtime.mode_evolution import (
            _integrate_scaled_continuous_history,
        )

        eta = numpy.linspace(0, 2, 81)
        calls = {"scalar": 0, "vector": 0}

        def rhs(eta_value, x):
            calls["scalar" if x.ndim == 1 else "vector"] += 1
            return numpy.array(
                (
                    -x[0],
                    1e-10 * x[0] - 2 * x[1],
                    1e-20 * x[1] - 3 * x[2],
                    numpy.zeros_like(x[3]),
                )
            )

        args = dict(initial_time_scale=0.01, rtol=1e-9, atol=1e-12)
        initial = numpy.array((1.0, 0.0, 0.0, 0.0))
        vector = _integrate_scaled_continuous_history(
            rhs, initial, eta, affine_system=True, vectorized_rhs=rhs, **args
        )
        self.assertGreater(calls["vector"], 0)
        scalar = _integrate_scaled_continuous_history(
            rhs, initial, eta, **args
        )
        exact = numpy.array(
            (
                numpy.exp(-eta),
                1e-10 * (numpy.exp(-eta) - numpy.exp(-2 * eta)),
                1e-30
                * (
                    0.5 * numpy.exp(-eta)
                    - numpy.exp(-2 * eta)
                    + 0.5 * numpy.exp(-3 * eta)
                ),
                numpy.zeros_like(eta),
            )
        )
        numpy.testing.assert_allclose(vector, exact, rtol=2e-7, atol=1e-40)
        numpy.testing.assert_allclose(vector, scalar, rtol=2e-7, atol=1e-40)
        numpy.testing.assert_array_equal(vector[3], numpy.zeros_like(eta))

    def test_affine_continuous_jacobian_preserves_tiny_forced_stiff_states(
        self,
    ):
        from copernican.lib.likelihoods.cmb.runtime.mode_evolution import (
            _integrate_scaled_continuous_history,
        )

        eta = numpy.linspace(0, 2, 81)

        def rhs(eta_value, state):
            return numpy.array((-state[0], 1e-20 * state[0] - 1000 * state[1]))

        arguments = dict(initial_time_scale=0.01, rtol=1e-9, atol=1e-12)
        affine = _integrate_scaled_continuous_history(
            rhs,
            numpy.array((1.0, 0.0)),
            eta,
            affine_system=True,
            vectorized_rhs=rhs,
            **arguments,
        )
        general = _integrate_scaled_continuous_history(
            rhs, numpy.array((1.0, 0.0)), eta, affine_system=False, **arguments
        )
        expected = numpy.array(
            (
                numpy.exp(-eta),
                1e-20 * (numpy.exp(-eta) - numpy.exp(-1000 * eta)) / 999,
            )
        )
        numpy.testing.assert_allclose(affine, expected, rtol=2e-7, atol=1e-30)
        numpy.testing.assert_allclose(affine, general, rtol=2e-7, atol=1e-30)
        radau = _integrate_scaled_continuous_history(
            rhs,
            numpy.array((1.0, 0.0)),
            eta,
            affine_system=True,
            vectorized_rhs=rhs,
            method="Radau",
            **arguments,
        )
        numpy.testing.assert_allclose(radau, expected, rtol=2e-7, atol=1e-30)
        numpy.testing.assert_allclose(radau, general, rtol=2e-7, atol=1e-30)

    def test_sparse_stiff_jacobian_retains_tiny_states_and_new_nonzeros(self):
        """Changing matrix density must preserve an independent exact path."""

        from scipy.sparse import issparse

        amplitudes = numpy.geomspace(1.0, 1e-25, 16)
        rates = numpy.geomspace(1.0, 1000.0, amplitudes.size)
        eta = numpy.linspace(0, 1, 31)

        def rhs(eta_value, state):
            common = 0.01 * (1 - numpy.exp(-eta_value))
            columns = amplitudes[:, None] if state.ndim == 2 else amplitudes
            normalized = state / columns
            forcing = (rates - 1 - amplitudes.size * common) * numpy.exp(
                -eta_value
            )
            if state.ndim == 2:
                return columns * (
                    -rates[:, None] * normalized
                    + common * normalized.sum(axis=0)
                    + forcing[:, None]
                )
            return columns * (
                -rates * normalized + common * normalized.sum() + forcing
            )

        solve = mode_evolution.solve_ivp
        matrices = []

        def observe(*args, **kwargs):
            solver_jacobian = kwargs["jac"]

            def retain(eta_value, state):
                matrix = solver_jacobian(eta_value, state)
                self.assertTrue(issparse(matrix))
                matrices.append(matrix.copy())
                return matrix

            kwargs["jac"] = retain
            # Force an additional, later evaluation: new nonzeros cannot
            # disappear merely because startup selected sparse storage.
            retain(0.5, args[2])
            return solve(*args, **kwargs)

        for method in ("BDF", "Radau"):
            matrices.clear()
            with self.subTest(method=method):
                # SciPy first chooses its backend from the initial matrix.
                # Observe subsequent matrices only after that first choice.
                def first_then_observe(*args, **kwargs):
                    first = kwargs["jac"](0.0, args[2])
                    self.assertTrue(issparse(first))
                    matrices.append(first.copy())
                    return observe(*args, **kwargs)

                with mock.patch.object(
                    mode_evolution, "solve_ivp", side_effect=first_then_observe
                ):
                    actual = (
                        mode_evolution._integrate_scaled_continuous_history(
                            rhs,
                            amplitudes,
                            eta,
                            initial_time_scale=0.01,
                            rtol=1e-9,
                            atol=1e-12,
                            affine_system=True,
                            vectorized_rhs=rhs,
                            method=method,
                        )
                    )
                self.assertLess(matrices[0].nnz, matrices[1].nnz)
                self.assertEqual(matrices[1].nnz, amplitudes.size**2)
                numpy.testing.assert_allclose(
                    actual,
                    amplitudes[:, None] * numpy.exp(-eta),
                    rtol=2e-7,
                    atol=1e-35,
                )

    def test_one_pending_probe_keeps_its_compatible_native_executor(self):
        """Cache pressure cannot switch a final pending row to another ODE."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract(
                include_massive_neutrino=True
            )
        )
        raw["numerical"].update(k_sample_count=4, k_max=0.03)
        contract = test_cmb._prepare_declared_contract(raw)
        build = mode_evolution.build_declared_mode_evolution
        scalar_calls = []
        observed_runtime = []

        def observe(inputs):
            runtime = build(inputs)
            observed_runtime.append(runtime)
            original = runtime.evolve_declared_mode

            def scalar(*args, **kwargs):
                scalar_calls.append(args[0])
                return original(*args, **kwargs)

            return replace(runtime, evolve_declared_mode=scalar)

        execution = test_cmb.cmb_execution
        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)
        with mock.patch.object(
            execution, "build_declared_mode_evolution", side_effect=observe
        ):
            single = execution._compute_custom_cmb_spectrum_data_impl(
                contract,
                (2,),
                performance_timer=execution.PhaseTimer(),
                source_probe=(0.001,),
                requested_spectra=("TT",),
            )
        self.assertEqual(scalar_calls, [])
        self.assertEqual(single.runtime_envelope["batch_mode_count"], 1)
        self.assertLess(
            single.runtime_envelope["equation_context_required_state_count"],
            single.runtime_envelope["state_slot_count"],
        )
        # The costly production defect occurred in a singleton *reference*
        # partition. Exercise its override and retained native history too.
        runtime = observed_runtime[0]
        sinks = {}
        reference = runtime.evolve_declared_modes_batched(
            numpy.asarray((0.001,)),
            evolution_sample_count_override=32,
            history_sinks=sinks,
        )
        self.assertEqual(set(reference), {0})
        self.assertEqual(set(sinks), {0.001})
        self.assertLess(
            sinks[0.001]["evolution_eta"].size,
            runtime.mode_grids_for_k(0.001)[0]["eta"].size,
        )
        self.assertEqual(scalar_calls, [])
        execution.cache.clear_cmb_caches()
        pair = execution._compute_custom_cmb_spectrum_data_impl(
            contract,
            (2,),
            performance_timer=execution.PhaseTimer(),
            source_probe=(0.001, 0.0011),
            requested_spectra=("TT",),
        )
        for name, values in single.sources[0.001].items():
            scale = numpy.max(abs(pair.sources[0.001][name]))
            self.assertLessEqual(
                numpy.max(abs(values - pair.sources[0.001][name])),
                2e-5 * scale,
            )

    def test_stiff_declared_block_uses_full_local_interval_equations(self):
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._split_collision_contract()
        raw["numerical"]["k_sample_count"] = 2
        contract = test_cmb._prepare_declared_contract(raw)
        integrate = mode_evolution._integrate_scaled_continuous_history
        with mock.patch.object(
            mode_evolution,
            "_integrate_scaled_continuous_history",
            wraps=integrate,
        ) as continuous:
            probe = (
                test_cmb.cmb_execution._compute_custom_cmb_spectrum_data_impl(
                    contract,
                    (2,),
                    performance_timer=test_cmb.cmb_execution.PhaseTimer(),
                    source_probe=(0.001, 0.002),
                    requested_spectra=("TT",),
                )
            )
        callbacks = {
            call.args[0].__name__ for call in continuous.call_args_list
        }
        self.assertIn("interval_rhs", callbacks)
        self.assertTrue(
            all(
                call.kwargs.get("method") == "Radau"
                for call in continuous.call_args_list
                if call.args[0].__name__ == "interval_rhs"
            )
        )
        self.assertTrue(probe.runtime_envelope["continuous_affine_system"])
        self.assertTrue(
            all(
                call.kwargs.get("affine_system")
                for call in continuous.call_args_list
            )
        )
        self.assertTrue(
            all(
                call.kwargs.get("vectorized_rhs") is not None
                for call in continuous.call_args_list
            )
        )
        self.assertGreater(
            probe.runtime_envelope["stiff_collision_intervals"], 0
        )
        for source in probe.sources.values():
            self.assertTrue(
                all(
                    numpy.all(numpy.isfinite(values))
                    for values in source.values()
                )
            )

    def test_prefix_is_additional_and_refinement_cannot_poison_shared_grid(
        self,
    ):
        """Real declared grids retain distinct controls and physical phase."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb

        build = mode_evolution.build_declared_mode_evolution
        observed = {}

        def capture(inputs):
            """Inspect actual owner meshes before starting any ODE work."""

            controls = replace(
                inputs.adaptive_controls,
                evolution_enabled=True,
                evolution_minimum_nodes=128,
            )
            runtime = build(replace(inputs, adaptive_controls=controls))
            k_value = float(inputs.k_values[-1])
            fine = runtime.mode_grids_for_k(k_value)[0]["eta"].copy()
            coarse = runtime.mode_grids_for_k(
                k_value,
                evolution_sample_count_override=32,
            )[0]["eta"]
            intermediate = runtime.mode_grids_for_k(
                k_value,
                evolution_sample_count_override=80,
            )[0]["eta"]
            unchanged = runtime.mode_grids_for_k(k_value)[0]["eta"]
            observed.update(
                fine=fine,
                coarse=coarse,
                intermediate=intermediate,
                unchanged=unchanged,
                k=k_value,
                tolerance=controls.evolution_relative_tolerance,
            )
            raise _ModeHistoryCaptured()

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract()
        )
        raw["numerical"]["evolution_eta_sample_count"] = 128
        contract = test_cmb._prepare_declared_contract(raw)
        with (
            mock.patch.object(
                test_cmb.cmb_execution,
                "build_declared_mode_evolution",
                side_effect=capture,
            ),
            self.assertRaises(_ModeHistoryCaptured),
        ):
            test_cmb.cmb_execution._compute_custom_cmb_spectrum_data(
                contract,
                (20, 21),
                requested_spectra=("TT",),
                workload="fixed_parameter_diagnostic",
            )
        numpy.testing.assert_array_equal(
            observed["fine"], observed["unchanged"]
        )
        self.assertLess(len(observed["coarse"]), len(observed["intermediate"]))
        self.assertLess(len(observed["intermediate"]), len(observed["fine"]))
        self.assertGreater(len(observed["coarse"]), 32)
        self.assertLessEqual(
            observed["k"] * numpy.max(numpy.diff(observed["fine"])),
            numpy.sqrt(observed["tolerance"]) * (1.0 + 1e-12),
        )

    def test_superhorizon_polarization_matches_frozen_camb_after_decoupling(
        self,
    ):
        """Matched opacity isolates late instantaneous-equilibrium bias."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb
        from tests.project.lib import camb_reference

        fixture = camb_reference.load_lcdm_history_reference_fixture()
        eta = numpy.asarray(fixture["eta_mpc"])
        selected = eta >= 100.0
        eta = eta[selected]
        expected = {
            "e_gamma2": -numpy.asarray(fixture["histories"]["E_2"])[
                0, selected
            ]
            / 4.0,
        }
        raw = test_cmb._declared_scalar_hierarchy_contract()
        raw["param_map"].update(
            camb_reference.FIXED_LCDM_REFERENCE_CONTRACT["param_map"]
        )
        raw["numerical"].update(
            k_max=0.04,
            k_sample_count=2,
            eta_sample_count=1024,
            evolution_eta_sample_count=2048,
            source_grid_multiplier=1,
            photon_hierarchy_l_max=12,
            photon_polarization_hierarchy_l_max=12,
            neutrino_hierarchy_l_max=12,
            evolution_phase_step=0.1,
            initial_redshift=1e5,
            tight_coupling_ratio=2000.0,
        )
        raw["perturbations"]["numerics"] = dict(raw["numerical"])
        contract = test_cmb._prepare_declared_contract(raw)
        build = test_cmb.cmb_execution.build_declared_mode_evolution
        build_background = test_cmb.cmb_execution._build_custom_cmb_background
        reference = camb_reference.camb.get_background(
            camb_reference._make_camb_params(fixture["contract"])
        )

        def matched_opacity(*args, **kwargs):
            # Isolate the collision operator using independent thermal input.
            # Native recombination and reionization have separate comparisons.
            background = build_background(*args, **kwargs)
            opacity = reference.get_background_redshift_evolution(
                background.z_grid, vars=("opacity",)
            )["opacity"]
            return replace(
                background,
                tau_dot_grid=-opacity,
                tau_dot_of_eta=PchipInterpolator(
                    background.eta_grid,
                    -opacity,
                ),
            )

        coupling = mode_evolution._tight_coupling_is_active
        k = float(fixture["k_mpc_inv"][0])
        errors = {}
        self.addCleanup(test_cmb.cache.clear_cmb_caches)
        for path in ("scalar", "batch", "streaming_only"):
            test_cmb.cache.clear_cmb_caches()
            captured = {}

            def capture(inputs):
                if path == "batch":
                    inputs = replace(
                        inputs,
                        k_values=numpy.asarray((k, 1.01 * k)),
                        stage_diagnostic_k_values=(k,),
                        stage_diagnostic_fields=tuple(expected),
                        stage_diagnostic_histories_by_k=captured,
                    )
                    runtime = build(inputs)
                    runtime.evolve_declared_modes_batched(inputs.k_values)
                    history = captured[f"{k:.12g}"]
                else:
                    runtime = build(inputs)
                    history = {}
                    runtime.evolve_declared_mode(
                        k, history_sink=history, collect_diagnostics=False
                    )
                captured["curves"] = {
                    name: numpy.interp(
                        eta,
                        history["evolution_eta"],
                        history["evolution_histories"][name],
                    )
                    for name in expected
                }
                raise _ModeHistoryCaptured()

            def select_coupling(**kwargs):
                if path == "streaming_only":
                    kwargs["expansion_rate"] = 0.0
                return coupling(**kwargs)

            with (
                self.subTest(path=path),
                mock.patch.object(
                    test_cmb.cmb_execution,
                    "build_declared_mode_evolution",
                    side_effect=capture,
                ),
                mock.patch.object(
                    test_cmb.cmb_execution,
                    "_build_custom_cmb_background",
                    side_effect=matched_opacity,
                ),
                mock.patch.object(
                    mode_evolution,
                    "_tight_coupling_is_active",
                    side_effect=select_coupling,
                ),
                self.assertRaises(_ModeHistoryCaptured),
            ):
                test_cmb.cmb_execution._compute_custom_cmb_spectrum_data(
                    contract, numpy.asarray((20,)), requested_spectra=("TT",)
                )
            errors[path] = {
                name: float(
                    numpy.max(abs(captured["curves"][name] - values))
                    / numpy.max(abs(values))
                )
                for name, values in expected.items()
            }
        for path in ("scalar", "batch"):
            for name in expected:
                self.assertLess(errors[path][name], 0.02, errors)
        self.assertGreater(errors["streaming_only"]["e_gamma2"], 0.1, errors)

    def test_batched_damping_updates_selected_accepted_moments(self):
        """Selected moments follow exact damping; other states retain data."""

        initial = numpy.arange(1.0, 13.0).reshape(3, 4)
        accepted = initial.copy()
        rows = numpy.asarray((0, 2))
        slots = numpy.asarray((1, 3))
        rates = (0.5, 1.5)
        duration = 0.4
        expected = initial.copy()
        for row, rate in zip(rows, rates):
            for slot in slots:
                expected[row, slot] = initial[row, slot] * math.exp(
                    -rate * duration
                )
        mode_evolution._apply_batched_collision_damping(
            accepted,
            rows,
            slots,
            numpy.exp(-numpy.asarray(rates) * duration),
        )
        numpy.testing.assert_allclose(accepted, expected, rtol=1e-15)

    def test_continuous_history_is_invariant_under_state_unit_changes(self):
        """Tiny coupled states must retain their order-one physical effect."""

        eta = numpy.linspace(0.0, 2.0, 21)
        expected = 2.0 * numpy.exp(-eta) - numpy.exp(-2.0 * eta)
        for unit in (1e-28, 1e-14, 1.0, 1e14):
            with self.subTest(unit=unit):

                def rhs(time, state):
                    return numpy.asarray(
                        (-state[0] + state[1] / unit, -2.0 * state[1])
                    )

                history = mode_evolution._integrate_scaled_continuous_history(
                    rhs,
                    numpy.asarray((1.0, unit)),
                    eta,
                    initial_time_scale=1.0,
                    rtol=1e-7,
                    atol=1e-9,
                )
                numpy.testing.assert_allclose(
                    history[0], expected, rtol=2e-6, atol=1e-8
                )
                numpy.testing.assert_allclose(
                    history[1] / unit,
                    numpy.exp(-2.0 * eta),
                    rtol=2e-6,
                    atol=1e-8,
                )

    def test_continuous_callbacks_match_independent_normalized_generator(self):
        """BDF receives the physical derivative and Jacobian in new units."""

        unit = 1e-24
        solve = mode_evolution.solve_ivp

        def rhs(time, state):
            return numpy.asarray((-state[0] + state[1] / unit, -2 * state[1]))

        def observe(scaled_rhs, *args, **kwargs):
            scaled_jacobian = kwargs["jac"]
            point = numpy.asarray((0.5, 0.25))
            self.assertTrue(
                numpy.allclose(scaled_rhs(0.0, point), (0.0, -0.5))
            )
            self.assertTrue(
                numpy.allclose(
                    scaled_jacobian(0.0, point),
                    ((-1.0, 2.0), (0.0, -2.0)),
                    rtol=1e-8,
                    atol=1e-10,
                )
            )
            return solve(scaled_rhs, *args, **kwargs)

        with mock.patch.object(
            mode_evolution, "solve_ivp", side_effect=observe
        ):
            history = mode_evolution._integrate_scaled_continuous_history(
                rhs,
                numpy.asarray((1.0, unit)),
                numpy.asarray((0.0, 1.0)),
                initial_time_scale=1.0,
                rtol=1e-7,
                atol=1e-9,
            )
        self.assertAlmostEqual(
            history[0, -1],
            2 * math.exp(-1.0) - math.exp(-2.0),
            places=6,
        )

    def test_continuous_history_scales_initially_zero_driven_states(self):
        """The initial drive supplies the scale of an unseeded oscillator."""

        unit = 1e-24
        eta = numpy.linspace(0.0, 2.0, 21)

        def rhs(time, state):
            return numpy.asarray((state[1] / unit, -unit * state[0], 0.0))

        errors = []
        for tolerance in (1e-4, 1e-7):
            history = mode_evolution._integrate_scaled_continuous_history(
                rhs,
                numpy.asarray((1.0, 0.0, 0.0)),
                eta,
                initial_time_scale=1.0,
                rtol=tolerance,
                atol=tolerance / 100.0,
            )
            errors.append(float(numpy.max(abs(history[0] - numpy.cos(eta)))))
            numpy.testing.assert_array_equal(history[2], 0.0)
        self.assertLess(errors[1], errors[0] / 10.0)
        numpy.testing.assert_allclose(
            history[1] / unit, -numpy.sin(eta), rtol=2e-6, atol=1e-7
        )

    def test_continuous_history_resolves_indirectly_driven_tiny_states(self):
        """A zero higher moment must inherit the scale of its driving chain."""

        eta = numpy.linspace(0.0, 2.0, 21)
        for unit in (1e-28, 1.0, 1e14):
            with self.subTest(unit=unit):

                def rhs(time, state):
                    return numpy.asarray(
                        (
                            0.0,
                            unit * state[0],
                            unit * state[1],
                            state[2] / (unit * unit),
                        )
                    )

                history = mode_evolution._integrate_scaled_continuous_history(
                    rhs,
                    numpy.asarray((1.0, 0.0, 0.0, 0.0)),
                    eta,
                    initial_time_scale=1.0,
                    rtol=1e-7,
                    atol=1e-9,
                )
                numpy.testing.assert_allclose(
                    history[3],
                    eta**3 / 6.0,
                    rtol=2e-5,
                    atol=1e-8,
                )

    def test_scalar_early_metric_agrees_between_refined_rk_and_bdf(self):
        """Real hierarchy prefixes retain their regular mode in both solvers.

        Only evolution through the first visible source time is needed to
        expose the small-shear error. Run the actual compiled equations and
        collision operators, then stop before late evolution and projection.
        """

        from tests.copernican.lib.likelihoods.cmb import test_cmb

        build = test_cmb.cmb_execution.build_declared_mode_evolution
        integrate_rk = mode_evolution._integrate_batched_rk4
        integrate_bdf = mode_evolution._integrate_scaled_continuous_history
        self.addCleanup(test_cmb.cache.clear_cmb_caches)
        measurements = {}
        for label, continuous, phase, tolerance in (
            ("rk", False, 0.1, 1e-5),
            ("rk_refined", False, 0.025, 1e-5),
            ("bdf", True, 0.1, 1e-5),
            ("bdf_refined", True, 0.1, 2.5e-6),
        ):
            test_cmb.cache.clear_cmb_caches()
            raw = test_cmb._speedup_contract(
                test_cmb._declared_scalar_hierarchy_contract()
            )
            raw["numerical"].update(
                k_sample_count=2,
                evolution_phase_step=phase,
                ode_rtol=tolerance,
            )
            raw["perturbations"]["numerics"] = dict(raw["numerical"])
            raw["perturbations"]["accuracy_controls"][
                "continuous_collision_solver"
            ] = continuous
            contract = test_cmb._prepare_declared_contract(raw)
            prefix = {}

            def capture_inputs(inputs):
                prefix["end"] = float(inputs.source_grids["eta"][0])
                prefix["phi_index"] = inputs.runtime_spec.state_index_by_key[
                    ("Phi", "tau", 0)
                ]
                return build(inputs)

            def capture_rk(initial, eta, **kwargs):
                count = int(numpy.searchsorted(eta, prefix["end"])) + 1
                for name in ("required_substeps", "active_intervals"):
                    kwargs[name] = kwargs[name][:, : count - 1]
                history, _, _ = integrate_rk(initial, eta[:count], **kwargs)
                phi = numpy.interp(
                    prefix["end"],
                    eta[:count],
                    history[0, :, prefix["phi_index"]],
                )
                raise _EarlyHistoryCaptured(phi)

            def capture_bdf(rhs, initial, eta, **kwargs):
                early_eta = numpy.append(
                    eta[eta < prefix["end"]], prefix["end"]
                )
                history = integrate_bdf(rhs, initial, early_eta, **kwargs)
                raise _EarlyHistoryCaptured(history[prefix["phi_index"], -1])

            with (
                self.subTest(solver=label),
                mock.patch.object(
                    test_cmb.cmb_execution,
                    "build_declared_mode_evolution",
                    side_effect=capture_inputs,
                ),
                mock.patch.object(
                    mode_evolution,
                    "_integrate_batched_rk4",
                    side_effect=capture_rk,
                ) as rk_call,
                mock.patch.object(
                    mode_evolution,
                    "_integrate_scaled_continuous_history",
                    side_effect=capture_bdf,
                ) as bdf_call,
                self.assertRaises(_EarlyHistoryCaptured) as captured,
            ):
                test_cmb.cmb_execution._compute_custom_cmb_spectrum_data(
                    contract,
                    numpy.asarray((20, 21)),
                    requested_spectra=("TT",),
                    workload="fixed_parameter_diagnostic",
                )
            measurements[label] = captured.exception.phi
            self.assertEqual(rk_call.call_count, int(not continuous))
            self.assertEqual(bdf_call.call_count, int(continuous))
            self.assertGreater(measurements[label], 0.3)
            self.assertLess(measurements[label], 0.8)
        self.assertLess(
            abs(measurements["bdf"] - measurements["bdf_refined"]),
            1e-4,
            measurements,
        )
        self.assertLess(
            abs(measurements["rk_refined"] - measurements["bdf_refined"]),
            abs(measurements["rk"] - measurements["bdf_refined"]),
            measurements,
        )
        self.assertLess(
            abs(measurements["rk_refined"] - measurements["bdf_refined"]),
            1e-3,
            measurements,
        )

    def test_generated_stage_hubble_rate_matches_declared_expansion(self):
        """RK stages must obey Hconf=aH/c after evaluating declared H(a)."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb

        metrics = {"maximum_relative_error": 0.0, "stage_count": 0}
        resolve = mode_evolution._resolve_declared_graph_context_ordered
        build = mode_evolution.build_declared_mode_evolution
        selected_modes = []

        def capture_runtime(inputs):
            """Retain one real accepted batch row without another solve."""

            runtime = build(inputs)
            selected_modes.append(float(inputs.k_values[0]))
            runtime.history_capture_k_values.add(selected_modes[-1])
            return runtime

        def observe(context, *args, **kwargs):
            if {"a", "H", "Hconf"}.issubset(context):
                expected = (
                    numpy.asarray(context["a"])
                    * numpy.asarray(context["H"])
                    / 299792.458
                )
                relative = numpy.abs(
                    numpy.asarray(context["Hconf"]) - expected
                ) / numpy.maximum(numpy.abs(expected), 1e-30)
                metrics["maximum_relative_error"] = max(
                    metrics["maximum_relative_error"],
                    float(numpy.max(relative)),
                )
                metrics["stage_count"] += 1
            return resolve(context, *args, **kwargs)

        contract = test_cmb._prepare_declared_contract(
            test_cmb._speedup_contract(
                test_cmb._declared_scalar_hierarchy_contract()
            )
        )
        with (
            mock.patch.object(
                mode_evolution,
                "_resolve_declared_graph_context_ordered",
                side_effect=observe,
            ),
            mock.patch.object(
                test_cmb.cmb_execution,
                "build_declared_mode_evolution",
                side_effect=capture_runtime,
            ),
            mock.patch.object(
                mode_evolution.cache,
                "set_cmb_evolution_history",
                wraps=mode_evolution.cache.set_cmb_evolution_history,
            ) as retain,
        ):
            result = test_cmb.cmb_execution._compute_custom_cmb_spectrum_data(
                contract,
                numpy.asarray((20, 21)),
                requested_spectra=("TT",),
                workload="fixed_parameter_diagnostic",
            )
        self.assertGreater(metrics["stage_count"], 10)
        self.assertLess(metrics["maximum_relative_error"], 1e-12)
        self.assertTrue(retain.called)
        identity, epoch, histories = retain.call_args.args
        self.assertEqual(identity[-1], selected_modes[-1])
        self.assertIn("Phi", histories)
        self.assertEqual(histories["Phi"].shape, epoch.shape)
        envelope = result.runtime_envelope
        self.assertGreater(envelope["evolution_peak_state_cells"], 0)
        self.assertLessEqual(
            envelope["evolution_peak_state_cells"],
            envelope["evolution_history_cell_bound"],
        )
        self.assertEqual(
            envelope["evolution_history_cell_bound"],
            envelope["evolution_chunk_size"]
            * envelope["evolution_eta_sample_count"]
            * envelope["state_slot_count"],
        )

    def test_batch_capability_requires_a_shared_declared_schedule(self):
        """Batching remains available only for compatible scalar modes."""

        common = {
            "generated_scalar_hierarchy": True,
            "shared_mode_grids_enabled": True,
            "mode_count": 2,
            "has_momentum_runtimes": False,
            "has_end_boundaries": False,
            "adaptive_evolution_enabled": False,
            "adaptive_source_enabled": False,
            "adaptive_transfer_enabled": False,
            "adaptive_projection_enabled": False,
            "adaptive_k_enabled": False,
            "continuous_collision_solver": False,
            "has_declared_collision_operators": False,
            "state_slots": (SimpleNamespace(wrt="eta"),),
            "collision_runtimes": (
                SimpleNamespace(activation_strategy="always"),
            ),
        }
        self.assertTrue(mode_evolution._can_batch_declared_evolution(**common))
        self.assertTrue(
            mode_evolution._can_batch_declared_evolution(
                **{**common, "mode_count": 1}
            )
        )
        self.assertFalse(
            mode_evolution._can_batch_declared_evolution(
                **{**common, "mode_count": 0}
            )
        )
        self.assertFalse(
            mode_evolution._can_batch_declared_evolution(
                **{**common, "shared_mode_grids_enabled": False}
            )
        )

    def test_evolution_runtime_exposes_explicit_input_and_output_types(self):
        """The hierarchy owner publishes typed composition boundaries."""

        self.assertTrue(callable(mode_evolution.build_declared_mode_evolution))
        input_type = mode_evolution.DeclaredModeEvolutionInputs
        self.assertIn(
            "execution_plan",
            {field.name for field in fields(input_type)},
        )
        runtime_names = {
            field.name
            for field in fields(mode_evolution.DeclaredModeEvolutionRuntime)
        }
        self.assertIn("evolve_declared_mode", runtime_names)
        self.assertIn("snapshot", runtime_names)

    def test_evolution_owner_has_no_diagnostics_or_evidence_import(self):
        """Physics execution cannot depend on its diagnostic consumers."""

        source = Path(mode_evolution.__file__).read_text(encoding="utf-8")
        self.assertNotIn(".diagnostics import", source)
        self.assertNotIn(".evidence import", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
