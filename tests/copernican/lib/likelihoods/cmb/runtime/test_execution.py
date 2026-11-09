"""Focused tests for declared CMB numerical execution."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb.runtime import execution


class ExecutionModuleTestCase(unittest.TestCase):
    """Exercise public numerical execution composition directly."""

    def test_independent_source_times_rebind_and_restore_physical_context(
        self,
    ):
        """Native reevaluation preserves shared nodes without another solve."""

        from copernican.lib.likelihoods.cmb.runtime import mode_evolution
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(test_cmb._custom_contract())
        contract = test_cmb._prepare_declared_contract(raw)
        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)
        captured = {}
        build = execution.build_declared_mode_evolution

        class Prepared(BaseException):
            """Stop after preparing the real runtime, before evolution."""

        def capture_runtime(inputs):
            captured["runtime"] = build(inputs)
            captured["source_eta"] = inputs.source_grids["eta"].copy()
            raise Prepared()

        wave = 0.001
        with mock.patch.object(
            execution, "build_declared_mode_evolution", capture_runtime
        ):
            with self.assertRaises(Prepared):
                execution._compute_custom_cmb_spectrum_data_impl(
                    contract,
                    (2,),
                    source_probe=numpy.array([wave]),
                    requested_spectra=("TT",),
                    performance_timer=execution.PhaseTimer(),
                )
        sink = {}
        _, expected_sources = captured["runtime"].evolve_declared_mode(
            wave, history_sink=sink, collect_diagnostics=False
        )
        epoch = captured["source_eta"]
        resample = captured["runtime"].resample_mode_history
        arguments = (wave, sink["evolution_eta"], sink["evolution_histories"])
        _, baseline = resample(*arguments, collect_diagnostics=False)
        target = numpy.unique(
            numpy.r_[
                epoch,
                (epoch[:-1] + epoch[1:]) / 2,
            ]
        )
        with mock.patch.object(
            mode_evolution,
            "_integrate_batched_rk4",
            side_effect=AssertionError("No new ODE work"),
        ):
            _, evaluated = resample(
                *arguments, target_eta=target, collect_diagnostics=False
            )
            shared = numpy.searchsorted(target, epoch)
            for name, expected in expected_sources.items():
                self.assertEqual(evaluated[name].shape, target.shape)
                numpy.testing.assert_allclose(
                    evaluated[name][shared], expected, rtol=1e-12, atol=1e-20
                )
            with self.assertRaises(ValueError):
                resample(*arguments, target_eta=numpy.array([numpy.nan]))
            _, restored = resample(*arguments, collect_diagnostics=False)
        for name, expected in baseline.items():
            numpy.testing.assert_array_equal(restored[name], expected)

    def test_source_interpolation_executes_independent_physical_modes(self):
        """A bounded declared graph uses real sources and radial operators."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(test_cmb._custom_contract())
        raw["numerical"].update(k_sample_count=8, k_max=0.03, ell_max=20)
        controls = {
            "adaptive_transfer": {
                "enabled": True,
                "minimum_nodes": 8,
                "maximum_nodes": 16,
                "relative_tolerance": 0.01,
                "algorithm": "independent_source_interpolation",
            }
        }
        raw["perturbations"]["accuracy_controls"] = controls
        contract = test_cmb._prepare_declared_contract(raw)
        contract["_engine_numerical_plan"] = contract.pop("numerical")
        contract["_engine_accuracy_controls"] = controls
        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)
        with mock.patch.object(
            execution,
            "project_interpolated_source_histories",
            wraps=execution.project_interpolated_source_histories,
        ) as projection:
            result = execution._compute_custom_cmb_spectrum_data(
                contract,
                (2, 10),
                requested_spectra=("TT",),
                workload="fixed_parameter_diagnostic",
            )
        envelope = result.runtime_envelope
        self.assertEqual(
            envelope["adaptive_transfer_algorithm"],
            "independent_source_interpolation",
        )
        self.assertGreater(
            envelope["adaptive_transfer_new_node_work_units"], 0
        )
        self.assertGreater(
            len(result.k_grid), len(envelope["native_source_k_grid"])
        )
        self.assertEqual(result.k_grid[-1], 0.03)
        self.assertEqual(
            envelope["projection_bessel_mode_count"], result.k_grid.size
        )
        self.assertTrue(
            numpy.all(numpy.isfinite(result.spectra["TT"])),
            "integrate_projected must retain finite physical spectra",
        )
        supplied = projection.call_args.kwargs
        x = supplied["angular_k"][0] * (supplied["eta"][-1] - supplied["eta"])
        from scipy.special import spherical_jn

        source_kernel_provider = supplied["kernel_provider"]
        kernels = source_kernel_provider((2, 10), x)
        numpy.testing.assert_allclose(
            kernels.j_l,
            numpy.stack([spherical_jn(ell, x) for ell in (2, 10)]),
            rtol=1e-12,
            atol=1e-15,
            err_msg="source_kernel_provider must use the actual radial phase",
        )
        component = next(iter(supplied["components"].values()))
        sources = {
            name: values[0]
            for name, values in supplied["base_sources"].items()
        }
        source_uncertainty_projector = supplied["uncertainty_projector"]
        missing_uncertainty = "Missing source uncertainty"
        # on_projection_grid must not invent missing source measurements.
        with self.assertRaisesRegex(ValueError, missing_uncertainty):
            source_uncertainty_projector(
                "source", component, sources, kernels, supplied["angular_k"][0]
            )
        with mock.patch.object(execution.PhaseTimer, "heartbeat") as heartbeat:
            supplied["heartbeat"](5, len(supplied["angular_k"]))
        self.assertEqual(
            heartbeat.call_args.kwargs["completed"],
            5,
            "projection_progress must retain actual completed work",
        )
        self.assertEqual(
            heartbeat.call_args.kwargs["total"], len(supplied["angular_k"])
        )

    def test_native_pilot_admission_starts_with_one_interior_mode(self):
        """An early pilot failure must precede remaining physical modes."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract()
        )
        raw["perturbations"]["accuracy_controls"] = {
            "adaptive_evolution": {
                "enabled": True,
                "minimum_nodes": 32,
                "maximum_nodes": 1024,
                "relative_tolerance": 0.02,
            }
        }
        contract = test_cmb._prepare_declared_contract(raw)
        calls = []
        build = execution.build_declared_mode_evolution

        class Admitted(BaseException):
            """Observe work admission before any numerical trajectory."""

        def admit(waves, **kwargs):
            calls.append(numpy.array(waves, copy=True))
            raise Admitted()

        def capture(inputs):
            return replace(build(inputs), evolve_declared_modes_batched=admit)

        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)
        with mock.patch.object(
            execution, "build_declared_mode_evolution", capture
        ):
            with self.assertRaises(Admitted):
                execution._compute_custom_cmb_spectrum_data_impl(
                    contract,
                    (2,),
                    requested_spectra=("TT",),
                    performance_timer=execution.PhaseTimer(),
                )
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0].size,
            1,
            "evolve_pilots must validate one mode before more work",
        )
        self.assertGreater(calls[0][0], raw["numerical"]["k_min"])
        self.assertLess(calls[0][0], raw["numerical"]["k_max"])

    def test_ordinary_background_and_source_controls_remain_independent(
        self,
    ):
        """A dense physical background cannot silently multiply other axes."""

        from copernican.lib.likelihoods.cmb import cmb
        from tests.project.lib.ccmbs_production_graph import _build_lcdm_plugin

        class Captured(BaseException):
            pass

        plugin = _build_lcdm_plugin()
        runtime = plugin.get_cmb_declared_runtime(plugin.INITIAL_GUESSES)
        observed = {}

        def capture(inputs):
            observed.update(
                background=len(inputs.background.eta_grid),
                source=len(inputs.source_grids["eta"]),
                evolution=inputs.numerics.evolution_eta_sample_count,
                numerical_overrides=inputs.contract_or_params.get(
                    "_numerical_overrides", {}
                ),
            )
            raise Captured()

        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)
        with mock.patch.object(
            execution, "build_declared_mode_evolution", side_effect=capture
        ):
            with self.assertRaises(Captured):
                cmb.compute_cmb_spectrum_from_contract(
                    runtime,
                    [2, 20, 100],
                    spectra=["TT", "TE", "EE"],
                    workload="full_spectrum",
                )
        self.assertEqual(observed["numerical_overrides"], {})
        self.assertGreater(observed["background"], observed["evolution"])
        self.assertLess(observed["source"], 2 * observed["background"])

    def test_optional_quadrature_capture_preserves_actual_kernel_results(self):
        from copernican.lib.likelihoods.cmb.runtime import (
            evidence,
            spectrum_projection,
        )

        arguments = dict(
            primordial_grid=numpy.ones(5),
            log_k_values=numpy.linspace(-2, 0, 5),
            primary=numpy.arange(5.0),
            secondary=numpy.arange(5.0),
        )
        expected = spectrum_projection._integrate_power_spectrum(**arguments)
        events = []
        with evidence.capture_completed_cmb_products(
            lambda *event: events.append(event)
        ):
            self.assertEqual(
                execution._integrate_power_spectrum(**arguments).tolist(),
                expected.tolist(),
            )
        self.assertEqual(events[0][0], "spectrum_quadrature")
        sentinel = {"TT": expected}
        with mock.patch.object(
            execution,
            "_integrate_declared_spectra_kernel",
            return_value=sentinel,
        ) as kernel:
            self.assertIs(execution._integrate_declared_spectra(), sentinel)
        self.assertIsNone(kernel.call_args.kwargs["product_observer"])

    def test_prepare_projection_chunk_bounds_live_kernels_and_retains_axes(
        self,
    ):
        """Dense angular adaptive work retains one chunk, not all modes."""

        import weakref

        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(test_cmb._custom_contract())
        raw["numerical"].update(
            k_sample_count=12,
            eta_sample_count=64,
            evolution_eta_sample_count=64,
            ell_max=100,
        )
        raw["perturbations"]["numerics"] = dict(raw["numerical"])
        raw["perturbations"]["accuracy_controls"] = {
            "adaptive_source": {
                "enabled": True,
                "minimum_nodes": 32,
                "maximum_nodes": 128,
                "relative_tolerance": 2.0,
            },
            "adaptive_projection": {
                "enabled": True,
                "minimum_nodes": 32,
                "maximum_nodes": 128,
                "relative_tolerance": 2.0,
            },
            "fail_on_nonconvergence": False,
        }
        contract = test_cmb._prepare_declared_contract(raw)
        self.addCleanup(execution.cache.clear_cmb_caches)
        original = execution._get_cached_declared_projection_kernel_batch
        references = []
        live = []

        def observe(*args, **kwargs):
            batch = original(*args, **kwargs)
            references.append(weakref.ref(batch.j_l))
            live.append(sum(ref() is not None for ref in references))
            return batch

        execution.cache.clear_cmb_caches()
        with (
            mock.patch.object(execution, "_BESSEL_WORK_CELL_BUDGET", 10_000),
            mock.patch.object(
                execution.cache._CMB_BESSEL_BATCH_CACHE, "max_bytes", 0
            ),
            mock.patch.object(
                execution,
                "_get_cached_declared_projection_kernel_batch",
                side_effect=observe,
            ),
        ):
            small = execution._compute_custom_cmb_spectrum_data(
                contract,
                tuple(range(2, 101)),
                requested_spectra=("TT", "TE", "EE"),
            )
        envelope = small.runtime_envelope
        self.assertTrue(envelope["adaptive_source_enabled"])
        self.assertTrue(envelope["adaptive_projection_enabled"])
        self.assertEqual(
            envelope["projection_peak_live_modes"],
            1,
            "prepare_projection_chunk retained excess live modes",
        )
        self.assertGreater(envelope["projection_peak_kernel_bytes"], 0)
        self.assertLessEqual(max(live), 2)
        self.assertGreater(envelope["projection_bessel_batch_count"], 1)
        execution.cache.clear_cmb_caches()
        with mock.patch.object(
            execution, "_BESSEL_WORK_CELL_BUDGET", 8_000_000
        ):
            large = execution._compute_custom_cmb_spectrum_data(
                contract,
                tuple(range(2, 101)),
                requested_spectra=("TT", "TE", "EE"),
            )
        for name in ("TT", "TE", "EE"):
            numpy.testing.assert_allclose(
                small.spectra[name],
                large.spectra[name],
                rtol=5e-14,
                atol=0,
            )
        for name in small.transfer_components:
            numpy.testing.assert_allclose(
                small.transfer_components[name],
                large.transfer_components[name],
                rtol=5e-14,
                atol=0,
            )

    def test_transfer_refinement_batches_new_kinetic_modes(self):
        """Independent refined nodes preserve the original scalar authority."""

        from copernican.lib.likelihoods.cmb.runtime import mode_evolution
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract(
                include_massive_neutrino=True
            )
        )
        raw["numerical"].update(k_sample_count=4, k_max=0.03)
        raw["perturbations"]["accuracy_controls"] = {
            "adaptive_transfer": {
                "minimum_nodes": 4,
                "maximum_nodes": 10,
                "relative_tolerance": 2.0,
                "absolute_tolerance": 1e-12,
            }
        }
        contract = test_cmb._prepare_declared_contract(raw)
        self.addCleanup(execution.cache.clear_cmb_caches)
        execution.cache.clear_cmb_caches()
        with mock.patch.object(
            execution.cache._CMB_SOURCE_HISTORY_CACHE, "limit", 1
        ):
            batched = execution._compute_custom_cmb_spectrum_data(
                contract, (2, 10), requested_spectra=("TT", "TE", "EE")
            )
        envelope = batched.runtime_envelope
        self.assertGreater(envelope["transfer_reference_batch_mode_count"], 1)
        self.assertEqual(
            envelope["transfer_reference_batch_mode_count"],
            envelope["adaptive_transfer_new_node_work_units"],
        )
        self.assertEqual(envelope["adaptive_transfer_warm_node_count"], 0)
        self.assertEqual(
            envelope["batch_mode_count"],
            envelope["adaptive_transfer_base_node_count"],
        )
        execution.cache.clear_cmb_caches()

        def original_equations(
            _grouped, original, context, states, derivative, rates
        ):
            original(context, states, derivative, rates)

        with (
            mock.patch.object(
                mode_evolution,
                "_can_batch_declared_evolution",
                return_value=False,
            ),
            mock.patch.object(
                mode_evolution,
                "evaluate_scalar_equations",
                side_effect=original_equations,
            ),
        ):
            scalar = execution._compute_custom_cmb_spectrum_data(
                contract, (2, 10), requested_spectra=("TT", "TE", "EE")
            )
        self.assertEqual(scalar.runtime_envelope["batch_mode_count"], 0)
        for name in ("TT", "TE", "EE"):
            numpy.testing.assert_allclose(
                batched.spectra[name],
                scalar.spectra[name],
                rtol=2e-5,
                atol=0,
            )

    def test_source_grid_reuses_native_work_but_evaluates_fresh_sources(self):
        """Raw native reuse survives eviction and matches fresh evaluation."""

        from copernican.lib.likelihoods.cmb.runtime import mode_evolution
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract(
                include_massive_neutrino=True
            )
        )
        raw["numerical"].update(k_sample_count=4, k_max=0.03)
        raw["_engine_numerical_plan"] = dict(raw["numerical"])
        raw["_engine_accuracy_controls"] = {
            "accuracy_tier": "final",
            "adaptive_evolution": {
                "minimum_nodes": 64,
                "maximum_nodes": 512,
                "relative_tolerance": 0.02,
            },
            "adaptive_source": {
                "minimum_nodes": 64,
                "maximum_nodes": 512,
                "relative_tolerance": 0.02,
            },
            "adaptive_projection": {
                "minimum_nodes": 64,
                "maximum_nodes": 512,
                "relative_tolerance": 0.02,
            },
        }
        contract = test_cmb._prepare_declared_contract(raw)
        waves = (0.001, 0.002)
        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)

        def probe(point):
            return execution._compute_custom_cmb_spectrum_data_impl(
                point,
                (2,),
                source_probe=waves,
                requested_spectra=("TT", "TE", "EE"),
                performance_timer=execution.PhaseTimer(),
            )

        @execution.cache.retain_cmb_source_histories
        def exercise():
            original = probe(contract)
            changed, _ = execution.refine_numerical_axis(
                contract,
                "source",
                evidence={"source_eta_sample_count": original.eta.size},
            )
            with mock.patch.object(
                mode_evolution,
                "_integrate_batched_rk4",
                side_effect=AssertionError("Compatible raw work must survive"),
            ):
                refreshed = probe(changed)
            self.assertEqual(
                refreshed.runtime_envelope["native_history_cache_hits"],
                2,
                "_native_history_key must ignore only output sampling",
            )
            self.assertEqual(refreshed.runtime_envelope["probe_new_modes"], 0)
            self.assertEqual(
                refreshed.runtime_envelope["probe_resampled_modes"], 2
            )
            self.assertIn(
                "Phi",
                refreshed.runtime_envelope["state_history_max_abs_by_k"][
                    "0.001"
                ],
            )
            self.assertGreater(refreshed.eta.size, original.eta.size)
            stats = execution.cache.cmb_source_history_scope_stats()
            self.assertGreater(stats["spills"], 0)
            self.assertGreater(stats["loads"], 0)
            # New evolution controls must require independent raw work.
            different = {
                **changed,
                "_numerical_overrides": {
                    **changed.get("_numerical_overrides", {}),
                    "evolution_eta_sample_count": 2
                    * int(raw["numerical"]["evolution_eta_sample_count"]),
                },
            }
            with mock.patch.object(
                mode_evolution,
                "_integrate_batched_rk4",
                side_effect=AssertionError("A new native solve is required"),
            ):
                with self.assertRaisesRegex(
                    AssertionError, "new native solve"
                ):
                    probe(different)
            phase_changed = {
                **changed,
                "_numerical_overrides": {
                    **changed.get("_numerical_overrides", {}),
                    "evolution_phase_step": 0.5
                    * float(raw["numerical"]["evolution_phase_step"]),
                },
            }
            with mock.patch.object(
                mode_evolution,
                "_integrate_batched_rk4",
                side_effect=AssertionError(
                    "Changed integration needs a solve"
                ),
            ):
                with self.assertRaisesRegex(
                    AssertionError, "integration needs"
                ):
                    probe(phase_changed)
            return changed, refreshed

        with (
            mock.patch.object(
                execution.cache._CMB_SOURCE_HISTORY_CACHE, "limit", 1
            ),
            mock.patch.object(
                execution.cache._CMB_SOURCE_HISTORY_CACHE, "max_bytes", 1024
            ),
        ):
            changed, refreshed = exercise()
        execution.cache.clear_cmb_caches()
        fresh = probe(changed)
        self.assertEqual(
            fresh.runtime_envelope["native_history_cache_hits"], 0
        )
        for wave in waves:
            for name, values in fresh.sources[wave].items():
                numpy.testing.assert_allclose(
                    refreshed.sources[wave][name],
                    values,
                    rtol=2e-5,
                    atol=0,
                    err_msg="_resample_mode_history evaluates current sources",
                )

    def test_source_refinement_preserves_actual_background_and_native_grid(
        self,
    ):
        """A real source ladder changes independently of physical evolution."""

        from copernican.lib.likelihoods.cmb import cmb
        from tests.copernican.lib.test_model_adapter import (
            DeclaredLCDMModelTestCase,
        )

        class Captured(BaseException):
            """Stop after inspecting real planned grids, before any ODE."""

        observations = []
        build = execution.build_declared_mode_evolution

        def capture(inputs):
            runtime = build(inputs)
            native = runtime.mode_grids_for_k(float(inputs.k_values[-1]))[0]
            observations.append(
                {
                    "contract": inputs.contract_or_params,
                    "background": inputs.background.eta_grid.copy(),
                    "background_seed": inputs.background.resolution_evidence[
                        "refinement"
                    ]["fine_eta_nodes"],
                    "source_seed": inputs.numerics.eta_sample_count,
                    "source": inputs.source_grids["eta"].copy(),
                    "radial_count": inputs.runtime_envelope[
                        "los_phase_eta_sample_count"
                    ],
                    "native": {k: v.copy() for k, v in native.items()},
                }
            )
            raise Captured()

        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)
        plugin = DeclaredLCDMModelTestCase._build_plugin()
        with mock.patch.object(
            execution, "build_declared_mode_evolution", side_effect=capture
        ):
            with self.assertRaises(Captured):
                cmb.compute_cmb_spectrum_cached(
                    plugin,
                    tuple(plugin.INITIAL_GUESSES),
                    (2, 10, 20),
                    spectra=("TT", "TE", "EE"),
                )
            before = observations[0]
            finer, _ = execution.refine_numerical_axis(
                before["contract"],
                "source",
                evidence={"source_eta_sample_count": before["source"].size},
            )
            with self.assertRaises(Captured):
                execution._compute_custom_cmb_spectrum_data_impl(
                    finer,
                    (2, 10, 20),
                    background_provider=plugin,
                    requested_spectra=("TT", "TE", "EE"),
                    performance_timer=execution.PhaseTimer(),
                )
            background_finer, _ = execution.refine_numerical_axis(
                before["contract"],
                "background",
                evidence={
                    "background_resolution_seed": before["background_seed"]
                },
            )
            with self.assertRaises(Captured):
                execution._compute_custom_cmb_spectrum_data_impl(
                    background_finer,
                    (2, 10, 20),
                    background_provider=plugin,
                    requested_spectra=("TT", "TE", "EE"),
                    performance_timer=execution.PhaseTimer(),
                )
        background_after = observations[2]
        self.assertGreater(
            background_after["background"].size, before["background"].size
        )
        self.assertEqual(
            background_after["source_seed"], before["source_seed"]
        )
        after = observations[1]
        self.assertGreater(after["source"].size, before["source"].size)
        self.assertTrue(
            numpy.all(numpy.isin(before["source"], after["source"]))
        )
        self.assertEqual(after["radial_count"], before["radial_count"])
        numpy.testing.assert_array_equal(
            after["background"], before["background"]
        )
        for name in before["native"]:
            numpy.testing.assert_array_equal(
                after["native"][name],
                before["native"][name],
                err_msg=name,
            )

    def test_hint_identity_tracks_physics_not_observation_density(self):
        from copernican.lib.likelihoods.cmb import cmb
        from tests.copernican.lib.test_model_adapter import (
            DeclaredLCDMModelTestCase,
        )

        class Planned(BaseException):
            """Stop after the real public planner and hint identity."""

        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)
        plugin = DeclaredLCDMModelTestCase._build_plugin()
        keys = []

        def capture(key):
            keys.append(key)
            return None

        parameters = tuple(plugin.INITIAL_GUESSES)
        with (
            mock.patch.object(
                execution.cache, "get_cmb_resolution_hint", side_effect=capture
            ),
            mock.patch.object(
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=Planned,
            ),
        ):
            for ells in ((2, 10, 20), (2, 3, 10, 20), (20, 2)):
                with self.assertRaises(Planned):
                    cmb.compute_cmb_spectrum_cached(
                        plugin, parameters, ells, spectra=("TT", "TE", "EE")
                    )
            changed = (parameters[0] * 1.002, *parameters[1:])
            with self.assertRaises(Planned):
                cmb.compute_cmb_spectrum_cached(
                    plugin, changed, (2, 10, 20), spectra=("TT", "TE", "EE")
                )
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(keys[0], keys[2])
        self.assertNotEqual(keys[0], keys[3])

    def test_joint_budget_retry_uses_earlier_measured_trials(self):
        """The public controller rotates flat axes and retains its trials."""

        contract = {
            "_engine_numerical_plan": {
                "eta_sample_count": 192,
                "evolution_eta_sample_count": 128,
            }
        }
        result = execution.CustomCMBSpectrumData(
            ell_grid=numpy.asarray((2,)),
            k_grid=numpy.asarray((0.01, 0.02)),
            transfer_components={},
            spectra={"TT": numpy.ones(1)},
            runtime_envelope={
                "observable_error_budget": {"converged": True},
                "production_scalar_k_convergence": {"converged": True},
            },
        )
        failures = [
            execution.AdaptiveNonConvergenceError(
                "Measured joint budget is unresolved",
                label="observable-error-budget",
                evidence={
                    "products": {
                        "TT": {
                            "relative_tolerance": 0.01,
                            "relative_error_bound": source + 0.0065,
                            "axis_relative_bounds": {
                                "source": source,
                                "evolution": 0.004,
                                "background": 0.0025,
                            },
                        }
                    }
                },
            )
            for source in (0.0045, 0.0046)
        ]
        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)
        with (
            mock.patch.object(
                execution,
                "resolve_production_scalar_convergence",
                return_value=SimpleNamespace(
                    enabled=True,
                    required_spectra=("TT",),
                    fail_on_nonconvergence=True,
                ),
            ),
            mock.patch.object(
                execution,
                "_custom_cmb_spectrum_cache_key",
                return_value="joint-budget-controller",
            ),
            mock.patch.object(
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=[*failures, result],
            ) as compute,
        ):
            completed = execution._compute_custom_cmb_spectrum_data(
                contract, (2,), requested_spectra=("TT",)
            )
        trials = completed.runtime_envelope["automatic_resolution_refinement"]
        self.assertEqual(
            [row["axis"] for row in trials], ["source", "evolution"]
        )
        last = compute.call_args.args[0]["_numerical_overrides"]
        self.assertNotIn("eta_sample_count", last)
        self.assertEqual(
            compute.call_args.args[0]["_engine_accuracy_controls"][
                "adaptive_source"
            ]["minimum_nodes"],
            384,
        )
        self.assertEqual(last["evolution_eta_sample_count"], 256)
        self.assertEqual(trials[1]["unproductive_budget_axes"], ["source"])
        for call in compute.call_args_list:
            self.assertNotIn("initialize_resolution", call.kwargs)

    def test_numerical_candidate_requires_full_independent_acceptance(
        self,
    ):
        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)
        contract = {"_engine_numerical_plan": {"eta_sample_count": 192}}
        base = execution.CustomCMBSpectrumData(
            ell_grid=numpy.array((2,)),
            k_grid=numpy.array((0.01, 0.02)),
            transfer_components={},
            spectra={"TT": numpy.ones(1)},
            runtime_envelope={},
        )
        failure = execution.AdaptiveNonConvergenceError(
            "The real independent check failed",
            label="unknown-axis",
            evidence={},
        )
        control = SimpleNamespace(
            enabled=True, required_spectra=("TT",), fail_on_nonconvergence=True
        )
        with (
            mock.patch.object(
                execution,
                "resolve_production_scalar_convergence",
                return_value=control,
            ),
            mock.patch.object(
                execution,
                "_custom_cmb_spectrum_cache_key",
                return_value="candidate-test",
            ),
            mock.patch.object(
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                return_value=base,
            ) as compute,
            mock.patch.object(
                execution,
                "_compute_production_refinement",
                side_effect=failure,
            ) as refine,
            self.assertRaises(execution.AdaptiveNonConvergenceError),
        ):
            execution._compute_custom_cmb_spectrum_data(
                contract, (2,), requested_spectra=("TT",)
            )
        self.assertEqual(compute.call_count, 1)
        refine.assert_called_once()
        self.assertEqual(refine.call_args.args[0], contract)
        self.assertFalse(execution.cache.has_cmb_spectrum("candidate-test"))

        accepted = execution.CustomCMBSpectrumData(
            ell_grid=base.ell_grid,
            k_grid=base.k_grid,
            transfer_components=base.transfer_components,
            spectra=base.spectra,
            runtime_envelope={
                "production_scalar_k_convergence": {"converged": True},
                "observable_error_budget": {"converged": True},
            },
        )
        with (
            mock.patch.object(
                execution,
                "resolve_production_scalar_convergence",
                return_value=control,
            ),
            mock.patch.object(
                execution,
                "_custom_cmb_spectrum_cache_key",
                return_value="candidate-test",
            ),
            mock.patch.object(
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                return_value=accepted,
            ),
        ):
            result = execution._compute_custom_cmb_spectrum_data(
                contract, (2,), requested_spectra=("TT",)
            )
        self.assertTrue(execution.cache.has_cmb_spectrum("candidate-test"))
        numpy.testing.assert_array_equal(
            result.spectra["TT"], accepted.spectra["TT"]
        )

    def test_failed_base_budget_stops_before_independent_k_solve(self):
        """Extra k work cannot repair an excessive nonnegative axis bound."""

        result = execution.CustomCMBSpectrumData(
            ell_grid=numpy.asarray((2,)),
            k_grid=numpy.asarray((0.01, 0.02)),
            transfer_components={},
            spectra={"TT": numpy.ones(1)},
            runtime_envelope={
                "observable_axis_error_bounds": {"evolution": {"TT": [0.02]}}
            },
        )
        with mock.patch.object(
            execution, "_compute_custom_cmb_spectrum_data_impl"
        ) as solve:
            with self.assertRaises(
                execution.AdaptiveNonConvergenceError
            ) as caught:
                execution._compute_production_refinement(
                    {},
                    result,
                    (2,),
                    background_provider=None,
                    effective_requested_spectra=("TT",),
                    requested=("TT",),
                    production_controls=SimpleNamespace(
                        relative_tolerances={"TT": 0.01}
                    ),
                    timer=execution.PhaseTimer(),
                    workload="full_spectrum",
                    diagnostic_request=False,
                )
        solve.assert_not_called()
        self.assertEqual(caught.exception.label, "observable-error-budget")
        self.assertEqual(
            caught.exception.evidence["assessment_stage"],
            "before_k_refinement",
        )

    def test_finer_axis_evidence_cannot_be_hidden_by_coarse_budget(self):
        base = execution.CustomCMBSpectrumData(
            ell_grid=numpy.asarray((2,)),
            k_grid=numpy.asarray((0.01, 0.02)),
            transfer_components={},
            spectra={"TT": numpy.ones(1)},
            runtime_envelope={
                "observable_axis_error_bounds": {"evolution": {"TT": [0.001]}}
            },
        )
        fine = execution.CustomCMBSpectrumData(
            ell_grid=base.ell_grid,
            k_grid=numpy.asarray((0.01, 0.015, 0.02)),
            transfer_components={},
            spectra=base.spectra,
            runtime_envelope={
                "evolution_modes_evolved": 1,
                "observable_axis_error_bounds": {"evolution": {"TT": [0.02]}},
            },
        )
        controls = SimpleNamespace(
            k_refinement_factor=2,
            required_spectra=("TT",),
            relative_tolerances={"TT": 0.01},
            fail_on_nonconvergence=True,
        )
        for axis in ("evolution", "k", "angular_k"):
            with self.subTest(axis=axis):
                coarse_case = replace(
                    base,
                    runtime_envelope={
                        "observable_axis_error_bounds": {axis: {"TT": [0.001]}}
                    },
                )
                fine_case = replace(
                    fine,
                    runtime_envelope={
                        **dict(fine.runtime_envelope),
                        "observable_axis_error_bounds": {axis: {"TT": [0.02]}},
                    },
                )
                with mock.patch.object(
                    execution,
                    "_compute_custom_cmb_spectrum_data_impl",
                    return_value=fine_case,
                ):
                    with self.assertRaises(
                        execution.AdaptiveNonConvergenceError
                    ) as caught:
                        execution._compute_production_refinement(
                            {"_engine_numerical_plan": {"k_sample_count": 2}},
                            coarse_case,
                            (2,),
                            background_provider=None,
                            effective_requested_spectra=("TT",),
                            requested=("TT",),
                            production_controls=controls,
                            timer=execution.PhaseTimer(),
                            workload="full_spectrum",
                            diagnostic_request=False,
                        )
                self.assertEqual(
                    caught.exception.label, "observable-error-budget"
                )

    def test_hint_seeds_resolution_but_recomputes_the_current_acceptance(self):
        cache = execution.cache
        cache.clear_cmb_caches()
        self.addCleanup(cache.clear_cmb_caches)
        contract = {"_engine_numerical_plan": {"eta_sample_count": 192}}
        original = execution.CustomCMBSpectrumData(
            ell_grid=numpy.array((2,)),
            k_grid=numpy.array((0.01, 0.02)),
            transfer_components={},
            spectra={"TT": numpy.ones(1)},
            runtime_envelope={
                "observable_error_budget": {"converged": True},
                "production_scalar_k_convergence": {"converged": True},
            },
        )
        control = SimpleNamespace(
            enabled=True, required_spectra=("TT",), fail_on_nonconvergence=True
        )
        hint = {
            "_engine_numerical_plan": {"eta_sample_count": 768},
            "_numerical_overrides": {"eta_sample_count": 768},
        }
        with (
            mock.patch.object(
                execution,
                "resolve_production_scalar_convergence",
                return_value=control,
            ),
            mock.patch.object(
                execution,
                "_custom_cmb_spectrum_cache_key",
                return_value="exact-point",
            ),
            mock.patch.object(
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                return_value=original,
            ) as compute,
        ):
            cache.set_cmb_resolution_hint("exact-point", hint)
            result = execution._compute_custom_cmb_spectrum_data(
                contract, (2,), requested_spectra=("TT",)
            )
        self.assertEqual(compute.call_count, 1)
        self.assertEqual(
            compute.call_args.args[0]["_numerical_overrides"][
                "eta_sample_count"
            ],
            768,
        )
        self.assertTrue(result.runtime_envelope["resolution_hint_reused"])

        # An exact result hit must not replace the finer hint with its
        # original request's obsolete seed controls.
        def exact(_contract, _ells, **kwargs):
            kwargs["performance_timer"].cache_state = "exact_cache_hit"
            return original

        with (
            mock.patch.object(
                execution,
                "resolve_production_scalar_convergence",
                return_value=control,
            ),
            mock.patch.object(
                execution,
                "_custom_cmb_spectrum_cache_key",
                return_value="exact-point",
            ),
            mock.patch.object(
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=exact,
            ),
        ):
            execution._compute_custom_cmb_spectrum_data(
                contract, (2,), requested_spectra=("TT",)
            )
        self.assertEqual(
            cache.get_cmb_resolution_hint("exact-point")[
                "_numerical_overrides"
            ]["eta_sample_count"],
            768,
        )
        cache.clear_cmb_result_caches()
        failure = execution.AdaptiveNonConvergenceError(
            "Current check failed", label="unknown-axis", evidence={}
        )
        with (
            mock.patch.object(
                execution,
                "resolve_production_scalar_convergence",
                return_value=control,
            ),
            mock.patch.object(
                execution,
                "_custom_cmb_spectrum_cache_key",
                return_value="exact-point",
            ),
            mock.patch.object(
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=failure,
            ),
        ):
            with self.assertRaises(execution.AdaptiveNonConvergenceError):
                execution._compute_custom_cmb_spectrum_data(
                    contract, (2,), requested_spectra=("TT",)
                )

    def test_native_source_probe_evolves_real_states_without_projection(self):
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._analytic_signal_contract(decay_rate=1e-4)
        raw["numerical"].update(
            ode_rtol=1e-9,
            ode_atol=1e-12,
            evolution_eta_sample_count=256,
        )
        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)
        contract = test_cmb._prepare_declared_contract(raw)
        with mock.patch.object(
            execution, "_declared_graph_projection"
        ) as projection:
            probe = execution._compute_custom_cmb_spectrum_data_impl(
                contract,
                (2,),
                performance_timer=execution.PhaseTimer(),
                requested_spectra=("TT",),
                source_probe=(0.001, 0.002),
            )
        self.assertIsInstance(probe, execution.NativeSourceProbe)
        self.assertEqual(probe.runtime_envelope["probe_new_modes"], 2)
        projection.assert_not_called()
        for sources in probe.sources.values():
            values = sources["signal_source"]
            expected = values[0] * numpy.exp(
                -1e-4 * (probe.eta - probe.eta[0])
            )
            numpy.testing.assert_allclose(
                values, expected, rtol=2e-5, atol=1e-8
            )

    def test_measured_evolution_error_promotes_only_its_axis(self):
        """Engine requests retry resolution with unchanged accuracy."""

        contract = {
            "_engine_numerical_plan": {"evolution_eta_sample_count": 128},
            "_engine_accuracy_controls": {
                "adaptive_evolution": {
                    "enabled": True,
                    "minimum_nodes": 128,
                    "maximum_nodes": 512,
                    "relative_tolerance": 0.01,
                },
            },
        }
        result = execution.CustomCMBSpectrumData(
            ell_grid=numpy.asarray((2,)),
            k_grid=numpy.asarray((0.01,)),
            transfer_components={},
            spectra={"TT": numpy.ones(1)},
        )
        failure = execution.AdaptiveNonConvergenceError(
            "Measured evolution error",
            label="scalar evolution history",
            evidence={"relative_error": 0.02, "absolute_error": 1e-7},
        )
        with mock.patch.object(
            execution,
            "_compute_custom_cmb_spectrum_data_impl",
            side_effect=(failure, result),
        ) as compute:
            accepted = execution._compute_custom_cmb_spectrum_data(
                contract,
                (2,),
                requested_spectra=("TT",),
            )
        self.assertEqual(compute.call_count, 2)
        promoted = compute.call_args.args[0]
        self.assertEqual(
            promoted["_numerical_overrides"],
            {
                "evolution_eta_sample_count": 256,
            },
        )
        self.assertEqual(
            promoted["_engine_accuracy_controls"]["adaptive_evolution"][
                "relative_tolerance"
            ],
            0.01,
        )
        self.assertNotIn("_numerical_overrides", contract)
        self.assertEqual(
            len(accepted.runtime_envelope["automatic_evolution_refinement"]), 1
        )
        with mock.patch.object(
            execution,
            "_compute_custom_cmb_spectrum_data_impl",
            side_effect=failure,
        ) as compute:
            with self.assertRaises(execution.AdaptiveNonConvergenceError):
                execution._compute_custom_cmb_spectrum_data(
                    contract,
                    (2,),
                    requested_spectra=("TT",),
                )
        self.assertEqual(compute.call_count, 4)

    def test_request_metadata_does_not_invalidate_physical_mode_histories(
        self,
    ):
        """Reordering observations shares modes; changed equations do not."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb

        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)
        raw = test_cmb._analytic_signal_contract(decay_rate=1.0e-4)
        raw["numerical"].update(
            ell_max=3,
            k_min=1.0e-5,
            k_max=0.02,
            k_sample_count=8,
            eta_sample_count=64,
            evolution_eta_sample_count=32,
        )
        contract = test_cmb._prepare_declared_contract(raw)
        contract["_engine_planner_evidence"] = {"signature": "first"}
        first = execution._compute_custom_cmb_spectrum_data(
            contract,
            (2, 3),
            requested_spectra=("TT",),
        )
        contract["_engine_planner_evidence"] = {"signature": "second"}
        second = execution._compute_custom_cmb_spectrum_data(
            contract,
            (3, 2),
            requested_spectra=("TT",),
        )
        numpy.testing.assert_array_equal(
            first.spectra["TT"], second.spectra["TT"][::-1]
        )
        self.assertEqual(
            second.runtime_envelope["evolution_cached_mode_count"],
            len(second.k_grid),
        )
        self.assertEqual(second.runtime_envelope["evolution_modes_evolved"], 0)
        changed_raw = test_cmb._analytic_signal_contract(decay_rate=2.0e-4)
        changed_raw["numerical"].update(raw["numerical"])
        changed = execution._compute_custom_cmb_spectrum_data(
            test_cmb._prepare_declared_contract(changed_raw),
            (3, 2),
            requested_spectra=("TT",),
        )
        self.assertEqual(
            changed.runtime_envelope["evolution_cached_mode_count"], 0
        )
        self.assertGreater(
            changed.runtime_envelope["evolution_modes_evolved"], 0
        )
        self.assertFalse(
            numpy.array_equal(changed.spectra["TT"], second.spectra["TT"])
        )

    def test_nested_k_refinement_retains_real_modes_with_small_cache(self):
        """A nested real solve evolves new nodes once under cache pressure."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb

        execution.cache.clear_cmb_caches()
        self.addCleanup(execution.cache.clear_cmb_caches)
        raw = test_cmb._analytic_signal_contract(decay_rate=1.0e-4)
        raw["numerical"].update(
            k_min=1.0e-5,
            k_max=0.02,
            k_sample_count=64,
            ell_max=3,
            eta_sample_count=64,
            evolution_eta_sample_count=32,
        )
        raw["perturbations"]["equations"]["evolve_signal_mode"].update(
            lhs={
                "kind": "derivative",
                "variable": "signal_mode",
                "wrt": "eta",
                "order": 1,
            },
            rhs="-decay_rate * k * signal_mode",
        )
        raw["perturbations"]["initial_conditions"]["signal_seed"]["target"][
            "wrt"
        ] = "eta"
        raw["perturbations"]["accuracy_controls"] = {
            "production_scalar_convergence": {
                "enabled": True,
                "k_refinement_factor": 2,
                "required_spectra": ["TT"],
                "fail_on_nonconvergence": True,
            }
        }
        contract = test_cmb._prepare_declared_contract(raw)
        evolved = []
        build = execution.build_declared_mode_evolution

        def record_runtime(inputs):
            runtime = build(inputs)

            def evolve_mode(k_value, **kwargs):
                result = runtime.evolve_declared_mode(k_value, **kwargs)
                if kwargs.get("count_as_primary_evolution", True):
                    evolved.append(float(k_value))
                return result

            def evolve_batch(k_values, **kwargs):
                result = runtime.evolve_declared_modes_batched(
                    k_values, **kwargs
                )
                evolved.extend(float(k_values[index]) for index in result)
                return result

            return replace(
                runtime,
                evolve_declared_mode=evolve_mode,
                evolve_declared_modes_batched=evolve_batch,
            )

        with (
            mock.patch.object(
                execution,
                "build_declared_mode_evolution",
                side_effect=record_runtime,
            ),
            mock.patch.object(
                execution.cache._CMB_SOURCE_HISTORY_CACHE,
                "limit",
                1,
            ),
            mock.patch.object(
                execution.cache._CMB_SOURCE_HISTORY_CACHE,
                "max_bytes",
                128,
            ),
        ):
            result = execution._compute_custom_cmb_spectrum_data(
                contract,
                (2, 3),
                requested_spectra=("TT",),
            )
        report = result.runtime_envelope["production_scalar_k_convergence"]
        self.assertTrue(report["converged"])
        self.assertEqual(
            len(evolved), len(set(evolved)), "primary mode was evolved twice"
        )
        self.assertEqual(
            len(evolved),
            report["refined_count"],
            "missing independently evolved refinement nodes",
        )
        self.assertGreaterEqual(
            report["refined_cached_mode_count"],
            report["base_count"],
            "base histories were not retained",
        )
        self.assertEqual(
            report["new_node_work_units"], report["new_node_count"]
        )
        self.assertEqual(len(result.k_grid), report["base_count"])
        self.assertGreater(
            result.runtime_envelope["request_source_history_retention"][
                "spills"
            ],
            0,
        )
        self.assertTrue(numpy.all(result.spectra["TT"] > 0.0))

    def test_custom_spectrum_data_accessors_return_named_payloads(self):
        """Transfer and spectrum accessors should expose stable arrays."""

        spectrum_data = execution.CustomCMBSpectrumData(
            ell_grid=numpy.array([20.0, 30.0]),
            k_grid=numpy.array([0.1, 0.2]),
            transfer_components={
                "temperature": numpy.array([1.0, 2.0]),
                "polarization_e": numpy.array([3.0, 4.0]),
            },
            spectra={
                "TT": numpy.array([5.0, 6.0]),
                "TE": numpy.array([7.0, 8.0]),
                "EE": numpy.array([9.0, 10.0]),
            },
        )

        self.assertTrue(
            numpy.array_equal(spectrum_data.Delta_l_T, numpy.array([1.0, 2.0]))
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.Delta_l_E, numpy.array([3.0, 4.0]))
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.C_l_TT, numpy.array([5.0, 6.0]))
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.C_l_TE, numpy.array([7.0, 8.0]))
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.C_l_EE, numpy.array([9.0, 10.0]))
        )

    def test_production_scalar_rule_rejects_nonconverged_doubled_grid(self):
        """The production wrapper must reject a failed k-grid refinement."""

        base_values = {
            "TT": numpy.asarray([10.0, 20.0]),
            "TE": numpy.asarray([1.0, 2.0]),
            "EE": numpy.asarray([3.0, 4.0]),
        }
        contract = {
            "model_name": "test",
            "numerical": {"k_sample_count": 8},
            "perturbation_data": SimpleNamespace(
                accuracy_controls={
                    "production_scalar_convergence": {
                        "enabled": True,
                        "k_refinement_factor": 2,
                        "required_spectra": ["TT", "TE", "EE"],
                        "fail_on_nonconvergence": True,
                    }
                }
            ),
        }

        def fake_impl(request, *args, **kwargs):
            del args, kwargs
            scale = 1.1 if request.get("_numerical_overrides") else 1.0
            return execution.CustomCMBSpectrumData(
                ell_grid=numpy.array([2, 3]),
                k_grid=numpy.array([0.1, 0.2]),
                transfer_components={},
                spectra={
                    name: values * scale
                    for name, values in base_values.items()
                },
            )

        with (
            mock.patch.object(
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=fake_impl,
            ),
            mock.patch.object(execution.cache, "set_cmb_spectrum"),
        ):
            with self.assertRaisesRegex(ValueError, "invalid k-grid evidence"):
                execution._compute_custom_cmb_spectrum_data(
                    contract,
                    (2, 3),
                    requested_spectra=("TT", "TE", "EE"),
                )

    def test_production_scalar_rule_records_converged_metrics(self):
        """A passing doubled-grid comparison remains in the envelope."""

        contract = {
            "model_name": "test",
            "numerical": {"k_sample_count": 8},
            "perturbation_data": SimpleNamespace(
                accuracy_controls={
                    "production_scalar_convergence": {
                        "enabled": True,
                        "k_refinement_factor": 2,
                        "required_spectra": ["TT", "TE", "EE"],
                        "relative_tolerances": {
                            "TT": 0.01,
                            "TE": 0.02,
                            "EE": 0.01,
                        },
                        "fail_on_nonconvergence": True,
                    }
                }
            ),
        }

        def fake_impl(request, *args, **kwargs):
            del args, kwargs
            scale = 1.001 if request.get("_numerical_overrides") else 1.0
            values = {
                "TT": numpy.asarray([10.0, 20.0]),
                "TE": numpy.asarray([1.0, 2.0]),
                "EE": numpy.asarray([3.0, 4.0]),
            }
            return execution.CustomCMBSpectrumData(
                ell_grid=numpy.array([2, 3]),
                k_grid=(
                    numpy.array([0.1, 0.15, 0.2])
                    if request.get("_numerical_overrides")
                    else numpy.array([0.1, 0.2])
                ),
                transfer_components={},
                spectra={
                    name: value * scale for name, value in values.items()
                },
                runtime_envelope={
                    "evolution_modes_evolved": (
                        1 if request.get("_numerical_overrides") else 0
                    )
                },
            )

        with (
            mock.patch.object(
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=fake_impl,
            ),
            mock.patch.object(execution.cache, "set_cmb_spectrum"),
        ):
            result = execution._compute_custom_cmb_spectrum_data(
                contract,
                (2, 3),
                requested_spectra=("TT", "TE", "EE"),
            )

        record = result.runtime_envelope["production_scalar_k_convergence"]
        self.assertTrue(record["converged"])
        self.assertEqual(record["base_count"], 2)
        self.assertEqual(record["refined_count"], 3)
        self.assertTrue(record["distinct"])
        self.assertTrue(record["nested"])
        self.assertEqual(record["new_node_count"], 1)
        self.assertGreater(record["new_node_work_units"], 0)
        self.assertTrue(record["cold_refinement"])
        self.assertFalse(record["warm_reuse"])
        self.assertNotEqual(
            record["base_grid_sha256"], record["refined_grid_sha256"]
        )
        self.assertEqual(len(record["refinement_identity"]), 64)
        self.assertEqual(record["declared_base_count"], 8)
        self.assertEqual(record["declared_refined_count"], 16)
        self.assertEqual(set(record["metrics"]), {"TT", "TE", "EE"})

    def test_cold_refinement_requires_measured_new_work(self):
        """A nominal finer count cannot certify a cold production run."""

        contract = {
            "numerical": {"k_sample_count": 8},
            "perturbation_data": SimpleNamespace(
                accuracy_controls={
                    "production_scalar_convergence": {
                        "enabled": True,
                        "required_spectra": ["TT", "TE", "EE"],
                    }
                }
            ),
        }

        def fake_impl(request, *args, **kwargs):
            del args, kwargs
            refined = bool(request.get("_numerical_overrides"))
            return execution.CustomCMBSpectrumData(
                ell_grid=numpy.asarray((2, 3)),
                k_grid=numpy.asarray(
                    (0.1, 0.15, 0.2) if refined else (0.1, 0.2)
                ),
                transfer_components={},
                spectra={
                    "TT": numpy.asarray((1.0, 2.0)),
                    "TE": numpy.asarray((0.1, 0.2)),
                    "EE": numpy.asarray((0.5, 0.8)),
                },
            )

        with mock.patch.object(
            execution,
            "_compute_custom_cmb_spectrum_data_impl",
            side_effect=fake_impl,
        ):
            with self.assertRaisesRegex(ValueError, "no measured new work"):
                execution._compute_custom_cmb_spectrum_data(
                    contract,
                    (2, 3),
                    requested_spectra=("TT", "TE", "EE"),
                )

    def test_warm_refinement_requires_matching_accepted_finer_grid(self):
        """Exact warm reuse records its matching nested finer calculation."""

        contract = {
            "numerical": {"k_sample_count": 8},
            "perturbation_data": SimpleNamespace(
                accuracy_controls={
                    "production_scalar_convergence": {
                        "enabled": True,
                        "required_spectra": ["TT", "TE", "EE"],
                    }
                }
            ),
        }

        def fake_impl(request, *args, **kwargs):
            del args
            refined = bool(request.get("_numerical_overrides"))
            if refined:
                kwargs["performance_timer"].mark_cache_state("exact_cache_hit")
            return execution.CustomCMBSpectrumData(
                ell_grid=numpy.asarray((2, 3)),
                k_grid=numpy.asarray(
                    (0.1, 0.15, 0.2) if refined else (0.1, 0.2)
                ),
                transfer_components={},
                spectra={
                    "TT": numpy.asarray((1.0, 2.0)),
                    "TE": numpy.asarray((0.1, 0.2)),
                    "EE": numpy.asarray((0.5, 0.8)),
                },
            )

        with (
            mock.patch.object(
                execution,
                "_compute_custom_cmb_spectrum_data_impl",
                side_effect=fake_impl,
            ),
            mock.patch.object(execution.cache, "set_cmb_spectrum"),
        ):
            result = execution._compute_custom_cmb_spectrum_data(
                contract,
                (2, 3),
                requested_spectra=("TT", "TE", "EE"),
            )

        record = result.runtime_envelope["production_scalar_k_convergence"]
        self.assertTrue(record["warm_reuse"])
        self.assertTrue(record["matching_accepted_finer_calculation"])
        self.assertEqual(record["new_node_work_units"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
