"""Focused tests for declared CMB numerical execution."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb.runtime import execution


class ExecutionModuleTestCase(unittest.TestCase):
    """Exercise public numerical execution composition directly."""

    def test_observable_budget_covers_all_declared_standard_surfaces(self):
        """BB, TP and EP have numerical budgets alongside TT/TE/EE/PP."""
        from copernican.lib.likelihoods.cmb.runtime import convergence

        names = ("TT", "TE", "EE", "BB", "PP", "TP", "EP")
        spectra = {
            name: numpy.asarray([0.0 if name == "BB" else 1.0])
            for name in names
        }
        bounds = {name: numpy.zeros(1) for name in names}
        result = SimpleNamespace(
            spectra=spectra,
            runtime_envelope={
                "observable_axis_error_bounds": {"source": bounds}
            },
        )
        controls = convergence.resolve_production_scalar_convergence(
            {
                "_engine_accuracy_controls": {
                    "production_scalar_convergence": {
                        "enabled": True,
                        "required_spectra": names,
                    }
                }
            }
        )
        budget = execution._require_observable_budget(result, controls)
        self.assertTrue(budget["converged"])
        self.assertEqual(set(budget["products"]), set(names))
        bounds["EP"] = numpy.asarray([0.01])
        with self.assertRaises(execution.AdaptiveNonConvergenceError):
            execution._require_observable_budget(result, controls)

    def test_observable_budget_enforces_combined_numerical_limit(self):
        """Every observable keeps the frozen joint numerical budget."""

        from copernican.lib.likelihoods.cmb.runtime import convergence

        for name in ("TT", "TE", "EE", "BB", "PP", "TP", "EP"):
            with self.subTest(spectrum=name):
                controls = convergence.resolve_production_scalar_convergence(
                    {
                        "_engine_accuracy_controls": {
                            "production_scalar_convergence": {
                                "enabled": True,
                                "required_spectra": (name,),
                            }
                        }
                    }
                )
                result = SimpleNamespace(
                    spectra={name: numpy.ones(1)},
                    runtime_envelope={
                        "observable_axis_error_bounds": {
                            "evolution": {name: numpy.asarray([0.006])}
                        }
                    },
                )
                with self.assertRaises(
                    execution.AdaptiveNonConvergenceError
                ) as caught:
                    execution._require_observable_budget(result, controls)
                self.assertEqual(
                    caught.exception.evidence["products"][name][
                        "relative_tolerance"
                    ],
                    0.005,
                )
                result.runtime_envelope["observable_axis_error_bounds"][
                    "evolution"
                ][name] = numpy.asarray([0.005])
                budget = execution._require_observable_budget(result, controls)
                self.assertTrue(budget["converged"])

        controls = convergence.resolve_production_scalar_convergence(
            {
                "_engine_accuracy_controls": {
                    "production_scalar_convergence": {
                        "enabled": True,
                        "required_spectra": ("TT",),
                        "relative_tolerances": {"TT": 0.001},
                    }
                }
            }
        )
        result = SimpleNamespace(
            spectra={"TT": numpy.ones(1)},
            runtime_envelope={
                "observable_axis_error_bounds": {
                    "evolution": {"TT": numpy.asarray([0.002])}
                }
            },
        )
        with self.assertRaises(
            execution.AdaptiveNonConvergenceError
        ) as caught:
            execution._require_observable_budget(result, controls)
        self.assertEqual(
            caught.exception.evidence["products"]["TT"]["relative_tolerance"],
            0.001,
        )

    def test_observable_budget_locates_spectrum_and_axis_peaks(self):
        """Retain physical multipoles and the actual normalization scales."""

        from copernican.lib.likelihoods.cmb.runtime import convergence

        spectra = {
            "TT": numpy.ones(3),
            "EE": numpy.ones(3),
            "TE": numpy.full(3, 0.2),
        }
        bounds = {
            "evolution": {
                "TT": numpy.zeros(3),
                "EE": numpy.zeros(3),
                "TE": numpy.asarray((0.0001, 0.001, 0.002)),
            },
            "projection": {
                "TT": numpy.zeros(3),
                "EE": numpy.zeros(3),
                "TE": numpy.asarray((0.004, 0.0, 0.0)),
            },
        }
        result = SimpleNamespace(
            ell_grid=numpy.asarray((2, 10, 20)),
            spectra=spectra,
            runtime_envelope={"observable_axis_error_bounds": bounds},
        )
        controls = convergence.resolve_production_scalar_convergence(
            {
                "_engine_accuracy_controls": {
                    "production_scalar_convergence": {"enabled": True}
                }
            }
        )
        with self.assertRaises(
            execution.AdaptiveNonConvergenceError
        ) as caught:
            execution._require_observable_budget(result, controls)
        evidence = caught.exception.evidence
        self.assertEqual(evidence["ell_interval"], (2, 20))
        product = evidence["products"]["TE"]
        self.assertEqual(product["absolute_error_peak_ell"], 2)
        self.assertEqual(product["axis_peak_ells"]["evolution"], 20)
        self.assertEqual(product["axis_peak_ells"]["projection"], 2)
        self.assertEqual(product["normalized_cross_peak_ell"], 2)
        self.assertEqual(
            product["relative_normalization"], "peak_absolute_spectrum"
        )
        self.assertEqual(
            product["cross_normalization"], "matching_auto_spectra"
        )
        self.assertAlmostEqual(product["normalization_scale"], 0.2)
        self.assertAlmostEqual(product["absolute_error_bound"], 0.0041)

    def test_native_axis_errors_locate_actual_failing_sources(self):
        """Report the source and epoch without rejecting negligible errors."""

        eta = numpy.asarray((10.0, 20.0, 30.0))
        base = {
            "temperature_isw": numpy.asarray((2.0, 2.0, 2.0)),
            "negligible": numpy.asarray((1e-14, 1e-14, 1e-14)),
        }
        reference = {
            "temperature_isw": numpy.asarray((2.0, 1.96, 2.0)),
            "negligible": numpy.zeros(3),
        }
        errors, maximum, failed = execution._measure_native_axis_errors(
            eta,
            base,
            reference,
            absolute_tolerance=1e-12,
            relative_tolerance=0.01,
        )
        self.assertEqual(failed, ("temperature_isw",))
        self.assertAlmostEqual(maximum, 0.02)
        self.assertAlmostEqual(
            errors["temperature_isw"]["absolute_error"], 0.04
        )
        self.assertEqual(errors["temperature_isw"]["worst_eta"], 20.0)
        self.assertAlmostEqual(errors["negligible"]["relative_error"], 1.0)
        with self.assertRaisesRegex(ValueError, "matching grids"):
            execution._measure_native_axis_errors(
                eta[:-1],
                base,
                reference,
                absolute_tolerance=1e-12,
                relative_tolerance=0.01,
            )

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

        from copernican.lib.likelihoods.cmb.runtime import evidence
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
        events = []
        numerical_products = []

        def retain_final(kind, payload):
            if kind == "accepted_numerical_spectrum":
                events.append(payload)
            elif kind in {"projection", "spectrum_quadrature"}:
                numerical_products.append((kind, payload))

        with (
            evidence.capture_completed_cmb_products(retain_final),
            mock.patch.object(
                execution,
                "project_interpolated_source_histories",
                wraps=execution.project_interpolated_source_histories,
            ) as projection,
        ):
            result = execution._compute_custom_cmb_spectrum_data(
                contract,
                (2, 10),
                requested_spectra=("TT",),
                workload="fixed_parameter_diagnostic",
            )
        self.assertEqual(len(events), 1)
        numpy.testing.assert_array_equal(events[0]["k"], result.k_grid)
        numpy.testing.assert_array_equal(events[0]["ell"], result.ell_grid)
        numpy.testing.assert_array_equal(
            events[0]["spectra"]["TT"], result.spectra["TT"]
        )
        projected = next(
            payload
            for kind, payload in reversed(numerical_products)
            if kind == "projection"
        )
        matching = [
            payload
            for kind, payload in numerical_products
            if kind == "spectrum_quadrature"
            and payload.get("observable") == "TT"
            and numpy.array_equal(payload["C_ell"], projected["spectra"]["TT"])
            and numpy.array_equal(payload["log_k"], numpy.log(projected["k"]))
        ]
        self.assertTrue(matching)
        self.assertTrue(
            any(
                numpy.array_equal(
                    row["weighted_transfer_products"],
                    row["primordial"][None, :]
                    * (
                        numpy.asarray(
                            projected["transfers"][row["primary_component"]],
                            dtype=numpy.longdouble,
                        )
                        * numpy.asarray(
                            projected["transfers"][row["secondary_component"]],
                            dtype=numpy.longdouble,
                        )
                    ),
                )
                for row in matching
            )
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

    def test_independent_projection_resumes_real_declared_products(self):
        """Durable angular chunks reproduce a fresh declared calculation."""

        import tempfile

        from copernican.lib.likelihoods.cmb.durable_history import (
            retain_durable_histories,
        )
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
        self.addCleanup(execution.cache.clear_cmb_caches)
        with tempfile.TemporaryDirectory() as directory:
            identity = {"physical": "a" * 64, "numerical": "b" * 64}
            outputs = []
            for attempt in ("cold", "warm"):
                execution.cache.clear_cmb_caches()
                with retain_durable_histories(
                    directory, identity, "cases/" + attempt
                ):
                    outputs.append(
                        execution._compute_custom_cmb_spectrum_data(
                            contract,
                            (2, 10),
                            requested_spectra=("TT",),
                            workload="fixed_parameter_diagnostic",
                        )
                    )
            cold, warm = outputs
            self.assertGreater(
                cold.runtime_envelope["durable_projection_checkpoints"][
                    "writes"
                ],
                0,
            )
            self.assertGreater(
                warm.runtime_envelope["durable_projection_checkpoints"][
                    "hits"
                ],
                0,
            )
            numpy.testing.assert_array_equal(
                cold.spectra["TT"], warm.spectra["TT"]
            )
            for name in cold.transfer_components:
                numpy.testing.assert_array_equal(
                    cold.transfer_components[name],
                    warm.transfer_components[name],
                )

    def test_direct_projection_resumes_real_declared_products(self):
        """Direct projection reuses only matching per-wave transfers."""

        import tempfile

        from copernican.lib.likelihoods.cmb.durable_history import (
            retain_durable_histories,
        )
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(test_cmb._custom_contract())
        raw["numerical"].update(k_sample_count=4, k_max=0.03, ell_max=10)
        contract = test_cmb._prepare_declared_contract(raw)
        self.addCleanup(execution.cache.clear_cmb_caches)
        with tempfile.TemporaryDirectory() as directory:
            identity = {"physical": "a" * 64, "numerical": "b" * 64}
            outputs = []
            for attempt in ("cold", "warm"):
                execution.cache.clear_cmb_caches()
                with retain_durable_histories(
                    directory, identity, "cases/" + attempt
                ):
                    outputs.append(
                        execution._compute_custom_cmb_spectrum_data(
                            contract,
                            (2, 10),
                            requested_spectra=("TT",),
                            workload="joint_mcmc",
                        )
                    )
            cold, warm = outputs
            self.assertGreater(
                cold.runtime_envelope["durable_projection_checkpoints"][
                    "writes"
                ],
                0,
            )
            self.assertEqual(
                warm.runtime_envelope["durable_projection_checkpoints"][
                    "hits"
                ],
                cold.runtime_envelope["durable_projection_checkpoints"][
                    "writes"
                ],
            )
            numpy.testing.assert_array_equal(
                cold.spectra["TT"], warm.spectra["TT"]
            )
            for name in cold.transfer_components:
                numpy.testing.assert_array_equal(
                    cold.transfer_components[name],
                    warm.transfer_components[name],
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

    def test_native_pilot_checks_endpoint_before_admitting_remaining_modes(
        self,
    ):
        """A failing endpoint cannot trigger unrelated high-k evolution."""

        from copernican.lib.likelihoods.cmb.runtime import evidence
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
        build = execution.build_declared_mode_evolution
        estimate = execution.estimate_history_convergence
        admitted, context = [], {}

        def capture(inputs):
            runtime = build(inputs)
            context["waves"] = numpy.asarray(inputs.k_values)
            evolve = runtime.evolve_declared_modes_batched

            def admit(waves, **kwargs):
                if kwargs.get("evolution_sample_count_override") is None:
                    admitted.extend(float(wave) for wave in waves)
                return evolve(waves, **kwargs)

            return replace(runtime, evolve_declared_modes_batched=admit)

        def observe(kind, products):
            if kind == "native_refinement":
                context["validated_wave"] = float(products["k"])

        def compare(*args, **kwargs):
            result = estimate(*args, **kwargs)
            fails = context["validated_wave"] == float(
                context["waves"][context["failure_index"]]
            )
            return replace(
                result,
                converged=not fails,
                relative_error=0.5 if fails else 0.0,
            )

        self.addCleanup(execution.cache.clear_cmb_caches)
        for failure_index, expected in ((0, (2, 0)), (4, (2, 0, 4))):
            with self.subTest(failing_endpoint=failure_index):
                execution.cache.clear_cmb_caches()
                admitted.clear()
                context["failure_index"] = failure_index
                with (
                    evidence.capture_completed_cmb_products(observe),
                    mock.patch.object(
                        execution, "build_declared_mode_evolution", capture
                    ),
                    mock.patch.object(
                        execution,
                        "informative_k_indices",
                        return_value=(0, 1, 2, 3, 4),
                    ),
                    mock.patch.object(
                        execution, "estimate_history_convergence", compare
                    ),
                ):
                    with self.assertRaises(
                        execution.AdaptiveNonConvergenceError
                    ) as caught:
                        execution._compute_custom_cmb_spectrum_data_impl(
                            contract,
                            (2,),
                            requested_spectra=("TT",),
                            performance_timer=execution.PhaseTimer(),
                        )
                self.assertEqual(
                    caught.exception.evidence["history_kind"], "state"
                )
                self.assertEqual(
                    caught.exception.evidence["relative_normalization"],
                    "history_amplitude",
                )
                self.assertEqual(
                    caught.exception.evidence["relative_tolerance"], 0.02
                )
                self.assertEqual(
                    caught.exception.evidence["absolute_error_scope"],
                    "all_histories",
                )
                self.assertGreaterEqual(
                    caught.exception.evidence["worst_history_absolute_error"],
                    0.0,
                )
                self.assertGreaterEqual(
                    caught.exception.evidence[
                        "worst_history_normalization_scale"
                    ],
                    0.0,
                )
                self.assertEqual(
                    caught.exception.evidence["mode_k"],
                    float(context["waves"][failure_index]),
                )
                self.assertEqual(
                    admitted,
                    [float(context["waves"][index]) for index in expected],
                )

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
        """Compare projection chunks on one real physical background."""

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
        build_background = execution._build_custom_cmb_background
        backgrounds = []
        references = []
        live = []

        def retain_background(*args, **kwargs):
            background = build_background(*args, **kwargs)
            backgrounds.append(background)
            return background

        def observe(*args, **kwargs):
            batch = original(*args, **kwargs)
            references.append(weakref.ref(batch.j_l))
            live.append(sum(ref() is not None for ref in references))
            return batch

        execution.cache.clear_cmb_caches()
        with (
            mock.patch.object(execution, "_BESSEL_WORK_CELL_BUDGET", 10_000),
            mock.patch.object(
                execution,
                "_build_custom_cmb_background",
                side_effect=retain_background,
            ),
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
        self.assertEqual(len(backgrounds), 1)
        execution.cache.clear_cmb_caches()
        with (
            mock.patch.object(
                execution, "_BESSEL_WORK_CELL_BUDGET", 8_000_000
            ),
            mock.patch.object(
                execution,
                "_build_custom_cmb_background",
                return_value=backgrounds[0],
            ),
        ):
            large = execution._compute_custom_cmb_spectrum_data(
                contract,
                tuple(range(2, 101)),
                requested_spectra=("TT", "TE", "EE"),
            )
        self.assertGreater(
            small.runtime_envelope["evolution_modes_evolved"], 0
        )
        self.assertGreater(
            large.runtime_envelope["evolution_modes_evolved"], 0
        )
        self.assertGreater(
            large.runtime_envelope["projection_peak_live_modes"],
            small.runtime_envelope["projection_peak_live_modes"],
        )
        self.assertGreater(
            small.runtime_envelope["projection_bessel_batch_count"],
            large.runtime_envelope["projection_bessel_batch_count"],
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

    def test_split_modes_match_independent_kinetic_and_ppf_histories(self):
        """Real four-mode state and source histories survive split/rejoin."""

        from copernican.lib.likelihoods.cmb.runtime import mode_evolution
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract(
                include_massive_neutrino=True
            )
        )
        raw["perturbations"]["species"]["arbitrary_component"] = {
            "sector": "scalar",
            "equation_of_state": "-0.9",
            "anisotropic_stress": "zero",
            "scalar_closure": {
                "kind": "ppf",
                "density_fraction": "0.7",
                "other_enthalpy": "0.3",
                "transition": "0.4",
                "initial": "0.0",
            },
        }
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
        waves = (0.02, 0.021, 0.022, 0.023)
        group = mode_evolution.group_mode_schedules
        integrate = mode_evolution._integrate_batched_rk4
        self.addCleanup(execution.cache.clear_cmb_caches)

        def run(*, independently):
            """Capture native states with selected grouping."""

            execution.cache.clear_cmb_caches()
            pending, states = [], {}

            def select(rows, **kwargs):
                if independently:
                    kwargs["maximum_modes"] = 1
                groups = group(rows, **kwargs)
                pending.extend(tuple(item["k_values"]) for item in groups)
                return groups

            def capture(initial, eta, **kwargs):
                result = integrate(initial, eta, **kwargs)
                selected = pending.pop(0)
                for index, wave in enumerate(selected):
                    states[float(wave)] = (
                        numpy.array(eta),
                        numpy.array(result[0][index]),
                    )
                return result

            with (
                mock.patch.object(
                    mode_evolution, "group_mode_schedules", side_effect=select
                ),
                mock.patch.object(
                    mode_evolution,
                    "_integrate_batched_rk4",
                    side_effect=capture,
                ),
            ):
                result = execution._compute_custom_cmb_spectrum_data_impl(
                    contract,
                    (2,),
                    source_probe=waves,
                    requested_spectra=("TT", "TE", "EE"),
                    performance_timer=execution.PhaseTimer(),
                )
            self.assertFalse(pending)
            return result, states

        reference, separate = run(independently=True)
        batched, shared = run(independently=False)
        envelope = batched.runtime_envelope
        self.assertEqual(envelope["schedule_max_group_size"], 4)
        self.assertGreater(envelope["schedule_split_interval_count"], 0)
        self.assertTrue(envelope["ppf_batched_relaxation"])
        self.assertEqual(envelope["schedule_max_overintegration_ratio"], 1.0)
        for wave in waves:
            numpy.testing.assert_array_equal(
                shared[wave][0], separate[wave][0]
            )
            state_scale = numpy.maximum(
                numpy.max(numpy.abs(shared[wave][1]), axis=0),
                numpy.max(numpy.abs(separate[wave][1]), axis=0),
            )
            numpy.testing.assert_array_less(
                numpy.abs(shared[wave][1] - separate[wave][1]),
                numpy.broadcast_to(
                    1e-11 * state_scale + 1e-300, shared[wave][1].shape
                ),
            )
            for name in reference.sources[wave]:
                left = batched.sources[wave][name]
                right = reference.sources[wave][name]
                scale = max(
                    numpy.max(numpy.abs(left)), numpy.max(numpy.abs(right))
                )
                numpy.testing.assert_array_less(
                    numpy.abs(left - right), 1e-11 * scale + 1e-300
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

    def test_ordinary_scalar_ppf_reuses_completed_native_history(self):
        """Ordinary scalar work reuses an interrupted producer's BDF mode."""

        import tempfile

        from copernican.lib.likelihoods.cmb.durable_history import (
            retain_durable_histories,
        )
        from copernican.lib.likelihoods.cmb.runtime import mode_evolution
        from copernican.lib.likelihoods.cmb.runtime.history_store import (
            retain_source_histories,
        )
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract()
        )
        raw["perturbations"]["species"]["arbitrary_component"] = {
            "sector": "scalar",
            "equation_of_state": "-0.9",
            "anisotropic_stress": "zero",
            "scalar_closure": {
                "kind": "ppf",
                "density_fraction": "0.7",
                "other_enthalpy": "0.3",
                "transition": "0.4",
                "initial": "0.0",
            },
        }
        raw["numerical"].update(k_sample_count=4, k_min=0.001, k_max=0.02)
        contract = test_cmb._prepare_declared_contract(raw)
        identity = {"physical": "a" * 64, "numerical": "b" * 64}
        backgrounds = []
        original_background = execution._build_custom_cmb_background
        original_set = execution.cache.set_cmb_evolution_history

        def remember_background(*args, **kwargs):
            result = original_background(*args, **kwargs)
            backgrounds.append(result)
            return result

        def stop_after_native(key, eta, histories):
            original_set(key, eta, histories)
            if key[0] == "native_evolution":
                raise InterruptedError("continuous history saved")

        def solve(*, selected=False):
            return execution._compute_custom_cmb_spectrum_data_impl(
                contract,
                (2,),
                source_probe=(0.001,) if selected else None,
                requested_spectra=("TT",),
                performance_timer=execution.PhaseTimer(),
            )

        self.addCleanup(execution.cache.clear_cmb_caches)
        execution.cache.clear_cmb_caches()
        with tempfile.TemporaryDirectory() as directory:
            with (
                retain_durable_histories(directory, identity, "cases/first"),
                retain_source_histories(max_bytes=0),
                mock.patch.object(
                    mode_evolution,
                    "_can_batch_declared_evolution",
                    return_value=False,
                ),
                mock.patch.object(
                    execution,
                    "_build_custom_cmb_background",
                    side_effect=remember_background,
                ),
                mock.patch.object(
                    execution.cache,
                    "set_cmb_evolution_history",
                    side_effect=stop_after_native,
                ),
                self.assertRaisesRegex(InterruptedError, "history saved"),
            ):
                solve(selected=True)
            execution.cache.clear_cmb_caches()
            with (
                retain_durable_histories(directory, identity, "cases/resume"),
                retain_source_histories(max_bytes=0),
                mock.patch.object(
                    mode_evolution,
                    "_can_batch_declared_evolution",
                    return_value=False,
                ),
            ):
                resumed = solve()
            self.assertGreaterEqual(
                resumed.runtime_envelope["native_history_cache_hits"], 1
            )
            execution.cache.clear_cmb_caches()
            with (
                mock.patch.object(
                    mode_evolution,
                    "_can_batch_declared_evolution",
                    return_value=False,
                ),
                mock.patch.object(
                    execution,
                    "_build_custom_cmb_background",
                    return_value=backgrounds[0],
                ),
            ):
                cold = solve()
            numpy.testing.assert_array_equal(
                resumed.spectra["TT"], cold.spectra["TT"]
            )
            for name in cold.transfer_components:
                numpy.testing.assert_array_equal(
                    resumed.transfer_components[name],
                    cold.transfer_components[name],
                )

    def test_make_bdf_checkpoint_reuses_completed_continuous_history(self):
        """A completed BDF mode reuses its native history and sources."""

        import tempfile

        from copernican.lib.likelihoods.cmb.durable_history import (
            retain_durable_histories,
        )
        from copernican.lib.likelihoods.cmb.runtime import mode_evolution
        from copernican.lib.likelihoods.cmb.runtime.history_store import (
            retain_source_histories,
        )
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract()
        )
        raw["perturbations"]["species"]["arbitrary_component"] = {
            "sector": "scalar",
            "equation_of_state": "-0.9",
            "anisotropic_stress": "zero",
            "scalar_closure": {
                "kind": "ppf",
                "density_fraction": "0.7",
                "other_enthalpy": "0.3",
                "transition": "0.4",
                "initial": "0.0",
            },
        }
        raw["numerical"].update(k_sample_count=4, k_max=0.03)
        contract = test_cmb._prepare_declared_contract(raw)
        identity = {"physical": "a" * 64, "numerical": "b" * 64}
        backgrounds = []
        original_background = execution._build_custom_cmb_background
        original_set = execution.cache.set_cmb_evolution_history

        def remember_background(*args, **kwargs):
            result = original_background(*args, **kwargs)
            backgrounds.append(result)
            return result

        def stop_after_native(key, eta, histories):
            original_set(key, eta, histories)
            if key[0] == "native_evolution":
                raise InterruptedError("continuous history saved")

        def probe():
            return execution._compute_custom_cmb_spectrum_data_impl(
                contract,
                (2,),
                source_probe=(0.001,),
                requested_spectra=("TT",),
                performance_timer=execution.PhaseTimer(),
            )

        self.addCleanup(execution.cache.clear_cmb_caches)
        execution.cache.clear_cmb_caches()
        with tempfile.TemporaryDirectory() as directory:
            with (
                retain_durable_histories(directory, identity, "cases/first"),
                retain_source_histories(max_bytes=0),
                mock.patch.object(
                    mode_evolution,
                    "_can_batch_declared_evolution",
                    return_value=False,
                ),
                mock.patch.object(
                    execution,
                    "_build_custom_cmb_background",
                    side_effect=remember_background,
                ),
                mock.patch.object(
                    execution.cache,
                    "set_cmb_evolution_history",
                    side_effect=stop_after_native,
                ),
                mock.patch.object(execution, "RKCheckpoint") as checkpoint,
                mock.patch.object(
                    execution,
                    "BDFCheckpoint",
                    wraps=execution.BDFCheckpoint,
                ) as make_bdf_checkpoint,
                mock.patch.object(
                    mode_evolution,
                    "_integrate_scaled_continuous_history",
                    wraps=mode_evolution._integrate_scaled_continuous_history,
                ) as continuous,
                self.assertRaisesRegex(InterruptedError, "history saved"),
            ):
                probe()
            checkpoint.assert_not_called()
            self.assertEqual(make_bdf_checkpoint.call_count, 1)
            self.assertEqual(
                make_bdf_checkpoint.call_args.args[1][0], "continuous_bdf"
            )
            self.assertEqual(continuous.call_count, 1)
            execution.cache.clear_cmb_caches()
            with (
                retain_durable_histories(directory, identity, "cases/resume"),
                retain_source_histories(max_bytes=0),
                mock.patch.object(
                    mode_evolution,
                    "_can_batch_declared_evolution",
                    return_value=False,
                ),
                mock.patch.object(
                    mode_evolution,
                    "_integrate_scaled_continuous_history",
                    wraps=mode_evolution._integrate_scaled_continuous_history,
                ) as continuous,
            ):
                resumed = probe()
            self.assertEqual(continuous.call_count, 0)
            self.assertEqual(
                resumed.runtime_envelope["native_history_cache_hits"], 1
            )
            execution.cache.clear_cmb_caches()
            with (
                mock.patch.object(
                    mode_evolution,
                    "_can_batch_declared_evolution",
                    return_value=False,
                ),
                mock.patch.object(
                    execution,
                    "_build_custom_cmb_background",
                    return_value=backgrounds[0],
                ),
            ):
                cold = probe()
            for name, values in cold.sources[0.001].items():
                numpy.testing.assert_array_equal(
                    resumed.sources[0.001][name], values
                )

    def test_ppf_batch_checkpoint_resumes_with_massive_neutrinos(self):
        """A PPF relaxation history resumes without changing its sources."""

        import tempfile

        from copernican.lib.likelihoods.cmb.durable_history import (
            RKCheckpoint,
            retain_durable_histories,
        )
        from copernican.lib.likelihoods.cmb.runtime.history_store import (
            retain_source_histories,
        )
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract(
                include_massive_neutrino=True
            )
        )
        raw["perturbations"]["species"]["arbitrary_component"] = {
            "sector": "scalar",
            "equation_of_state": "-0.9",
            "anisotropic_stress": "zero",
            "scalar_closure": {
                "kind": "ppf",
                "density_fraction": "0.7",
                "other_enthalpy": "0.3",
                "transition": "0.4",
                "initial": "0.0",
            },
        }
        raw["numerical"].update(k_sample_count=4, k_max=0.03)
        contract = test_cmb._prepare_declared_contract(raw)
        identity = {"physical": "a" * 64, "numerical": "b" * 64}
        backgrounds = []
        original_background = execution._build_custom_cmb_background

        def remember_background(*args, **kwargs):
            result = original_background(*args, **kwargs)
            backgrounds.append(result)
            return result

        def interrupting_checkpoint(store, key, **kwargs):
            checkpoint = RKCheckpoint(store, key, interval_seconds=0, **kwargs)
            original_record = checkpoint.record

            def interrupt(histories, interval, stats):
                original_record(histories, interval, stats)
                if interval == 3:
                    raise InterruptedError("PPF checkpoint saved")

            checkpoint.record = interrupt
            return checkpoint

        def probe():
            return execution._compute_custom_cmb_spectrum_data_impl(
                contract,
                (2,),
                source_probe=(0.001, 0.003),
                requested_spectra=("TT",),
                performance_timer=execution.PhaseTimer(),
            )

        self.addCleanup(execution.cache.clear_cmb_caches)
        execution.cache.clear_cmb_caches()
        with tempfile.TemporaryDirectory() as directory:
            with (
                retain_durable_histories(directory, identity, "cases/first"),
                retain_source_histories(max_bytes=0),
                mock.patch.object(
                    execution,
                    "_build_custom_cmb_background",
                    side_effect=remember_background,
                ),
                mock.patch.object(
                    execution,
                    "RKCheckpoint",
                    side_effect=interrupting_checkpoint,
                ),
                self.assertRaisesRegex(InterruptedError, "PPF checkpoint"),
            ):
                probe()
            execution.cache.clear_cmb_caches()
            with (
                retain_durable_histories(directory, identity, "cases/resume"),
                retain_source_histories(max_bytes=0),
            ):
                resumed = probe()
            self.assertTrue(resumed.runtime_envelope["ppf_batched_relaxation"])
            self.assertEqual(
                resumed.runtime_envelope["rk_checkpoint_restored_intervals"],
                3,
            )
            execution.cache.clear_cmb_caches()
            with mock.patch.object(
                execution,
                "_build_custom_cmb_background",
                return_value=backgrounds[0],
            ):
                cold = probe()
            for wave in cold.sources:
                for name, values in cold.sources[wave].items():
                    numpy.testing.assert_array_equal(
                        resumed.sources[wave][name], values
                    )

    def test_scalar_split_checkpoint_resumes_massive_neutrino_mode(self):
        """A real scalar split history resumes with its coupling state."""

        import inspect
        import tempfile

        from copernican.lib.likelihoods.cmb.durable_history import (
            RKCheckpoint,
            retain_durable_histories,
        )
        from copernican.lib.likelihoods.cmb.runtime import mode_evolution
        from copernican.lib.likelihoods.cmb.runtime.history_store import (
            retain_source_histories,
        )
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract(
                include_massive_neutrino=True
            )
        )
        raw["numerical"].update(k_sample_count=4, k_max=0.03)
        contract = test_cmb._prepare_declared_contract(raw)
        identity = {"physical": "a" * 64, "numerical": "b" * 64}
        backgrounds, native, saved, callers = [], [], [], []
        original_background = execution._build_custom_cmb_background
        original_build = execution.build_declared_mode_evolution

        def remember_background(*args, **kwargs):
            result = original_background(*args, **kwargs)
            backgrounds.append(result)
            return result

        def capture_mode(inputs):
            runtime = original_build(inputs)

            def evolve(wave, **kwargs):
                sink = kwargs.setdefault("history_sink", {})
                result = runtime.evolve_declared_mode(wave, **kwargs)
                native.append(sink)
                return result

            return replace(runtime, evolve_declared_mode=evolve)

        def interrupting_checkpoint(store, key, **kwargs):
            callers.extend(frame.function for frame in inspect.stack())
            checkpoint = RKCheckpoint(store, key, interval_seconds=0, **kwargs)
            original_record = checkpoint.record

            def interrupt(histories, interval, stats):
                original_record(histories, interval, stats)
                if interval == 3:
                    saved.append(dict(stats))
                    raise InterruptedError("scalar checkpoint saved")

            checkpoint.record = interrupt
            return checkpoint

        def probe():
            return execution._compute_custom_cmb_spectrum_data_impl(
                contract,
                (2,),
                source_probe=(0.001,),
                requested_spectra=("TT",),
                performance_timer=execution.PhaseTimer(),
            )

        self.addCleanup(execution.cache.clear_cmb_caches)
        execution.cache.clear_cmb_caches()
        with tempfile.TemporaryDirectory() as directory:
            with (
                retain_durable_histories(directory, identity, "cases/first"),
                retain_source_histories(max_bytes=0),
                mock.patch.object(
                    mode_evolution,
                    "_can_batch_declared_evolution",
                    return_value=False,
                ),
                mock.patch.object(
                    execution,
                    "_build_custom_cmb_background",
                    side_effect=remember_background,
                ),
                mock.patch.object(
                    execution, "build_declared_mode_evolution", capture_mode
                ),
                mock.patch.object(
                    execution,
                    "RKCheckpoint",
                    side_effect=interrupting_checkpoint,
                ),
                self.assertRaisesRegex(InterruptedError, "checkpoint saved"),
            ):
                probe()
            self.assertEqual(len(saved), 1)
            self.assertIn("make_scalar_checkpoint", callers)
            self.assertEqual(saved[0]["tight_coupling_active"], 1)
            execution.cache.clear_cmb_caches()
            with (
                retain_durable_histories(directory, identity, "cases/resume"),
                retain_source_histories(max_bytes=0),
                mock.patch.object(
                    mode_evolution,
                    "_can_batch_declared_evolution",
                    return_value=False,
                ),
                mock.patch.object(
                    execution, "build_declared_mode_evolution", capture_mode
                ),
            ):
                resumed = probe()
            self.assertEqual(
                resumed.runtime_envelope[
                    "scalar_checkpoint_restored_intervals"
                ],
                3,
            )
            self.assertGreater(
                resumed.runtime_envelope[
                    "scalar_checkpoint_restored_rhs_calls"
                ],
                0,
            )
            execution.cache.clear_cmb_caches()
            with (
                mock.patch.object(
                    mode_evolution,
                    "_can_batch_declared_evolution",
                    return_value=False,
                ),
                mock.patch.object(
                    execution,
                    "_build_custom_cmb_background",
                    return_value=backgrounds[0],
                ),
                mock.patch.object(
                    execution, "build_declared_mode_evolution", capture_mode
                ),
            ):
                cold = probe()
            for name, values in cold.sources[0.001].items():
                numpy.testing.assert_array_equal(
                    resumed.sources[0.001][name], values
                )
            self.assertEqual(len(native), 2)
            numpy.testing.assert_array_equal(
                native[0]["evolution_eta"], native[1]["evolution_eta"]
            )
            for name, values in native[1]["evolution_histories"].items():
                numpy.testing.assert_array_equal(
                    native[0]["evolution_histories"][name], values
                )

    def test_interrupted_native_rk_resumes_the_same_physical_modes(self):
        """A real hierarchy probe resumes mid-mode without changing sources."""

        import inspect
        import tempfile

        from copernican.lib.likelihoods.cmb.durable_history import (
            RKCheckpoint,
            retain_durable_histories,
        )
        from copernican.lib.likelihoods.cmb.runtime.history_store import (
            retain_source_histories,
        )
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract(
                include_massive_neutrino=True
            )
        )
        raw["numerical"].update(k_sample_count=4, k_max=0.03)
        contract = test_cmb._prepare_declared_contract(raw)
        waves = (0.001, 0.03)
        identity = {"physical": "a" * 64, "numerical": "b" * 64}
        callbacks, backgrounds, restored = [], [], []
        original_background = execution._build_custom_cmb_background
        original_restore = RKCheckpoint.restore

        def remember_background(*args, **kwargs):
            result = original_background(*args, **kwargs)
            backgrounds.append(result)
            return result

        def interrupting_checkpoint(store, key):
            callbacks.extend(frame.function for frame in inspect.stack())
            checkpoint = RKCheckpoint(store, key, interval_seconds=0)
            original_record = checkpoint.record

            def interrupt(histories, interval, stats):
                original_record(histories, interval, stats)
                if interval == 3:
                    callbacks.extend(
                        frame.function for frame in inspect.stack()
                    )
                    raise InterruptedError("native RK checkpoint saved")

            checkpoint.record = interrupt
            return checkpoint

        def observe_restore(checkpoint, histories):
            result = original_restore(checkpoint, histories)
            if result is not None:
                restored.append(result[0])
            return result

        def probe():
            return execution._compute_custom_cmb_spectrum_data_impl(
                contract,
                (2,),
                source_probe=waves,
                requested_spectra=("TT",),
                performance_timer=execution.PhaseTimer(),
            )

        self.addCleanup(execution.cache.clear_cmb_caches)
        execution.cache.clear_cmb_caches()
        with tempfile.TemporaryDirectory() as directory:
            with (
                retain_durable_histories(directory, identity, "cases/first"),
                retain_source_histories(max_bytes=0),
                mock.patch.object(
                    execution,
                    "_build_custom_cmb_background",
                    side_effect=remember_background,
                ),
                mock.patch.object(
                    execution,
                    "RKCheckpoint",
                    side_effect=interrupting_checkpoint,
                ),
                self.assertRaisesRegex(InterruptedError, "checkpoint saved"),
            ):
                probe()
            self.assertIn("make_rk_checkpoint", callbacks)
            self.assertIn("retain_interval", callbacks)
            execution.cache.clear_cmb_caches()
            with (
                retain_durable_histories(directory, identity, "cases/resume"),
                retain_source_histories(max_bytes=0),
                mock.patch.object(RKCheckpoint, "restore", observe_restore),
            ):
                resumed = probe()
            self.assertIn(3, restored)
            self.assertEqual(
                resumed.runtime_envelope["rk_checkpoint_restored_intervals"],
                3,
            )
            self.assertGreater(
                resumed.runtime_envelope["rk_checkpoint_restored_rk_stages"],
                0,
            )
            execution.cache.clear_cmb_caches()
            with mock.patch.object(
                execution,
                "_build_custom_cmb_background",
                return_value=backgrounds[0],
            ):
                cold = probe()
            for wave in waves:
                for name, values in cold.sources[wave].items():
                    numpy.testing.assert_array_equal(
                        resumed.sources[wave][name], values
                    )

    def test_interrupted_primary_batch_retains_completed_sources(self):
        """A stopped ordinary batch resumes its independently saved mode."""

        import inspect
        import tempfile

        from copernican.lib.likelihoods.cmb.durable_history import (
            retain_durable_histories,
        )
        from copernican.lib.likelihoods.cmb.runtime import mode_evolution
        from copernican.lib.likelihoods.cmb.runtime.history_store import (
            retain_source_histories,
        )
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract(
                include_massive_neutrino=True
            )
        )
        raw["numerical"].update(k_sample_count=4, k_max=0.03)
        contract = test_cmb._prepare_declared_contract(raw)
        identity = {"physical": "a" * 64, "numerical": "b" * 64}
        saved_waves = []
        callback_names = []
        backgrounds = []
        original_set = execution.cache.set_cmb_source_history
        original_background = execution._build_custom_cmb_background

        def remember_background(*args, **kwargs):
            result = original_background(*args, **kwargs)
            backgrounds.append(result)
            return result

        def stop_after_saved(key, sources):
            original_set(key, sources)
            callers = {frame.function for frame in inspect.stack()}
            if "retain_completed_primary_mode" in callers:
                callback_names.extend(callers)
                saved_waves.append(float(key[-1]))
                raise InterruptedError("completed primary mode saved")

        def solve():
            return execution._compute_custom_cmb_spectrum_data_impl(
                contract,
                (2,),
                requested_spectra=("TT",),
                performance_timer=execution.PhaseTimer(),
            )

        self.addCleanup(execution.cache.clear_cmb_caches)
        execution.cache.clear_cmb_caches()
        with tempfile.TemporaryDirectory() as directory:
            with (
                retain_durable_histories(directory, identity, "cases/first"),
                retain_source_histories(max_bytes=0),
                mock.patch.object(
                    execution,
                    "_build_custom_cmb_background",
                    side_effect=remember_background,
                ),
                mock.patch.object(
                    execution.cache,
                    "set_cmb_source_history",
                    side_effect=stop_after_saved,
                ),
                mock.patch.object(
                    mode_evolution,
                    "_integrate_batched_rk4",
                    wraps=mode_evolution._integrate_batched_rk4,
                ) as integrate,
                self.assertRaisesRegex(InterruptedError, "primary mode"),
            ):
                solve()
            self.assertEqual(len(saved_waves), 1)
            self.assertIn("retain_completed_primary_mode", callback_names)
            self.assertEqual(integrate.call_count, 1)
            execution.cache.clear_cmb_caches()
            with (
                retain_durable_histories(directory, identity, "cases/resume"),
                retain_source_histories(max_bytes=0),
            ):
                resumed = solve()
            self.assertGreaterEqual(
                resumed.runtime_envelope["source_history_cache_hit_count"], 1
            )
            self.assertIn(saved_waves[0], resumed.k_grid)
            execution.cache.clear_cmb_caches()
            with mock.patch.object(
                execution,
                "_build_custom_cmb_background",
                return_value=backgrounds[0],
            ):
                cold = solve()
            numpy.testing.assert_array_equal(
                resumed.spectra["TT"], cold.spectra["TT"]
            )
            for name in cold.transfer_components:
                numpy.testing.assert_array_equal(
                    resumed.transfer_components[name],
                    cold.transfer_components[name],
                )

    def test_interrupted_source_probe_retains_each_completed_mode(self):
        """A later physical probe resumes without rerunning completed modes."""

        import inspect
        import tempfile

        from copernican.lib.likelihoods.cmb.durable_history import (
            retain_durable_histories,
        )
        from copernican.lib.likelihoods.cmb.runtime import mode_evolution
        from copernican.lib.likelihoods.cmb.runtime.history_store import (
            retain_source_histories,
        )
        from tests.copernican.lib.likelihoods.cmb import test_cmb

        raw = test_cmb._speedup_contract(
            test_cmb._declared_scalar_hierarchy_contract(
                include_massive_neutrino=True
            )
        )
        raw["numerical"].update(k_sample_count=4, k_max=0.03)
        contract = test_cmb._prepare_declared_contract(raw)
        waves = (0.001, 0.03)
        identity = {"physical": "a" * 64, "numerical": "b" * 64}
        saved_waves = []
        callback_names = []
        saved_backgrounds = []
        original_set = execution.cache.set_cmb_source_history
        original_background = execution._build_custom_cmb_background

        def remember_background(*args, **kwargs):
            result = original_background(*args, **kwargs)
            saved_backgrounds.append(result)
            return result

        def stop_after_saved(key, sources):
            original_set(key, sources)
            saved_waves.append(float(key[-1]))
            callback_names.extend(frame.function for frame in inspect.stack())
            raise InterruptedError("stopped after first saved source")

        def probe():
            return execution._compute_custom_cmb_spectrum_data_impl(
                contract,
                (2,),
                source_probe=waves,
                requested_spectra=("TT",),
                performance_timer=execution.PhaseTimer(),
            )

        self.addCleanup(execution.cache.clear_cmb_caches)
        execution.cache.clear_cmb_caches()
        with tempfile.TemporaryDirectory() as directory:
            with (
                retain_durable_histories(directory, identity, "cases/first"),
                retain_source_histories(max_bytes=0),
                mock.patch.object(
                    execution,
                    "_build_custom_cmb_background",
                    side_effect=remember_background,
                ),
                mock.patch.object(
                    execution.cache,
                    "set_cmb_source_history",
                    side_effect=stop_after_saved,
                ),
                mock.patch.object(
                    mode_evolution,
                    "_integrate_batched_rk4",
                    wraps=mode_evolution._integrate_batched_rk4,
                ) as integrate,
                self.assertRaisesRegex(InterruptedError, "first saved source"),
            ):
                probe()
            self.assertEqual(integrate.call_count, 1)
            self.assertIn("retain_completed_native_mode", callback_names)
            self.assertEqual(len(saved_waves), 1)
            self.assertIn(saved_waves[0], waves)
            self.assertEqual(len(saved_backgrounds), 1)
            execution.cache.clear_cmb_caches()
            with (
                retain_durable_histories(directory, identity, "cases/resume"),
                retain_source_histories(max_bytes=0),
            ):
                resumed = probe()
            self.assertEqual(resumed.runtime_envelope["probe_new_modes"], 1)
            self.assertEqual(
                resumed.runtime_envelope["durable_source_modes_checkpointed"],
                1,
            )
            execution.cache.clear_cmb_caches()
            with mock.patch.object(
                execution,
                "_build_custom_cmb_background",
                return_value=saved_backgrounds[0],
            ):
                cold = probe()
            for wave in waves:
                for name, values in cold.sources[wave].items():
                    numpy.testing.assert_array_equal(
                        resumed.sources[wave][name], values
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

        from copernican.lib.likelihoods.cmb.runtime import evidence

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
                failed_products=("TT",),
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
        observed = []
        with (
            evidence.capture_completed_cmb_products(
                lambda kind, products: observed.append((kind, products))
            ),
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
        failures_observed = [
            products
            for kind, products in observed
            if kind == "numerical_refinement_failure"
        ]
        self.assertEqual(len(failures_observed), 2)
        self.assertEqual(failures_observed[1]["axis"], "evolution")
        self.assertEqual(failures_observed[1]["failed_products"], ("TT",))
        self.assertEqual(
            failures_observed[1]["metric"]["kind"], "observable_budget_ratio"
        )
        self.assertAlmostEqual(failures_observed[1]["metric"]["value"], 1.11)
        self.assertAlmostEqual(
            failures_observed[1]["evidence"]["products"]["TT"][
                "relative_error_bound"
            ],
            0.0111,
        )
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

    def test_durable_finer_grid_requires_verified_independent_histories(self):
        """Raw finer modes are reusable only with their exact wave identity."""

        import tempfile

        from copernican.lib.likelihoods.cmb.durable_history import (
            retain_durable_histories,
        )

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
        token = None

        def fake_impl(request, *args, **kwargs):
            refined = bool(request.get("_numerical_overrides"))
            return execution.CustomCMBSpectrumData(
                ell_grid=numpy.array([2, 3]),
                k_grid=numpy.array(
                    [0.1, 0.15, 0.2] if refined else [0.1, 0.2]
                ),
                transfer_components={},
                spectra={
                    "TT": numpy.array([1.0, 2.0]),
                    "TE": numpy.array([0.1, 0.2]),
                    "EE": numpy.array([0.5, 0.8]),
                },
                runtime_envelope={
                    "native_source_k_grid": (
                        [0.1, 0.15, 0.2] if refined else [0.1, 0.2]
                    ),
                    "durable_source_waves": (
                        [{"k": 0.15, "key_sha256": token}] if refined else []
                    ),
                },
            )

        self.addCleanup(execution.cache.clear_cmb_caches)
        with tempfile.TemporaryDirectory() as directory:
            with retain_durable_histories(
                directory,
                {"physical": "a" * 64, "numerical": "b" * 64},
                "cases/producer",
            ) as durable:
                key = ("source", 0.15)
                durable.set(key, {"source": numpy.array([1.0, 2.0])})
                token = durable.verified_history_token(key)
                with (
                    mock.patch.object(
                        execution,
                        "_compute_custom_cmb_spectrum_data_impl",
                        side_effect=fake_impl,
                    ),
                    mock.patch.object(execution.cache, "set_cmb_spectrum"),
                ):
                    result = execution._compute_custom_cmb_spectrum_data(
                        contract, (2, 3), requested_spectra=("TT", "TE", "EE")
                    )
                    record = result.runtime_envelope[
                        "production_scalar_k_convergence"
                    ]
                    self.assertTrue(record["warm_reuse"])
                    self.assertEqual(
                        record["durable_independent_source_waves"], (0.15,)
                    )
                    self.assertFalse(
                        record["matching_accepted_finer_calculation"]
                    )
                    token = "invalid"
                    with self.assertRaisesRegex(
                        ValueError, "no measured new work"
                    ):
                        execution._compute_custom_cmb_spectrum_data(
                            contract,
                            (2, 3),
                            requested_spectra=("TT", "TE", "EE"),
                        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
