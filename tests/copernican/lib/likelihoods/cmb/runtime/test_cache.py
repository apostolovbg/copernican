"""Focused tests for the declared CMB cache module."""

import unittest
from pathlib import Path
from unittest import mock

import numpy

from copernican.lib.cmb_identity import CCMBS_ID
from copernican.lib.likelihoods.cmb.runtime import cache


class CacheModuleTestCase(unittest.TestCase):
    """Exercise declared cache helpers directly."""

    def test_resolution_hints_are_isolated_controls_and_parameter_scoped(self):
        cache.clear_cmb_caches()
        self.addCleanup(cache.clear_cmb_caches)
        self.assertFalse(cache.has_cmb_spectrum("point"))
        self.assertIsNone(cache.get_cmb_resolution_hint("point"))
        controls = {"_numerical_overrides": {"eta_sample_count": 768}}
        cache.set_cmb_resolution_hint("point", controls)
        controls["_numerical_overrides"]["eta_sample_count"] = 1
        self.assertEqual(
            cache.get_cmb_resolution_hint("point")["_numerical_overrides"][
                "eta_sample_count"
            ],
            768,
        )
        self.assertIsNone(
            cache.get_cmb_resolution_hint("different-parameters")
        )
        with self.assertRaises(ValueError):
            cache.set_cmb_resolution_hint("point", {"certificate": True})
        cache.clear_cmb_result_caches()
        self.assertIsNotNone(cache.get_cmb_resolution_hint("point"))
        cache.clear_cmb_parameter_caches()
        self.assertIsNone(cache.get_cmb_resolution_hint("point"))

    def test_selected_native_histories_survive_spill_without_certificates(
        self,
    ):
        """The same RK arrays survive refinement pressure and leave at exit."""

        from copernican.lib.likelihoods.cmb.runtime.history_store import (
            retain_source_histories,
        )

        eta = numpy.linspace(0.0, 1.0, 5)
        state = numpy.arange(5, dtype=float)
        self.assertIsNone(cache.get_cmb_evolution_history("absent"))
        cache.set_cmb_evolution_history("outside", eta, {"eta": state})
        with retain_source_histories(max_bytes=0) as store:
            cache.set_cmb_evolution_history("mode", eta, {"eta": state})
            cache.set_cmb_source_history("pressure", {"source": state})
            epoch, histories = cache.get_cmb_evolution_history("mode")
            numpy.testing.assert_array_equal(epoch, eta)
            numpy.testing.assert_array_equal(histories["eta"], state)
            self.assertFalse(histories["eta"].flags.writeable)
            self.assertIsNone(cache.get_cmb_evolution_history("changed_mode"))
            self.assertGreater(store.snapshot()["loads"], 0)
            self.assertEqual(set(histories), {"eta"})
        self.assertIsNone(cache.get_cmb_evolution_history("mode"))

    def test_performance_record_retains_scoped_work_separately_from_times(
        self,
    ):
        """Nested work records stay serializable without becoming seconds."""

        self.addCleanup(cache.clear_cmb_caches)
        cache.clear_cmb_caches()
        scopes = {"evolution:rk": {"completed": 10, "total": 20}}
        record = cache.record_cmb_performance(
            {
                "evolution_seconds": 3.0,
                "request_elapsed_seconds": 5.0,
                "work_scopes": scopes,
            }
        )
        self.assertEqual(record["phase_seconds"], {"evolution_seconds": 3.0})
        self.assertEqual(record["heartbeat"]["work_scopes"], scopes)
        self.assertEqual(record["heartbeat"]["request_elapsed_seconds"], 5.0)

    def test_entry_limited_graph_cache_does_not_walk_payloads(self):
        """Entry eviction must not inspect large compiled graph payloads."""

        cache_store = cache._BoundedCacheStore(limit=1)
        with mock.patch.object(
            cache,
            "_numpy_payload_bytes",
            side_effect=AssertionError("entry-only cache walked a graph"),
        ):
            first = object()
            second = object()
            cache_store.set("first", first)
            self.assertIs(cache_store.get("first"), first)
            cache_store.set("first", second)
            cache_store.set("second", first)
            self.assertIsNone(cache_store.get("first"))
            self.assertIs(cache_store.get("second"), first)
            cache_store.prune(lambda key: key == "second")
            self.assertEqual(cache_store.snapshot()["entries"], 0)

    def test_byte_budget_survives_replacement_eviction_and_pruning(self):
        """Measured array bytes remain bounded after every store operation."""

        cache_store = cache._BoundedCacheStore(limit=10, max_bytes=32)
        small = numpy.ones(2, dtype=numpy.float64)
        large = numpy.ones(4, dtype=numpy.float64)
        cache_store.set("first", small)
        cache_store.set("first", large)
        self.assertEqual(cache_store._bytes, 32)
        cache_store.set("second", small)
        self.assertIsNone(cache_store.get("first"))
        self.assertEqual(cache_store._bytes, 16)
        cache_store.prune(lambda key: key == "second")
        self.assertEqual(cache_store._bytes, 0)
        cache_store.set("oversized", numpy.ones(5))
        self.assertIsNone(cache_store.get("oversized"))
        cache_store.set("last", large)
        cache_store.clear()
        self.assertEqual(cache_store._bytes, 0)
        self.assertEqual(cache_store._entry_bytes, {})

    def test_request_refinement_survives_global_history_eviction(self):
        """Nested solves retain exact raw work without enlarging global LRU."""

        self.addCleanup(cache.clear_cmb_caches)
        cache.clear_cmb_caches()

        def refined_request():
            for mode in range(4):
                cache.set_cmb_source_history(
                    ("physical_graph", mode),
                    {"source": numpy.arange(8, dtype=float) + mode},
                )
            for mode in range(4):
                product = cache.get_cmb_source_history(
                    ("physical_graph", mode)
                )
                numpy.testing.assert_array_equal(
                    product["source"], numpy.arange(8, dtype=float) + mode
                )
            self.assertIsNone(
                cache.get_cmb_source_history(("changed_physical_graph", 0))
            )
            accounting = cache.cmb_source_history_scope_stats()
            self.assertEqual(accounting["entries"], 4)
            self.assertGreater(accounting["spills"], 0)
            self.assertGreater(
                cache.cmb_cache_stats()["source_history"]["evictions"], 0
            )
            cache.clear_cmb_parameter_caches()
            self.assertIsNone(
                cache.get_cmb_source_history(("physical_graph", 0))
            )
            return accounting

        with (
            mock.patch.object(cache._CMB_SOURCE_HISTORY_CACHE, "limit", 1),
            mock.patch.object(
                cache._CMB_SOURCE_HISTORY_CACHE, "max_bytes", 64
            ),
        ):
            evaluate = cache.retain_cmb_source_histories(refined_request)
            self.assertEqual(evaluate()["entries"], 4)
        self.assertEqual(cache.cmb_source_history_scope_stats(), {})

    def test_bounded_cache_store_tracks_hits_misses_and_evictions(self):
        """The internal bounded cache should expose explicit accounting."""

        cache_store = cache._BoundedCacheStore(limit=1)
        self.assertIsNone(cache_store.get("missing"))

        cache_store.set("alpha", 1)
        self.assertEqual(cache_store.get("alpha"), 1)
        self.assertEqual(cache_store.snapshot()["entries"], 1)

        cache_store.set("beta", 2)
        self.assertEqual(cache_store.snapshot()["evictions"], 1)
        cache_store.prune(lambda key: key == "beta")
        self.assertEqual(cache_store.snapshot()["entries"], 0)

        cache_store.clear()
        snapshot = cache_store.snapshot()
        self.assertEqual(snapshot["entries"], 0)
        self.assertEqual(snapshot["hits"], 0)
        self.assertEqual(snapshot["misses"], 0)

    def test_cache_module_functions_round_trip_values(self):
        """The module cache helpers should store every cache family."""

        cache.clear_cmb_caches()
        self.assertIsNone(cache.get_cmb_initial_state("direct-missing"))
        self.assertIsNone(
            cache.set_cmb_initial_state("direct-initial", ("state",))
        )
        self.assertEqual(
            cache.get_cmb_initial_state("direct-initial"),
            ("state",),
        )
        self.assertIsNone(cache.get_cmb_hierarchy_schedule("direct-missing"))
        self.assertIsNone(
            cache.set_cmb_hierarchy_schedule("direct-schedule", (1, 2, 4))
        )
        self.assertEqual(
            cache.get_cmb_hierarchy_schedule("direct-schedule"),
            (1, 2, 4),
        )
        self.assertIsNone(cache.get_declared_symbol_plan("plan"))
        cache.set_declared_symbol_plan("plan", {"step": 1})
        self.assertEqual(
            cache.get_declared_symbol_plan("plan"),
            {"step": 1},
        )

        self.assertIsNone(cache.get_declared_graph_execution_plan("graph"))
        cache.set_declared_graph_execution_plan("graph", {"node": 1})
        self.assertEqual(
            cache.get_declared_graph_execution_plan("graph"),
            {"node": 1},
        )

        self.assertIsNone(cache.get_runtime_assets("runtime"))
        cache.set_runtime_assets("runtime", {"asset": 1})
        self.assertEqual(
            cache.get_runtime_assets("runtime"),
            {"asset": 1},
        )

        self.assertIsNone(cache.get_declared_momentum_topology("topology"))
        cache.set_declared_momentum_topology(
            "topology",
            {"nodes": 2},
        )
        self.assertEqual(
            cache.get_declared_momentum_topology("topology"),
            {"nodes": 2},
        )

        self.assertIsNone(cache.get_declared_momentum_grid("momentum"))
        cache.set_declared_momentum_grid(
            "momentum",
            {"grid": 1},
        )
        self.assertEqual(
            cache.get_declared_momentum_grid("momentum"),
            {"grid": 1},
        )

        self.assertIsNone(cache.get_cmb_background("bg"))
        cache.set_cmb_background("bg", {"a": 1.0})
        self.assertEqual(
            cache.get_cmb_background("bg"),
            {"a": 1.0},
        )

        self.assertIsNone(
            cache.get_reionization_calibration_seed("reionization")
        )
        cache.set_reionization_calibration_seed(
            "reionization",
            2.5,
        )
        self.assertEqual(
            cache.get_reionization_calibration_seed("reionization"),
            2.5,
        )

        self.assertIsNone(cache.get_cmb_spectrum("spec"))
        cache.set_cmb_spectrum("spec", {"tt": 2.0})
        self.assertEqual(
            cache.get_cmb_spectrum("spec"),
            {"tt": 2.0},
        )

        self.assertIsNone(cache.get_cmb_lensing("lensing"))
        cache.set_cmb_lensing("lensing", {"lensed_TT": numpy.ones(2)})
        self.assertTrue(
            numpy.array_equal(
                cache.get_cmb_lensing("lensing")["lensed_TT"],
                numpy.ones(2),
            )
        )

        self.assertIsNone(cache.get_cmb_transfer("transfer"))
        cache.set_cmb_transfer(
            "transfer",
            {"temperature": numpy.ones((2, 2))},
        )
        self.assertTrue(
            numpy.array_equal(
                cache.get_cmb_transfer("transfer")["temperature"],
                numpy.ones((2, 2)),
            )
        )

        x_values = numpy.array([0.1, 0.2])
        cache.store_bessel_inputs("xsig", x_values)
        self.assertTrue(
            numpy.array_equal(
                cache.get_bessel_inputs("xsig"),
                x_values,
            )
        )
        self.assertIsNone(cache.get_bessel_values((2, "xsig")))
        cache.set_bessel_values((2, "xsig"), ("j", "jp"))
        self.assertEqual(
            cache.get_bessel_values((2, "xsig")),
            ("j", "jp"),
        )

        kernel_key = ((2, 3), "xsig")
        self.assertIsNone(
            cache.get_declared_projection_kernel_batch(kernel_key)
        )
        cache.set_declared_projection_kernel_batch(
            kernel_key,
            "batch",
        )
        self.assertTrue(hasattr(cache, "set_declared_projection_kernel_batch"))
        self.assertEqual(
            cache.get_declared_projection_kernel_batch(kernel_key),
            "batch",
        )

        cache.record_cmb_performance(
            {"projection_seconds": 0.25},
            cache_hit=True,
        )
        cache.record_cmb_phase("lensing", 0.5)
        self.assertTrue(callable(cache.record_cmb_performance))
        self.assertTrue(callable(cache.record_cmb_phase))

        stats = cache.cmb_cache_stats()
        self.assertIn("background", stats)
        self.assertIn("declared_spectrum", stats)
        self.assertIn("declared_transfer", stats)
        self.assertIn("source_history", stats)
        self.assertIn("runtime_assets", stats)
        self.assertIn("declared_momentum_topology", stats)
        self.assertNotIn("custom_background", stats)
        self.assertNotIn("custom_spectrum", stats)
        self.assertEqual(stats["declared_symbol_plan"]["entries"], 1)
        self.assertEqual(
            stats["declared_graph_execution_plan"]["entries"],
            1,
        )
        self.assertEqual(
            stats["declared_momentum_grid"]["entries"],
            1,
        )
        performance = cache.cmb_performance_stats()
        self.assertEqual(int(performance["requests"]), 1)
        self.assertEqual(int(performance["cache_hits"]), 1)
        self.assertEqual(
            performance["phase_seconds"]["projection_seconds"],
            0.25,
        )
        self.assertEqual(performance["phase_seconds"]["lensing"], 0.5)

        cache.clear_cmb_caches()
        cleared_stats = cache.cmb_cache_stats()
        self.assertEqual(cleared_stats["declared_symbol_plan"]["entries"], 0)
        self.assertEqual(
            cleared_stats["declared_graph_execution_plan"]["entries"],
            0,
        )
        self.assertEqual(
            cleared_stats["runtime_assets"]["entries"],
            0,
        )
        self.assertEqual(
            cleared_stats["declared_momentum_grid"]["entries"],
            0,
        )

    def test_cache_inventory_and_parameter_invalidation_are_explicit(self):
        """Parameter invalidation must retain process-local structure."""

        cache.clear_cmb_caches()
        cache.set_runtime_assets("runtime", "assets")
        cache.set_declared_momentum_topology("topology", "nodes")
        cache.set_declared_momentum_grid("grid", "bound-grid")
        cache.set_cmb_background("background", "tables")
        source_key = ("contract", "model", "solver", "grid", 0.1)
        cache.set_cmb_source_history(
            source_key,
            {"temperature": numpy.asarray([1.0, 2.0])},
        )
        cache.set_cmb_initial_state(
            ("initial", 0.1),
            (numpy.asarray([1.0]), (), {}),
        )
        cache.set_cmb_hierarchy_schedule(
            ("schedule", 0.1),
            numpy.asarray([1, 2, 4], dtype=int),
        )
        self.assertTrue(callable(cache.set_cmb_source_history))
        self.assertTrue(
            numpy.array_equal(
                cache.get_cmb_source_history(source_key)["temperature"],
                numpy.asarray([1.0, 2.0]),
            )
        )
        cache.set_cmb_spectrum("spectrum", "result")

        inventory = cache.cmb_cache_inventory()
        self.assertEqual(
            inventory["runtime_assets"]["category"],
            "structural",
        )
        self.assertEqual(
            inventory["declared_momentum_grid"]["category"],
            "parameter",
        )
        self.assertEqual(
            inventory["declared_spectrum"]["category"],
            "result",
        )
        self.assertGreater(int(inventory["runtime_assets"]["owner_pid"]), 0)

        cache.clear_cmb_parameter_caches()
        stats = cache.cmb_cache_stats()
        self.assertEqual(stats["runtime_assets"]["entries"], 1)
        self.assertEqual(stats["declared_momentum_topology"]["entries"], 1)
        self.assertEqual(stats["declared_momentum_grid"]["entries"], 0)
        self.assertEqual(stats["background"]["entries"], 0)
        self.assertEqual(stats["declared_spectrum"]["entries"], 0)
        self.assertEqual(stats["source_history"]["entries"], 0)
        self.assertEqual(stats["initial_state"]["entries"], 0)
        self.assertEqual(stats["hierarchy_schedule"]["entries"], 0)

    def test_result_and_request_accounting_helpers_are_explicit(self):
        """Result invalidation and request mutation should stay observable."""

        cache.clear_cmb_caches()
        cache.set_cmb_spectrum("spectrum", "result")
        cache.clear_cmb_result_caches()
        self.assertIsNone(cache.get_cmb_spectrum("spectrum"))

        cache.record_cmb_performance(
            {"projection_seconds": 0.25},
            workload="joint_mcmc",
        )
        cache.extend_latest_cmb_request_phase("lensing", 0.5)
        self.assertEqual(
            cache.latest_cmb_performance_record()["phase_seconds"]["lensing"],
            0.5,
        )
        fail_request = cache.fail_latest_cmb_request
        self.assertTrue(callable(fail_request))
        fail_request(
            {"category": "test"},
            stop_phase="lensing",
        )
        failed = cache.latest_cmb_performance_record()
        self.assertEqual(failed["outcome"], "failure")
        self.assertEqual(failed["stop_phase"], "lensing")

        topology = {"nodes": 2}
        cache.set_declared_momentum_topology("topology", topology)
        self.assertIs(
            cache.get_declared_momentum_topology("topology"),
            topology,
        )

    def test_warm_performance_quantiles_are_deterministic(self):
        """Warm acceptance samples must report stable median and p95 values."""

        cache.clear_cmb_caches()
        self.assertTrue(callable(cache.cmb_performance_quantiles))
        self.assertTrue(callable(cache.set_reionization_calibration_seed))
        for elapsed_seconds in (1.0, 2.0, 3.0, 4.0, 5.0):
            cache.record_cmb_performance(
                {"total_seconds": elapsed_seconds},
                workload="joint_mcmc",
                cache_state="warm",
            )

        report = cache.cmb_performance_quantiles(
            workload="joint_mcmc",
            cache_state="warm",
        )

        self.assertEqual(report["sample_count"], 5)
        self.assertEqual(report["median_seconds"], 3.0)
        self.assertEqual(report["p95_seconds"], 4.8)

    def test_cache_source_does_not_import_camb(self):
        """The declared cache module should remain CAMB-free."""

        source_text = Path(cache.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import camb", source_text)

    def test_runtime_cache_identity_separates_static_and_request_work(self):
        """Runtime identities must keep request work out of static keys."""

        identity = cache.RuntimeCacheIdentity(
            contract_static=("graph", 1),
            model_static=("model_values", 2),
            request_specific=("ells", (20, 30)),
        )
        changed_request = cache.RuntimeCacheIdentity(
            contract_static=identity.contract_static,
            model_static=identity.model_static,
            request_specific=("ells", (20, 40)),
        )

        self.assertNotEqual(identity, changed_request)
        self.assertIs(
            type(identity),
            cache.RuntimeCacheIdentity,
        )
        self.assertEqual(
            identity.contract_static,
            changed_request.contract_static,
        )
        self.assertEqual(
            identity.model_static,
            changed_request.model_static,
        )
        self.assertNotEqual(hash(identity), hash(changed_request))
        self.assertEqual(identity.execution_solver, CCMBS_ID)

        cache.clear_cmb_caches()
        cache.remember_cmb_request_identity(identity)
        self.assertEqual(
            cache.latest_cmb_request_identity(),
            identity,
        )
        cache.set_cmb_spectrum(identity, "spectrum")
        self.assertEqual(
            cache.latest_cmb_request_identity(),
            identity,
        )
        cache.clear_cmb_result_caches()
        self.assertEqual(
            cache.latest_cmb_request_identity(),
            identity,
        )
        cache.clear_cmb_caches()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
