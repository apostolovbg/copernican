"""Focused tests for the declared CMB cache module."""

import unittest
from pathlib import Path

import numpy

from copernican.lib.cmb_identity import CCMBS_ID
from copernican.lib.likelihoods.cmb.runtime import cache


class CacheModuleTestCase(unittest.TestCase):
    """Exercise declared cache helpers directly."""

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
