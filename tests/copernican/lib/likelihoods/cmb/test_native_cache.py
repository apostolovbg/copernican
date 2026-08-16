"""Focused tests for the native CMB cache module."""

import unittest
from pathlib import Path

import numpy

from copernican.lib.cmb_identity import NATIVE_CMB_ENGINE_ID
from copernican.lib.likelihoods.cmb import native_cache


class NativeCacheModuleTestCase(unittest.TestCase):
    """Exercise native cache helpers directly."""

    def test_bounded_cache_store_tracks_hits_misses_and_evictions(self):
        """The internal bounded cache should expose explicit accounting."""

        cache = native_cache._BoundedCacheStore(limit=1)
        self.assertIsNone(cache.get("missing"))

        cache.set("alpha", 1)
        self.assertEqual(cache.get("alpha"), 1)
        self.assertEqual(cache.snapshot()["entries"], 1)

        cache.set("beta", 2)
        self.assertEqual(cache.snapshot()["evictions"], 1)
        cache.prune(lambda key: key == "beta")
        self.assertEqual(cache.snapshot()["entries"], 0)

        cache.clear()
        snapshot = cache.snapshot()
        self.assertEqual(snapshot["entries"], 0)
        self.assertEqual(snapshot["hits"], 0)
        self.assertEqual(snapshot["misses"], 0)

    def test_native_cache_module_functions_round_trip_values(self):
        """The module cache helpers should store every cache family."""

        native_cache.clear_native_cmb_caches()
        self.assertIsNone(native_cache.get_declared_symbol_plan("plan"))
        native_cache.set_declared_symbol_plan("plan", {"step": 1})
        self.assertEqual(
            native_cache.get_declared_symbol_plan("plan"),
            {"step": 1},
        )

        self.assertIsNone(
            native_cache.get_declared_graph_execution_plan("graph")
        )
        native_cache.set_declared_graph_execution_plan("graph", {"node": 1})
        self.assertEqual(
            native_cache.get_declared_graph_execution_plan("graph"),
            {"node": 1},
        )

        self.assertIsNone(native_cache.get_native_runtime_assets("runtime"))
        native_cache.set_native_runtime_assets("runtime", {"asset": 1})
        self.assertEqual(
            native_cache.get_native_runtime_assets("runtime"),
            {"asset": 1},
        )

        self.assertIsNone(
            native_cache.get_declared_momentum_topology("topology")
        )
        native_cache.set_declared_momentum_topology(
            "topology",
            {"nodes": 2},
        )
        self.assertEqual(
            native_cache.get_declared_momentum_topology("topology"),
            {"nodes": 2},
        )

        self.assertIsNone(native_cache.get_declared_momentum_grid("momentum"))
        native_cache.set_declared_momentum_grid(
            "momentum",
            {"grid": 1},
        )
        self.assertEqual(
            native_cache.get_declared_momentum_grid("momentum"),
            {"grid": 1},
        )

        self.assertIsNone(native_cache.get_native_cmb_background("bg"))
        native_cache.set_native_cmb_background("bg", {"a": 1.0})
        self.assertEqual(
            native_cache.get_native_cmb_background("bg"),
            {"a": 1.0},
        )

        self.assertIsNone(
            native_cache.get_reionization_calibration_seed("reionization")
        )
        native_cache.set_reionization_calibration_seed(
            "reionization",
            2.5,
        )
        self.assertEqual(
            native_cache.get_reionization_calibration_seed("reionization"),
            2.5,
        )

        self.assertIsNone(native_cache.get_native_cmb_spectrum("spec"))
        native_cache.set_native_cmb_spectrum("spec", {"tt": 2.0})
        self.assertEqual(
            native_cache.get_native_cmb_spectrum("spec"),
            {"tt": 2.0},
        )

        self.assertIsNone(native_cache.get_native_cmb_transfer("transfer"))
        native_cache.set_native_cmb_transfer(
            "transfer",
            {"temperature": numpy.ones((2, 2))},
        )
        self.assertTrue(
            numpy.array_equal(
                native_cache.get_native_cmb_transfer("transfer")[
                    "temperature"
                ],
                numpy.ones((2, 2)),
            )
        )

        x_values = numpy.array([0.1, 0.2])
        native_cache.store_bessel_inputs("xsig", x_values)
        self.assertTrue(
            numpy.array_equal(
                native_cache.get_bessel_inputs("xsig"),
                x_values,
            )
        )
        self.assertIsNone(native_cache.get_bessel_values((2, "xsig")))
        native_cache.set_bessel_values((2, "xsig"), ("j", "jp"))
        self.assertEqual(
            native_cache.get_bessel_values((2, "xsig")),
            ("j", "jp"),
        )

        kernel_key = ((2, 3), "xsig")
        self.assertIsNone(
            native_cache.get_declared_projection_kernel_batch(kernel_key)
        )
        native_cache.set_declared_projection_kernel_batch(
            kernel_key,
            "batch",
        )
        self.assertTrue(
            hasattr(native_cache, "set_declared_projection_kernel_batch")
        )
        self.assertEqual(
            native_cache.get_declared_projection_kernel_batch(kernel_key),
            "batch",
        )

        native_cache.record_native_cmb_performance(
            {"projection_seconds": 0.25},
            cache_hit=True,
        )
        native_cache.record_native_cmb_phase("lensing", 0.5)
        self.assertTrue(callable(native_cache.record_native_cmb_performance))
        self.assertTrue(callable(native_cache.record_native_cmb_phase))

        stats = native_cache.native_cmb_cache_stats()
        self.assertIn("native_background", stats)
        self.assertIn("native_spectrum", stats)
        self.assertIn("native_transfer", stats)
        self.assertIn("native_runtime_assets", stats)
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
        performance = native_cache.native_cmb_performance_stats()
        self.assertEqual(int(performance["requests"]), 1)
        self.assertEqual(int(performance["cache_hits"]), 1)
        self.assertEqual(
            performance["phase_seconds"]["projection_seconds"],
            0.25,
        )
        self.assertEqual(performance["phase_seconds"]["lensing"], 0.5)

        native_cache.clear_native_cmb_caches()
        cleared_stats = native_cache.native_cmb_cache_stats()
        self.assertEqual(cleared_stats["declared_symbol_plan"]["entries"], 0)
        self.assertEqual(
            cleared_stats["declared_graph_execution_plan"]["entries"],
            0,
        )
        self.assertEqual(
            cleared_stats["native_runtime_assets"]["entries"],
            0,
        )
        self.assertEqual(
            cleared_stats["declared_momentum_grid"]["entries"],
            0,
        )

    def test_cache_inventory_and_parameter_invalidation_are_explicit(self):
        """Parameter invalidation must retain process-local structure."""

        native_cache.clear_native_cmb_caches()
        native_cache.set_native_runtime_assets("runtime", "assets")
        native_cache.set_declared_momentum_topology("topology", "nodes")
        native_cache.set_declared_momentum_grid("grid", "bound-grid")
        native_cache.set_native_cmb_background("background", "tables")
        native_cache.set_native_cmb_spectrum("spectrum", "result")

        inventory = native_cache.native_cmb_cache_inventory()
        self.assertEqual(
            inventory["native_runtime_assets"]["category"],
            "structural",
        )
        self.assertEqual(
            inventory["declared_momentum_grid"]["category"],
            "parameter",
        )
        self.assertEqual(
            inventory["native_spectrum"]["category"],
            "result",
        )
        self.assertGreater(
            int(inventory["native_runtime_assets"]["owner_pid"]), 0
        )

        native_cache.clear_native_cmb_parameter_caches()
        stats = native_cache.native_cmb_cache_stats()
        self.assertEqual(stats["native_runtime_assets"]["entries"], 1)
        self.assertEqual(stats["declared_momentum_topology"]["entries"], 1)
        self.assertEqual(stats["declared_momentum_grid"]["entries"], 0)
        self.assertEqual(stats["native_background"]["entries"], 0)
        self.assertEqual(stats["native_spectrum"]["entries"], 0)

    def test_result_and_request_accounting_helpers_are_explicit(self):
        """Result invalidation and request mutation should stay observable."""

        native_cache.clear_native_cmb_caches()
        native_cache.set_native_cmb_spectrum("spectrum", "result")
        native_cache.clear_native_cmb_result_caches()
        self.assertIsNone(native_cache.get_native_cmb_spectrum("spectrum"))

        native_cache.record_native_cmb_performance(
            {"projection_seconds": 0.25},
            workload="joint_mcmc",
        )
        native_cache.extend_latest_native_cmb_request_phase("lensing", 0.5)
        self.assertEqual(
            native_cache.latest_native_cmb_performance_record()[
                "phase_seconds"
            ]["lensing"],
            0.5,
        )
        fail_request = native_cache.fail_latest_native_cmb_request
        self.assertTrue(callable(fail_request))
        fail_request(
            {"category": "test"},
            stop_phase="lensing",
        )
        failed = native_cache.latest_native_cmb_performance_record()
        self.assertEqual(failed["outcome"], "failure")
        self.assertEqual(failed["stop_phase"], "lensing")

        topology = {"nodes": 2}
        native_cache.set_declared_momentum_topology("topology", topology)
        self.assertIs(
            native_cache.get_declared_momentum_topology("topology"),
            topology,
        )

    def test_warm_performance_quantiles_are_deterministic(self):
        """Warm acceptance samples must report stable median and p95 values."""

        native_cache.clear_native_cmb_caches()
        self.assertTrue(
            callable(native_cache.native_cmb_performance_quantiles)
        )
        self.assertTrue(
            callable(native_cache.set_reionization_calibration_seed)
        )
        for elapsed_seconds in (1.0, 2.0, 3.0, 4.0, 5.0):
            native_cache.record_native_cmb_performance(
                {"total_seconds": elapsed_seconds},
                workload="joint_mcmc",
                cache_state="warm",
            )

        report = native_cache.native_cmb_performance_quantiles(
            workload="joint_mcmc",
            cache_state="warm",
        )

        self.assertEqual(report["sample_count"], 5)
        self.assertEqual(report["median_seconds"], 3.0)
        self.assertEqual(report["p95_seconds"], 4.8)

    def test_native_cache_source_does_not_import_camb(self):
        """The native cache module should remain CAMB-free."""

        source_text = Path(native_cache.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import camb", source_text)

    def test_runtime_cache_identity_separates_static_and_request_work(self):
        """Runtime identities must keep request work out of static keys."""

        identity = native_cache.NativeRuntimeCacheIdentity(
            contract_static=("graph", 1),
            cosmology_static=("cosmology", 2),
            request_specific=("ells", (20, 30)),
        )
        changed_request = native_cache.NativeRuntimeCacheIdentity(
            contract_static=identity.contract_static,
            cosmology_static=identity.cosmology_static,
            request_specific=("ells", (20, 40)),
        )

        self.assertNotEqual(identity, changed_request)
        self.assertIs(
            type(identity),
            native_cache.NativeRuntimeCacheIdentity,
        )
        self.assertEqual(
            identity.contract_static,
            changed_request.contract_static,
        )
        self.assertEqual(
            identity.cosmology_static,
            changed_request.cosmology_static,
        )
        self.assertNotEqual(hash(identity), hash(changed_request))
        self.assertEqual(identity.execution_engine, NATIVE_CMB_ENGINE_ID)

        native_cache.clear_native_cmb_caches()
        native_cache.remember_native_cmb_request_identity(identity)
        self.assertEqual(
            native_cache.latest_native_cmb_request_identity(),
            identity,
        )
        native_cache.set_native_cmb_spectrum(identity, "spectrum")
        self.assertEqual(
            native_cache.latest_native_cmb_request_identity(),
            identity,
        )
        native_cache.clear_native_cmb_result_caches()
        self.assertEqual(
            native_cache.latest_native_cmb_request_identity(),
            identity,
        )
        native_cache.clear_native_cmb_caches()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
