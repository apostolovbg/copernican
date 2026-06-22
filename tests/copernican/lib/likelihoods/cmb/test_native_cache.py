"""Focused tests for the native CMB cache module."""

import unittest
from pathlib import Path

import numpy

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

        self.assertIsNone(native_cache.get_custom_cmb_background("bg"))
        native_cache.set_custom_cmb_background("bg", {"a": 1.0})
        self.assertEqual(
            native_cache.get_custom_cmb_background("bg"),
            {"a": 1.0},
        )

        self.assertIsNone(native_cache.get_custom_cmb_spectrum("spec"))
        native_cache.set_custom_cmb_spectrum("spec", {"tt": 2.0})
        self.assertEqual(
            native_cache.get_custom_cmb_spectrum("spec"),
            {"tt": 2.0},
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

        stats = native_cache.native_cmb_cache_stats()
        self.assertEqual(stats["declared_symbol_plan"]["entries"], 1)
        self.assertEqual(
            stats["declared_graph_execution_plan"]["entries"],
            1,
        )

        native_cache.clear_native_cmb_caches()
        cleared_stats = native_cache.native_cmb_cache_stats()
        self.assertEqual(cleared_stats["declared_symbol_plan"]["entries"], 0)
        self.assertEqual(
            cleared_stats["declared_graph_execution_plan"]["entries"],
            0,
        )

    def test_native_cache_source_does_not_import_camb(self):
        """The native cache module should remain CAMB-free."""

        source_text = Path(native_cache.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import camb", source_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
