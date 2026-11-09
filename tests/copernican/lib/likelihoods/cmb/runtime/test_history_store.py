"""Request-lifetime raw-history reuse and integrity controls."""

import unittest
from contextvars import Context
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb.runtime import history_store
from copernican.lib.likelihoods.cmb.runtime.history_store import (
    RequestSourceHistoryStore,
    _SpilledSourceHistory,
    active_source_history_store,
    retain_source_histories,
)


class RequestSourceHistoryStoreTestCase(unittest.TestCase):
    """Check exact identities, bounded retention and unconditional cleanup."""

    def test_scheduled_columns_survive_working_cache_eviction(self):
        """Scalar and momentum rows keep exact future values compactly."""

        keys = [(0, 0.0), (0, 0.5), (0, 1.0), (1, 0.0)]
        epochs = numpy.array([2.0, 3.0, 4.0, 4.0])
        velocities = numpy.arange(12.0).reshape(4, 3)
        columns = history_store.StageContextColumns(
            keys,
            {"eta": epochs, "velocity": velocities},
            static={"mass": 0.06},
        )
        working_cache = {}
        for key, epoch, velocity in zip(keys, epochs, velocities):
            working_cache[key] = columns.get(key)
            self.assertEqual(working_cache[key]["eta"], epoch)
            numpy.testing.assert_array_equal(
                working_cache[key]["velocity"], velocity
            )
            self.assertEqual(working_cache[key]["mass"], 0.06)
            working_cache.clear()
        self.assertIsNone(columns.get((2, 0.0)))
        self.assertEqual(columns.get((0, 0.0))["eta"], 2.0)
        row = columns.get((0, 0.5))
        row.pop("eta")
        self.assertIn("eta", columns.get((0, 0.5)))
        self.assertTrue(
            numpy.shares_memory(columns.columns["velocity"], velocities)
        )
        with self.assertRaises(ValueError):
            columns.get((0, 0.5))["velocity"][0] = -1.0
        with self.assertRaisesRegex(ValueError, "unique"):
            history_store.StageContextColumns([keys[0]] * 2, {})
        with self.assertRaisesRegex(ValueError, "match stages"):
            history_store.StageContextColumns(keys, {"eta": epochs[:2]})

    def test_native_mapping_preserves_full_arrays_and_cleans_up_on_error(self):
        self.assertEqual(
            history_store.allocate_native_history((3, 2)).shape, (3, 2)
        )
        mappings = []

        def operation(fail=False):
            values = history_store.allocate_native_history((2, 5, 3))
            mappings.append(values)
            values[:] = numpy.arange(30).reshape(2, 5, 3)
            if fail:
                raise RuntimeError("forced consumer failure")
            return values.copy()

        with mock.patch.object(
            history_store, "NATIVE_HISTORY_MEMORY_CELLS", 4
        ):
            execute = history_store.with_native_history_storage(operation)
            result = execute()
            self.assertIsInstance(result, numpy.ndarray)
            numpy.testing.assert_array_equal(
                result, numpy.arange(30).reshape(2, 5, 3)
            )
            self.assertTrue(mappings[-1]._mmap.closed)
            with self.assertRaisesRegex(RuntimeError, "consumer failure"):
                history_store.with_native_history_storage(operation)(True)
            self.assertTrue(mappings[-1]._mmap.closed)
        self.assertIsNone(history_store._NATIVE_HISTORY_RESOURCES.get())

    def test_spilled_histories_reuse_exact_arrays_after_memory_pressure(self):
        """A small memory budget changes storage, not the retained values."""

        with retain_source_histories(max_bytes=16) as store:
            self.assertIsInstance(store, RequestSourceHistoryStore)
            expected = numpy.linspace(-1.0, 1.0, 33)
            for index in range(5):
                store.set(("physics", index), {"file": expected + index})
            for index in range(5):
                loaded = store.get(("physics", index))
                numpy.testing.assert_array_equal(
                    loaded["file"], expected + index
                )
            self.assertIsNone(store.get(("changed_physics", 0)))
            snapshot = store.snapshot()
            self.assertEqual(snapshot["entries"], 5)
            self.assertEqual(snapshot["spills"], 5)
            self.assertEqual(snapshot["loads"], 5)
            self.assertLessEqual(snapshot["resident_bytes"], 16)
            directory = store._directory.name
        self.assertEqual(store.snapshot()["entries"], 0)
        from pathlib import Path

        self.assertFalse(Path(directory).exists())

    def test_resident_sources_cannot_alias_the_producer(self):
        """Producer and consumer mutation cannot change retained raw work."""

        store = RequestSourceHistoryStore(max_bytes=1024)
        original = numpy.arange(4, dtype=float)
        store.set("exact", {"source": original})
        original[:] = -1
        product = store.get("exact")
        numpy.testing.assert_array_equal(product["source"], numpy.arange(4))
        with self.assertRaises(ValueError):
            product["source"][0] = 99
        product.pop("source")
        self.assertIn("source", store.get("exact"))
        store.clear()
        self.assertEqual(store.snapshot()["resident_bytes"], 0)

    def test_corrupt_spill_is_rejected_without_borrowing_a_certificate(self):
        """Invalid retained bytes produce a miss and must be evolved again."""

        with retain_source_histories(max_bytes=0) as store:
            store.set("exact", {"source": numpy.arange(4, dtype=float)})
            reference = store._entries["exact"]
            self.assertIsInstance(reference, _SpilledSourceHistory)
            reference.path.write_bytes(b"invalid history archive")
            self.assertIsNone(store.get("exact"))
            self.assertEqual(store.snapshot()["invalidations"], 1)
            self.assertEqual(store.snapshot()["entries"], 0)

    def test_nested_refinement_shares_only_the_current_execution_context(self):
        """Nested work shares its owner; independent contexts start empty."""

        self.assertIsNone(active_source_history_store())
        with retain_source_histories(max_bytes=1024) as outer:
            outer.set("exact", {"source": numpy.arange(4, dtype=float)})
            with retain_source_histories(max_bytes=0) as inner:
                self.assertIs(inner, outer)
                self.assertIsNotNone(inner.get("exact"))
            self.assertIsNone(Context().run(active_source_history_store))
            self.assertIs(active_source_history_store(), outer)
        self.assertIsNone(active_source_history_store())

    def test_exception_exit_releases_spills_and_context(self):
        """A failed solve cannot leak retained arrays into another request."""

        with self.assertRaisesRegex(RuntimeError, "failed solve"):
            with retain_source_histories(max_bytes=0) as store:
                store.set("exact", {"source": numpy.arange(4, dtype=float)})
                raise RuntimeError("failed solve")
        self.assertIsNone(active_source_history_store())
        self.assertEqual(store.snapshot()["entries"], 0)

    def test_invalid_sources_are_rejected_before_retention(self):
        """Malformed products cannot become reusable source histories."""

        with self.assertRaises(ValueError):
            RequestSourceHistoryStore(max_bytes=-1)
        with retain_source_histories(max_bytes=1024) as store:
            for sources in (
                {},
                {"source": numpy.array([])},
                {"source": numpy.array([numpy.nan])},
                {"source": numpy.ones((2, 2))},
                {"first": numpy.ones(2), "second": numpy.ones(3)},
            ):
                with self.subTest(names=tuple(sources)):
                    with self.assertRaises(ValueError):
                        store.set("invalid", sources)
            self.assertEqual(store.snapshot()["entries"], 0)

    def test_replacement_releases_old_spills_and_tracks_new_values(self):
        """Replacing a retained identity removes its previous spill file."""

        with retain_source_histories(max_bytes=0) as store:
            store.set("exact", {"source": numpy.zeros(4)})
            first = store._entries["exact"].path
            store.set("exact", {"source": numpy.ones(4)})
            self.assertFalse(first.exists())
            numpy.testing.assert_array_equal(
                store.get("exact")["source"], numpy.ones(4)
            )
