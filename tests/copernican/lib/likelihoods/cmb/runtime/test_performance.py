"""Focused tests for declared CMB performance accounting."""

import unittest
from unittest import mock

from copernican.lib.likelihoods.cmb.runtime import performance


class PerformanceModuleTestCase(unittest.TestCase):
    """Exercise declared phase timing and cache-state accounting."""

    def test_repeated_scopes_preserve_previous_attempt_work(self):
        timer = performance.PhaseTimer()
        timer.start_work("evolution", "pilots")
        timer.heartbeat("evolution", scope="pilots", completed=2, total=3)
        timer.start_work("evolution", "pilots")
        timer.heartbeat("evolution", scope="pilots", completed=1, total=5)
        scopes = timer.snapshot()["work_scopes"]
        self.assertEqual(scopes["evolution:pilots@attempt_1"]["completed"], 2)
        self.assertEqual(scopes["evolution:pilots"]["completed"], 1)

    def test_nested_phases_charge_each_interval_once(self):
        """Inner owners pause their parent and retain the failing owner."""

        now = [0.0]
        with mock.patch.object(
            performance, "perf_counter", side_effect=lambda: now[0]
        ):
            timer = performance.PhaseTimer()
            with timer.phase("evolution"):
                now[0] = 2.0
                with timer.phase("initial_data"):
                    now[0] = 5.0
                    self.assertEqual(timer.total_seconds(), 5.0)
                now[0] = 7.0
            self.assertEqual(timer.phase_seconds["evolution"], 4.0)
            self.assertEqual(timer.phase_seconds["initial_data"], 3.0)
            self.assertEqual(timer.total_seconds(), 7.0)
            with self.assertRaisesRegex(RuntimeError, "physics"):
                with timer.phase("evolution"):
                    with timer.phase("initial_data"):
                        raise RuntimeError("physics")
            self.assertEqual(timer.failed_phase, "initial_data")

    def test_weighted_scope_estimate_excludes_other_phases(self):
        """Unequal RK costs and intervening projection do not corrupt ETA."""

        now = [0.0]
        with mock.patch.object(
            performance, "perf_counter", side_effect=lambda: now[0]
        ):
            timer = performance.PhaseTimer()
            timer._started_at = timer._last_heartbeat_at = 0.0
            with timer.phase("evolution"):
                timer.start_work("evolution", "rk")
                now[0] = 10.0
                timer.heartbeat(
                    "evolution",
                    scope="rk",
                    completed=1,
                    total=2,
                    weighted_completed=10,
                    weighted_total=100,
                    force=True,
                )
                work = timer.snapshot()["work_scopes"]["evolution:rk"]
                self.assertEqual(work["remaining_seconds"], 90.0)
                with timer.phase("projection"):
                    now[0] = 110.0
                now[0] = 120.0
                timer.heartbeat(
                    "evolution",
                    scope="rk",
                    completed=2,
                    total=3,
                    weighted_completed=20,
                    weighted_total=150,
                    force=True,
                )
                work = timer.snapshot()["work_scopes"]["evolution:rk"]
                self.assertEqual(work["remaining_seconds"], 130.0)
                self.assertEqual(timer.last_heartbeat_seconds, 20.0)
                timer.start_work("evolution", "cache_hits")
                timer.heartbeat(
                    "evolution",
                    scope="cache_hits",
                    completed=8,
                    total=8,
                    force=True,
                )
                self.assertIsNone(
                    timer.snapshot()["work_scopes"]["evolution:cache_hits"][
                        "remaining_seconds"
                    ]
                )

    def test_progress_rejects_invalid_and_backwards_work(self):
        """Dynamic totals may grow, but counters cannot silently reset."""

        timer = performance.PhaseTimer()
        timer.start_work("evolution", "rk")
        timer.heartbeat("evolution", scope="rk", completed=2, total=4)
        for completed, total in ((5, 4), (-1, 4), (1, 4), (2, 3)):
            with self.subTest(completed=completed, total=total):
                with self.assertRaises(ValueError):
                    timer.heartbeat(
                        "evolution",
                        scope="rk",
                        completed=completed,
                        total=total,
                    )
        for value in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                timer.add("projection", value)
        with self.assertRaises(ValueError):
            timer.heartbeat("evolution", completed=0)
        with self.assertRaises(ValueError):
            timer.heartbeat("evolution", weighted_completed=1)
        with self.assertRaises(ValueError):
            timer.heartbeat(
                "evolution",
                weighted_completed=2,
                weighted_total=1,
            )

    def test_refinement_merge_keeps_phase_time_work_and_failure(self):
        """Outer refinement contributes to the same request's accounting."""

        parent = performance.PhaseTimer()
        child = performance.PhaseTimer(_started_at=parent._started_at)
        parent.add("evolution", 2.0)
        child.add("evolution", 3.0)
        child.add("projection", 4.0)
        child.start_work("projection", "quadrature")
        child.heartbeat(
            "projection",
            scope="quadrature",
            completed=4,
            total=4,
            weighted_completed=8,
            weighted_total=8,
            force=True,
        )
        child.failed_phase = "projection"
        parent.merge(child, scope_prefix="refined")
        self.assertEqual(parent.total_seconds(), 9.0)
        self.assertEqual(parent.phase_seconds["evolution"], 5.0)
        self.assertEqual(parent.failed_phase, "projection")
        self.assertEqual(parent.heartbeat_count, 1)
        self.assertEqual(
            parent.snapshot()["work_scopes"]["projection:refined/quadrature"][
                "completed"
            ],
            4,
        )
        with child.phase("evolution"):
            with self.assertRaises(ValueError):
                parent.merge(child, scope_prefix="active")

    def test_phase_timer_exposes_no_wall_time_budget(self):
        """Timing remains observable without an execution-time limit."""

        self.assertTrue(callable(performance.PhaseTimer.add))
        self.assertTrue(callable(performance.PhaseTimer.phase))
        self.assertFalse(hasattr(performance, "PerformanceBudget"))
        self.assertFalse(hasattr(performance, "enforce_performance_budget"))
        self.assertFalse(hasattr(performance, "resolve_performance_budget"))

    def test_phase_timer_accumulates_named_work(self):
        """Phase timing must retain separate compilation and projection."""

        timer = performance.PhaseTimer()
        with timer.phase("compilation"):
            pass
        timer.add("projection", 0.25)
        timer.mark_cache_state("warm")
        self.assertEqual(timer.cache_state, "warm")
        timer.set_work_units({"evolution": 4})
        self.assertEqual(timer.work_units["evolution"], 4)

        snapshot = timer.snapshot(total_seconds=0.5)
        self.assertGreaterEqual(snapshot["compilation_seconds"], 0.0)
        self.assertEqual(snapshot["projection_seconds"], 0.25)
        self.assertEqual(snapshot["total_seconds"], 0.5)

    def test_long_elapsed_requests_are_recordable(self):
        """A slow request remains a successful diagnostic record."""

        timer = performance.PhaseTimer()
        snapshot = timer.snapshot(total_seconds=181.0)
        self.assertEqual(snapshot["total_seconds"], 181.0)

    def test_heartbeat_is_throttled_and_serialized(self):
        """Long phases expose progress without flooding the log."""

        timer = performance.PhaseTimer(heartbeat_interval_seconds=60.0)
        self.assertTrue(
            timer.heartbeat("evolution", completed=2, total=4, force=True)
        )
        self.assertFalse(timer.heartbeat("evolution", completed=3, total=4))
        snapshot = timer.snapshot(total_seconds=1.0)
        self.assertEqual(snapshot["heartbeat_count"], 1)
        self.assertEqual(snapshot["last_heartbeat_phase"], "evolution")
        self.assertGreaterEqual(snapshot["last_heartbeat_seconds"], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
