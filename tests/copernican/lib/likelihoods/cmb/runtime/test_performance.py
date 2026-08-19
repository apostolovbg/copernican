"""Focused tests for declared CMB performance accounting."""

import unittest

from copernican.lib.likelihoods.cmb.runtime import performance


class PerformanceModuleTestCase(unittest.TestCase):
    """Exercise declared phase timing and cache-state accounting."""

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
