"""Focused tests for declared CMB performance accounting."""

import unittest

from copernican.lib.likelihoods.cmb.runtime import performance


class PerformanceModuleTestCase(unittest.TestCase):
    """Exercise declared workload budgets and phase accounting."""

    def test_bounded_budget_exposes_required_cache_state_limits(self):
        """The bounded preset must expose every accepted cache-state limit."""

        budget = performance.resolve_performance_budget(
            {"runtime_envelope": "bounded"}
        )

        self.assertTrue(callable(performance.enforce_performance_budget))
        self.assertTrue(callable(performance.PhaseTimer.add))
        self.assertTrue(callable(performance.PhaseTimer.phase))
        self.assertIsInstance(
            budget,
            performance.PerformanceBudget,
        )
        self.assertEqual(budget.limit_for("full_spectrum"), 180.0)
        self.assertEqual(budget.limit_for("warm_parameter"), 5.0)
        self.assertEqual(budget.limit_for("exact_cache_hit"), 1.0)

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

    def test_budget_rejects_overrun_with_workload_name(self):
        """An over-budget warm request must retain its workload label."""

        budget = performance.PerformanceBudget(
            full_spectrum_seconds=1.0,
            warm_parameter_seconds=0.5,
            exact_cache_hit_seconds=0.25,
        )
        with self.assertRaisesRegex(
            performance.PerformanceBudgetError,
            r"joint_mcmc: 0\.750s > 0\.500s",
        ):
            performance.enforce_performance_budget(
                0.75,
                workload="joint_mcmc",
                budget=budget,
                cache_state="warm",
            )

    def test_cache_state_selects_the_matching_budget(self):
        """Cold, warm, and exact requests must retain distinct limits."""

        budget = performance.PerformanceBudget(
            full_spectrum_seconds=1.0,
            warm_parameter_seconds=0.5,
            exact_cache_hit_seconds=0.25,
        )
        performance.enforce_performance_budget(
            0.75,
            workload="joint_mcmc",
            budget=budget,
            cache_state="cold",
        )
        with self.assertRaises(performance.PerformanceBudgetError):
            performance.enforce_performance_budget(
                0.75,
                workload="joint_mcmc",
                budget=budget,
                cache_state="warm",
            )
        with self.assertRaises(performance.PerformanceBudgetError):
            performance.enforce_performance_budget(
                0.3,
                workload="joint_mcmc",
                budget=budget,
                cache_state="exact_cache_hit",
            )

    def test_unbounded_controls_do_not_invent_wall_time_limit(self):
        """Unspecified controls must preserve caller-owned runtime policy."""

        self.assertIsNone(performance.resolve_performance_budget({}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
