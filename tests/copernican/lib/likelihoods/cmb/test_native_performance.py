"""Focused tests for native CMB performance accounting."""

import unittest

from copernican.lib.likelihoods.cmb import native_performance


class NativePerformanceModuleTestCase(unittest.TestCase):
    """Exercise native workload budgets and phase accounting."""

    def test_bounded_budget_exposes_required_workload_limits(self):
        """The bounded preset must expose both acceptance budgets."""

        budget = native_performance.resolve_native_performance_budget(
            {"runtime_envelope": "bounded"}
        )

        self.assertTrue(
            callable(native_performance.enforce_native_performance_budget)
        )
        self.assertTrue(callable(native_performance.NativePhaseTimer.add))
        self.assertTrue(callable(native_performance.NativePhaseTimer.phase))
        self.assertIsInstance(
            budget,
            native_performance.NativePerformanceBudget,
        )
        self.assertEqual(budget.limit_for("full_spectrum"), 180.0)
        self.assertEqual(
            budget.limit_for("joint_mcmc"),
            60.0,
        )

    def test_phase_timer_accumulates_named_work(self):
        """Phase timing must retain separate compilation and projection."""

        timer = native_performance.NativePhaseTimer()
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
        """An over-budget run must fail with the measured workload."""

        budget = native_performance.NativePerformanceBudget(
            full_spectrum_seconds=1.0,
            joint_mcmc_seconds=0.5,
        )
        with self.assertRaisesRegex(
            native_performance.NativePerformanceBudgetError,
            r"joint_mcmc: 0\.750s > 0\.500s",
        ):
            native_performance.enforce_native_performance_budget(
                0.75,
                workload="joint_mcmc",
                budget=budget,
            )

    def test_cold_joint_start_uses_startup_budget_only_once(self):
        """Structural startup and warm proposal limits must stay distinct."""

        budget = native_performance.NativePerformanceBudget(
            full_spectrum_seconds=1.0,
            joint_mcmc_seconds=0.5,
        )
        native_performance.enforce_native_performance_budget(
            0.75,
            workload="joint_mcmc",
            budget=budget,
            cache_state="cold",
        )
        with self.assertRaises(
            native_performance.NativePerformanceBudgetError
        ):
            native_performance.enforce_native_performance_budget(
                0.75,
                workload="joint_mcmc",
                budget=budget,
                cache_state="warm",
            )

    def test_unbounded_controls_do_not_invent_wall_time_limit(self):
        """Unspecified controls must preserve caller-owned runtime policy."""

        self.assertIsNone(
            native_performance.resolve_native_performance_budget({})
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
