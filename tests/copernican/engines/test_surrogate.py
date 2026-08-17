"""Tests for the deterministic surrogate and delayed-acceptance helpers."""

import unittest

import numpy

from copernican.engines.surrogate import (
    DelayedAcceptanceController,
    DeterministicLocalSurrogate,
    SurrogateResult,
    run_delayed_acceptance_chain,
)


class SurrogateContractTestCase(unittest.TestCase):
    """Exercise surrogate support, exact fallback, and correction records."""

    def test_exact_training_points_and_outside_domain_are_attributable(self):
        surrogate = DeterministicLocalSurrogate(
            lower=(0.0, 0.0),
            upper=(1.0, 1.0),
            min_support=2,
        )
        surrogate.add_exact_sample((0.0, 0.0), 0.0, sample_id="origin")
        surrogate.add_exact_sample((1.0, 1.0), -1.0, sample_id="corner")

        exact = surrogate.predict((0.0, 0.0))
        outside = surrogate.predict((1.5, 0.5))

        self.assertIsInstance(exact, SurrogateResult)
        self.assertEqual(exact.domain_status, "supported")
        self.assertEqual(exact.training_sample_ids, ("origin",))
        self.assertTrue(numpy.isclose(exact.prediction, 0.0))
        self.assertEqual(outside.domain_status, "outside_domain")
        self.assertTrue(outside.exact_required)
        self.assertEqual(outside.training_sample_ids, ())

    def test_delayed_acceptance_applies_exact_second_stage_correction(self):
        surrogate = DeterministicLocalSurrogate(
            lower=(-2.0,),
            upper=(2.0,),
            min_support=2,
            uncertainty_threshold=10.0,
        )
        surrogate.add_exact_sample((-1.0,), -0.5, sample_id="left")
        surrogate.add_exact_sample((1.0,), -0.5, sample_id="right")
        calls = []

        def exact_target(params):
            calls.append(tuple(params))
            return -0.5 * float(params[0] ** 2)

        controller = DelayedAcceptanceController(
            exact_target,
            surrogate,
            rng=numpy.random.default_rng(3),
            proposal_scale=0.1,
        )
        outcome = controller.step((0.0,), 0.0, (0.5,))

        self.assertTrue(outcome.exact_called)
        self.assertIn(outcome.stage, {"exact_correction", "exact_fallback"})
        self.assertTrue(calls)
        self.assertIn("exact_corrections", controller.counters)
        self.assertEqual(len(controller.proposal_records), 1)
        self.assertIn("exact_called", controller.proposal_records[0])

    def test_unsupported_candidate_uses_exact_fallback(self):
        surrogate = DeterministicLocalSurrogate(
            lower=(0.0,), upper=(1.0,), min_support=2
        )
        surrogate.add_exact_sample((0.25,), -1.0, sample_id="seed")
        exact_calls = []

        def exact_target(params):
            exact_calls.append(tuple(params))
            return -float(params[0])

        controller = DelayedAcceptanceController(
            exact_target,
            surrogate,
            rng=numpy.random.default_rng(4),
        )
        outcome = controller.step((0.25,), -0.25, (0.75,))

        self.assertEqual(outcome.stage, "exact_fallback")
        self.assertTrue(outcome.exact_called)
        self.assertEqual(len(exact_calls), 1)
        self.assertGreaterEqual(controller.counters["support_fallbacks"], 1)

    def test_analytic_chain_records_finite_exact_target_values(self):
        """A bounded Gaussian target remains finite through correction
        steps.
        """

        surrogate = DeterministicLocalSurrogate(
            lower=(-4.0,),
            upper=(4.0,),
            min_support=2,
            uncertainty_threshold=100.0,
        )

        result = run_delayed_acceptance_chain(
            (0.0,),
            lambda params: -0.5 * float(params[0] ** 2),
            surrogate,
            n_steps=500,
            rng=numpy.random.default_rng(5),
            proposal_scale=0.2,
        )

        self.assertEqual(result["positions"].shape, (500, 1))
        self.assertTrue(numpy.isfinite(result["log_probability"]).all())
        self.assertEqual(
            result["counters"]["proposals"],
            result["counters"]["accepted"] + result["counters"]["rejected"],
        )
        self.assertLess(
            result["counters"]["exact_calls"],
            result["counters"]["proposals"],
        )
        self.assertLess(abs(float(result["positions"].mean())), 0.3)
        self.assertGreater(float(result["positions"].std()), 0.7)
        self.assertLess(float(result["positions"].std()), 1.2)
        self.assertEqual(len(result["proposal_records"]), 500)

    def test_correlated_target_and_bounded_proposals_remain_finite(self):
        """Correlated bounded targets use the same exact correction
        contract.
        """

        surrogate = DeterministicLocalSurrogate(
            lower=(-2.0, -2.0),
            upper=(2.0, 2.0),
            min_support=2,
            uncertainty_threshold=100.0,
        )

        def correlated_target(params):
            x, y = params
            return -0.5 * (x * x - 1.4 * x * y + y * y)

        result = run_delayed_acceptance_chain(
            (0.0, 0.0),
            correlated_target,
            surrogate,
            n_steps=25,
            rng=numpy.random.default_rng(6),
            proposal_scale=0.25,
        )

        self.assertTrue(numpy.isfinite(result["positions"]).all())
        self.assertTrue(numpy.all(result["positions"] >= surrogate.lower))
        self.assertTrue(numpy.all(result["positions"] <= surrogate.upper))

    def test_invalid_domain_proposal_is_typed_as_exact_failure(self):
        """An exact target failure is recorded without silent acceptance."""

        surrogate = DeterministicLocalSurrogate(
            lower=(0.0,), upper=(1.0,), min_support=1
        )
        surrogate.add_exact_sample((0.5,), 0.0, sample_id="seed")

        def exact_target(params):
            if not 0.0 <= params[0] <= 1.0:
                raise ValueError("outside analytic target domain")
            return -float(params[0] ** 2)

        controller = DelayedAcceptanceController(
            exact_target,
            surrogate,
            rng=numpy.random.default_rng(7),
        )
        outcome = controller.step((0.5,), -0.25, (2.0,))

        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.stage, "exact_fallback")
        self.assertTrue(
            outcome.surrogate.domain_status.startswith("exact_failure:")
        )


if __name__ == "__main__":
    unittest.main()
