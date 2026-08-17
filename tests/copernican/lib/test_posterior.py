"""Smoke tests for copernican.lib.posterior."""

from __future__ import annotations

import unittest
from unittest import mock

from copernican.lib import posterior as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.posterior")

    def test_make_logposterior_returns_posterior_evaluator(self) -> None:
        def like(params):
            return -sum(params)

        evaluator = module.make_logposterior(like, [])
        self.assertIsInstance(evaluator, module.PosteriorEvaluator)
        self.assertAlmostEqual(evaluator([1.0, 2.0]), -3.0)

    def test_posterior_evaluator_preserves_ordered_batch_results(self) -> None:
        """Batch evaluation keeps order while using the wrapped batch hook."""

        like = mock.Mock()
        like.parameter_bounds = None
        like.parameter_transforms = None
        like.evaluate_batch.return_value = (-3.0, -5.0)
        evaluator = module.make_logposterior(like, [])

        values = evaluator.evaluate_batch(((1.0,), (2.0,)))

        self.assertEqual(values, (-3.0, -5.0))
        like.evaluate_batch.assert_called_once_with([(1.0,), (2.0,)])


if __name__ == "__main__":
    unittest.main()
