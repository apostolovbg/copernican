"""Smoke tests for copernican_lib.posterior."""

from __future__ import annotations

import unittest

from copernican_lib import posterior as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.posterior")

    def test_make_logposterior_returns_posterior_evaluator(self) -> None:
        def like(params):
            return -sum(params)

        evaluator = module.make_logposterior(like, [])
        self.assertIsInstance(evaluator, module.PosteriorEvaluator)
        self.assertAlmostEqual(evaluator([1.0, 2.0]), -3.0)


if __name__ == "__main__":
    unittest.main()
