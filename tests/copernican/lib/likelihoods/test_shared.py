"""Smoke tests for copernican.lib.likelihoods.shared."""

import unittest

from copernican.lib.likelihoods import shared as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.likelihoods.shared")

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "LikelihoodProtocol"))
        self.assertTrue(hasattr(module, "LikelihoodState"))
        self.assertTrue(hasattr(module.LikelihoodProtocol, "loglike"))
        self.assertTrue(hasattr(module.LikelihoodProtocol, "state"))
        self.assertTrue(hasattr(module.LikelihoodState, "as_mapping"))


if __name__ == "__main__":
    unittest.main()
