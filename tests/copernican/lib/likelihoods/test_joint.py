"""Smoke tests for copernican.lib.likelihoods.joint."""

import unittest

from copernican.lib.likelihoods import joint as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.likelihoods.joint")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the joint-likelihood surface to the coverage policy."""

    def test_public_symbols_are_present(self) -> None:
        self.assertTrue(hasattr(module, "JointLike"))
        self.assertTrue(hasattr(module.JointLike, "loglike"))
        self.assertTrue(hasattr(module.JointLike, "state"))


if __name__ == "__main__":
    unittest.main()
