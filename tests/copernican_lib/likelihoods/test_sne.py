"""Smoke tests for copernican_lib.likelihoods.sne."""

import unittest

from copernican_lib.likelihoods import sne as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.likelihoods.sne")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the SNe helper API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "SNeLike"))
        self.assertTrue(callable(module.SNeLike))

    def test_loglike_and_state_symbols_are_exposed(self) -> None:
        loglike = module.SNeLike.loglike
        state = module.SNeLike.state
        self.assertTrue(callable(loglike))
        self.assertTrue(hasattr(state, "__get__"))


if __name__ == "__main__":
    unittest.main()
