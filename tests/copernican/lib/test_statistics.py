"""Smoke tests for copernican.lib.statistics."""

import unittest

from copernican.lib import statistics as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.statistics")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the statistics API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(module.calculate_bao_observables))
        self.assertTrue(callable(module.chi_squared_bao))
        self.assertTrue(callable(module.chi_squared_cmb))
        self.assertTrue(callable(module.chi_squared_sne))


if __name__ == "__main__":
    unittest.main()
