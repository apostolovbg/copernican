"""Smoke tests for copernican_lib.csv_writer."""

import unittest

from copernican_lib import csv_writer as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.csv_writer")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the CSV writer API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(module.save_bao_results_csv))
        self.assertTrue(callable(module.save_cmb_results_csv))
        self.assertTrue(callable(module.save_sne_results_detailed_csv))


if __name__ == "__main__":
    unittest.main()
