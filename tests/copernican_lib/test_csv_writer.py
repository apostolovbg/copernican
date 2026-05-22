"""Smoke tests for copernican_lib.csv_writer."""

import unittest

from copernican_lib import csv_writer as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.csv_writer")


if __name__ == "__main__":
    unittest.main()
