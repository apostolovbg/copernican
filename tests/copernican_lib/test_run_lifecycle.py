"""Smoke tests for copernican_lib.run_lifecycle."""

import unittest

from copernican_lib import run_lifecycle as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.run_lifecycle")


if __name__ == "__main__":
    unittest.main()
