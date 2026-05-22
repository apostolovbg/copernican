"""Smoke tests for copernican_lib.posterior."""

import unittest

from copernican_lib import posterior as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.posterior")


if __name__ == "__main__":
    unittest.main()
