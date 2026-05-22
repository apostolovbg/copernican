"""Smoke tests for copernican_lib.error_handler."""

import unittest

from copernican_lib import error_handler as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.error_handler")


if __name__ == "__main__":
    unittest.main()
