"""Smoke tests for copernican.lib.error_handler."""

import unittest

from copernican.lib import error_handler as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.error_handler")

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "configure_warnings"))
        self.assertTrue(hasattr(module, "report_error"))


if __name__ == "__main__":
    unittest.main()
