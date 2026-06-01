"""Smoke tests for copernican.lib.console_output."""

import unittest

from copernican.lib import console_output as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.console_output")

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "ask"))
        self.assertTrue(hasattr(module, "write"))


if __name__ == "__main__":
    unittest.main()
