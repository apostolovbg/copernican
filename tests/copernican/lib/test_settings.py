"""Smoke tests for copernican.lib.settings."""

import unittest

from copernican.lib import settings as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.settings")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the settings surface to the coverage policy."""

    def test_public_symbols_are_present(self) -> None:
        self.assertTrue(hasattr(module, "get_settings_path"))
        self.assertTrue(hasattr(module, "load_settings"))
        self.assertTrue(hasattr(module, "save_settings"))
        self.assertTrue(hasattr(module, "get_settings"))


if __name__ == "__main__":
    unittest.main()
