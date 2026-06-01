"""Smoke tests for copernican.lib.engine_capabilities."""

import unittest

from copernican.lib import engine_capabilities as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.engine_capabilities")

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "EngineSetting"))
        self.assertTrue(hasattr(module, "EngineProgressChunk"))
        self.assertTrue(hasattr(module, "EngineCapabilities"))
        self.assertTrue(hasattr(module, "get_engine_capabilities"))


if __name__ == "__main__":
    unittest.main()
