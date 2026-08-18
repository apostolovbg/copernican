"""Smoke tests for copernican.lib.sampler_capabilities."""

import unittest

from copernican.lib import sampler_capabilities as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__, "copernican.lib.sampler_capabilities"
        )

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "SamplerSetting"))
        self.assertTrue(hasattr(module, "SamplerProgressChunk"))
        self.assertTrue(hasattr(module, "SamplerCapabilities"))
        self.assertTrue(hasattr(module, "get_sampler_capabilities"))


if __name__ == "__main__":
    unittest.main()
