"""Smoke tests for copernican.rng_minigames.alien_invasion.ai_config."""

import unittest

from copernican.rng_minigames.alien_invasion import ai_config as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__,
            "copernican.rng_minigames.alien_invasion.ai_config",
        )

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "load_settings"))


if __name__ == "__main__":
    unittest.main()
