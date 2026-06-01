"""Smoke tests for copernican.rng_minigames.alien_invasion.game."""

import inspect
import unittest

from copernican.rng_minigames.alien_invasion import game as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__,
            "copernican.rng_minigames.alien_invasion.game",
        )


class TestPublicSymbols(unittest.TestCase):
    """Assert the module exposes the expected public symbols."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(module.launch_alien_invasion))
        source = inspect.getsource(module.launch_alien_invasion)
        self.assertIn("class AIPilotController", source)
        self.assertIn("def start(", source)
        self.assertIn("def stop(", source)


if __name__ == "__main__":
    unittest.main()
