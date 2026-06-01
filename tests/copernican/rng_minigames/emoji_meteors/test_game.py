"""Smoke tests for copernican.rng_minigames.emoji_meteors.game."""

import unittest

from copernican.rng_minigames.emoji_meteors import game as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__, "copernican.rng_minigames.emoji_meteors.game"
        )


class TestPublicSymbols(unittest.TestCase):
    """Assert the module exposes the expected public symbols."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(module.launch_emoji_meteors))


if __name__ == "__main__":
    unittest.main()
