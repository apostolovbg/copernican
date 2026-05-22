"""Smoke tests for rng_minigames.emoji_meteors.game."""

import unittest

from rng_minigames.emoji_meteors import game as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "rng_minigames.emoji_meteors.game")


if __name__ == "__main__":
    unittest.main()
