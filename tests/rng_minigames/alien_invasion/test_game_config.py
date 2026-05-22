"""Smoke tests for rng_minigames.alien_invasion.game_config."""

import unittest

from rng_minigames.alien_invasion import game_config as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__, "rng_minigames.alien_invasion.game_config"
        )


if __name__ == "__main__":
    unittest.main()
