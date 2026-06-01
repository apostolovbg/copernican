"""Smoke tests for copernican.rng_minigames.alien_invasion.hall_of_fame."""

import unittest

from copernican.rng_minigames.alien_invasion import hall_of_fame as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__,
            "copernican.rng_minigames.alien_invasion.hall_of_fame",
        )

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "HallOfFame"))
        self.assertTrue(hasattr(module.HallOfFame, "record"))
        self.assertTrue(hasattr(module.HallOfFame, "show"))


if __name__ == "__main__":
    unittest.main()
