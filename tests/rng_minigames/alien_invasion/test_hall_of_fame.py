"""Smoke tests for rng_minigames.alien_invasion.hall_of_fame."""

import unittest

from rng_minigames.alien_invasion import hall_of_fame as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__, "rng_minigames.alien_invasion.hall_of_fame"
        )


if __name__ == "__main__":
    unittest.main()
