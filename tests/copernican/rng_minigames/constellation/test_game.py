"""Smoke tests for copernican.rng_minigames.constellation.game."""

import unittest

from copernican.rng_minigames.constellation import game as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__, "copernican.rng_minigames.constellation.game"
        )


class TestPublicSymbols(unittest.TestCase):
    """Assert the module exposes the expected public symbols."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(module.launch_constellation))


if __name__ == "__main__":
    unittest.main()
