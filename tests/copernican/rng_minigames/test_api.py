"""Smoke tests for copernican.rng_minigames.api."""

import unittest

from copernican.rng_minigames import api as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.rng_minigames.api")

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "MinigameContext"))


if __name__ == "__main__":
    unittest.main()
