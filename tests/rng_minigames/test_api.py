"""Smoke tests for rng_minigames.api."""

import unittest

from rng_minigames import api as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "rng_minigames.api")


if __name__ == "__main__":
    unittest.main()
