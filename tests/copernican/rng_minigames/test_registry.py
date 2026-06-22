"""Smoke tests for copernican.rng_minigames.registry."""

from __future__ import annotations

import unittest

from copernican.rng_minigames import (
    load_launcher,
    load_registry,
    refresh_registry,
)
from copernican.rng_minigames import registry as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__,
            "copernican.rng_minigames.registry",
        )

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "MinigameDescriptor"))
        self.assertTrue(hasattr(module, "get_descriptor"))
        self.assertTrue(hasattr(load_registry, "__call__"))
        self.assertTrue(hasattr(refresh_registry, "__call__"))
        self.assertTrue(hasattr(load_launcher, "__call__"))

    def test_registry_refresh_roundtrip(self) -> None:
        """Refreshing the registry should expose launchers for each game."""

        entries = refresh_registry()
        ids = {entry.game_id for entry in entries}
        self.assertTrue(
            {"emoji_meteors", "constellation", "alien_invasion"} <= ids
        )
        for descriptor in load_registry():
            launcher = load_launcher(descriptor.game_id)
            self.assertTrue(callable(launcher))


if __name__ == "__main__":
    unittest.main()
