"""Smoke tests for rng_minigames.alien_invasion.ai_agent."""

import inspect
import unittest

from rng_minigames.alien_invasion import ai_agent as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__, "rng_minigames.alien_invasion.ai_agent"
        )


class TestPublicSymbols(unittest.TestCase):
    """Assert the module exposes the expected public symbols."""

    def test_public_symbols_are_exposed(self) -> None:
        source = inspect.getsource(module.AlienInvasionAI)
        self.assertTrue(callable(module.AlienInvasionAI))
        self.assertIn("def decide(", source)
        self.assertIn("def begin_run(", source)
        self.assertIn("def reward_enemy_destroyed(", source)
        self.assertIn("def penalize_enemy_respawned(", source)
        self.assertIn("def penalize_edge(", source)
        self.assertIn("def cool_edge_streak(", source)
        self.assertIn("def record_run(", source)
        self.assertIn("def forget(", source)


if __name__ == "__main__":
    unittest.main()
