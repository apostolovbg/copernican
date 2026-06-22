"""Smoke tests for copernican.rng_minigames.alien_invasion.ai_agent."""

import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from copernican.rng_minigames.alien_invasion import ai_agent as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__,
            "copernican.rng_minigames.alien_invasion.ai_agent",
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

    def test_alien_invasion_ai_persists_learning(self) -> None:
        """The AI brain should write its weights repeatedly."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with mock.patch.object(module.random, "random", return_value=0.5):
                brain = module.AlienInvasionAI(tmp_path)
                original_weights = brain.state["weights"].copy()
                original_network = copy.deepcopy(brain.state["network"])

                snapshot = {
                    "player_x": 100.0,
                    "player_y": 300.0,
                    "canvas_width": 400,
                    "enemies": [
                        {
                            "x": 90.0,
                            "y": 120.0,
                            "rank": "lieutenant",
                            "hp": 1,
                        },
                        {
                            "x": 170.0,
                            "y": 80.0,
                            "rank": "colonel",
                            "hp": 3,
                        },
                    ],
                    "incoming": [{"x": 110.0, "y": 40.0, "vy": 5.0}],
                    "charges": 1,
                    "player_shots": [{"x": 95.0}],
                }
                decision = brain.decide(snapshot)
                self.assertTrue({"move", "shoot", "charge"} <= decision.keys())

                brain.record_run(success=True, duration=7.5)
                reloaded = module.AlienInvasionAI(tmp_path)
                self.assertAlmostEqual(reloaded.state["best_time"], 7.5)
                self.assertGreaterEqual(reloaded.state["runs"], 1)
                self.assertGreaterEqual(reloaded.state["worlds_saved"], 1)
                self.assertGreater(
                    reloaded.state["weights"]["aggression"],
                    original_weights["aggression"],
                )
                self.assertNotEqual(
                    reloaded.state["network"]["weights"][0][0][0],
                    original_network["weights"][0][0][0],
                )
                self.assertEqual(
                    reloaded.state["network"]["weights"][0][0][0],
                    brain.state["network"]["weights"][0][0][0],
                )

                reloaded.record_run(success=False, duration=0)
                final = module.AlienInvasionAI(tmp_path)
                self.assertGreaterEqual(final.state["worlds_lost"], 1)
                self.assertGreaterEqual(
                    final.state["weights"]["caution"],
                    reloaded.state["weights"]["caution"],
                )

    def test_alien_invasion_ai_forget(self) -> None:
        """Reset the AI and restore the default weights."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            brain = module.AlienInvasionAI(tmp_path)
            brain.record_run(success=True, duration=4.2)
            self.assertGreaterEqual(brain.state["runs"], 1)
            previous_network = copy.deepcopy(brain.state["network"])
            brain.forget()
            self.assertEqual(brain.state["runs"], 0)
            self.assertIsNone(brain.state["best_time"])
            self.assertEqual(
                brain.state["weights"],
                {"aggression": 0.5, "caution": 0.5, "charge": 0.3},
            )
            self.assertNotEqual(
                brain.state["network"]["weights"][0][0][0],
                previous_network["weights"][0][0][0],
            )


if __name__ == "__main__":
    unittest.main()
