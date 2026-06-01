"""Tests for the RNG mini-games package."""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from copernican.rng_minigames import api as api_module
from copernican.rng_minigames import (
    load_launcher,
    load_registry,
    refresh_registry,
)
from copernican.rng_minigames import registry as registry_module
from copernican.rng_minigames.alien_invasion import ai_agent, hall_of_fame


class RNGMiniGamesTestCase(unittest.TestCase):
    """Exercise the package API and the Alien Invasion helpers."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(api_module, "MinigameContext"))
        self.assertTrue(hasattr(hall_of_fame, "HallOfFame"))
        self.assertTrue(hasattr(ai_agent, "AlienInvasionAI"))
        self.assertTrue(hasattr(registry_module, "MinigameDescriptor"))
        self.assertTrue(hasattr(registry_module, "get_descriptor"))
        self.assertTrue(hasattr(load_registry, "__call__"))
        self.assertTrue(hasattr(refresh_registry, "__call__"))
        self.assertTrue(hasattr(load_launcher, "__call__"))

    def test_alien_invasion_ai_persists_learning(self) -> None:
        """The AI brain should write its weights repeatedly."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with mock.patch.object(
                ai_agent.random, "random", return_value=0.5
            ):
                brain = ai_agent.AlienInvasionAI(tmp_path)
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
                reloaded = ai_agent.AlienInvasionAI(tmp_path)
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

                # A failed run should increase the caution weight and
                # persist it.
                reloaded.record_run(success=False, duration=0)
                final = ai_agent.AlienInvasionAI(tmp_path)
                self.assertGreaterEqual(final.state["worlds_lost"], 1)
                self.assertGreaterEqual(
                    final.state["weights"]["caution"],
                    reloaded.state["weights"]["caution"],
                )

    def test_alien_invasion_ai_forget(self) -> None:
        """Reset the AI and restore the default weights."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            brain = ai_agent.AlienInvasionAI(tmp_path)
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

    def test_hall_of_fame_sorts_and_limits_entries(self) -> None:
        """The hall of fame should keep the fastest runs and persist them."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            board = hall_of_fame.HallOfFame(tmp_path, limit=3)
            board.record("NI", 15.0)
            board.record("AI", 6.0)
            board.record("NI", 9.0)
            board.record("NI", 30.0)

            restored = hall_of_fame.HallOfFame(tmp_path, limit=3)
            times_left = [entry["time_left"] for entry in restored.entries]
            initials = [entry["initials"] for entry in restored.entries]

            self.assertEqual(times_left, sorted(times_left, reverse=True))
            self.assertEqual(initials[0], "NI")
            self.assertEqual(len(times_left), 3)
            self.assertTrue(math.isclose(max(times_left), 30.0, abs_tol=0.01))

    def test_registry_refresh_roundtrip(self) -> None:
        """Refreshing the registry should produce descriptors for each game."""

        entries = refresh_registry()
        ids = {entry.game_id for entry in entries}
        self.assertTrue(
            {"emoji_meteors", "constellation", "alien_invasion"} <= ids
        )
        for descriptor in load_registry():
            launcher = load_launcher(descriptor.game_id)
            self.assertTrue(callable(launcher))
