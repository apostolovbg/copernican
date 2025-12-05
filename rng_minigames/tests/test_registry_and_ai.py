"""Tests for the RNG mini-games package."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from rng_minigames.alien_invasion import ai_agent, hall_of_fame


def test_alien_invasion_ai_persists_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The AI brain should write its weights and reload them on the next run."""

    # Keep the decision path deterministic by disabling the exploration branch.
    monkeypatch.setattr(ai_agent.random, "random", lambda: 0.5)
    brain = ai_agent.AlienInvasionAI(tmp_path)
    original_weights = brain.state["weights"].copy()
    original_network = copy.deepcopy(brain.state["network"])

    snapshot = {
        "player_x": 100.0,
        "player_y": 300.0,
        "canvas_width": 400,
        "enemies": [
            {"x": 90.0, "y": 120.0, "rank": "lieutenant", "hp": 1},
            {"x": 170.0, "y": 80.0, "rank": "colonel", "hp": 3},
        ],
        "incoming": [{"x": 110.0, "y": 40.0, "vy": 5.0}],
        "charges": 1,
        "player_shots": [{"x": 95.0}],
    }
    decision = brain.decide(snapshot)
    assert {"move", "shoot", "charge"} <= decision.keys()

    brain.record_run(success=True, duration=7.5)
    reloaded = ai_agent.AlienInvasionAI(tmp_path)
    assert reloaded.state["best_time"] == pytest.approx(7.5)
    assert reloaded.state["runs"] >= 1
    assert reloaded.state["worlds_saved"] >= 1
    assert (
        reloaded.state["weights"]["aggression"]
        > original_weights["aggression"]
    )
    assert (
        reloaded.state["network"]["weights"][0][0][0]
        != original_network["weights"][0][0][0]
    )
    assert (
        reloaded.state["network"]["weights"][0][0][0]
        == brain.state["network"]["weights"][0][0][0]
    )

    # A failed run should increase the caution weight and persist it.
    reloaded.record_run(success=False, duration=0)
    final = ai_agent.AlienInvasionAI(tmp_path)
    assert final.state["worlds_lost"] >= 1
    assert (
        final.state["weights"]["caution"]
        >= reloaded.state["weights"]["caution"]
    )


def test_alien_invasion_ai_forget(tmp_path: Path) -> None:
    """Resetting the AI should wipe progress and restore default weights."""

    brain = ai_agent.AlienInvasionAI(tmp_path)
    brain.record_run(success=True, duration=4.2)
    assert brain.state["runs"] >= 1
    previous_network = copy.deepcopy(brain.state["network"])
    brain.forget()
    assert brain.state["runs"] == 0
    assert brain.state["best_time"] is None
    assert brain.state["weights"] == {
        "aggression": 0.5,
        "caution": 0.5,
        "charge": 0.3,
    }
    assert (
        brain.state["network"]["weights"][0][0][0]
        != previous_network["weights"][0][0][0]
    )


def test_hall_of_fame_sorts_and_limits_entries(tmp_path: Path) -> None:
    """The hall of fame should keep the fastest runs and persist them."""

    board = hall_of_fame.HallOfFame(tmp_path, limit=3)
    board.record("NI", 15.0)
    board.record("AI", 6.0)
    board.record("NI", 9.0)

    # Slower runs should be discarded once the limit is reached.
    board.record("NI", 30.0)

    restored = hall_of_fame.HallOfFame(tmp_path, limit=3)
    times_left = [entry["time_left"] for entry in restored.entries]
    initials = [entry["initials"] for entry in restored.entries]

    assert times_left == sorted(times_left, reverse=True)
    assert initials[0] == "NI"
    assert len(times_left) == 3
    assert math.isclose(max(times_left), 30.0, rel_tol=0, abs_tol=0.01)


def test_registry_refresh_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refreshing the registry should produce descriptors for each game."""

    from rng_minigames import load_launcher, load_registry, refresh_registry

    entries = refresh_registry()
    ids = {entry.id for entry in entries}
    assert {"emoji_meteors", "constellation", "alien_invasion"} <= ids
    for descriptor in load_registry():
        launcher = load_launcher(descriptor.id)
        assert callable(launcher)
