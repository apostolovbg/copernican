"""Utility helpers for loading Alien Invasion AI settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_SETTINGS: Dict[str, Any] = {
    "exploration_rate": 0.7,
    "learning_speed": 30,
    "run_duration_seconds": 300,
    "hidden_units": [40, 32, 24, 15, 12],
    "history_limit": 320,
    "time_pressure": {
        "base": 0.6,
        "scale": 0.4,
        "exponent": 0.5,
        "fallback": 0.8,
    },
    "kill_reward": {
        "base": 1.4,
        "general_bonus": 2.2,
        "increment": 0.35,
        "max_increment": 7,
    },
    "respawn_penalty": {
        "lieutenant": 0.05,
        "major": 0.1,
        "colonel": 0.2,
    },
    "edge_penalty_multiplier": 8.0,
    "edge_streak_scale": 3.0,
    "edge_streak_decay": 1.5,
    "initial_weights": {"aggression": 0.7, "caution": 0.3, "charge": 0.5},
    "win_bonus": {"aggression": 0.2, "charge": 0.15, "caution": -0.05},
    "loss_caution_cap": 1.1,
    "kill_time_bonus": {"multiplier": 2.5, "exponent": 1.5},
    "kill_drought_penalty": {"multiplier": 1.7, "kills": 1},
}
SETTINGS_PATH = Path(__file__).with_name("ai_settings.yml")


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override settings into the base dict recursively."""
    result: Dict[str, Any] = dict(base)
    for key, override_value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(override_value, dict)
        ):
            result[key] = _merge(result[key], override_value)
        else:
            result[key] = override_value
    return result


def _write_default_settings_file(path: Path) -> None:
    """Persist the default AI settings so users can edit them."""

    try:
        path.write_text(
            yaml.safe_dump(DEFAULT_SETTINGS, sort_keys=False).strip() + "\n"
        )
    except Exception:
        pass


def load_settings() -> Dict[str, Any]:
    """Read the AI settings YAML, merging it with sensible defaults."""

    path = SETTINGS_PATH
    if not path.exists():
        _write_default_settings_file(path)
        raw: Dict[str, Any] = {}
    else:
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except Exception:
            raw = {}
            _write_default_settings_file(path)
    return _merge(DEFAULT_SETTINGS, raw)
