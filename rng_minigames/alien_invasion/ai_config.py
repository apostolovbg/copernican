"""Utility helpers for loading Alien Invasion AI settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_SETTINGS: Dict[str, Any] = {
    "exploration_rate": 0.75,
    "learning_speed": 10,
    "run_duration_seconds": 300,
    "time_pressure": {
        "base": 0.6,
        "scale": 0.4,
        "exponent": 0.5,
        "fallback": 0.8,
    },
    "kill_reward": {
        "base": 0.7,
        "general_bonus": 1.5,
        "increment": 0.08,
        "max_increment": 2.5,
    },
    "respawn_penalty": {
        "lieutenant": 0.25,
        "major": 0.35,
        "colonel": 0.45,
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings() -> Dict[str, Any]:
    """Read the AI settings YAML, merging it with sensible defaults."""

    path = Path(__file__).with_name("ai_settings.yml")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raw = {}
    except Exception:
        raw = {}
    return _merge(DEFAULT_SETTINGS, raw)

