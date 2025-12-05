"""Utility helpers for loading Alien Invasion AI settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_SETTINGS: Dict[str, Any] = {
    "exploration_rate": 0.9,
    "learning_speed": 10,
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
        "base": 0.7,
        "general_bonus": 1.5,
        "increment": 0.15,
        "max_increment": 4,
    },
    "respawn_penalty": {
        "lieutenant": 0.3,
        "major": 0.5,
        "colonel": 0.8,
    },
}
SETTINGS_PATH = Path(__file__).with_name("ai_settings.yml")


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
