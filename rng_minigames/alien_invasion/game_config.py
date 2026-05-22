"""Game settings loader for Alien Invasion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_SETTINGS: Dict[str, Any] = {
    "player": {"shield": 50},
    "general": {"shield": 20, "max_speed": 7.0},
    "charges": {"capacity": 3},
    "player_motion": {
        "max_speed": 14.0,
        "accel": 0.45,
        "decel": 0.4,
        "snap_error": 1.2,
    },
    "explosion": {
        "shard_count": 90,
        "frame_ms": 40,
        "violence_scale": 1.0,
    },
    "player_explosion": {
        "hold_seconds": 5.0,
    },
    "debris": {
        "count": 14,
        "damages_all": False,
    },
}
_STORAGE_DIR = Path(__file__).resolve().parent / "_storage"
_SETTINGS_NAME = "game_settings.yml"


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override settings into the base dictionary recursively."""
    result = dict(base)
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


def _write_default_settings(path: Path) -> None:
    """Write the defaults to disk so the YAML loader always finds a file."""
    try:
        path.write_text(
            yaml.safe_dump(DEFAULT_SETTINGS, sort_keys=False).strip() + "\n"
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        pass


def load_settings() -> Dict[str, Any]:
    """Read game settings YAML, falling back to built-in defaults."""

    _STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = _STORAGE_DIR / _SETTINGS_NAME
    try:
        disk_settings = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        disk_settings = {}
        _write_default_settings(path)
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        disk_settings = {}
        _write_default_settings(path)
    if not path.exists():
        _write_default_settings(path)
    return _merge(DEFAULT_SETTINGS, disk_settings)
