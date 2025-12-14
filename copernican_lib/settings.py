"""Helper for loading/storing Copernican settings."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

SETTINGS_ENV_VAR = "COPERNICAN_SETTINGS_PATH"
DEFAULT_SETTINGS_FILE = "copernican_settings.yml"

DEFAULT_SETTINGS: dict[str, dict[str, object]] = {
    "logs": {
        "log_retention_count": 10,
        "capture_console": True,
        "log_level": "INFO",
    },
    "datasets": {
        "auto_dataset_discovery": True,
        "dataset_hash_auto_rebuild": False,
        "dataset_hash_ttl_hours": 24,
    },
    "gui": {
        "detach_gui": True,
        "require_managed_venv": True,
        "show_environment_hints": True,
    },
    "tools": {
        "rebuild_model_cache_on_start": False,
        "revalidate_parsers_before_run": False,
    },
}


_SETTINGS_CACHE: dict[str, dict[str, object]] | None = None


def get_settings_path() -> Path:
    """Return the path where the settings file lives."""

    env_path = os.environ.get(SETTINGS_ENV_VAR)
    if env_path:
        return Path(env_path)
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / DEFAULT_SETTINGS_FILE


def _merge_settings(
    base: dict[str, object], override: dict[str, object]
) -> dict[str, object]:
    """Recursively apply ``override`` values on top of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_settings(
                merged[key].copy(), value  # type: ignore[arg-type]
            )
        else:
            merged[key] = value
    return merged


def load_settings() -> dict[str, dict[str, object]]:
    """Load settings from disk, creating defaults if missing."""

    path = get_settings_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                DEFAULT_SETTINGS,
                fh,
                sort_keys=False,
                default_flow_style=False,
            )
        return copy.deepcopy(DEFAULT_SETTINGS)
    raw = yaml.safe_load(path.read_text(encoding="utf-8") or "{}") or {}
    settings: dict[str, dict[str, object]] = {}
    for section, defaults in DEFAULT_SETTINGS.items():
        section_override = raw.get(section, {})
        settings[section] = _merge_settings(defaults, section_override)
    return settings


def save_settings(settings: dict[str, dict[str, object]]) -> None:
    """Persist settings back to disk."""

    path = get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(settings, fh, sort_keys=False, default_flow_style=False)
    global _SETTINGS_CACHE
    _SETTINGS_CACHE = copy.deepcopy(settings)


def get_settings() -> dict[str, dict[str, object]]:
    """Return the cached settings, loading them if necessary."""

    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None:
        _SETTINGS_CACHE = load_settings()
    return copy.deepcopy(_SETTINGS_CACHE)
