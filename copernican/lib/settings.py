"""Helper for loading and storing Copernican settings."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

SETTINGS_ENV_VAR = "COPERNICAN_SETTINGS_PATH"
DEFAULT_SETTINGS_FILE = "defaults.yml"
DEFAULT_SETTINGS_DIR = Path(__file__).resolve().parent / "global_settings"
DEFAULT_SETTINGS_PATH = DEFAULT_SETTINGS_DIR / DEFAULT_SETTINGS_FILE
USER_SETTINGS_FILE = "copernican_settings.yml"

DEFAULT_SETTINGS_TEMPLATE: dict[str, dict[str, object]] = {
    "datasets": {
        "auto_dataset_discovery": True,
        "dataset_hash_auto_rebuild": False,
        "dataset_hash_ttl_hours": 24,
    },
    "gui": {
        "require_managed_venv": True,
        "show_environment_hints": True,
    },
    "tools": {
        "rebuild_model_cache_on_start": False,
        "revalidate_parsers_before_run": False,
    },
}


def _merge_settings(
    base: dict[str, object], override: dict[str, object]
) -> dict[str, object]:
    """Recursively apply ``override`` values on top of ``base``."""

    merged = copy.deepcopy(base)
    for key, override_value in override.items():
        if isinstance(override_value, dict) and isinstance(
            merged.get(key), dict
        ):
            merged[key] = _merge_settings(
                merged[key].copy(),
                override_value,  # type: ignore[arg-type]
            )
        else:
            merged[key] = override_value
    return merged


def _load_packaged_defaults() -> dict[str, dict[str, object]]:
    """Load the shipped defaults file or fall back to the template."""

    if not DEFAULT_SETTINGS_PATH.exists():
        return copy.deepcopy(DEFAULT_SETTINGS_TEMPLATE)
    raw = yaml.safe_load(
        DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8") or "{}"
    )
    if not isinstance(raw, dict):
        raw = {}
    settings: dict[str, dict[str, object]] = {}
    for section, defaults in DEFAULT_SETTINGS_TEMPLATE.items():
        section_override = raw.get(section, {})
        if not isinstance(section_override, dict):
            section_override = {}
        settings[section] = _merge_settings(defaults, section_override)
    return settings


DEFAULT_SETTINGS = _load_packaged_defaults()


def _get_user_config_dir() -> Path:
    """Return the platform config directory for persistent settings."""

    if os.name == "nt":
        for env_var in ("APPDATA", "LOCALAPPDATA"):
            env_path = os.environ.get(env_var)
            if env_path:
                return Path(env_path) / "copernican"
        return Path.home() / "AppData" / "Roaming" / "copernican"
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "copernican"
    return Path.home() / ".config" / "copernican"


def get_settings_path() -> Path:
    """Return the path where the user settings file lives."""

    env_path = os.environ.get(SETTINGS_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    return _get_user_config_dir() / USER_SETTINGS_FILE


def load_settings() -> dict[str, dict[str, object]]:
    """Load user settings and fall back to the packaged defaults."""

    path = get_settings_path()
    if not path.exists():
        return copy.deepcopy(DEFAULT_SETTINGS)
    raw = yaml.safe_load(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(raw, dict):
        raw = {}
    settings: dict[str, dict[str, object]] = {}
    for section, defaults in DEFAULT_SETTINGS.items():
        section_override = raw.get(section, {})
        if not isinstance(section_override, dict):
            section_override = {}
        settings[section] = _merge_settings(defaults, section_override)
    return settings


def save_settings(settings: dict[str, dict[str, object]]) -> None:
    """Persist user settings to the platform config directory."""

    path = get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(
            settings, file_handle, sort_keys=False, default_flow_style=False
        )
    global _SETTINGS_CACHE
    _SETTINGS_CACHE = copy.deepcopy(settings)


_SETTINGS_CACHE: dict[str, dict[str, object]] | None = None


def get_settings() -> dict[str, dict[str, object]]:
    """Return the cached settings, loading them if necessary."""

    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None:
        _SETTINGS_CACHE = load_settings()
    return copy.deepcopy(_SETTINGS_CACHE)
