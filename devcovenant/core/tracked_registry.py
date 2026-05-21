"""Tracked registry path and persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from devcovenant.core.repository_paths import display_path, load_yaml

DEV_COVENANT_DIR = "devcovenant"
REGISTRY_DIR = f"{DEV_COVENANT_DIR}/registry"
REGISTRY_FILENAME = "registry.yaml"
REGISTRY_REL_PATH = f"{REGISTRY_DIR}/{REGISTRY_FILENAME}"


def registry_root(repo_root: Path) -> Path:
    """Return the path to the tracked registry root directory."""
    return repo_root / REGISTRY_DIR


def policy_registry_path(repo_root: Path) -> Path:
    """Return the tracked registry document path."""
    return registry_root(repo_root) / REGISTRY_FILENAME


def profile_registry_path(repo_root: Path) -> Path:
    """Return the tracked registry document path for profile data."""
    return policy_registry_path(repo_root)


def base_registry_document() -> Dict[str, Any]:
    """Return the canonical top-level tracked-registry document skeleton."""
    return {
        "metadata": {
            "schema_version": 1,
            "registry_layout": "single-root",
        },
        "project-governance": {},
        "managed-docs": {},
        "workflow_contract": {},
        "policies": {},
        "profiles": {},
        "inventory": {},
    }


def load_registry_document(path: Path) -> Dict[str, Any]:
    """Load one tracked-registry YAML mapping or return the base skeleton."""
    rendered = display_path(path)
    if not path.exists():
        return base_registry_document()
    try:
        payload = load_yaml(path)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML in registry file {rendered}: {exc}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Unable to read registry file {rendered}: {exc}"
        ) from exc
    if payload is None:
        return base_registry_document()
    if not isinstance(payload, dict):
        raise ValueError(
            "Registry payload must be a YAML mapping: " f"{rendered}"
        )
    normalized = base_registry_document()
    for key in (
        "metadata",
        "project-governance",
        "managed-docs",
        "workflow_contract",
        "policies",
        "profiles",
        "inventory",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            normalized[key] = value
    return normalized


class _TrackedRegistryYamlDumper(
    getattr(yaml, "CSafeDumper", yaml.SafeDumper)
):
    """YAML dumper for tracked-registry files with readable strings."""


def _represent_registry_string(
    dumper: yaml.Dumper, text_value: str
) -> yaml.nodes.ScalarNode:
    """Render multiline strings as literal blocks."""
    style = "|" if "\n" in text_value else None
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", text_value, style=style
    )


_TrackedRegistryYamlDumper.add_representer(str, _represent_registry_string)


def write_registry_document(path: Path, payload: Dict[str, Any]) -> Path:
    """Persist one normalized tracked-registry mapping deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.dump(
        payload,
        Dumper=_TrackedRegistryYamlDumper,
        sort_keys=False,
        allow_unicode=False,
    )
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == rendered:
            return path
    path.write_text(rendered, encoding="utf-8")
    return path
