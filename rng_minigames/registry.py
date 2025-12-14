"""Registry helpers for discovering RNG mini-games."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

PACKAGE_ROOT = Path(__file__).resolve().parent
REGISTRY_FILE = PACKAGE_ROOT / "registry.json"


@dataclass
class MinigameDescriptor:
    """Describe an RNG mini-game."""

    id: str
    name: str
    module: str
    callable: str
    description: str
    metadata_hash: str
    module_hash: str


def _hash_file(path: Path) -> str:
    """Return the SHA256 digest for a metadata or module file."""

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _module_path(module: str) -> Path:
    """Resolve the on-disk path for a given RNG module identifier."""

    parts = module.split(".")
    if parts[0] == "rng_minigames":
        parts = parts[1:]
    candidate = PACKAGE_ROOT.joinpath(*parts)
    if candidate.with_suffix(".py").exists():
        return candidate.with_suffix(".py")
    return candidate / "__init__.py"


def _load_metadata(path: Path) -> Dict[str, Any]:
    """Load metadata JSON from the provided path."""

    return json.loads(path.read_text())


def _build_registry() -> List[Dict[str, Any]]:
    """Rebuild metadata entries for every RNG mini-game."""

    entries: List[Dict[str, Any]] = []
    for meta_path in PACKAGE_ROOT.glob("*/metadata.json"):
        data = _load_metadata(meta_path)
        module_path = _module_path(data["module"])
        entries.append(
            {
                "id": data["id"],
                "name": data["name"],
                "module": data["module"],
                "callable": data["callable"],
                "description": data.get("description", ""),
                "metadata_hash": _hash_file(meta_path),
                "module_hash": _hash_file(module_path),
            }
        )
    entries.sort(key=lambda entry: entry["id"])
    return entries


def refresh_registry() -> List[MinigameDescriptor]:
    """Rebuild the registry file after rehashing all metadata."""

    entries = _build_registry()
    REGISTRY_FILE.write_text(json.dumps(entries, indent=2))
    return [MinigameDescriptor(**entry) for entry in entries]


def load_registry() -> List[MinigameDescriptor]:
    """Load the cached registry or rebuild it when missing."""

    if not REGISTRY_FILE.exists():
        return refresh_registry()
    try:
        entries = json.loads(REGISTRY_FILE.read_text())
    except Exception:
        return refresh_registry()
    return [MinigameDescriptor(**entry) for entry in entries]


def get_descriptor(game_id: str) -> MinigameDescriptor | None:
    """Return the descriptor for ``game_id`` if present."""

    for entry in load_registry():
        if entry.id == game_id:
            return entry
    return None


def load_launcher(game_id: str) -> Callable[..., Any]:
    """Import (and reload) the launcher callable for ``game_id``."""

    descriptor = get_descriptor(game_id)
    if not descriptor:
        raise KeyError(game_id)
    module = importlib.import_module(descriptor.module)
    module = importlib.reload(module)
    try:
        return getattr(module, descriptor.callable)
    except AttributeError as exc:
        raise AttributeError(
            f"Mini-game {game_id} missing callable {descriptor.callable}"
        ) from exc
