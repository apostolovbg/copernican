"""Engine capability descriptors shared by GUI, CLI and orchestration
helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

MAX_ENGINE_SETTINGS = 16
MAX_PROGRESS_CHUNKS = 3
_ALLOWED_TYPES = {"int", "float", "str", "bool"}


@dataclass(frozen=True)
class EngineSetting:
    """Describe a single adjustable knob exposed by an engine."""

    key: str
    label: str
    description: str = ""
    dtype: str = "str"
    default: Any | None = None
    hint: str | None = None

    def __post_init__(self) -> None:
        dtype = self.dtype.lower()
        if dtype not in _ALLOWED_TYPES:
            raise ValueError(
                "EngineSetting dtype must be one of "
                f"{sorted(_ALLOWED_TYPES)}; received {self.dtype!r}"
            )


@dataclass(frozen=True)
class EngineProgressChunk:
    """Describe a named progress chunk that engines emit."""

    name: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class EngineCapabilities:
    """Aggregate an engine's settings and progress chunk descriptors."""

    settings: Tuple[EngineSetting, ...] = ()
    progress_chunks: Tuple[EngineProgressChunk, ...] = ()


def _ensure_limit(
    collection: Sequence[Any], *, limit: int, label: str
) -> None:
    if len(collection) > limit:
        raise ValueError(
            f"{label} must contain at most {limit} entries; "
            f"detected {len(collection)}"
        )


def _normalize_setting(
    value: EngineSetting | Mapping[str, Any]
) -> EngineSetting:
    if isinstance(value, EngineSetting):
        return value
    if isinstance(value, Mapping):
        return EngineSetting(
            key=value["key"],
            label=value["label"],
            description=value.get("description", ""),
            dtype=value.get("dtype", value.get("type", "str")),
            default=value.get("default"),
            hint=value.get("hint"),
        )
    raise TypeError("ENGINE_SETTINGS entries must be EngineSetting or mapping")


def _normalize_chunk(
    value: EngineProgressChunk | Mapping[str, Any]
) -> EngineProgressChunk:
    if isinstance(value, EngineProgressChunk):
        return value
    if isinstance(value, Mapping):
        return EngineProgressChunk(
            name=value["name"],
            label=value["label"],
            description=value.get("description", ""),
        )
    raise TypeError("ENGINE_PROGRESS_CHUNKS entries must be chunk or mapping")


def get_engine_capabilities(module: object) -> EngineCapabilities:
    """Return a module's declared settings and progress chunks."""

    raw_settings = getattr(module, "ENGINE_SETTINGS", ()) or ()
    normalized_settings = tuple(
        _normalize_setting(entry) for entry in raw_settings
    )
    _ensure_limit(
        normalized_settings,
        limit=MAX_ENGINE_SETTINGS,
        label="ENGINE_SETTINGS",
    )
    raw_chunks = getattr(module, "ENGINE_PROGRESS_CHUNKS", ()) or ()
    normalized_chunks = tuple(_normalize_chunk(entry) for entry in raw_chunks)
    _ensure_limit(
        normalized_chunks,
        limit=MAX_PROGRESS_CHUNKS,
        label="ENGINE_PROGRESS_CHUNKS",
    )
    return EngineCapabilities(
        settings=normalized_settings,
        progress_chunks=normalized_chunks,
    )


__all__ = [
    "EngineCapabilities",
    "EngineProgressChunk",
    "EngineSetting",
    "MAX_ENGINE_SETTINGS",
    "MAX_PROGRESS_CHUNKS",
    "get_engine_capabilities",
]
