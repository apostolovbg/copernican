"""Sampler capability descriptors shared by GUI, CLI and orchestration
helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

MAX_SAMPLER_SETTINGS = 16
MAX_PROGRESS_CHUNKS = 3
_ALLOWED_TYPES = {"int", "float", "str", "bool"}


@dataclass(frozen=True)
class SamplerSetting:
    """Describe a single adjustable knob exposed by a sampler."""

    key: str
    label: str
    description: str = ""
    dtype: str = "str"
    default: Any | None = None
    hint: str | None = None

    def __post_init__(self) -> None:
        """Validate that dtype belongs to the supported set."""
        dtype = self.dtype.lower()
        if dtype not in _ALLOWED_TYPES:
            raise ValueError(
                "SamplerSetting dtype must be one of "
                f"{sorted(_ALLOWED_TYPES)}; received {self.dtype!r}"
            )


@dataclass(frozen=True)
class SamplerProgressChunk:
    """Describe a named progress chunk that samplers emit."""

    name: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class SamplerCapabilities:
    """Aggregate a sampler's settings and progress chunk descriptors."""

    settings: Tuple[SamplerSetting, ...] = ()
    progress_chunks: Tuple[SamplerProgressChunk, ...] = ()


def _ensure_limit(
    collection: Sequence[Any], *, limit: int, label: str
) -> None:
    """Raise when ``collection`` exceeds the specified ``limit``."""
    if len(collection) > limit:
        raise ValueError(
            f"{label} must contain at most {limit} entries; "
            f"detected {len(collection)}"
        )


def _normalize_setting(
    candidate_setting: SamplerSetting | Mapping[str, Any],
) -> SamplerSetting:
    """Normalize an entry into a canonical :class:`SamplerSetting`."""
    if isinstance(candidate_setting, SamplerSetting):
        return candidate_setting
    if isinstance(candidate_setting, Mapping):
        return SamplerSetting(
            key=candidate_setting["key"],
            label=candidate_setting["label"],
            description=candidate_setting.get("description", ""),
            dtype=candidate_setting.get(
                "dtype", candidate_setting.get("type", "str")
            ),
            default=candidate_setting.get("default"),
            hint=candidate_setting.get("hint"),
        )
    raise TypeError(
        "SAMPLER_SETTINGS entries must be SamplerSetting or mapping"
    )


def _normalize_chunk(
    candidate_chunk: SamplerProgressChunk | Mapping[str, Any],
) -> SamplerProgressChunk:
    """Normalize a configuration entry into a progress chunk."""
    if isinstance(candidate_chunk, SamplerProgressChunk):
        return candidate_chunk
    if isinstance(candidate_chunk, Mapping):
        return SamplerProgressChunk(
            name=candidate_chunk["name"],
            label=candidate_chunk["label"],
            description=candidate_chunk.get("description", ""),
        )
    raise TypeError("SAMPLER_PROGRESS_CHUNKS entries must be chunk or mapping")


def get_sampler_capabilities(module: object) -> SamplerCapabilities:
    """Return a module's declared settings and progress chunks."""

    raw_settings = getattr(module, "SAMPLER_SETTINGS", ()) or ()
    normalized_settings = tuple(
        _normalize_setting(entry) for entry in raw_settings
    )
    _ensure_limit(
        normalized_settings,
        limit=MAX_SAMPLER_SETTINGS,
        label="SAMPLER_SETTINGS",
    )
    raw_chunks = getattr(module, "SAMPLER_PROGRESS_CHUNKS", ()) or ()
    normalized_chunks = tuple(_normalize_chunk(entry) for entry in raw_chunks)
    _ensure_limit(
        normalized_chunks,
        limit=MAX_PROGRESS_CHUNKS,
        label="SAMPLER_PROGRESS_CHUNKS",
    )
    return SamplerCapabilities(
        settings=normalized_settings,
        progress_chunks=normalized_chunks,
    )


__all__ = [
    "SamplerCapabilities",
    "SamplerProgressChunk",
    "SamplerSetting",
    "MAX_SAMPLER_SETTINGS",
    "MAX_PROGRESS_CHUNKS",
    "get_sampler_capabilities",
]
