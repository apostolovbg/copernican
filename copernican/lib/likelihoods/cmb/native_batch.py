"""Ordered native CMB batch results and stable diagnostic serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy

from .native_errors import NativeCMBError


def _jsonable(value: Any) -> Any:
    """Convert native diagnostics and spectra into JSON-compatible values."""

    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, NativeCMBError):
        return value.diagnostic()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=str)]
    return value


@dataclass(frozen=True, slots=True)
class NativeCMBBatchResult:
    """Store one ordered native CMB batch outcome and its provenance."""

    index: int
    spectrum: numpy.ndarray | Mapping[str, numpy.ndarray] | None = None
    failure: NativeCMBError | None = None
    performance_envelope: Mapping[str, Any] = field(default_factory=dict)
    cache_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate one-and-only-one success or typed failure outcome."""

        if int(self.index) < 0:
            raise ValueError("Native CMB batch indices must be non-negative")
        has_spectrum = self.spectrum is not None
        has_failure = self.failure is not None
        if has_spectrum == has_failure:
            raise ValueError(
                "Native CMB batch results require a spectrum or typed failure"
            )
        if self.failure is not None and not isinstance(
            self.failure, NativeCMBError
        ):
            raise TypeError("Native CMB batch failures must be typed errors")
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(
            self,
            "performance_envelope",
            dict(self.performance_envelope or {}),
        )
        object.__setattr__(
            self,
            "cache_provenance",
            dict(self.cache_provenance or {}),
        )

    @property
    def success(self) -> bool:
        """Return whether this item contains a spectrum, not a failure."""

        return self.spectrum is not None

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible batch provenance."""

        return {
            "cache_provenance": _jsonable(self.cache_provenance),
            "failure": (
                None if self.failure is None else _jsonable(self.failure)
            ),
            "index": self.index,
            "performance_envelope": _jsonable(self.performance_envelope),
            "spectrum": (
                None if self.spectrum is None else _jsonable(self.spectrum)
            ),
            "success": self.success,
        }


__all__ = ["NativeCMBBatchResult"]
