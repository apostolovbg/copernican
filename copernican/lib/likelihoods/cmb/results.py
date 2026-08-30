"""Ordered declared CMB batch results and stable diagnostic serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy

from .errors import CMBError


def _jsonable(value: Any) -> Any:
    """Convert declared diagnostics and spectra into JSON-compatible values."""

    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, CMBError):
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
class CMBBatchResult:
    """Store one ordered declared CMB batch outcome and its provenance."""

    index: int
    spectrum: numpy.ndarray | Mapping[str, numpy.ndarray] | None = None
    failure: CMBError | None = None
    performance_envelope: Mapping[str, Any] = field(default_factory=dict)
    cache_provenance: Mapping[str, Any] = field(default_factory=dict)
    requested_ells: tuple[int, ...] = ()
    requested_spectra: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    phase_timings: Mapping[str, float] = field(default_factory=dict)
    solver_id: str = ""
    solver_label: str = ""
    raw_spectra: Mapping[str, numpy.ndarray] | None = None

    def __post_init__(self) -> None:
        """Validate one-and-only-one success or typed failure outcome."""

        if int(self.index) < 0:
            raise ValueError("Declared CMB batch indices must be non-negative")
        has_spectrum = self.spectrum is not None
        has_failure = self.failure is not None
        if has_spectrum == has_failure:
            raise ValueError(
                "Declared CMB batch results require a spectrum or "
                "typed failure"
            )
        if self.failure is not None and not isinstance(self.failure, CMBError):
            raise TypeError("Declared CMB batch failures must be typed errors")
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
        object.__setattr__(
            self,
            "requested_ells",
            tuple(int(value) for value in self.requested_ells),
        )
        object.__setattr__(
            self,
            "requested_spectra",
            tuple(str(value) for value in self.requested_spectra),
        )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))
        object.__setattr__(
            self,
            "phase_timings",
            dict(self.phase_timings or {}),
        )
        if self.raw_spectra is not None:
            if not isinstance(self.raw_spectra, Mapping):
                raise TypeError("Batch raw spectra must be a named mapping")
            object.__setattr__(self, "raw_spectra", dict(self.raw_spectra))
        object.__setattr__(self, "solver_id", str(self.solver_id))
        object.__setattr__(self, "solver_label", str(self.solver_label))

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
            "diagnostics": _jsonable(self.diagnostics),
            "phase_timings": _jsonable(self.phase_timings),
            "performance_envelope": _jsonable(self.performance_envelope),
            "requested_ells": self.requested_ells,
            "requested_spectra": self.requested_spectra,
            "raw_spectra": (
                None
                if self.raw_spectra is None
                else _jsonable(self.raw_spectra)
            ),
            "solver_id": self.solver_id,
            "solver_label": self.solver_label,
            "spectrum": (
                None if self.spectrum is None else _jsonable(self.spectrum)
            ),
            "success": self.success,
        }


__all__ = ["CMBBatchResult"]
