"""Contracts shared by selectable Copernican CMB solvers.

The solver boundary is deliberately independent of the numerical backend.
The reference NumPy/SciPy implementation and a future Taichi implementation
must exchange the same prepared contracts, ordered results, diagnostics, and
typed failures through this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy

from .errors import CMBError


def _jsonable(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""

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
class CMBSolverCapabilities:
    """Describe one solver's backend and declared numerical capabilities."""

    solver_id: str
    solver_label: str
    execution_backend: str = "cpu"
    implementation: str = "reference"
    supported_spectra: tuple[str, ...] = ()
    supported_grids: Mapping[str, Any] = field(default_factory=dict)
    accuracy_tiers: tuple[str, ...] = ()
    batch_mode: str = "ordered_scalar_adapter"
    preparation: bool = True
    cleanup: bool = True
    device_probe: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        """Return a manifest-safe capability mapping."""

        return {
            "solver_id": self.solver_id,
            "solver_label": self.solver_label,
            "execution_backend": self.execution_backend,
            "implementation": self.implementation,
            "supported_spectra": self.supported_spectra,
            "supported_grids": _jsonable(self.supported_grids),
            "accuracy_tiers": self.accuracy_tiers,
            "batch_mode": self.batch_mode,
            "preparation": self.preparation,
            "cleanup": self.cleanup,
            "device_probe": _jsonable(self.device_probe),
        }


@dataclass(frozen=True, slots=True)
class CMBResult:
    """One ordered scalar solver outcome with complete provenance."""

    spectra: numpy.ndarray | Mapping[str, numpy.ndarray] | None = None
    requested_ells: tuple[int, ...] = ()
    requested_spectra: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    cache_provenance: Mapping[str, Any] = field(default_factory=dict)
    phase_timings: Mapping[str, float] = field(default_factory=dict)
    failure: CMBError | None = None
    solver_id: str = ""
    solver_label: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize one successful or failed outcome."""

        has_spectra = self.spectra is not None
        has_failure = self.failure is not None
        if has_spectra == has_failure:
            raise ValueError(
                "CMB results require spectra or one typed solver failure"
            )
        if has_failure and not isinstance(self.failure, CMBError):
            raise TypeError(
                "CMB result failures must be typed declared errors"
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
            "cache_provenance",
            dict(self.cache_provenance or {}),
        )
        object.__setattr__(
            self, "phase_timings", dict(self.phase_timings or {})
        )

    @property
    def success(self) -> bool:
        """Return whether this result contains spectra."""

        return self.spectra is not None

    @property
    def spectrum(self) -> numpy.ndarray | Mapping[str, numpy.ndarray] | None:
        """Return the spectra payload using the singular public spelling."""

        return self.spectra

    def raise_for_failure(self) -> None:
        """Raise the typed failure when this outcome is unsuccessful."""

        if self.failure is not None:
            raise self.failure

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic serialization for manifests and results."""

        return {
            "cache_provenance": _jsonable(self.cache_provenance),
            "diagnostics": _jsonable(self.diagnostics),
            "failure": (
                None if self.failure is None else self.failure.diagnostic()
            ),
            "phase_timings": _jsonable(self.phase_timings),
            "requested_ells": self.requested_ells,
            "requested_spectra": self.requested_spectra,
            "solver_id": self.solver_id,
            "solver_label": self.solver_label,
            "spectra": (
                None if self.spectra is None else _jsonable(self.spectra)
            ),
            "success": self.success,
        }


@runtime_checkable
class CMBSolverProtocol(Protocol):
    """Protocol implemented by every selectable CMB solver backend."""

    solver_id: str
    solver_label: str

    def capabilities(self) -> Mapping[str, object]:
        """Return backend, grid, accuracy, and device capabilities."""

    def prepare(self, contract: Mapping[str, object]) -> object:
        """Prepare immutable structural assets for one model contract."""

    def evaluate(
        self,
        prepared: object,
        ells: Sequence[int],
        *,
        spectra: Sequence[str],
        workload: str,
    ) -> CMBResult:
        """Evaluate one prepared contract and return a typed result."""

    def evaluate_batch(
        self,
        prepared: Sequence[object],
        ells: Sequence[int],
        *,
        spectra: Sequence[str],
        workload: str,
    ) -> tuple[CMBResult, ...]:
        """Evaluate prepared contracts in input order."""

    def cleanup(self) -> None:
        """Release process-local or device-local solver resources."""


__all__ = [
    "CMBResult",
    "CMBSolverCapabilities",
    "CMBSolverProtocol",
]
