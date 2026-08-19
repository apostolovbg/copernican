"""Runtime phase timing and cache-state accounting for the declared CMB
solver.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator, Mapping

CMB_PHASE_NAMES = (
    "compilation",
    "background",
    "initial_data",
    "evolution",
    "projection",
    "lensing",
    "likelihood_assembly",
)


@dataclass(slots=True)
class PhaseTimer:
    """Accumulate wall time for named declared execution phases."""

    phase_seconds: dict[str, float] = field(default_factory=dict)
    failed_phase: str | None = None
    cache_state: str = "cold"
    work_units: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure one phase and add its elapsed time to the timer."""

        phase_name = str(name)
        started = perf_counter()
        try:
            yield
        # DEVCOV_ALLOW_BROAD_ONCE phase failure accounting boundary.
        except BaseException:
            self.failed_phase = phase_name
            raise
        finally:
            self.phase_seconds[phase_name] = (
                self.phase_seconds.get(phase_name, 0.0)
                + perf_counter()
                - started
            )

    def add(self, name: str, elapsed_seconds: float) -> None:
        """Add an externally measured interval to one named phase."""

        value = float(elapsed_seconds)
        if value < 0.0:
            raise ValueError(
                "Declared phase elapsed time must be non-negative"
            )
        phase_name = str(name)
        self.phase_seconds[phase_name] = (
            self.phase_seconds.get(phase_name, 0.0) + value
        )

    def total_seconds(self) -> float:
        """Return the accumulated time across all recorded phases."""

        return float(sum(self.phase_seconds.values()))

    def mark_cache_state(self, state: str) -> None:
        """Record whether the request was cold, warm, or an exact hit."""

        normalized = str(state).strip().lower()
        if normalized not in {"cold", "warm", "exact_cache_hit"}:
            raise ValueError(f"Unknown declared cache state: {state}")
        self.cache_state = normalized

    def set_work_units(self, work_units: Mapping[str, Any]) -> None:
        """Record non-negative governed work-unit counters."""

        for name, raw_value in work_units.items():
            value = int(raw_value)
            if value < 0:
                raise ValueError(
                    "Declared work-unit counts must be non-negative"
                )
            self.work_units[str(name)] = value

    def snapshot(
        self, *, total_seconds: float | None = None
    ) -> dict[str, float]:
        """Return a stable scalar timing payload for runtime manifests."""

        snapshot = {
            f"{name}_seconds": float(value)
            for name, value in sorted(self.phase_seconds.items())
        }
        for name in CMB_PHASE_NAMES:
            snapshot.setdefault(f"{name}_seconds", 0.0)
        snapshot["total_seconds"] = float(
            self.total_seconds() if total_seconds is None else total_seconds
        )
        return snapshot


__all__ = [
    "CMB_PHASE_NAMES",
    "PhaseTimer",
]
