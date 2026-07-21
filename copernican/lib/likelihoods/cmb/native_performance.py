"""Runtime timing and acceptance budgets for the native CMB solver."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator, Mapping


class NativePerformanceBudgetError(ValueError):
    """Identify a native run that exceeded its declared wall-time budget."""


@dataclass(frozen=True, slots=True)
class NativePerformanceBudget:
    """Declare the wall-time limits for accepted native workloads."""

    full_spectrum_seconds: float = 180.0
    joint_mcmc_seconds: float = 60.0

    def limit_for(self, workload: str) -> float:
        """Return the positive limit for one named workload."""

        normalized = str(workload).strip().lower()
        if normalized in {"full", "full_spectrum", "native_spectrum"}:
            return float(self.full_spectrum_seconds)
        if normalized in {"joint", "joint_mcmc", "mcmc"}:
            return float(self.joint_mcmc_seconds)
        raise ValueError(f"Unknown native performance workload: {workload}")


@dataclass(slots=True)
class NativePhaseTimer:
    """Accumulate wall time for named native execution phases."""

    phase_seconds: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure one phase and add its elapsed time to the timer."""

        phase_name = str(name)
        started = perf_counter()
        try:
            yield
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
            raise ValueError("Native phase elapsed time must be non-negative")
        phase_name = str(name)
        self.phase_seconds[phase_name] = (
            self.phase_seconds.get(phase_name, 0.0) + value
        )

    def total_seconds(self) -> float:
        """Return the accumulated time across all recorded phases."""

        return float(sum(self.phase_seconds.values()))

    def snapshot(
        self, *, total_seconds: float | None = None
    ) -> dict[str, float]:
        """Return a stable scalar timing payload for runtime manifests."""

        snapshot = {
            f"{name}_seconds": float(value)
            for name, value in sorted(self.phase_seconds.items())
        }
        snapshot["total_seconds"] = float(
            self.total_seconds() if total_seconds is None else total_seconds
        )
        return snapshot


def _positive_seconds(value: Any, *, name: str, default: float) -> float:
    """Coerce one finite positive duration or return ``default``."""

    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if (
        result <= 0.0
        or result != result
        or result in {float("inf"), float("-inf")}
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return result


def resolve_native_performance_budget(
    accuracy_controls: Mapping[str, Any] | None,
) -> NativePerformanceBudget | None:
    """Resolve an optional native wall-time budget from accuracy controls."""

    controls = accuracy_controls or {}
    raw_budget = controls.get("performance_budget")
    if raw_budget is None and controls.get("runtime_envelope") == "bounded":
        raw_budget = {}
    if raw_budget is None:
        return None
    if raw_budget == "bounded":
        raw_budget = {}
    if not isinstance(raw_budget, Mapping):
        raise ValueError(
            "cmb.perturbations.accuracy_controls.performance_budget must be "
            "a mapping or the preset 'bounded'"
        )
    return NativePerformanceBudget(
        full_spectrum_seconds=_positive_seconds(
            raw_budget.get("full_spectrum_seconds"),
            name=(
                "cmb.perturbations.accuracy_controls.performance_budget."
                "full_spectrum_seconds"
            ),
            default=180.0,
        ),
        joint_mcmc_seconds=_positive_seconds(
            raw_budget.get("joint_mcmc_seconds"),
            name=(
                "cmb.perturbations.accuracy_controls.performance_budget."
                "joint_mcmc_seconds"
            ),
            default=60.0,
        ),
    )


def enforce_native_performance_budget(
    elapsed_seconds: float,
    *,
    workload: str,
    budget: NativePerformanceBudget | None,
) -> None:
    """Raise when one measured workload exceeds its declared budget."""

    if budget is None:
        return
    elapsed = float(elapsed_seconds)
    if elapsed < 0.0 or elapsed != elapsed:
        raise ValueError("Native workload elapsed time must be finite")
    limit = budget.limit_for(workload)
    if elapsed > limit:
        raise NativePerformanceBudgetError(
            "Native CMB performance budget exceeded for "
            f"{workload}: {elapsed:.3f}s > {limit:.3f}s"
        )
