"""Runtime timing and acceptance budgets for the native CMB solver."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator, Mapping

from .native_errors import NativePerformanceBudgetError

NATIVE_PHASE_NAMES = (
    "compilation",
    "background",
    "initial_data",
    "evolution",
    "projection",
    "lensing",
    "likelihood_assembly",
)


@dataclass(frozen=True, slots=True)
class NativePerformanceBudget:
    """Declare the wall-time limits for accepted native cache states."""

    full_spectrum_seconds: float = 180.0
    warm_parameter_seconds: float = 5.0
    exact_cache_hit_seconds: float = 1.0

    def limit_for(self, workload: str) -> float:
        """Return the positive limit for one named workload."""

        normalized = str(workload).strip().lower()
        if normalized in {
            "cold",
            "cold_full_spectrum",
            "full",
            "full_spectrum",
            "native_spectrum",
        }:
            return float(self.full_spectrum_seconds)
        if normalized in {
            "joint",
            "joint_mcmc",
            "mcmc",
            "warm",
            "warm_parameter",
        }:
            return float(self.warm_parameter_seconds)
        if normalized in {"cache_hit", "exact", "exact_cache_hit"}:
            return float(self.exact_cache_hit_seconds)
        raise ValueError(f"Unknown native performance workload: {workload}")


@dataclass(slots=True)
class NativePhaseTimer:
    """Accumulate wall time for named native execution phases."""

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
            raise ValueError("Native phase elapsed time must be non-negative")
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
            raise ValueError(f"Unknown native cache state: {state}")
        self.cache_state = normalized

    def set_work_units(self, work_units: Mapping[str, Any]) -> None:
        """Record non-negative governed work-unit counters."""

        for name, raw_value in work_units.items():
            value = int(raw_value)
            if value < 0:
                raise ValueError(
                    "Native work-unit counts must be non-negative"
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
        for name in NATIVE_PHASE_NAMES:
            snapshot.setdefault(f"{name}_seconds", 0.0)
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
    if "joint_mcmc_seconds" in raw_budget:
        raise ValueError(
            "cmb.perturbations.accuracy_controls.performance_budget."
            "joint_mcmc_seconds was removed; use "
            "warm_parameter_seconds"
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
        warm_parameter_seconds=_positive_seconds(
            raw_budget.get("warm_parameter_seconds"),
            name=(
                "cmb.perturbations.accuracy_controls.performance_budget."
                "warm_parameter_seconds"
            ),
            default=5.0,
        ),
        exact_cache_hit_seconds=_positive_seconds(
            raw_budget.get("exact_cache_hit_seconds"),
            name=(
                "cmb.perturbations.accuracy_controls.performance_budget."
                "exact_cache_hit_seconds"
            ),
            default=1.0,
        ),
    )


def enforce_native_performance_budget(
    elapsed_seconds: float,
    *,
    workload: str,
    budget: NativePerformanceBudget | None,
    cache_state: str | None = None,
) -> None:
    """Raise when one measured workload exceeds its declared budget.

    Cache state defines the measured workload boundary. Cold requests own
    structural setup, warm requests are parameter rebounds, and exact hits
    must not redo numerical work.
    """

    if budget is None:
        return
    elapsed = float(elapsed_seconds)
    if elapsed < 0.0 or elapsed != elapsed:
        raise ValueError("Native workload elapsed time must be finite")
    normalized_cache_state = str(cache_state or "cold").strip().lower()
    budget_workloads = {
        "cold": "full_spectrum",
        "warm": "warm_parameter",
        "exact_cache_hit": "exact_cache_hit",
    }
    try:
        budget_workload = budget_workloads[normalized_cache_state]
    except KeyError as exc:
        raise ValueError(
            "Native performance cache state is invalid: " f"{cache_state}"
        ) from exc
    limit = budget.limit_for(budget_workload)
    if elapsed > limit:
        raise NativePerformanceBudgetError(
            "Native CMB performance budget exceeded for "
            f"{workload}: {elapsed:.3f}s > {limit:.3f}s",
            context={
                "budget_workload": str(budget_workload),
                "cache_state": normalized_cache_state or None,
                "elapsed_seconds": elapsed,
                "limit_seconds": limit,
                "workload": str(workload),
            },
        )


__all__ = [
    "NATIVE_PHASE_NAMES",
    "NativePerformanceBudget",
    "NativePerformanceBudgetError",
    "NativePhaseTimer",
    "enforce_native_performance_budget",
    "resolve_native_performance_budget",
]
