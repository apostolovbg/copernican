"""Runtime phase timing and cache-state accounting for the declared CMB
solver.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator, Mapping

_LOGGER = logging.getLogger(__name__)

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
    heartbeat_interval_seconds: float = 30.0
    heartbeat_count: int = 0
    last_heartbeat_phase: str | None = None
    last_heartbeat_seconds: float = 0.0
    _started_at: float = field(default_factory=perf_counter)
    _last_heartbeat_at: float = field(default_factory=perf_counter)
    _phase_stack: list[str] = field(default_factory=list)
    _active_since: float | None = None
    _work_scopes: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict
    )

    def _account_active(self, now: float) -> None:
        """Charge an interval only to its innermost active owner."""

        if self._phase_stack and self._active_since is not None:
            name = self._phase_stack[-1]
            self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + (
                now - self._active_since
            )
        self._active_since = now

    def _phase_elapsed(self, name: str, now: float) -> float:
        """Include currently running exclusive work without mutating it."""

        elapsed = self.phase_seconds.get(name, 0.0)
        if self._phase_stack and self._phase_stack[-1] == name:
            elapsed += now - float(self._active_since)
        return elapsed

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure one phase and add its elapsed time to the timer."""

        phase_name = str(name)
        self._account_active(perf_counter())
        self._phase_stack.append(phase_name)
        try:
            yield
        # DEVCOV_ALLOW_BROAD_ONCE phase failure accounting boundary.
        except BaseException:
            if self.failed_phase is None:
                self.failed_phase = phase_name
            raise
        finally:
            self._account_active(perf_counter())
            self._phase_stack.pop()

    def add(self, name: str, elapsed_seconds: float) -> None:
        """Add an externally measured interval to one named phase."""

        value = float(elapsed_seconds)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "Declared phase elapsed time must be non-negative"
            )
        phase_name = str(name)
        self.phase_seconds[phase_name] = (
            self.phase_seconds.get(phase_name, 0.0) + value
        )

    def total_seconds(self) -> float:
        """Return the accumulated time across all recorded phases."""

        now = perf_counter()
        return float(
            sum(self._phase_elapsed(name, now) for name in self._phase_names())
        )

    def _phase_names(self) -> set[str]:
        """Include an active phase before its first completed interval."""

        return set(self.phase_seconds) | set(self._phase_stack)

    def start_work(self, phase: str, scope: str) -> None:
        """Start a distinct fixed or adaptively growing work scope."""

        now = perf_counter()
        key = (str(phase), str(scope))
        if key in self._work_scopes:
            attempt = 1
            while (key[0], f"{key[1]}@attempt_{attempt}") in self._work_scopes:
                attempt += 1
            self._work_scopes[(key[0], f"{key[1]}@attempt_{attempt}")] = dict(
                self._work_scopes[key]
            )
        self._work_scopes[(str(phase), str(scope))] = {
            "started_at": now,
            "phase_start": self._phase_elapsed(str(phase), now),
            "completed": 0,
            "total": 0,
            "weighted_completed": 0.0,
            "weighted_total": 0.0,
            "remaining_seconds": None,
        }

    def merge(self, other: PhaseTimer, *, scope_prefix: str) -> None:
        """Retain completed refinement timings and independently named work."""

        if other._phase_stack:
            raise ValueError("Only completed phase timers may be merged")
        for name, elapsed in other.phase_seconds.items():
            self.add(name, elapsed)
        for (phase, scope), work in other._work_scopes.items():
            self._work_scopes[(phase, f"{scope_prefix}/{scope}")] = dict(work)
        self.failed_phase = other.failed_phase or self.failed_phase
        self.heartbeat_count += other.heartbeat_count
        self._last_heartbeat_at = max(
            self._last_heartbeat_at, other._last_heartbeat_at
        )
        if other.heartbeat_count:
            self.last_heartbeat_phase = other.last_heartbeat_phase
            self.last_heartbeat_seconds = other.last_heartbeat_seconds

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

    def heartbeat(
        self,
        phase: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        scope: str = "scheduled_modes",
        weighted_completed: float | None = None,
        weighted_total: float | None = None,
        details: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        """Emit a low-frequency progress record for a long-running phase."""

        now = perf_counter()
        phase_name = str(phase)
        scope_key = (phase_name, str(scope))
        if scope_key not in self._work_scopes:
            self.start_work(phase_name, str(scope))
            now = perf_counter()
        work = self._work_scopes[scope_key]
        if (completed is None) != (total is None):
            raise ValueError("Progress requires completed and total counts")
        if completed is not None:
            if not 0 <= completed <= total:
                raise ValueError(
                    "Progress must satisfy 0 <= completed <= total"
                )
            if completed < work["completed"] or total < work["total"]:
                raise ValueError("Work counters must grow within their scope")
            work.update(completed=int(completed), total=int(total))
        if (weighted_completed is None) != (weighted_total is None):
            raise ValueError("Weighted progress requires both work values")
        if weighted_completed is not None:
            if not (
                math.isfinite(weighted_completed)
                and math.isfinite(weighted_total)
                and 0.0 <= weighted_completed <= weighted_total
                and weighted_completed >= work["weighted_completed"]
                and weighted_total >= work["weighted_total"]
            ):
                raise ValueError("Weighted work must be finite and monotone")
            work.update(
                weighted_completed=float(weighted_completed),
                weighted_total=float(weighted_total),
            )
        phase_elapsed = self._phase_elapsed(phase_name, now)
        work_elapsed = max(0.0, phase_elapsed - work["phase_start"])
        # Equal-cost counts alone cannot predict nonuniform RK intervals or
        # oscillatory kernels. Only explicitly weighted scheduled work earns
        # an estimate; future refinement decisions remain unknown.
        if work["weighted_completed"] > 0.0:
            work["remaining_seconds"] = work_elapsed * (
                work["weighted_total"] / work["weighted_completed"] - 1.0
            )
        if not force and now - self._last_heartbeat_at < float(
            self.heartbeat_interval_seconds
        ):
            return False
        self._last_heartbeat_at = now
        self.heartbeat_count += 1
        self.last_heartbeat_phase = phase_name
        self.last_heartbeat_seconds = float(phase_elapsed)
        progress = (
            f"; scope={scope}; scope_elapsed={now - work['started_at']:.1f}s"
            f"; request_elapsed={now - self._started_at:.1f}s"
            "; future_adaptive_work=unknown"
        )
        if completed is not None and total is not None:
            progress += f"; completed={int(completed)}/{int(total)}"
        if weighted_completed is not None:
            progress += (
                f"; weighted_work={weighted_completed:g}/{weighted_total:g}"
                f"; estimated_scope_remaining_wall_time="
                f"{work['remaining_seconds']:.1f}s"
                if work["remaining_seconds"] is not None
                else "; estimated_scope_remaining_wall_time=unknown"
            )
        if details:
            progress += "".join(
                f"; {name}={value}" for name, value in sorted(details.items())
            )
        _LOGGER.info(
            "CCMBS phase heartbeat: phase=%s; elapsed=%.1fs%s",
            self.last_heartbeat_phase,
            self.last_heartbeat_seconds,
            progress,
        )
        return True

    def snapshot(
        self, *, total_seconds: float | None = None
    ) -> dict[str, Any]:
        """Return a stable scalar timing payload for runtime manifests."""

        now = perf_counter()
        snapshot = {
            f"{name}_seconds": self._phase_elapsed(name, now)
            for name in sorted(self._phase_names())
        }
        for name in CMB_PHASE_NAMES:
            snapshot.setdefault(f"{name}_seconds", 0.0)
        snapshot["total_seconds"] = float(
            self.total_seconds() if total_seconds is None else total_seconds
        )
        snapshot["heartbeat_count"] = int(self.heartbeat_count)
        snapshot["last_heartbeat_phase"] = self.last_heartbeat_phase
        snapshot["last_heartbeat_seconds"] = float(self.last_heartbeat_seconds)
        snapshot["request_elapsed_seconds"] = float(now - self._started_at)
        snapshot["work_scopes"] = {
            f"{phase}:{scope}": {
                name: value
                for name, value in work.items()
                if name not in {"started_at", "phase_start"}
            }
            for (phase, scope), work in sorted(self._work_scopes.items())
        }
        return snapshot


__all__ = [
    "CMB_PHASE_NAMES",
    "PhaseTimer",
]
