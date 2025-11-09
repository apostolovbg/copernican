# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.
# Last Updated: 2025-11-09

"""Helpers that turn Stage 2 sampler metrics into runtime estimates.

**Last Updated:** 2025-11-09

The Stage 2 console output streams progress information for both the
ΛCDM and alternative models.  This module centralises the arithmetic that
combines per-stage progress callbacks into a single runtime estimate so
other components can react without importing the full ``copernican``
launcher, which enforces strict environment guards at import time.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


class StageTwoRuntimeTracker:
    """Aggregate sampler metrics across multiple Stage 2 phases.

    Instances consume structured dictionaries describing the current
    sampler state—batch timings, completed steps, total steps and the
    model/stage identifiers.  The tracker keeps a running estimate of the
    elapsed time, remaining steps and sampler throughput, emitting a
    formatted status line whenever enough time has passed since the last
    update.  Callers can print the returned string directly to update the
    console without reimplementing the aggregation logic.
    """

    __slots__ = (
        "_offsets",
        "_total_steps",
        "_initial_timestamp",
        "_completed_steps",
        "_last_emit",
        "_throttle",
    )

    def __init__(
        self,
        stages: Iterable[tuple[str, str, int]],
        *,
        throttle_seconds: float = 1.0,
    ) -> None:
        """Record the Stage 2 plan for combined runtime reporting.

        Parameters
        ----------
        stages:
            Iterable of ``(model_name, stage_name, steps)`` tuples ordered
            exactly as the sampler will execute them.  Zero-step entries
            are ignored so dry stages do not affect timing.
        throttle_seconds:
            Minimum interval between emitted status lines.  A one-second
            cadence keeps the console informative without overwhelming
            operators.
        """

        plan: list[tuple[str, str, int]] = []
        for model, stage, steps in stages:
            count = int(steps)
            if count <= 0:
                continue
            plan.append((model, stage, count))
        self._offsets: dict[tuple[str, str], int] = {}
        offset = 0
        for model, stage, steps in plan:
            self._offsets[(model, stage)] = offset
            offset += steps
        self._total_steps = offset
        self._initial_timestamp: float | None = None
        self._completed_steps = 0
        self._last_emit: float | None = None
        self._throttle = max(float(throttle_seconds), 0.0)

    @staticmethod
    def _format_seconds(value: float) -> str:
        """Return ``value`` seconds formatted as ``HH:MM:SS``."""

        value = max(0.0, float(value))
        total_seconds = int(round(value))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update(self, event: Mapping[str, Any]) -> str | None:
        """Return a status line summarising the Stage 2 runtime estimate."""

        if self._total_steps <= 0:
            return None
        model = str(event.get("model_name", ""))
        stage = str(event.get("stage", ""))
        key = (model, stage)
        if key not in self._offsets:
            return None
        timestamp = float(event.get("timestamp", 0.0))
        if self._initial_timestamp is None:
            self._initial_timestamp = timestamp
        elapsed = max(timestamp - (self._initial_timestamp or timestamp), 0.0)
        steps_completed = int(event.get("steps_completed", 0))
        offset = self._offsets[key]
        completed = min(offset + steps_completed, self._total_steps)
        if completed <= self._completed_steps:
            completed = self._completed_steps
        else:
            self._completed_steps = completed
        if completed <= 0:
            return None
        speed = None
        if elapsed > 0.0:
            speed = completed / elapsed
        if speed is None or speed <= 0.0:
            raw_speed = event.get("speed")
            try:
                speed = float(raw_speed)
            except (TypeError, ValueError):
                speed = None
        if speed is None or speed <= 0.0:
            return None
        remaining_steps = self._total_steps - completed
        remaining = remaining_steps / speed if speed > 0 else 0.0
        total = elapsed + remaining
        if (
            self._last_emit is not None
            and timestamp - self._last_emit < self._throttle
            and completed < self._total_steps
        ):
            return None
        self._last_emit = timestamp
        message = (
            "Stage 2 runtime estimate: "
            f"elapsed {self._format_seconds(elapsed)}, "
            f"remaining {self._format_seconds(remaining)}, "
            f"total {self._format_seconds(total)}, "
            f"throughput {speed:.2f} step/s"
        )
        return message
