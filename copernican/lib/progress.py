# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Simplified CLI progress reporting helpers.

The module keeps the shared listener contract used by both the GUI and CLI
workers while replacing the carriage-return renderer with a lightweight counter
that logs stage completion as ``<stage>: <current>/<total> batches complete``.
Clients still receive structured ``batch_start``, ``progress_update`` and
``batch_finish`` events so the Run Builder/Monitor can mirror the sampler state
without co-opting the terminal.
"""

from __future__ import annotations

import contextlib
import math
import threading
from time import perf_counter
from typing import Callable, Iterator

from copernican.lib import console_output

__all__ = ["BatchProgressBar"]


class BatchProgressBar:
    """Counter-based progress reporter for CLI samplers."""

    def __init__(
        self,
        stage_label: str,
        total_steps: int,
        *,
        display: bool = True,
        progress_listener: Callable[[dict[str, object]], None] | None = None,
        stage_metadata: dict[str, str] | None = None,
        subunit_labels: tuple[str, str] | None = None,
        walker_total: int | None = None,
    ) -> None:
        """Initialize the progress bar with stage metadata and listeners."""
        self._stage_label = str(stage_label)
        self._total_steps = max(int(total_steps), 0)
        self._display = bool(display and self._total_steps > 0)
        self._progress_listener = progress_listener
        self._stage_metadata = dict(stage_metadata or {})
        self._subunit_labels = (
            tuple(subunit_labels) if subunit_labels else None
        )
        self._lock = threading.RLock()
        self._batch_index = 0
        self._current_start = 1
        self._current_end = 0
        self._current_span = 0
        self._current_step_total = max(self._total_steps, 1)
        self._current_step_processed = 0
        self._current_walker_processed = 0
        if walker_total is None:
            self._default_walker_total = self._current_step_total
        else:
            self._default_walker_total = max(int(walker_total), 1)
        self._current_walker_total = self._default_walker_total
        self._active = False
        self._last_logged_percent = -1
        self._batch_interval: int | None = None
        self._expected_batches = 1
        self._started_at: float | None = None

    def _batch_fraction(self, step_index: int, fraction: float) -> float:
        """Return the normalized position inside the current batch."""
        span = max(self._current_span, 1)
        offset = max(step_index - self._current_start, 0)
        raw = (offset + fraction) / span
        return min(max(raw, 0.0), 1.0)

    def _rope_fraction(self, step_index: int, fraction: float) -> float:
        """Return progress fraction measured across all steps."""
        total = max(self._total_steps, 1)
        raw = (max(step_index - 1, 0) + fraction) / total
        return min(max(raw, 0.0), 1.0)

    def _percent_for(self, step_index: int, fraction: float) -> int:
        """Convert a rope fraction into an integer percentage."""
        return int(round(self._rope_fraction(step_index, fraction) * 100))

    def _log_batch_completion(self) -> None:
        """Log the batch completion line when display is enabled."""
        if not self._display:
            return
        batches = max(self._expected_batches, 1)
        console_output.write(
            f"{self._stage_label} progress: "
            f"{min(self._batch_index, batches)}/{batches} batches completed."
        )

    def _format_progress_message(
        self, processed: int, total: int, percent: int
    ) -> str:
        """Build the textual progress summary echoed to the console."""
        started_at = self._started_at
        elapsed_seconds = (
            max(perf_counter() - started_at, 0.0)
            if started_at is not None
            else 0.0
        )
        items_per_second = (
            float(processed) / elapsed_seconds
            if elapsed_seconds > 0 and processed > 0
            else 0.0
        )
        remaining = max(int(total) - int(processed), 0)
        eta_seconds = (
            float(remaining) / items_per_second
            if items_per_second > 0
            else None
        )
        eta_text = "unknown" if eta_seconds is None else f"{eta_seconds:.1f}s"
        if self._subunit_labels is None:
            unit_label = "steps"
        else:
            singular, plural = self._subunit_labels
            unit_label = singular if total == 1 else plural
        return (
            f"{self._stage_label} batch {self._batch_index}: "
            f"{processed}/{total} {unit_label} completed ({percent}%), "
            f"elapsed={elapsed_seconds:.1f}s, "
            f"rate={items_per_second:.2f}/s, eta={eta_text}"
        )

    def _notify_listener(
        self,
        *,
        event: str,
        step_index: int,
        processed: int,
        total: int,
        percent: int,
        batch_fraction: float,
        step_fraction: float,
        walker_processed: int | None = None,
        walker_total: int | None = None,
    ) -> None:
        """Notify the optional listener with structured progress data."""
        if self._progress_listener is None:
            return
        walker_processed = (
            processed if walker_processed is None else int(walker_processed)
        )
        walker_total = (
            total if walker_total is None else max(int(walker_total), 1)
        )
        walker_processed = min(max(walker_processed, 0), walker_total)
        walker_fraction = walker_processed / max(walker_total, 1)
        step_remaining = max(int(total) - int(processed), 0)
        walker_remaining = max(walker_total - walker_processed, 0)
        record = {
            "event": event,
            "stage_label": self._stage_label,
            "stage_metadata": dict(self._stage_metadata),
            "batch_index": self._batch_index,
            "batch_size": self._current_span,
            "batch_start": self._current_start,
            "batch_end": self._current_end,
            "batch_percent": percent,
            "batch_fraction": min(max(batch_fraction, 0.0), 1.0),
            "step_index": step_index,
            "step_processed": processed,
            "step_total": total,
            "step_fraction": min(max(step_fraction, 0.0), 1.0),
            "step_remaining": step_remaining,
            "walker_processed": walker_processed,
            "walker_total": walker_total,
            "walker_fraction": walker_fraction,
            "walker_percent": int(round(walker_fraction * 100)),
            "walker_remaining": walker_remaining,
            "remaining": step_remaining,
            "status": "active" if self._active else "inactive",
        }
        started_at = self._started_at
        elapsed_seconds = (
            max(perf_counter() - started_at, 0.0)
            if started_at is not None
            else 0.0
        )
        items_per_second = (
            float(processed) / elapsed_seconds
            if elapsed_seconds > 0 and processed > 0
            else 0.0
        )
        remaining = step_remaining
        eta_seconds = (
            float(remaining) / items_per_second
            if items_per_second > 0
            else None
        )
        record.update(
            {
                "elapsed_seconds": elapsed_seconds,
                "items_per_second": items_per_second,
                "eta_seconds": eta_seconds,
            }
        )
        try:
            self._progress_listener(record)
        except (RuntimeError, TypeError, ValueError, KeyError):
            pass

    def start_batch(self, batch_start: int, batch_end: int) -> None:
        """Announce a new batch spanning ``batch_start`` to ``batch_end``."""

        with self._lock:
            if self._total_steps <= 0 or batch_end < batch_start:
                self._active = False
                return
            self._batch_index += 1
            self._current_start = int(batch_start)
            self._current_end = int(batch_end)
            self._current_span = max(
                self._current_end - self._current_start + 1, 0
            )
            if self._batch_interval is None and self._current_span > 0:
                self._batch_interval = self._current_span
                self._expected_batches = math.ceil(
                    max(self._total_steps, 1) / self._batch_interval
                )
            self._active = self._current_span > 0
            self._current_step_total = max(self._total_steps, 1)
            self._current_step_processed = 0
            self._current_walker_processed = 0
            self._current_walker_total = self._default_walker_total
            self._last_logged_percent = 0
            self._started_at = perf_counter()
            self._notify_listener(
                event="batch_start",
                step_index=self._current_start,
                processed=0,
                total=self._current_step_total,
                percent=0,
                batch_fraction=0.0,
                step_fraction=0.0,
            )

    def start_step(
        self, step_index: int, walker_total: int | None = None
    ) -> str | None:
        """Register the walker total for ``step_index`` and render progress."""

        with self._lock:
            if not self._active:
                return None
            walker_total = (
                max(int(walker_total), 1)
                if walker_total is not None
                else max(self._current_step_total, 1)
            )
            self._current_step_total = walker_total
            self._current_step_processed = 0
            self._current_walker_processed = 0
            self._current_walker_total = walker_total
        return self.update(
            step_index,
            processed=0,
            total=self._current_step_total,
        )

    def update(
        self,
        step_index: int,
        *,
        processed: int | None = None,
        total: int | None = None,
        step_progress: float | None = None,
        force: bool = False,
        walker_processed: int | None = None,
        walker_total: int | None = None,
    ) -> str | None:
        """Return the progress summary for the current batch."""

        with self._lock:
            if not self._active:
                return None
            batch_size = self._current_span
            if batch_size <= 0:
                return None
            if total is not None:
                total_int = max(int(total), 1)
                self._current_step_total = total_int
            else:
                total_int = self._current_step_total
            if processed is None:
                if step_progress is None or not math.isfinite(step_progress):
                    processed_int = total_int
                else:
                    processed_int = int(
                        round(min(max(step_progress, 0.0), 1.0) * total_int)
                    )
            else:
                processed_int = int(processed)
            processed_int = min(max(processed_int, 0), total_int)
            self._current_step_processed = processed_int
            if walker_processed is None:
                walker_processed_int = processed_int
            else:
                walker_processed_int = int(walker_processed)
            if walker_total is None:
                walker_total_int = total_int
            else:
                walker_total_int = max(int(walker_total), 1)
            self._current_walker_processed = min(
                max(walker_processed_int, 0), walker_total_int
            )
            self._current_walker_total = walker_total_int
            if step_progress is None or not math.isfinite(step_progress):
                step_fraction = (
                    processed_int / max(total_int, 1) if total_int else 0.0
                )
            else:
                step_fraction = float(min(max(step_progress, 0.0), 1.0))
            percent = self._percent_for(step_index, step_fraction)
            batch_fraction = self._batch_fraction(step_index, step_fraction)
            if not force and percent == self._last_logged_percent:
                self._notify_listener(
                    event="progress_update",
                    step_index=step_index,
                    processed=processed_int,
                    total=total_int,
                    percent=percent,
                    batch_fraction=batch_fraction,
                    step_fraction=step_fraction,
                    walker_processed=self._current_walker_processed,
                    walker_total=self._current_walker_total,
                )
                return None
            self._last_logged_percent = percent
            message: str | None = None
            if self._display:
                message = self._format_progress_message(
                    processed_int, total_int, percent
                )
                console_output.write(message)
            self._notify_listener(
                event="progress_update",
                step_index=step_index,
                processed=processed_int,
                total=total_int,
                percent=percent,
                batch_fraction=batch_fraction,
                step_fraction=step_fraction,
                walker_processed=self._current_walker_processed,
                walker_total=self._current_walker_total,
            )
            return message

    def finish_batch(self) -> None:
        """Close the current batch, inserting required spacing."""

        with self._lock:
            if not self._active:
                return
            self._log_batch_completion()
            self._notify_listener(
                event="batch_finish",
                step_index=self._current_end or self._current_start,
                processed=self._current_step_processed,
                total=max(self._current_step_total, 1),
                percent=100,
                batch_fraction=1.0,
                step_fraction=1.0,
                walker_processed=self._current_walker_processed,
                walker_total=self._current_walker_total,
            )
            self._active = False
            self._current_span = 0
            self._current_start = 1
            self._current_end = 0
            self._current_step_total = max(self._total_steps, 1)
            self._current_step_processed = 0
            self._current_walker_processed = 0
            self._current_walker_total = self._default_walker_total
            self._started_at = None
            self._last_logged_percent = -1

    @contextlib.contextmanager
    def suspend_display(self) -> Iterator[None]:
        """No-op context manager kept for compatibility."""

        yield
