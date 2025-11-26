# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Console progress helpers shared across Copernican engines.

**Last Updated:** 2025-11-22

The previous implementations lived directly inside
``engines.cosmo_engine_mcmc`` which made the sampler difficult to reuse.
This module centralises the console renderer, spinner pump integration and
``emcee`` move instrumentation so future engines can opt into the same live
progress feed without inheriting unrelated MCMC logic.  The helpers keep the
carriage-return based renderer that mirrors the historical transcript format
while guaranteeing that concurrent log messages never leave stale bars behind
on the console.  A small context manager temporarily suspends the repaint to
let callers write additional output and then restores the most recent line so
operators always see a coherent snapshot of the sampler state.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from typing import Any, Callable, Iterator

import numpy as np
from emcee import moves
from emcee.state import State

from copernican_lib import console_output as console

CARRIAGE_RETURN = chr(13)

__all__ = [
    "BatchProgressBar",
    "StepProgressEmitter",
    "configure_sampler_progress_reporting",
]


class BatchProgressBar:
    """Render Stage 2 progress with direct carriage-return repainting.

    The renderer mirrors the legacy Stage 2 console layout while exposing a
    public API so any engine can instantiate it.  Updates originate from both
    sampler callbacks and a background repaint pump.  A re-entrant lock keeps
    those writers in sequence so Unicode glyphs never interleave on the
    console, and a suspension context ensures other messages can briefly take
    over the terminal without leaving the progress line corrupted.
    """

    _BAR_WIDTH = 50
    _WALKER_BAR_WIDTH = 12
    _PARTIAL_GLYPHS = "▏▎▍▌▋▊▉"
    _SPINNER_FRAMES = (
        "⠋",
        "⠙",
        "⠹",
        "⠸",
        "⠼",
        "⠴",
        "⠦",
        "⠧",
        "⠇",
        "⠏",
    )

    def __init__(
        self,
        stage_label: str,
        total_steps: int,
        *,
        display: bool = True,
        subunit_labels: tuple[str, str] | None = ("walker", "walkers"),
    ) -> None:
        self._stage_label = stage_label
        self._total_steps = max(int(total_steps), 0)
        self._display = bool(display and self._total_steps > 0)
        # The per-step subunit defaults to "walker" terminology so ensemble
        # samplers keep their historical output, while alternative engines can
        # override the labels to describe their own iteration units.
        if subunit_labels is None:
            self._subunit_labels: tuple[str, str] | None = None
        else:
            singular, plural = subunit_labels
            self._subunit_labels = (str(singular), str(plural))
        self._batch_index = 0
        self._current_start = 1
        self._current_end = 0
        self._current_span = 0
        self._active = False
        self._last_percent = -1
        self._last_line = ""
        self._last_rendered_length = 0
        self._current_step_total = 1
        self._current_step_processed = 0
        self._spinner_index = -1
        self._last_rendered = ""
        self._lock = threading.RLock()

    def _build_bar(self, fraction: float, width: int) -> str:
        """Return a Unicode bar ``width`` cells wide for ``fraction``."""

        fraction = min(max(fraction, 0.0), 1.0)
        exact_units = min(max(fraction * width, 0.0), float(width))
        full_units = int(math.floor(exact_units))
        remainder = exact_units - full_units
        partial_levels = len(self._PARTIAL_GLYPHS)
        partial_index = int(round(remainder * partial_levels))
        if partial_index > partial_levels:
            partial_index = partial_levels
        if (
            partial_index == partial_levels
            and remainder > 0
            and full_units < width
        ):
            full_units += 1
            partial_index = 0
        full_units = min(full_units, width)
        if partial_index and full_units < width:
            partial = self._PARTIAL_GLYPHS[partial_index - 1]
        else:
            partial = ""
        remaining_cells = width - full_units - len(partial)
        return f"{'█' * full_units}{partial}{'-' * max(remaining_cells, 0)}"

    def _next_spinner(self) -> str:
        """Advance and return the animated spinner glyph."""

        if not self._SPINNER_FRAMES:
            return ""
        self._spinner_index = (self._spinner_index + 1) % len(
            self._SPINNER_FRAMES
        )
        return self._SPINNER_FRAMES[self._spinner_index]

    def _render_line(
        self,
        step_index: int,
        *,
        processed: int,
        total: int,
        batch_size: int,
    ) -> tuple[str, int, str]:
        """Return the console line, percentage and display text."""

        walker_total = max(total, 1)
        walker_processed = min(max(processed, 0), walker_total)
        step_progress = walker_processed / walker_total
        completed_before = max(0, step_index - self._current_start)
        completed_before = min(completed_before, batch_size)
        fraction = (completed_before + step_progress) / max(batch_size, 1)
        fraction = min(max(fraction, 0.0), 1.0)
        percent = int(round(fraction * 100))
        bar = self._build_bar(fraction, self._BAR_WIDTH)
        walker_bar = self._build_bar(step_progress, self._WALKER_BAR_WIDTH)
        completed_steps = completed_before + (1 if step_progress >= 1.0 else 0)
        completed_steps = min(completed_steps, batch_size)
        remaining = max(batch_size - completed_steps, 0)
        remaining_word = "step" if remaining == 1 else "steps"
        progress_word = "step" if batch_size == 1 else "steps"
        walker_remaining = max(walker_total - walker_processed, 0)
        walker_word = "walker" if walker_remaining == 1 else "walkers"
        spinner = self._next_spinner()
        postfix = (
            f"step {min(step_index, self._current_end)} of {batch_size} "
            f"{progress_word}, {remaining} {remaining_word} remaining"
        )
        walker_postfix = f"{walker_bar} {walker_processed}/{walker_total}"
        if self._subunit_labels is None:
            walker_fragment = walker_postfix
        else:
            singular, plural = self._subunit_labels
            walker_word = singular if walker_remaining == 1 else plural
            walker_fragment = (
                f"{walker_postfix}, {walker_remaining} {walker_word} left"
            )
        display_line = (
            f"{bar} {percent:>3d}% {spinner} ("
            f"{postfix}; {walker_fragment})"
        )
        line = f"{CARRIAGE_RETURN}{display_line}"
        return line, percent, display_line

    def _render_raw(self, rendered_text: str) -> None:
        """Write ``rendered_text`` to the console using a carriage return."""

        if not self._display:
            return
        # Prefix the payload with ``\r`` and request no trailing characters so
        # the terminal rewinds to column zero without emitting implicit
        # newlines. The old implementation relied on ``end="\r"`` which worked
        # interactively but left blank spacer rows in logs that captured the
        # trailing carriage return as a standalone line feed.
        console.write(f"{CARRIAGE_RETURN}{rendered_text}", end="")
        self._last_rendered = rendered_text

    def _emit_display_line(self, display_line: str) -> None:
        """Render text while padding trailing remnants."""

        if not self._display:
            return
        previous_width = self._last_rendered_length
        current_width = len(display_line)
        if previous_width > current_width:
            padded = display_line + (" " * (previous_width - current_width))
        else:
            padded = display_line
        self._last_rendered_length = max(previous_width, current_width)
        self._render_raw(padded)

    def _clear_line(self) -> None:
        """Erase the previously rendered progress line from the console."""

        if not self._display:
            return
        if not self._last_line:
            return
        blank = " " * self._last_rendered_length
        self._render_raw(blank)
        self._last_rendered = ""

    def start_batch(self, batch_start: int, batch_end: int) -> None:
        """Announce a new batch spanning ``batch_start`` to ``batch_end``."""

        with self._lock:
            if not self._display or batch_end < batch_start:
                self._active = False
                return
            self._batch_index += 1
            self._current_start = int(batch_start)
            self._current_end = int(batch_end)
            self._current_span = max(
                self._current_end - self._current_start + 1, 0
            )
            self._active = True
            self._last_percent = -1
            self._last_line = ""
            self._last_rendered_length = 0
            self._current_step_total = 1
            self._current_step_processed = 0
            self._spinner_index = -1
            span = self._current_end - self._current_start + 1
            step_word = "step" if span == 1 else "steps"
            console.write(
                f"{self._stage_label} batch {self._batch_index} "
                f"({span} {step_word}) progress:"
            )
            if self._current_span > 0 and self._display:
                line, percent, display_line = self._render_line(
                    self._current_start,
                    processed=0,
                    total=max(self._current_step_total, 1),
                    batch_size=self._current_span,
                )
                self._emit_display_line(display_line)
                self._last_line = display_line
                self._last_percent = percent

    def start_step(
        self, step_index: int, walker_total: int | None = None
    ) -> str | None:
        """Register the walker total for ``step_index`` and render progress."""

        with self._lock:
            if not self._active:
                return None
            if walker_total is None:
                walker_total = self._current_step_total
            walker_total = max(int(walker_total), 1)
            self._current_step_total = walker_total
            self._current_step_processed = 0
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
    ) -> str | None:
        """Return the rendered progress line for the active batch."""

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
            if step_progress is None or not math.isfinite(step_progress):
                step_progress = processed_int / max(total_int, 1)
            else:
                step_progress = float(min(max(step_progress, 0.0), 1.0))

            line, percent, display_line = self._render_line(
                step_index,
                processed=processed_int,
                total=total_int,
                batch_size=batch_size,
            )

            if (
                not force
                and percent == self._last_percent
                and display_line == self._last_line
            ):
                return None
            self._last_percent = percent
            self._last_line = display_line
            self._emit_display_line(display_line)
            return line

    def finish_batch(self) -> None:
        """Close the current batch, inserting required spacing."""

        with self._lock:
            if not self._active:
                return
            if self._last_line:
                self._clear_line()
            if self._display:
                # Finalise the cleared line and leave a spacer so subsequent
                # console output never collides with the retired bar. Keeping
                # the spacer separate from the clearing pass ensures captured
                # transcripts still include the blank line even when the
                # renderer was idle at 0%.
                console.write("")
                console.write("")
            self._active = False
            self._last_percent = -1
            self._last_line = ""
            self._last_rendered = ""
            self._last_rendered_length = 0
            self._current_span = 0
            self._current_step_total = 1
            self._current_step_processed = 0
            self._spinner_index = -1

    @contextlib.contextmanager
    def suspend_display(self) -> Iterator[None]:
        """Temporarily hide the active line while other output prints."""

        rendered_line = ""
        active = False
        with self._lock:
            active = self._active and bool(self._last_line)
            if active and self._display:
                rendered_line = self._last_rendered
                self._clear_line()
        try:
            yield
        finally:
            if active and self._display and rendered_line:
                with self._lock:
                    self._render_raw(rendered_line)

    @property
    def batch_index(self) -> int:
        """Return the index of the current batch for diagnostics."""

        return self._batch_index

    @property
    def uses_live_display(self) -> bool:
        """Return ``True`` when the bar repaints the console directly."""

        return self._display


class StepProgressEmitter:
    """Bridge sampler move callbacks to batch progress updates."""

    _IDLE_REPAINT_INTERVAL = 0.1

    __slots__ = (
        "_progress_bar",
        "_active_step",
        "_walker_total",
        "_timer",
        "_idle_interval",
        "_last_repaint",
        "_last_processed",
        "_last_total",
    )

    def __init__(self, progress_bar: BatchProgressBar) -> None:
        self._progress_bar = progress_bar
        self._active_step: int | None = None
        self._walker_total = 1
        self._timer = time.monotonic
        self._idle_interval = float(self._IDLE_REPAINT_INTERVAL)
        self._last_repaint: float | None = None
        self._last_processed = 0
        self._last_total = 1

    def start(self, step_index: int, walker_total: int) -> None:
        """Prepare to track ``step_index`` with ``walker_total`` updates."""

        self._active_step = int(step_index)
        self._walker_total = max(int(walker_total), 1)
        self._last_processed = 0
        self._last_total = self._walker_total
        self._progress_bar.start_step(step_index, self._walker_total)
        self._last_repaint = self._timer()

    def clear(self) -> None:
        """Disable updates until the next step begins."""

        self._active_step = None
        self._walker_total = 1
        self._last_repaint = None
        self._last_processed = 0
        self._last_total = 1

    def __call__(self, processed: int, total: int) -> None:
        """Forward partial progress updates to the active batch."""

        if self._active_step is None:
            return
        effective_total = max(int(total) if total else self._walker_total, 1)
        self._last_processed = int(processed)
        self._last_total = effective_total
        self._progress_bar.update(
            self._active_step,
            processed=processed,
            total=effective_total,
        )
        if self._progress_bar.uses_live_display:
            self._last_repaint = self._timer()

    def tick(self) -> None:
        """Rotate the spinner when idle so live displays keep animating."""

        if self._active_step is None:
            return
        if not self._progress_bar.uses_live_display:
            return
        if self._last_repaint is None:
            self._last_repaint = self._timer()
            return
        now = self._timer()
        if now - self._last_repaint < self._idle_interval:
            return
        line = self._progress_bar.update(
            self._active_step,
            processed=self._last_processed,
            total=self._last_total,
            force=True,
        )
        if line:
            self._last_repaint = now

    @property
    def idle_interval(self) -> float:
        """Return the maximum delay between spinner refresh attempts."""

        return self._idle_interval


class _ReportingStretchMove(moves.StretchMove):
    """Stretch move variant that emits per-walker progress notifications."""

    def __init__(
        self,
        *args: Any,
        progress_notifier: Callable[[int, int], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._progress_notifier = progress_notifier

    @classmethod
    def from_existing(
        cls,
        move: moves.StretchMove,
        *,
        progress_notifier: Callable[[int, int], None] | None = None,
    ) -> "_ReportingStretchMove":
        """Clone ``move`` while attaching ``progress_notifier``."""

        new_move = cls(
            getattr(move, "a", 2.0),
            nsplits=getattr(move, "nsplits", 2),
            randomize_split=getattr(move, "randomize_split", True),
            live_dangerously=getattr(move, "live_dangerously", False),
            progress_notifier=progress_notifier,
        )
        return new_move

    def set_progress_notifier(
        self, notifier: Callable[[int, int], None] | None
    ) -> None:
        """Update the callable receiving per-walker updates."""

        self._progress_notifier = notifier

    def _notify(self, processed: int, total: int) -> None:
        if self._progress_notifier is None:
            return
        try:
            self._progress_notifier(processed, total)
        except Exception:  # pragma: no cover - defensive safeguard
            pass

    def propose(self, model, state):  # type: ignore[override]
        """Generate proposals while reporting per-walker progress."""

        nwalkers, ndim = state.coords.shape
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError(
                "It is unadvisable to use a red-blue move "
                "with fewer walkers than twice the number of dimensions."
            )

        self.setup(state.coords)
        accepted = np.zeros(nwalkers, dtype=bool)
        all_inds = np.arange(nwalkers)
        inds = all_inds % self.nsplits
        if self.randomize_split:
            model.random.shuffle(inds)
        total_updates = max(nwalkers, 1)
        processed = 0

        for split in range(self.nsplits):
            S1 = inds == split
            sets = [state.coords[inds == j] for j in range(self.nsplits)]
            s = sets[split]
            c = [
                sets[index] for index in range(self.nsplits) if index != split
            ]

            q, factors = self.get_proposal(s, c, model.random)
            new_log_probs, new_blobs = model.compute_log_prob_fn(q)

            for j, f, nlp in zip(all_inds[S1], factors, new_log_probs):
                lnpdiff = f + nlp - state.log_prob[j]
                if lnpdiff > np.log(model.random.rand()):
                    accepted[j] = True
                processed += 1
                self._notify(processed, total_updates)

            new_state = State(q, log_prob=new_log_probs, blobs=new_blobs)
            state = self.update(state, new_state, accepted, S1)

        return state, accepted


def configure_sampler_progress_reporting(
    sampler: Any,
    notifier: Callable[[int, int], None] | None,
) -> None:
    """Ensure sampler moves forward updates to ``notifier`` when available."""

    moves_attr = getattr(sampler, "_moves", [])
    if not moves_attr:
        return

    updated_moves: list[Any] = []
    for entry in moves_attr:
        orientation = "bare"
        weight: Any | None = None
        move_obj: Any = entry

        if isinstance(entry, tuple) and len(entry) == 2:
            first, second = entry
            if isinstance(first, moves.Move):
                move_obj = first
                weight = second
                orientation = "move_weight"
            elif isinstance(second, moves.Move):
                move_obj = second
                weight = first
                orientation = "weight_move"
        elif isinstance(entry, list) and len(entry) == 2:
            first, second = entry
            if isinstance(first, moves.Move):
                move_obj = first
                weight = second
                orientation = "move_weight_list"
            elif isinstance(second, moves.Move):
                move_obj = second
                weight = first
                orientation = "weight_move_list"
        else:
            move_field = getattr(entry, "move", None)
            weight_field = getattr(entry, "weight", None)
            if isinstance(move_field, moves.Move) and weight_field is not None:
                move_obj = move_field
                weight = weight_field
                orientation = "weighted_move"

        if isinstance(move_obj, _ReportingStretchMove):
            move_obj.set_progress_notifier(notifier)
        elif isinstance(move_obj, moves.StretchMove):
            move_obj = _ReportingStretchMove.from_existing(
                move_obj, progress_notifier=notifier
            )

        if orientation == "move_weight":
            updated_moves.append((move_obj, weight))
        elif orientation == "weight_move":
            updated_moves.append((weight, move_obj))
        elif orientation == "move_weight_list":
            updated_moves.append([move_obj, weight])
        elif orientation == "weight_move_list":
            updated_moves.append([weight, move_obj])
        elif orientation == "weighted_move":
            cloned = type(entry)(move=move_obj, weight=weight)
            updated_moves.append(cloned)
        else:
            updated_moves.append(move_obj)

    if isinstance(moves_attr, tuple):
        sampler._moves = tuple(updated_moves)
    elif hasattr(moves_attr, "__class__") and hasattr(moves_attr, "moves"):
        moves_attr.moves = updated_moves
        sampler._moves = moves_attr
    else:
        sampler._moves = list(updated_moves)
