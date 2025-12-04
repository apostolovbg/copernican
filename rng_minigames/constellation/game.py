"""Constellation mini-game for seed creation."""

from __future__ import annotations

import hashlib
import random
import time
from typing import TYPE_CHECKING

try:  # pragma: no cover - Tk is optional
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - executed when Tk is missing
    tk = None
    ttk = None

from rng_minigames.api import MinigameContext


def launch_constellation(context: MinigameContext) -> None:
    """Render the constellation mini-game."""

    target_connections = 10

    def _apply_seed(selection: list[int], duration: float) -> None:
        if not selection:
            return
        payload = "".join(f"{index:02d}" for index in selection)
        payload += f"{int(duration * 1000):06d}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        seed_value = str(int(digest[:12], 16))
        context.set_seed(seed_value)
        context.notify(
            f"Constellation forged seed {seed_value}.",
            "INFO",
        )

    if not context.render or tk is None or context.tk_root is None:
        random_selection = random.sample(range(300), target_connections)
        _apply_seed(random_selection, random.random() * 10)
        return

    window = tk.Toplevel(context.tk_root)
    window.title("Constellation")
    window.resizable(False, False)
    window.transient(context.tk_root)
    canvas_width = 760
    canvas_height = 480
    canvas = tk.Canvas(
        window,
        width=canvas_width,
        height=canvas_height,
        highlightthickness=0,
        background="#030a16",
    )
    canvas.pack(padx=16, pady=(0, 8))
    star_count = 260
    stars: list[dict[str, object]] = []
    random.seed()

    def _clamp(value: float) -> int:
        return max(0, min(255, int(round(value))))

    for index in range(star_count):
        x_pos = random.randint(10, canvas_width - 10)
        y_pos = random.randint(10, canvas_height - 10)
        magnitude = random.uniform(1.2, 3.5)
        tint = random.randint(200, 255)
        hue_shift = random.randint(-25, 25)
        r_val = _clamp(tint + hue_shift)
        g_val = _clamp(tint + hue_shift // 2)
        b_val = _clamp(tint + hue_shift // 3)
        color = f"#{r_val:02x}{g_val:02x}{b_val:02x}"
        star_id = canvas.create_oval(
            x_pos - magnitude,
            y_pos - magnitude,
            x_pos + magnitude,
            y_pos + magnitude,
            fill=color,
            outline="",
        )
        stars.append(
            {
                "id": star_id,
                "x": x_pos,
                "y": y_pos,
                "radius": magnitude,
                "color": color,
            }
        )

    instructions = ttk.Label(
        window,
        text=(
            "Trace a constellation by connecting ten stars. "
            "Left-click to add a star, right-click to remove one, and follow "
            "the glowing lines until all ten are linked."
        ),
        wraplength=720,
        padding=(12, 6),
    )
    instructions.pack(anchor="w", padx=16)
    selection_frame = ttk.Frame(window)
    selection_frame.pack(fill="x", padx=16, pady=(0, 6))
    status_var = tk.StringVar(value=f"Stars connected: 0/{target_connections}")
    ttk.Label(
        selection_frame,
        textvariable=status_var,
        padding=(0, 2),
    ).pack(side="left", anchor="w")
    button_frame = ttk.Frame(selection_frame)
    button_frame.pack(side="right", anchor="e")
    action_var = tk.StringVar(
        value="Connect 10 stars to forge your constellation."
    )
    ttk.Label(window, textvariable=action_var, padding=(16, 0)).pack(
        anchor="w"
    )
    ttk.Frame(window, height=4).pack(fill="x")
    button_bar = ttk.Frame(window)
    button_bar.pack(fill="x", padx=16, pady=(0, 12))
    selected_preview = ttk.Frame(button_bar)
    selected_preview.pack(side="left", anchor="w")

    selected_indices: list[int] = []
    halo_items: dict[int, int] = {}
    line_items: list[int] = []
    start_time = time.time()
    accept_button: ttk.Button | None = None

    def _render_selection_status() -> None:
        count = len(selected_indices)
        status_var.set(f"Stars connected: {count}/{target_connections}")
        if count < target_connections:
            remaining = target_connections - count
            suffix = "" if remaining == 1 else "s"
            action_var.set(f"Connect {remaining} more star{suffix}.")
        else:
            message = 'Constellation ready! Click "Ad astra!" to confirm.'
            action_var.set(message)

    def _redraw_preview() -> None:
        for child in selected_preview.winfo_children():
            child.destroy()
        if selected_indices:
            for idx in selected_indices:
                ttk.Label(
                    selected_preview,
                    text=f"★{idx}",
                    font=("Helvetica", 16),
                ).pack(side="left", padx=2)
        else:
            ttk.Label(selected_preview, text="No stars selected yet.").pack(
                side="left"
            )

    def _redraw_lines() -> None:
        for line in line_items:
            canvas.delete(line)
        line_items.clear()
        for first, second in zip(selected_indices[:-1], selected_indices[1:]):
            start_star = stars[first]
            end_star = stars[second]
            line_items.append(
                canvas.create_line(
                    start_star["x"],
                    start_star["y"],
                    end_star["x"],
                    end_star["y"],
                    fill="#6fc3ff",
                    width=2,
                )
            )

    def _highlight_star(index: int) -> None:
        star = stars[index]
        halo = canvas.create_oval(
            star["x"] - star["radius"] - 6,
            star["y"] - star["radius"] - 6,
            star["x"] + star["radius"] + 6,
            star["y"] + star["radius"] + 6,
            outline="#ffe28a",
            width=2,
        )
        canvas.itemconfigure(star["id"], fill="#ffd966")
        halo_items[index] = halo

    def _remove_highlight(index: int) -> None:
        star = stars[index]
        canvas.itemconfigure(star["id"], fill=star["color"])
        halo = halo_items.pop(index, None)
        if halo:
            canvas.delete(halo)

    def _reset_selection() -> None:
        for line in line_items:
            canvas.delete(line)
        line_items.clear()
        for index in selected_indices:
            _remove_highlight(index)
        selected_indices.clear()
        _render_selection_status()
        _redraw_preview()
        if accept_button is not None:
            accept_button.state(["disabled"])

    def _finalize() -> None:
        duration = time.time() - start_time
        _apply_seed(selected_indices[:], duration)
        window.destroy()

    def _cancel_window() -> None:
        window.destroy()

    def _handle_click(event: "tk.Event") -> None:
        if len(selected_indices) >= target_connections:
            return
        nearest = None
        nearest_dist = float("inf")
        for idx, star in enumerate(stars):
            dx = event.x - star["x"]
            dy = event.y - star["y"]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < star["radius"] + 8 and dist < nearest_dist:
                nearest = idx
                nearest_dist = dist
        if nearest is None or nearest in selected_indices:
            return
        selected_indices.append(nearest)
        _highlight_star(nearest)
        _redraw_lines()
        _render_selection_status()
        _redraw_preview()
        if len(selected_indices) >= target_connections and accept_button:
            accept_button.state(["!disabled"])

    def _handle_right_click(event: "tk.Event") -> None:
        if not selected_indices:
            return
        nearest = None
        nearest_dist = float("inf")
        for idx in selected_indices:
            star = stars[idx]
            dx = event.x - star["x"]
            dy = event.y - star["y"]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < star["radius"] + 10 and dist < nearest_dist:
                nearest = idx
                nearest_dist = dist
        if nearest is None:
            return
        selected_indices.remove(nearest)
        _remove_highlight(nearest)
        _redraw_lines()
        _render_selection_status()
        _redraw_preview()
        if (
            accept_button is not None
            and len(selected_indices) < target_connections
        ):
            accept_button.state(["disabled"])

    canvas.bind("<Button-1>", _handle_click)
    canvas.bind("<Button-3>", _handle_right_click)
    window.protocol("WM_DELETE_WINDOW", _cancel_window)
    window.grab_set()
    _render_selection_status()
    _redraw_preview()

    accept_button = ttk.Button(
        button_bar,
        text="Ad astra!",
        command=_finalize,
        state=tk.DISABLED,
    )
    accept_button.pack(side="right", padx=(8, 0))
    ttk.Button(
        button_bar,
        text="Try again",
        command=_reset_selection,
    ).pack(side="right")
    ttk.Button(
        button_bar,
        text="Cancel",
        command=_cancel_window,
    ).pack(side="right", padx=(0, 8))
