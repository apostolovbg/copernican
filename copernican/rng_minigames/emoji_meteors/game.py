"""Emoji meteors mini-game used to forge GUI seeds."""

from __future__ import annotations

from random import SystemRandom

try:  # pragma: no cover - Tk only available when GUI rendering is enabled
    import tkinter as tkinter_module
    from tkinter import ttk
except ImportError:  # pragma: no cover - executed on headless environments
    tkinter_module = None
    ttk = None

from ..api import MinigameContext

SECURE_RANDOM = SystemRandom()

_EMOJI_METEOR_CHOICES = [
    "🐱",
    "🐶",
    "🦊",
    "🐻",
    "🐼",
    "🐨",
    "🐯",
    "🦁",
    "🐸",
    "🐙",
    "🦄",
    "🦉",
    "🐢",
    "🐧",
    "🐰",
    "🐭",
    "🦕",
    "🐋",
    "🦜",
    "🐝",
    "🦓",
    "🐎",
    "🐪",
    "🦥",
    "🐿️",
    "🦔",
    "🐌",
    "🦬",
    "🦩",
]


def launch_emoji_meteors(context: MinigameContext) -> None:
    """Open an interactive emoji picker that forges a whimsical seed."""

    def _apply_seed(picks: list[str]) -> None:
        """Map the selected emoji IDs into a numeric seed."""
        if not picks:
            return
        encoded_payload = "".join(
            f"{ord(symbol) % 1000:03d}" for symbol in picks
        )
        seed_value = encoded_payload.lstrip("0") or "0"
        context.set_seed(seed_value)
        context.notify(
            f"Emoji meteors {' '.join(picks)} forged seed {seed_value}.",
            "INFO",
        )

    if not context.render or tkinter_module is None or context.tk_root is None:
        _apply_seed(SECURE_RANDOM.sample(_EMOJI_METEOR_CHOICES, 5))
        return

    window = tkinter_module.Toplevel(context.tk_root)
    window.title("Emoji meteors")
    window.resizable(False, False)
    window.transient(context.tk_root)
    canvas_width = 760
    canvas_height = 480
    canvas = tkinter_module.Canvas(
        window,
        width=canvas_width,
        height=canvas_height,
        highlightthickness=0,
        background="#1d2b3a",
    )
    canvas.pack(padx=16, pady=(0, 8))
    instructions = ttk.Label(
        window,
        text=(
            "Pet five of us and we'll weave a seed from the color of our fur!"
        ),
        wraplength=720,
        padding=(12, 6),
    )
    instructions.pack(anchor="w", padx=16)
    selection_frame = ttk.Frame(window)
    selection_frame.pack(fill="x", padx=16, pady=(0, 6))
    status_var = tkinter_module.StringVar(value="Selections: (none yet)")
    ttk.Label(
        selection_frame,
        textvariable=status_var,
        padding=(0, 2),
    ).pack(side="left", anchor="w")
    button_frame = ttk.Frame(selection_frame)
    button_frame.pack(side="right", anchor="e")
    action_var = tkinter_module.StringVar()
    selections: list[str] = []
    meteor_items: dict[int, dict[str, object]] = {}

    def _spawn_meteor() -> tuple[int, dict[str, object]]:
        """Spawn a random emoji meteor with jittery motion."""
        emoji = SECURE_RANDOM.choice(_EMOJI_METEOR_CHOICES)
        x_pos = SECURE_RANDOM.randint(40, canvas_width - 40)
        y_pos = SECURE_RANDOM.randint(-canvas_height, -20)
        item_id = canvas.create_text(
            x_pos,
            y_pos,
            text=emoji,
            font=("Helvetica", 58),
        )
        meta = {
            "emoji": emoji,
            "speed": SECURE_RANDOM.uniform(1.5, 3.5),
        }
        return item_id, meta

    for _ in range(20):
        meteor_id, meteor_meta = _spawn_meteor()
        meteor_items[meteor_id] = meteor_meta

    after_id: str | None = None

    def _animate() -> None:
        """Advance all meteors down the canvas and wrap them."""
        nonlocal after_id
        for meteor_id, meteor_meta in meteor_items.items():
            canvas.move(meteor_id, 0, meteor_meta["speed"])
            x_pos, y_pos = canvas.coords(meteor_id)
            if y_pos > canvas_height + 50:
                new_x = SECURE_RANDOM.randint(40, canvas_width - 40)
                canvas.coords(meteor_id, new_x, -30)
                new_emoji = SECURE_RANDOM.choice(_EMOJI_METEOR_CHOICES)
                meteor_meta["emoji"] = new_emoji
                meteor_meta["speed"] = SECURE_RANDOM.uniform(1.5, 3.5)
                canvas.itemconfigure(meteor_id, text=new_emoji)
        after_id = canvas.after(60, _animate)

    def _finalize_and_close(picks: list[str]) -> None:
        """Apply the seed and tear down the window when selection completes."""
        if after_id:
            canvas.after_cancel(after_id)
        _apply_seed(picks)
        window.destroy()

    def _render_selection_status() -> None:
        """Update the UI label describing the current picks."""
        display = " ".join(selections)
        status_var.set(f"Selections: {display if display else '(none yet)'}")

    def _handle_click(event: "tkinter_module.Event") -> None:
        """Handle meteors being clicked to add them to the selection."""
        if len(selections) >= 5:
            return
        hits = canvas.find_closest(event.x, event.y)
        if not hits:
            return
        hit_id = hits[0]
        hit_meta = meteor_items.get(hit_id)
        if not hit_meta:
            return
        emoji = hit_meta["emoji"]
        if not isinstance(emoji, str):
            return
        selections.append(emoji)
        canvas.itemconfigure(hit_id, font=("Helvetica", 80))
        _render_selection_status()
        _redraw_preview()

    def _on_close() -> None:
        """Cancel the animation timer and close the window."""
        if after_id:
            canvas.after_cancel(after_id)
        window.destroy()

    canvas.bind("<Button-1>", _handle_click)
    window.protocol("WM_DELETE_WINDOW", _on_close)
    window.grab_set()
    _render_selection_status()
    ttk.Label(
        window,
        textvariable=action_var,
        padding=(16, 0),
    ).pack(anchor="w")
    ttk.Frame(window, height=4).pack(fill="x")
    button_bar = ttk.Frame(window)
    button_bar.pack(fill="x", padx=16, pady=(0, 12))
    selected_preview = ttk.Frame(button_bar)
    selected_preview.pack(side="left", anchor="w")

    def _redraw_preview() -> None:
        """Refresh the preview of selected emojis in the footer."""
        for child in selected_preview.winfo_children():
            child.destroy()
        if selections:
            for symbol in selections:
                ttk.Label(
                    selected_preview,
                    text=symbol,
                    font=("Helvetica", 28),
                ).pack(side="left", padx=2)
        else:
            ttk.Label(selected_preview, text="No animals yet.").pack(
                side="left"
            )

    def _handle_try_again() -> None:
        """Clear the selections and respawn all meteors."""
        nonlocal after_id
        selections.clear()
        _render_selection_status()
        _redraw_preview()
        for meteor_id, meteor_meta in meteor_items.items():
            new_x = SECURE_RANDOM.randint(40, canvas_width - 40)
            canvas.coords(meteor_id, new_x, -20)
            new_emoji = SECURE_RANDOM.choice(_EMOJI_METEOR_CHOICES)
            meteor_meta["emoji"] = new_emoji
            meteor_meta["speed"] = SECURE_RANDOM.uniform(1.5, 3.5)
            canvas.itemconfigure(
                meteor_id, text=new_emoji, font=("Helvetica", 58)
            )
        if after_id is None:
            _animate()

    def _handle_accept() -> None:
        """Confirm the selection if five emojis are chosen."""
        if len(selections) >= 5:
            _finalize_and_close(selections[:5])
        else:
            action_var.set("Catch at least five animals before confirming.")

    ttk.Button(
        button_bar,
        text="Cute enough",
        command=_handle_accept,
    ).pack(side="right", padx=(8, 0))
    ttk.Button(
        button_bar,
        text="Try again",
        command=_handle_try_again,
    ).pack(side="right")
    ttk.Button(
        button_bar,
        text="Cancel",
        command=_on_close,
    ).pack(side="right", padx=(0, 8))
    _redraw_preview()
    _animate()
