"""Shared data structures for RNG mini-games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


NotifyCallback = Callable[[str, str], None]
SetSeedCallback = Callable[[str], None]


@dataclass
class MinigameContext:
    """Runtime context passed to every mini-game launcher.

    Attributes:
        set_seed: Callback invoked when the game produces a deterministic seed.
        notify: Callback used to display information/warnings/errors.
        render: Whether Tk rendering is available.
        tk_root: Optional root window to attach child dialogs to.
    """

    set_seed: SetSeedCallback
    notify: NotifyCallback
    render: bool = False
    tk_root: Any | None = None
