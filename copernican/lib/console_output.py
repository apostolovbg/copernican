# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Console I/O helpers shared across the Copernican Suite.

All user-facing text is funneled through this module. Once run logging is
configured, output becomes a logging record and the configured console
handler displays it exactly once. Before configuration, the helper writes
directly to the requested stream. Input remains routed through ``input`` so
the logger's prompt capture can record interactive exchanges.
"""

from __future__ import annotations

import logging
import sys


def _direct_print(msg: str, *, end: str, error: bool) -> None:
    """Print ``msg`` to the desired stream while handling Unicode safely."""

    stream = sys.stderr if error else sys.stdout
    try:
        print(msg, end=end, file=stream, flush=True)
    except UnicodeEncodeError:
        fallback = msg.encode("ascii", errors="replace").decode("ascii")
        print(fallback, end=end, file=stream, flush=True)


def write(msg: str = "", *, end: str = "\n", error: bool = False) -> None:
    """Display ``msg`` through the active logger or a direct stream."""

    active_logger = logging.getLogger()
    if active_logger.handlers:
        level = logging.ERROR if error else logging.INFO
        rendered = msg if end == "\n" else f"{msg}{end}"
        active_logger.log(level, rendered.rstrip("\n"))
        return
    _direct_print(msg, end=end, error=error)


def ask(prompt: str = "") -> str:
    """Prompt the user and return their input while logging the exchange."""

    return input(prompt)
