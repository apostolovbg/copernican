# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Console I/O helpers shared across the Copernican Suite.

All user facing text is funneled through this module so that the patched
``print``/``input`` hooks in :mod:`copernican_lib.logger` can capture every
message exactly once while the helpers centralize which outputs should be
mirrored to the active log file.  Wrapping the calls also keeps the terminal
display consistent and allows per-message control over whether a line should
be persisted.  The helpers always flush so carriage-return based progress bars
remain visible on every platform.
"""

from __future__ import annotations

import contextlib
import sys
import threading

_LOGGING_SUPPRESSION = threading.local()


@contextlib.contextmanager
def _suppress_console_logging():
    """Temporarily disable logging for patched ``print`` calls."""

    previous = getattr(_LOGGING_SUPPRESSION, "value", False)
    _LOGGING_SUPPRESSION.value = True
    try:
        yield
    finally:
        _LOGGING_SUPPRESSION.value = previous


def console_logging_suppressed() -> bool:
    """Return whether console output currently suppresses logging."""

    return getattr(_LOGGING_SUPPRESSION, "value", False)


def _direct_print(msg: str, *, end: str, error: bool) -> None:
    """Print ``msg`` to the desired stream while handling Unicode safely."""

    stream = sys.stderr if error else sys.stdout
    try:
        print(msg, end=end, file=stream, flush=True)
    except UnicodeEncodeError:
        fallback = msg.encode("ascii", errors="replace").decode("ascii")
        print(fallback, end=end, file=stream, flush=True)


def write(
    msg: str = "",
    *,
    end: str = "\n",
    error: bool = False,
    log: bool = True,
) -> None:
    """Display ``msg`` and optionally mirror it to the log file.

    Parameters
    ----------
    msg : str
        The text to print.
    end : str, optional
        String appended after the message. Defaults to a newline.
    error : bool, optional
        When ``True`` the message is sent to ``stderr`` instead of ``stdout``.
    log : bool, optional
        When ``False`` the patched ``print`` hook temporarily suppresses
        logging so the message remains visible without growing the log file.
        This is useful for high-frequency progress bars or status refreshes.
    """

    if log:
        _direct_print(msg, end=end, error=error)
        return
    with _suppress_console_logging():
        _direct_print(msg, end=end, error=error)


def ask(prompt: str = "") -> str:
    """Prompt the user and return their input while logging the exchange."""

    return input(prompt)
