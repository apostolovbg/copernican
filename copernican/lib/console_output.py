# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Console I/O helpers shared across the Copernican Suite.

All user facing text is funneled through this module so that the patched
``print``/``input`` hooks in :mod:`copernican.lib.logger` can capture every
message exactly once while the helpers centralize which outputs should be
persisted.  Wrapping the calls keeps the terminal display consistent and
ensures console text is mirrored to the shared application logger.
"""

from __future__ import annotations

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
    """Display ``msg`` and let patched streams mirror it to the log."""

    _direct_print(msg, end=end, error=error)


def ask(prompt: str = "") -> str:
    """Prompt the user and return their input while logging the exchange."""

    return input(prompt)
