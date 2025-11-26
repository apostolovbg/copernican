# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Console I/O helpers shared across the Copernican Suite.

All user facing text is funneled through this module so that console
messages and prompts are handled in one place.  The logger patches
``print`` and ``input`` to capture output verbatim; these helpers provide
the indirection necessary to keep that behaviour consistent everywhere.
Keeping the wrapper centralised also ensures the project never writes
directly to ``stdout`` or ``stderr`` without passing through the
logging-aware hooks defined in :mod:`copernican_lib.logger`.
"""

import logging
import os
import sys


def write(
    msg: str = "",
    *,
    end: str = os.linesep,
    error: bool = False,
) -> None:
    """Display ``msg`` on the console and mirror it to the log file.

    Routing all prints through this function ensures the patched
    ``print``/``input`` hooks in :mod:`copernican_lib.logger` can record
    every message exactly once.  Direct calls to ``print`` should be
    avoided inside the project so that logs remain faithful. The stream
    is always flushed so progress lines using carriage returns remain
    visible on all platforms.

    Parameters
    ----------
    msg : str
        The text to print.
    end : str, optional
        String appended after the message. Defaults to a newline.
    error : bool, optional
        When ``True`` the message is sent to ``stderr`` rather than
        ``stdout``.

    The write is wrapped in a ``try`` block so terminals that cannot
    represent certain Unicode characters still receive output. Unencodable
    characters are replaced with ``?`` to avoid raising a
    :class:`UnicodeEncodeError`.
    """
    stream = sys.stderr if error else sys.stdout
    text = f"{msg}{end}"
    log_text = msg if end == os.linesep else text
    try:
        stream.write(text)
    except UnicodeEncodeError:
        fallback = msg.encode("ascii", errors="replace").decode("ascii")
        stream.write(f"{fallback}{end}")
    stream.flush()
    logger = logging.getLogger()
    logger.log(
        logging.ERROR if error else logging.INFO,
        log_text,
        extra={"console_capture": True},
    )


def ask(prompt: str = "") -> str:
    """Prompt the user and return their input while logging the exchange.

    The patched :func:`builtins.input` records both the prompt and the
    response to the active log file.  Wrapping the call here clarifies the
    intent and avoids scattering raw ``input`` calls across the codebase.
    """
    return input(prompt)
