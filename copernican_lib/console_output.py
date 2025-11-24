# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.
# Last Updated: 2025-11-24

"""Console I/O helpers shared across the Copernican Suite.

All user facing text is funneled through this module so that console
messages and prompts are handled in one place.  The logger patches
``print`` and ``input`` to capture output verbatim; these helpers provide
the indirection necessary to keep that behaviour consistent everywhere.
Keeping the wrapper centralised also ensures the project never writes
directly to ``stdout`` or ``stderr`` without passing through the
logging-aware hooks defined in :mod:`copernican_lib.logger`.
"""

import os
import sys


def _read_from_windows(valid_inputs: set[str]) -> str:
    """Read a single keypress on Windows without echoing the input."""

    import msvcrt

    while True:
        key = msvcrt.getwch()
        if key == "\x03":  # Ctrl+C
            raise KeyboardInterrupt
        lower = key.lower()
        if lower in valid_inputs:
            return lower


def _read_from_posix(valid_inputs: set[str]) -> str:
    """Read a single keypress on POSIX terminals without echoing."""

    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    new_settings = termios.tcgetattr(fd)
    new_settings[3] &= ~(termios.ECHO | termios.ICANON)
    termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
    try:
        while True:
            key = sys.stdin.read(1)
            if not key:
                continue
            if key == "\x03":  # Ctrl+C
                raise KeyboardInterrupt
            lower = key.lower()
            if lower in valid_inputs:
                return lower
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def write(msg: str = "", *, end: str = "\n", error: bool = False) -> None:
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
    try:
        print(msg, end=end, file=stream, flush=True)
    except UnicodeEncodeError:
        fallback = msg.encode("ascii", errors="replace").decode("ascii")
        print(fallback, end=end, file=stream, flush=True)


def ask(prompt: str = "") -> str:
    """Prompt the user and return their input while logging the exchange.

    The patched :func:`builtins.input` records both the prompt and the
    response to the active log file.  Wrapping the call here clarifies the
    intent and avoids scattering raw ``input`` calls across the codebase.
    """
    return input(prompt)


def read_keypress(valid_inputs: set[str], *, prompt: str = "") -> str:
    """Capture a single keypress without requiring the Enter key.

    The dashboard menus keep their layout intact while waiting for input by
    suppressing character echo and consuming exactly one keystroke. The
    helper normalises keys to lower-case to simplify downstream comparisons
    and falls back to the standard :func:`input` prompt when stdin is not a
    terminal, such as during unit tests or automated runs.

    Parameters
    ----------
    valid_inputs : set[str]
        Normalised list of accepted keys. The function blocks until one of
        these values is pressed.
    prompt : str, optional
        Prompt displayed before listening for keypresses. Defaults to a blank
        string so menus can remain on screen.

    Returns
    -------
    str
        The selected key, always lower-case.
    """

    write(prompt, end="")
    normalised = {value.lower() for value in valid_inputs}
    if not normalised:
        return ""

    if not sys.stdin.isatty():
        return ask(prompt).strip().lower()

    if os.name == "nt":
        selected = _read_from_windows(normalised)
    else:
        selected = _read_from_posix(normalised)

    write("")
    return selected
