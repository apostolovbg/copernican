"""Console output utilities for the Copernican Suite."""

import sys


def write(msg: str = "", *, end: str = "\n", error: bool = False) -> None:
    """Display ``msg`` on the console and let the logger capture it.

    Parameters
    ----------
    msg : str
        The text to print.
    end : str, optional
        String appended after the message. Defaults to a newline.
    error : bool, optional
        When ``True`` output is written to ``stderr`` instead of ``stdout``.
    """
    stream = sys.stderr if error else sys.stdout
    print(msg, end=end, file=stream)


def ask(prompt: str = "") -> str:
    """Prompt the user and return their input."""
    return input(prompt)
