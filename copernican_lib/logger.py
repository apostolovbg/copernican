# Copernican Suite Logger
"""Logging utilities for the Copernican Suite.

The logger records all console input and output verbatim while also emitting
standard log messages. ``print`` and ``input`` are patched so their text is
captured to the log file without being echoed twice on the console.
"""

import builtins
import logging
import os
import sys

from .utils import ensure_dir_exists, get_timestamp


class _PathFilter(logging.Filter):
    """Filter that strips absolute paths above the project root."""

    def __init__(self, base_dir: str):
        """Store repository root for later path stripping."""
        super().__init__()
        self.base_dir = os.path.abspath(base_dir)

    def filter(self, record: logging.LogRecord) -> bool:
        """Remove leading ``base_dir`` segments from log messages."""
        if isinstance(record.msg, str):
            # Replace absolute base paths with project-relative forms
            clean = record.msg.replace(self.base_dir + os.sep, "")
            record.msg = clean.replace(self.base_dir, ".")
        return True


class _ConsoleFilter(logging.Filter):
    """Filter to exclude captured console messages from the StreamHandler."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Skip records already printed via patched ``print``/``input``."""
        # When ``console_capture`` is True the message originated from
        # ``print`` or ``input``. Those messages are already displayed on the
        # console via the original call, so the stream handler should ignore
        # them to avoid duplicate lines.
        return not getattr(record, "console_capture", False)


def _patch_builtins(base_dir: str) -> None:
    """Mirror print and input to the logger with path sanitisation."""

    logger = logging.getLogger()
    if getattr(builtins.print, "__copernican_patched__", False):
        return

    orig_print = builtins.print
    orig_input = builtins.input

    def _shorten(msg: str) -> str:
        """Replace absolute paths in ``msg`` with project-relative forms."""
        base = os.path.abspath(base_dir)
        return msg.replace(base + os.sep, "").replace(base, ".")

    def print_patch(*args, **kwargs):
        """Proxy ``print`` that mirrors output to the log file."""
        orig_print(*args, **kwargs)
        if kwargs.get("file", sys.stdout) is sys.stdout:
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            message = sep.join(str(a) for a in args)
            if end != "\n":
                message += end
            logger.info(
                _shorten(message),
                extra={"console_capture": True},
            )

    def input_patch(prompt: str = ""):
        """Proxy ``input`` that logs the prompt and response."""
        response = orig_input(prompt)
        logger.info(
            _shorten(f"{prompt}{response}"),
            extra={"console_capture": True},
        )
        return response

    print_patch.__copernican_patched__ = True
    input_patch.__copernican_patched__ = True
    builtins.print = print_patch
    builtins.input = input_patch


def setup_logging(log_dir: str = ".", base_dir: str | None = None) -> str:
    """Initializes logging handlers and patches ``print``/``input``."""

    ensure_dir_exists(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Name log file using an execution timestamp
    run_tag = f"copernican-run_{get_timestamp()}.txt"
    log_filename = os.path.join(log_dir, run_tag)

    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    if base_dir:
        fh.addFilter(_PathFilter(base_dir))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    # Exclude messages that already appeared on the console via patched
    # ``print``/``input`` calls.
    ch.addFilter(_ConsoleFilter())
    logger.addHandler(ch)

    logging.info(f"Logging initialized. Log file: {log_filename}")

    if base_dir:
        _patch_builtins(base_dir)

    return log_filename


def get_logger():
    """Returns the active logger instance."""
    return logging.getLogger()
