# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# Copernican Suite Logger
"""Logging utilities for the Copernican Suite.

This module wraps :mod:`logging` configuration so that every run produces a
fully self-contained log file.  Console messages and user prompts are
mirrored verbatim by patching ``print`` and ``input``; path information is
sanitised so absolute directories outside the repository are not leaked.
Consumers can therefore rely on the log for complete provenance of a
session without clutter or duplicated lines.
"""

import builtins
import importlib
import logging
import os
import platform
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
    """Initialise logging and return the log file path.

    A file handler stores timestamped records while a stream handler echoes
    messages to the console.  The routine also patches ``print`` and
    ``input`` so that all interactive exchanges are captured.  When
    ``base_dir`` is provided, absolute paths inside log messages are
    shortened to keep the output relocatable.
    """

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


def log_environment_info() -> None:
    """Log Python, OS, CPU and key package versions.

    Detailed information is written to the log file while a short
    summary prints to the console. This aids in reproducing results
    across different systems.
    """
    logger = logging.getLogger()
    py_ver = platform.python_version()
    os_info = platform.platform()
    cpu = platform.processor() or "Unknown CPU"
    pkgs = {}
    for name in ("numpy", "scipy", "matplotlib"):
        try:
            mod = importlib.import_module(name)
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkgs[name] = "not installed"
    logger.info("Environment details:")
    logger.info(f"  Python: {py_ver}")
    logger.info(f"  OS: {os_info}")
    logger.info(f"  CPU: {cpu}")
    for n, v in pkgs.items():
        logger.info(f"  {n}: {v}")
    summary = f"Python {py_ver}; {os_info}; CPU {cpu}; " + ", ".join(
        f"{n} {v}" for n, v in pkgs.items()
    )
    print(f"Environment summary: {summary}")


def get_logger():
    """Returns the active logger instance."""
    return logging.getLogger()
