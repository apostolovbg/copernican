# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.
# Copernican Suite Logger
"""Logging utilities for the Copernican Suite.

This module wraps :mod:`logging` configuration so that every run produces a
fully self-contained log file.  Console messages and user prompts are
mirrored verbatim by patching ``print`` and ``input``; path information is
sanitised so absolute directories outside the repository are not leaked.
Consumers can therefore rely on the log for complete provenance of a
session without clutter or duplicated lines.  Log timestamps are emitted in
Coordinated Universal Time (UTC) so diagnostics remain comparable across
machines in different time zones.
"""

import builtins
import importlib
import logging
import os
import platform
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, TextIO

from . import console_output
from .run_lifecycle import MAX_PROGRAM_LOGS, prepare_program_log_path
from .utils import ensure_dir_exists, get_timestamp

_PROGRAM_LOGGER_NAME = "copernican.program"
_PROGRAM_LOGGER: logging.Logger | None = None
_PROGRAM_LOG_PATH: str | None = None


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
    # Avoid patching multiple times when modules reload during tests. The
    # marker attribute keeps repeated calls idempotent while still allowing
    # other libraries to introspect the wrapped functions.
    if getattr(builtins.print, "__copernican_patched__", False):
        return

    orig_print = builtins.print
    orig_input = builtins.input

    def _shorten(msg: str) -> str:
        """Replace absolute paths in ``msg`` with project-relative forms."""
        base = os.path.abspath(base_dir)
        return msg.replace(base + os.sep, "").replace(base, ".")

    def _log_console_message(text: str) -> None:
        """Log console lines that would otherwise only hit stdout/stderr."""

        if console_output.console_logging_suppressed():
            return
        cleaned = _shorten(text).rstrip("\n")
        if not cleaned.strip():
            return
        logger.info(
            cleaned,
            extra={"console_capture": True},
        )
        program_logger = _PROGRAM_LOGGER
        if program_logger is not None:
            program_logger.info(
                cleaned,
                extra={"console_capture": True},
            )

    def print_patch(*args, **kwargs):
        """Proxy ``print`` that mirrors output to the log file."""
        orig_print(*args, **kwargs)
        if (
            kwargs.get("file", sys.stdout) is sys.stdout
            and not console_output.console_logging_suppressed()
        ):
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            message = sep.join(str(a) for a in args)
            if end != "\n":
                message += end
            _log_console_message(message)

    def input_patch(prompt: str = ""):
        """Proxy ``input`` that logs the prompt and response."""
        response = orig_input(prompt)
        _log_console_message(f"{prompt}{response}")
        return response

    print_patch.__copernican_patched__ = True
    input_patch.__copernican_patched__ = True
    builtins.print = print_patch
    builtins.input = input_patch

    class _StreamProxy:
        """Proxy that mirrors stream writes to the logger."""

        def __init__(self, stream: TextIO) -> None:
            self._stream = stream

        def write(self, data: str) -> None:
            self._stream.write(data)
            if data:
                _log_console_message(data)

        def flush(self) -> None:
            self._stream.flush()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._stream, name)

    if not isinstance(sys.stdout, _StreamProxy):
        sys.stdout = _StreamProxy(sys.stdout)
    if not isinstance(sys.stderr, _StreamProxy):
        sys.stderr = _StreamProxy(sys.stderr)


def ensure_console_capture(base_dir: str) -> None:
    """Install the patched ``print``/``input`` functions if needed."""

    _patch_builtins(base_dir)


def setup_program_logging(
    log_dir: str = "logs",
    *,
    base_dir: str | None = None,
    rollover_mb: float = 10.0,
    backup_count: int = 3,
    log_level: str = "INFO",
    max_logs: int | None = None,
) -> str:
    """Initialise the rotating diagnostics log for the suite shell.

    Program-level events such as menu navigation, queue operations and
    configuration loading are captured separately from per-run logs so
    developers can inspect launcher issues without combing through
    scientific results.  The handler rotates automatically when
    ``rollover_mb`` is exceeded to prevent unbounded growth during long
    sessions.  Logs live outside the run directories to keep Git history
    clean and to avoid bundling diagnostics with reproducibility
    artifacts.  ``log_level`` controls the logged severity threshold while
    ``max_logs`` dictates how many archived files are retained.
    """

    ensure_dir_exists(log_dir)
    max_logs = max_logs if max_logs is not None else MAX_PROGRAM_LOGS
    max_logs = max(int(max_logs), 1)

    log_path = str(
        prepare_program_log_path(
            Path(log_dir),
            max_logs=max_logs,
        )
    )

    logger_obj = logging.getLogger(_PROGRAM_LOGGER_NAME)
    level_name = str(log_level).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger_obj.setLevel(level)
    logger_obj.propagate = False
    for handler in logger_obj.handlers[:]:
        logger_obj.removeHandler(handler)

    max_bytes = max(1, int(rollover_mb * 1024 * 1024))
    rotating_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    formatter.converter = time.gmtime
    rotating_handler.setFormatter(formatter)
    if base_dir:
        rotating_handler.addFilter(_PathFilter(base_dir))
    logger_obj.addHandler(rotating_handler)

    logger_obj.info(
        (
            "Program diagnostics log initialised at %s (rollover %.2f MB, "
            "backup count %d)"
        ),
        log_path,
        max_bytes / (1024 * 1024),
        backup_count,
    )

    global _PROGRAM_LOGGER, _PROGRAM_LOG_PATH
    _PROGRAM_LOGGER = logger_obj
    _PROGRAM_LOG_PATH = log_path
    return log_path


def setup_logging(
    log_dir: str = ".",
    base_dir: str | None = None,
    *,
    log_tag: str | None = None,
) -> str:
    """Initialise logging and return the log file path.

    A file handler stores timestamped records while a stream handler echoes
    messages to the console.  The routine also patches ``print`` and
    ``input`` so that all interactive exchanges are captured.  When
    ``base_dir`` is provided, absolute paths inside log messages are
    shortened to keep the output relocatable.  ``log_tag`` may be provided to
    force the log filename (with or without ``.txt``) so callers can align the
    log with a predetermined timestamp.
    """

    ensure_dir_exists(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Name log file using an execution timestamp or explicit tag
    file_tag = log_tag or f"copernican-run_{get_timestamp()}.txt"
    if not file_tag.endswith(".txt"):
        file_tag = f"{file_tag}.txt"
    log_filename = os.path.join(log_dir, file_tag)

    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    formatter.converter = time.gmtime
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

    logging.info(
        f"Logging initialized with UTC timestamps. Log file: {log_filename}"
    )

    if base_dir:
        _patch_builtins(base_dir)

    return log_filename


def log_environment_info(
    target_logger: logging.Logger | None = None,
    *,
    console: bool = True,
) -> None:
    """Log Python, OS, CPU and key package versions.

    When ``console`` is ``True`` the stream handler prints each line so the
    caller sees the environment details immediately; otherwise the records stay
    in the program log only.
    """

    logger = target_logger or logging.getLogger()
    log_kwargs = {"extra": {"console_capture": not console}}
    py_ver = platform.python_version()
    os_info = platform.platform()
    cpu = platform.processor() or "Unknown CPU"
    pkgs: dict[str, str] = {}
    for name in ("numpy", "scipy", "matplotlib"):
        try:
            mod = importlib.import_module(name)
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkgs[name] = "not installed"
    logger.info("Environment details:", **log_kwargs)
    logger.info(f"  Python: {py_ver}", **log_kwargs)
    logger.info(f"  OS: {os_info}", **log_kwargs)
    logger.info(f"  CPU: {cpu}", **log_kwargs)
    for name, version in pkgs.items():
        logger.info(f"  {name} {version}", **log_kwargs)
    summary = f"Python {py_ver}; {os_info}; CPU {cpu}; " + ", ".join(
        f"{n} {v}" for n, v in pkgs.items()
    )
    logger.info(f"Environment summary: {summary}", **log_kwargs)


def get_program_logger() -> logging.Logger:
    """Return the suite-level diagnostics logger."""

    global _PROGRAM_LOGGER
    if _PROGRAM_LOGGER is None:
        _PROGRAM_LOGGER = logging.getLogger(_PROGRAM_LOGGER_NAME)
    return _PROGRAM_LOGGER


def get_logger():
    """Returns the active logger instance."""
    return logging.getLogger()


def get_program_log_path() -> str | None:
    """Return the active program log path, if available."""

    return _PROGRAM_LOG_PATH
