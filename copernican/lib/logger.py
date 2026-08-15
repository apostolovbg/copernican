# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.
# Copernican Suite Logger
"""Logging utilities for the Copernican Suite.

This module configures one worker-owned file log for each run. Console
messages and user prompts are captured without wrapping ``sys.stdout`` or
``sys.stderr``, while GUI workers emit structured console records for the
in-memory Run Monitor. Repository paths may be rendered relative to the
checkout, but external output paths remain absolute. Log timestamps use
Coordinated Universal Time (UTC) so diagnostics remain comparable across
machines.
"""

import builtins
import importlib
import json
import logging
import os
import platform
import sys
import threading
import time

from .utils import ensure_dir_exists, get_timestamp

_CONSOLE_CAPTURE_STATE = threading.local()
_WORKER_EVENT_PREFIX = "COPERNICAN_EVENT\t"


class _PathFormatter(logging.Formatter):
    """Render repository paths relatively without mutating log records."""

    def __init__(self, fmt: str, base_dir: str):
        """Store the repository root used only for rendered file output."""
        super().__init__(fmt)
        self.base_dir = os.path.abspath(base_dir)

    def format(self, record: logging.LogRecord) -> str:
        """Shorten repository paths in this handler's rendered text."""

        rendered = super().format(record)
        rendered = rendered.replace(self.base_dir + os.sep, "")
        return rendered.replace(self.base_dir, ".")


class _WorkerEventFormatter(logging.Formatter):
    """Serialize worker console records for the GUI monitor transport."""

    def format(self, record: logging.LogRecord) -> str:
        """Return one parseable event carrying severity and message."""

        message = super().format(record)
        payload = {
            "severity": record.levelname,
            "message": message,
        }
        return _WORKER_EVENT_PREFIX + json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
        )


class _ConsoleFilter(logging.Filter):
    """Filter to exclude captured console messages from the StreamHandler."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Skip records already printed via patched ``print``/``input``."""
        # When ``console_capture`` is True the message originated from
        # ``print`` or ``input``. Those messages are already displayed on the
        # console via the original call, so the stream handler should ignore
        # them to avoid duplicate lines.
        return not getattr(record, "console_capture", False)


def _close_handlers(logger: logging.Logger) -> None:
    """Detach and close handlers that are being replaced."""

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def _console_capture_active() -> bool:
    """Return ``True`` while mirrored console output is being logged."""

    return bool(getattr(_CONSOLE_CAPTURE_STATE, "active", False))


def _set_console_capture_active(active: bool) -> None:
    """Track whether console mirroring is already in progress."""

    _CONSOLE_CAPTURE_STATE.active = active


def _patch_builtins(base_dir: str) -> None:
    """Mirror direct print and input calls without wrapping streams."""

    logger = logging.getLogger()
    # Patch each builtin independently so repeated setup calls can recover
    # from partial test-time monkeypatching without double-wrapping.
    print_is_patched = getattr(builtins.print, "__copernican_patched__", False)
    input_is_patched = getattr(builtins.input, "__copernican_patched__", False)
    orig_print = builtins.print
    orig_input = builtins.input

    def _log_console_message(text: str, level: int = logging.INFO) -> None:
        """Log console lines that would otherwise only hit stdout/stderr."""

        cleaned = text.rstrip("\n")
        if not cleaned.strip():
            return
        if _console_capture_active():
            return
        _set_console_capture_active(True)
        try:
            logger.log(
                level,
                cleaned,
                extra={"console_capture": True},
            )
        finally:
            _set_console_capture_active(False)

    def print_patch(*args, **kwargs):
        """Proxy ``print`` that mirrors output to the log file."""
        orig_print(*args, **kwargs)
        destination = kwargs.get("file", sys.stdout)
        if destination is sys.stdout or destination is sys.stderr:
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            message = sep.join(str(argument) for argument in args)
            if end != "\n":
                message += end
            level = (
                logging.ERROR if destination is sys.stderr else logging.INFO
            )
            _log_console_message(message, level)

    def input_patch(prompt: str = ""):
        """Proxy ``input`` that logs the prompt and response."""
        response = orig_input(prompt)
        _log_console_message(f"{prompt}{response}")
        return response

    if not print_is_patched:
        print_patch.__copernican_patched__ = True
        builtins.print = print_patch
    if not input_is_patched:
        input_patch.__copernican_patched__ = True
        builtins.input = input_patch


def ensure_console_capture(base_dir: str) -> None:
    """Install the patched ``print``/``input`` functions if needed."""

    _patch_builtins(base_dir)


def parse_worker_event(line: str) -> tuple[int, str] | None:
    """Parse one structured GUI-worker console record."""

    if not line.startswith(_WORKER_EVENT_PREFIX):
        return None
    try:
        payload = json.loads(line[len(_WORKER_EVENT_PREFIX) :])
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    severity = payload.get("severity")
    if not isinstance(message, str) or not isinstance(severity, str):
        return None
    level = getattr(logging, severity.upper(), None)
    if not isinstance(level, int):
        return None
    return level, message


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
    file_tag = log_tag or f"copernican-run_{get_timestamp()}.txt"
    if not file_tag.endswith(".txt"):
        file_tag = f"{file_tag}.txt"
    log_filename = os.path.abspath(os.path.join(log_dir, file_tag))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        if getattr(handler, "_copernican_run_log_path", None) == log_filename:
            if base_dir:
                _patch_builtins(base_dir)
            return log_filename
    _close_handlers(logger)

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    format_string = "%(asctime)s - %(levelname)s - %(message)s"
    if base_dir:
        formatter = _PathFormatter(format_string, base_dir)
    else:
        formatter = logging.Formatter(format_string)
    formatter.converter = time.gmtime
    file_handler.setFormatter(formatter)
    file_handler._copernican_run_log_path = os.path.abspath(log_filename)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    if os.environ.get("COPERNICAN_GUI_EVENT_STREAM") == "1":
        console_handler.setFormatter(_WorkerEventFormatter("%(message)s"))
    else:
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    # Exclude messages that already appeared on the console via patched
    # ``print``/``input`` calls.
    console_handler.addFilter(_ConsoleFilter())
    logger.addHandler(console_handler)

    logging.info(
        f"Logging initialized with UTC timestamps. Log file: {log_filename}"
    )

    if base_dir:
        _patch_builtins(base_dir)

    return log_filename


def setup_monitor_logger() -> logging.Logger:
    """Return the memory-only logger used by the GUI Run Monitor."""

    logger_obj = logging.getLogger("copernican.gui.run")
    logger_obj.setLevel(logging.INFO)
    logger_obj.propagate = False
    _close_handlers(logger_obj)
    return logger_obj


def log_environment_info(
    target_logger: logging.Logger | None = None,
    *,
    console: bool = True,
) -> None:
    """Log Python, OS, CPU and key package versions.

    When ``console`` is ``True`` the stream handler prints each line so the
    caller sees the environment details immediately; otherwise the records stay
    on the active logger only.
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
        except ImportError:
            pkgs[name] = "not installed"
    logger.info("Environment details:", **log_kwargs)
    logger.info(f"  Python: {py_ver}", **log_kwargs)
    logger.info(f"  OS: {os_info}", **log_kwargs)
    logger.info(f"  CPU: {cpu}", **log_kwargs)
    for name, version in pkgs.items():
        logger.info(f"  {name} {version}", **log_kwargs)
    summary = f"Python {py_ver}; {os_info}; CPU {cpu}; " + ", ".join(
        f"{package_name} {package_version}"
        for package_name, package_version in pkgs.items()
    )
    logger.info(f"Environment summary: {summary}", **log_kwargs)


def get_logger():
    """Returns the active logger instance."""
    return logging.getLogger()
