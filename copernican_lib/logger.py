# Copernican Suite Logger
"""Logging utilities for the Copernican Suite."""
# The application writes human-readable logs both to the console and to a file
# in the output directory. This module centralises the setup so every module can
# retrieve the same logger instance.

import logging
import os
import sys
import builtins
from .utils import get_timestamp, ensure_dir_exists


class ConsoleCapture:
    """Tee stdout and user input to the active log file."""

    def __init__(self, log_file_handle):
        self.log_file = log_file_handle
        self._orig_stdout = sys.stdout
        self._orig_input = builtins.input

    def write(self, data):
        self.log_file.write(data)
        self._orig_stdout.write(data)

    def flush(self):
        self.log_file.flush()
        self._orig_stdout.flush()

    def input(self, prompt=""):
        self.write(prompt)
        response = self._orig_input(prompt)
        self.write(response + "\n")
        return response

    def start(self):
        sys.stdout = self
        builtins.input = self.input

    def stop(self):
        sys.stdout = self._orig_stdout
        builtins.input = self._orig_input


def setup_logging(log_dir="."):
    """Initializes logging to both console and a file."""
    ensure_dir_exists(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_filename = os.path.join(log_dir, f"copernican-run_{get_timestamp()}.txt")

    fh = open(log_filename, "w", encoding="utf-8")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    message = f"Logging initialized. Log file: {log_filename}"
    fh.write(message + "\n")
    logger.info(message)
    return log_filename, fh


def get_logger():
    """Returns the active logger instance."""
    return logging.getLogger()
