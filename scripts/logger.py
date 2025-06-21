# Copernican Suite Logger
"""Logging utilities for the Copernican Suite."""

import logging
import os
import sys
from .utils import get_timestamp, ensure_dir_exists


def setup_logging(log_dir="."):
    """Initializes logging to both console and a file."""
    ensure_dir_exists(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_filename = os.path.join(log_dir, f"copernican-run_{get_timestamp()}.txt")

    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)

    logging.info(f"Logging initialized. Log file: {log_filename}")
    return log_filename


def get_logger():
    """Returns the active logger instance."""
    return logging.getLogger()
