"""Simple logging helper used by model parsing code.

This module contains a single function ``report_error`` which delegates
error messages to Python's :mod:`logging` system.  It exists so that
parsers can emit errors without caring about the global logger setup.
"""

import logging


def report_error(message: str) -> None:
    """Log ``message`` to the shared application logger."""
    logging.getLogger().error(message)
