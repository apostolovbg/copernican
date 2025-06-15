"""Simple error logging utility for the JSON pipeline."""
# DEV NOTE (v1.5a): Initial placeholder for structured error reporting.

import logging


def report_error(message):
    logging.getLogger().error(message)
