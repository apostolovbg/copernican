"""Simple error logging utility for the JSON pipeline."""
# DEV NOTE (v1.5b): Logs all errors through the main logger. Future versions may
# expand this with structured codes.

import logging


def report_error(message):
    logging.getLogger().error(message)
