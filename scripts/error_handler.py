"""Simple error logging utility for the JSON pipeline."""

import logging


def report_error(message):
    logging.getLogger().error(message)
