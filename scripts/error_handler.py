"""Centralized error handling utilities for the Copernican Suite."""
# DEV NOTE (v1.5c): Error reporting helper from Phase 0.
import logging

def report_error(message):
    """Log an error message through the main logger."""
    logging.getLogger().error(message)
