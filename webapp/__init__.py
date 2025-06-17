# DEV NOTE (v1.5h): Package initialization for the Flask web interface.
"""Expose the Flask application for external runners."""
from .app import app

__all__ = ["app"]
