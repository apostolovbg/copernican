# DEV NOTE (v1.5i hotfix 10): Package now lazy-loads the Flask app to
# avoid runpy warnings when executed with ``python -m webapp.app``.
"""Expose the Flask application for external runners."""

__all__ = ["app"]

def __getattr__(name: str):
    """Lazily import and return the Flask application instance."""
    if name == "app":
        from .app import app as flask_app
        return flask_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
