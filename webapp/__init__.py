"""Flask app factory for the Copernican Suite web interface."""
# DEV NOTE (v1.5h): Added initial Flask app factory as part of web transition Phases 0-3.

from flask import Flask


def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)

    from . import routes  # Local import so Flask is only required when webapp is used
    routes.init_app(app)
    return app
