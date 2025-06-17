"""Runs the Copernican Suite Flask web application."""
# DEV NOTE (v1.5h): Added Flask entrypoint for the experimental web UI.

from webapp import create_app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
