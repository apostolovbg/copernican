#!/bin/bash
# Start the Copernican Suite on Unix-like systems.
#
# The script locates a Python 3 interpreter, creates a local virtual
# environment if needed, installs the project and re-executes itself inside
# that environment. Subsequent runs reuse the cached dependencies.

set -e
cd "$(dirname "$0")"

# If we are already inside the virtual environment simply launch the suite.
if [ -n "$VIRTUAL_ENV" ]; then
    exec python copernican.py "$@"
fi

# Detect a usable Python 3 interpreter.
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    echo "Python 3 is not installed." >&2
    echo "Install it with 'sudo apt install python3' or" >&2
    echo "'brew install python'." >&2
    echo "Get it at https://www.python.org/downloads/." >&2
    exit 1
fi

# Create the virtual environment on first run.
if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
fi

# Activate the environment, upgrade pip, install the project and re-run.
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
exec "$0" "$@"
