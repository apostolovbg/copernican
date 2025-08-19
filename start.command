#!/bin/bash
# Start the Copernican Suite on macOS.
#
# The script finds a Python 3 interpreter, prepares a virtual environment and
# re-executes itself inside that environment so later runs reuse the cached
# installation.

set -e
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"

# Relaunch from inside the virtual environment when already activated.
if [ -n "$VIRTUAL_ENV" ]; then
    exec python copernican.py "$@"
fi

## Detect a usable Python 3.11+ interpreter.
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    echo "Python 3.11 is not installed." >&2
    echo "Install it with 'brew install python@3.11'." >&2
    echo "Get it at https://www.python.org/downloads/." >&2
    exit 1
fi

# Verify interpreter version by parsing '--version' output.
PY_VERSION="$($PYTHON --version 2>&1 | awk '{print $2}')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION#*.}"
PY_MINOR="${PY_MINOR%%.*}"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && \
    [ "$PY_MINOR" -lt 11 ]; }; then
    echo "Python 3.11 or newer is required." >&2
    echo "Install it with 'brew install python@3.11'." >&2
    echo "Get it at https://www.python.org/downloads/." >&2
    exit 1
fi

# Build the environment when missing.
if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
fi

# Activate, update pip, install the project and restart the script. Delete
# any 'build/' directory before and after 'pip install .'
# to avoid stale build artifacts.
source .venv/bin/activate
python -m pip install --upgrade pip
rm -rf build
python -m pip install .
rm -rf build
exec "$SCRIPT" "$@"

