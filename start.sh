#!/bin/bash
# Start the Copernican Suite on Unix-like systems.
#
# The script locates a Python 3 interpreter, creates a local virtual
# environment if needed, installs the project and re-executes itself inside
# that environment. Subsequent runs reuse the cached dependencies.

set -e
# Resolve absolute path to this script before changing directories.
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"

# If we are already inside the virtual environment simply launch the suite.
if [ -n "$VIRTUAL_ENV" ]; then
    exec python copernican.py "$@"
fi

## Detect a usable Python 3.11+ interpreter.
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    echo "Python 3.11 is not installed." >&2
    if [ "$(uname)" = "Darwin" ]; then
        echo "Install it with 'brew install python@3.11'." >&2
    else
        echo "Install with 'sudo apt install python3.11 python3.11-venv'." >&2
    fi
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
    if [ "$(uname)" = "Darwin" ]; then
        echo "Install it with 'brew install python@3.11'." >&2
    else
        echo "Install with 'sudo apt install python3.11 python3.11-venv'." >&2
    fi
    exit 1
fi

# Create the virtual environment on first run.
if [ ! -d ".venv" ]; then
    # Allow venv creation to fail so we can emit a clearer message if the
    # activation script is missing.
    "$PYTHON" -m venv .venv || true
fi

# Ensure the virtual environment was created successfully. On Debian-based
# systems the 'python3.11-venv' package may be missing, leaving out the
# activation script. Advise the user to install it and abort in that case.
if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment support is missing." >&2
    echo "Install with 'sudo apt install python3.11-venv'." >&2
    exit 1
fi

# Activate the environment, upgrade pip, install the project and restart the
# script.
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
exec "$SCRIPT" "$@"
