#!/bin/bash
# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# Start the Copernican Suite on Unix-like systems.
#
# The script locates a Python 3 interpreter, creates a local virtual
# environment if needed, installs the project and re-executes itself inside
# that environment. Subsequent runs reuse the cached dependencies.

# Abort on errors and on references to unset variables to guard against
# mistyped names.
set -eu
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
    # Allow the initial creation to fail so we can retry if needed.
    "$PYTHON" -m venv .venv || true
fi

# Ensure the virtual environment was created successfully. On Debian-based
# systems the 'python3.11-venv' package may be missing, leaving out the
# activation script. If it is missing after the first try delete '.venv' and
# recreate it once. Abort with guidance if the second attempt still lacks the
# activation script.
if [ ! -f ".venv/bin/activate" ]; then
    rm -rf .venv
    "$PYTHON" -m venv .venv || true
    if [ ! -f ".venv/bin/activate" ]; then
        echo "Virtual environment support is missing." >&2
        echo "Install with 'sudo apt install python3.11-venv'." >&2
        exit 1
    fi
fi

# Activate the environment, upgrade pip, install dependencies with hash
# verification, install the project and restart the script. Delete any
# 'build/' directory before and after 'pip install .' to avoid stale build
# artifacts.
source .venv/bin/activate
python -m pip install --upgrade pip
# Install pinned dependencies to ensure reproducible environments.
python -m pip install --require-hashes -r requirements.lock
rm -rf build
python -m pip install .
rm -rf build
exec "$SCRIPT" "$@"
