#!/bin/bash
# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.
# Last Updated: 2025-08-30

# Start the Copernican Suite on Unix-like systems.
#
# The script downloads a private Python 3.12+ interpreter into '.python',
# creates a local virtual environment and re-executes itself inside that
# environment. System-wide Python installations are ignored.

# Abort on errors and on references to unset variables to guard against
# mistyped names.
set -eu
# Resolve absolute path to this script before changing directories.
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"

# If we are already inside the virtual environment simply launch the suite.
# Use parameter expansion to avoid an "unbound variable" error when
# "VIRTUAL_ENV" is unset and "set -u" is active.
# Enforce use of the repository's own virtual environment.
EXPECTED_VENV="$(pwd)/.venv"
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$EXPECTED_VENV" ]; then
    echo "Deactivate the active virtual environment before running" >&2
    echo "start.sh." >&2
    exit 1
fi
if [ "${VIRTUAL_ENV:-}" = "$EXPECTED_VENV" ]; then
    exec python copernican.py "$@"
fi

# Always bootstrap a dedicated interpreter.
PY_DIR="$(pwd)/.python"
PY_BIN="$PY_DIR/bin/python3"
if [ ! -x "$PY_BIN" ]; then
    mkdir -p "$PY_DIR"
    BASE="https://github.com/indygreg/python-build-standalone/releases"
    REL="20240710"
    VER="3.12.4"
    ARCH="$(uname -m)"
    if [ "$(uname)" = "Darwin" ]; then
        PLAT="apple-darwin"
    else
        PLAT="unknown-linux-gnu"
    fi
    URL="$BASE/download/$REL/"
    URL="${URL}cpython-${VER}+${REL}-${ARCH}-${PLAT}-install_only.tar.gz"
    curl -L "$URL" | tar -xz -C "$PY_DIR" --strip-components=1
fi
PYTHON="$PY_BIN"

# Create the virtual environment on first run.
if [ ! -d ".venv" ]; then
    # Allow the initial creation to fail so we can retry if needed.
    "$PYTHON" -m venv .venv || true
fi

# Ensure the virtual environment exists. Retry once before giving up to catch
# rare failures when extracting the bundled interpreter.
if [ ! -f ".venv/bin/activate" ]; then
    rm -rf .venv
    "$PYTHON" -m venv .venv || true
    if [ ! -f ".venv/bin/activate" ]; then
        echo "Virtual environment creation failed." >&2
        exit 1
    fi
fi

# Activate the environment, upgrade pip, install dependencies with hash
# verification, then install the project without dependencies and restart
# the script. Delete any 'build/' directory before and after installing the
# project to avoid stale build artifacts.
source .venv/bin/activate
python -m pip install --upgrade pip
# Install pinned dependencies with hash verification.
python -m pip install --require-hashes -r requirements.lock
rm -rf build
python -m pip install --no-deps .
rm -rf build
exec "$SCRIPT" "$@"
