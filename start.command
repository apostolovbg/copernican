#!/bin/bash
# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.
# Last Updated: 2025-10-30

# Start the Copernican Suite on macOS.
#
# The script downloads a private Python 3.11+ interpreter into '.python',
# creates a local virtual environment and re-executes itself inside that
# environment. System-wide Python installations are ignored.

# Abort on errors and on references to unset variables to guard against
# mistyped names.
set -eu
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"

pkg_notice() {
    echo 'A package manager may request your password.'
    echo 'The Copernican Suite never reads or stores it.'
}

sudo_pkg() {
    pkg_notice
    sudo -k -p '[sudo] password for package manager: ' "$@"
}

brew_pkg() {
    pkg_notice
    brew "$@"
}

# Relaunch from inside the virtual environment when already activated.
# Use parameter expansion to avoid an "unbound variable" error when
# "VIRTUAL_ENV" is unset and "set -u" is active.
# Enforce use of the repository's own virtual environment.
EXPECTED_VENV="$(pwd)/.venv"
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$EXPECTED_VENV" ]; then
    echo "Deactivate the active virtual environment before running" >&2
    echo "start.command." >&2
    exit 1
fi
if [ "${VIRTUAL_ENV:-}" = "$EXPECTED_VENV" ]; then
    STRICT=0
    AUTO=0
    while true; do
        echo "Copernican Suite"
        echo "1) Launch Copernican Suite"
        echo "2) Run the unit test suite"
        if [ "$STRICT" -eq 1 ]; then
            echo "3) Disable strict warning mode"
        else
            echo "3) Enable strict warning mode"
        fi
        if [ "$AUTO" -eq 1 ]; then
            echo "4) Disable automatic dependency installation"
        else
            echo "4) Enable automatic dependency installation"
        fi
        echo "5) Exit"
        read -r -p "Select an option: " choice
        case "$choice" in
            1)
                COPERNICAN_STRICT_WARNINGS=$STRICT \
                COPERNICAN_AUTO_INSTALL=$AUTO \
                exec python copernican.py ;;
            2)
                COPERNICAN_STRICT_WARNINGS=$STRICT \
                COPERNICAN_AUTO_INSTALL=$AUTO \
                exec python -m unittest discover -v ;;
            3)
                if [ "$STRICT" -eq 1 ]; then STRICT=0; else STRICT=1; fi ;;
            4)
                if [ "$AUTO" -eq 1 ]; then AUTO=0; else AUTO=1; fi ;;
            5)
                exit 0 ;;
        esac
    done
fi

# Always bootstrap a dedicated interpreter.
PY_DIR="$(pwd)/.python"
PY_BIN="$PY_DIR/bin/python3"
# Delete any interpreter older than Python 3.11 before reuse so legacy
# downloads never survive across upgrades.
if [ -x "$PY_BIN" ] && ! "$PY_BIN" -c 'import sys; exit(0 if sys.version_info >= (3, 11) else 1)'; then
    rm -rf "$PY_DIR"
fi
if [ ! -x "$PY_BIN" ]; then
    mkdir -p "$PY_DIR"
    BASE="https://github.com/astral-sh/python-build-standalone/releases"
    REL="20251028"
    VER="3.11.14"
    ARCH="$(uname -m)"
    PLAT="apple-darwin"
    # Build the release URL once so we can validate it before invoking curl.
    # An empty URL means the release metadata above is stale.
    URL_PATH="cpython-${VER}+${REL}-${ARCH}-${PLAT}-install_only.tar.gz"
    DOWNLOAD_URL="$BASE/download/$REL/$URL_PATH"
    if [ -z "$DOWNLOAD_URL" ]; then
        echo "Copernican Suite download URL is empty." >&2
        exit 1
    fi
    curl -fL "$DOWNLOAD_URL" | tar -xz -C "$PY_DIR" --strip-components=1
fi
PYTHON="$PY_BIN"

# Build the environment when missing.
if [ -x ".venv/bin/python" ] && ! .venv/bin/python -c 'import sys; exit(0 if sys.version_info >= (3, 11) else 1)'; then
    rm -rf .venv
fi
if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
fi

# Retry virtual environment creation once when the activation script is
# missing. A missing script usually means the Python installation lacks
# the ``venv`` module. Recreating the environment gives the interpreter
# another chance before advising the user to reinstall Python.
if [ ! -f ".venv/bin/activate" ]; then
    rm -rf .venv
    "$PYTHON" -m venv .venv
    if [ ! -f ".venv/bin/activate" ]; then
        echo "Virtual environment creation failed." >&2
        exit 1
    fi
fi

# Activate, update pip, install dependencies,
# then install the project without dependencies and restart the script.
# Delete any 'build/' directory before and after installing the project to
# avoid stale build artifacts.
source .venv/bin/activate
python -m pip install --upgrade pip
# Install pinned dependencies.
python -m pip install -r requirements.lock
# Remove any 'build/' directory before and after installing the project
# to avoid stale build artifacts.
rm -rf build
python -m pip install --no-deps .
rm -rf build
exec "$SCRIPT" "$@"

