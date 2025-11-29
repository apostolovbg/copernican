#!/bin/bash
# Last Updated: 2025-11-24
# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# Start the Copernican Suite on Unix-like systems.
#
# The script downloads a private Python 3.11 interpreter into '.python',
# creates a local virtual environment and re-executes itself inside that
# environment. System-wide Python installations are ignored so Python
# 3.12 never leaks into the managed bootstrap sequence.

# Abort on errors and on references to unset variables to guard against
# mistyped names.
set -eu
# Resolve absolute path to this script before changing directories.
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
SCRIPT_ARGS=("$@")
cd "$(dirname "$0")"

EXPECTED_VENV="$(pwd)/.venv"
PY_DIR="$(pwd)/.python"
if [ -f "copernican_lib/VERSION" ]; then
    SUITE_VERSION="$(cat copernican_lib/VERSION)"
else
    SUITE_VERSION="unknown"
fi

STRICT=0
if [ "${COPERNICAN_STRICT_WARNINGS:-}" = "1" ]; then
    STRICT=1
fi

update_dependencies() {
    echo
    echo "Updating managed dependencies..."
    local venv_python
    venv_python="$EXPECTED_VENV/bin/python"
    if [ ! -x "$venv_python" ]; then
        echo "The managed virtual environment is missing." >&2
        echo "Choose 'Create the managed virtual environment' first." >&2
        return 0
    fi
    if ! "$venv_python" -m pip install --upgrade pip; then
        echo "Failed to upgrade pip." >&2
        return 1
    fi
    if ! "$venv_python" -m pip install -r requirements.lock; then
        echo "Failed to install dependencies." >&2
        return 1
    fi
    rm -rf build
    if ! "$venv_python" -m pip install --no-deps .; then
        echo "Failed to reinstall the Copernican Suite." >&2
        return 1
    fi
    rm -rf build
    echo "Dependencies updated successfully."
}

remove_environment() {
    echo
    echo "Removing the managed virtual environment..."
    rm -rf "$EXPECTED_VENV"
    echo "Managed environment removed. The launcher will now exit."
    exit 0
}

rebuild_environment() {
    echo
    echo "Rebuilding the managed virtual environment..."
    rm -rf "$EXPECTED_VENV"
    unset VIRTUAL_ENV || true
    exec "$SCRIPT" "${SCRIPT_ARGS[@]}"
}

environment_menu() {
    while true; do
        echo
        echo "Environment and dependency management"
        echo
        if [ -d "$EXPECTED_VENV" ]; then
            echo "1) Update dependencies in the managed virtual environment"
            echo "2) Remove the managed virtual environment"
            echo "3) Rebuild the managed virtual environment"
            echo "4) Back"
            echo
            read -r -p "Write the number of choice: " env_choice
            env_choice=${env_choice:-4}
            case "$env_choice" in
                1)
                    if ! update_dependencies; then
                        echo "Dependency update failed." >&2
                    fi
                    ;;
                2)
                    remove_environment
                    ;;
                3)
                    rebuild_environment
                    ;;
                4)
                    return
                    ;;
                *)
                    echo "Please enter a number between 1 and 4."
                    ;;
            esac
        else
            echo "1) Create the managed virtual environment and install dependencies"
            echo "2) Back"
            echo
            read -r -p "Write the number of choice: " env_choice
            env_choice=${env_choice:-2}
            case "$env_choice" in
                1)
                    rebuild_environment
                    ;;
                2)
                    return
                    ;;
                *)
                    echo "Please enter 1 or 2."
                    ;;
            esac
        fi
    done
}

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

ensure_pip() {
    if python -m ensurepip --upgrade; then
        return 0
    fi

    local tmpfile
    tmpfile="$(mktemp "${TMPDIR:-/tmp}/copernican-get-pip-XXXXXXXX.py")"
    if ! curl -fL "https://bootstrap.pypa.io/get-pip.py" -o "$tmpfile"; then
        rm -f "$tmpfile"
        echo "Failed to download get-pip.py." >&2
        return 1
    fi

    if ! python "$tmpfile"; then
        rm -f "$tmpfile"
        echo "Failed to bootstrap pip via get-pip.py." >&2
        return 1
    fi

    rm -f "$tmpfile"
}

# Check whether the supplied Python binary reports a version within the 3.11
# series. Purging anything outside this window blocks Python 3.12 from entering
# the managed environment while allowing future 3.11 maintenance releases.
python_in_311_series() {
    "$1" -c 'import sys; exit(0 if (3, 11) <= sys.version_info < (3, 12) else 1)'
}

# If we are already inside the virtual environment simply launch the suite.
# Use parameter expansion to avoid an "unbound variable" error when
# "VIRTUAL_ENV" is unset and "set -u" is active.
# Enforce use of the repository's own virtual environment.
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$EXPECTED_VENV" ]; then
    echo "Deactivate the active virtual environment before running" >&2
    echo "start.sh." >&2
    exit 1
fi
if [ "${VIRTUAL_ENV:-}" = "$EXPECTED_VENV" ]; then
    while true; do
        echo
        echo "Copernican Suite ${SUITE_VERSION} Launcher:"
        echo
        echo "Choose an option or press Enter to launch the Suite"
        echo "1) Launch Copernican Suite"
        echo "2) Run the unit test suite"
        if [ "$STRICT" -eq 1 ]; then
            echo "3) Disable strict warning mode"
        else
            echo "3) Enable strict warning mode"
        fi
        echo "4) Environment and dependency management"
        echo "5) Exit"
        echo
        read -r -p "Write the number of choice: " choice
        choice=${choice:-1}
        case "$choice" in
            1)
                COPERNICAN_STRICT_WARNINGS=$STRICT \
                exec python copernican.py ;;
            2)
                COPERNICAN_STRICT_WARNINGS=$STRICT \
                exec python -m unittest discover -v ;;
            3)
                if [ "$STRICT" -eq 1 ]; then STRICT=0; else STRICT=1; fi ;;
            4)
                environment_menu ;;
            5)
                exit 0 ;;
            *)
                echo "Please enter a number between 1 and 5."
                ;;
        esac
    done
fi

# Always bootstrap a dedicated interpreter.
PY_BIN="$PY_DIR/bin/python3"
# Remove any bundled interpreter outside the Python 3.11 series before reuse
# so the virtual environment is always built from the supported runtime. The
# interpreter may exist when users pull a newer release without cleaning
# `.python` first.
if [ -x "$PY_BIN" ] && ! python_in_311_series "$PY_BIN"; then
    rm -rf "$PY_DIR"
fi
if [ ! -x "$PY_BIN" ]; then
    mkdir -p "$PY_DIR"
    BASE="https://github.com/astral-sh/python-build-standalone/releases"
    REL="20251028"
    VER="3.11.14"
    ARCH="$(uname -m)"
    if [ "$(uname)" = "Darwin" ]; then
        PLAT="apple-darwin"
    else
        PLAT="unknown-linux-gnu"
    fi
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

# Create the virtual environment on first run.
# Remove any legacy virtual environment built from an older interpreter. The
# bundled interpreter check above ensures new environments always use
# Python 3.11 or newer.
if [ -x ".venv/bin/python" ] && ! python_in_311_series ".venv/bin/python"; then
    rm -rf .venv
fi
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

# Activate the environment, upgrade pip, install dependencies,
# then install the project without dependencies and restart the script.
# Delete any 'build/' directory before and after installing the project
# to avoid stale build artifacts.
source .venv/bin/activate
if ! ensure_pip; then
    echo "Unable to bootstrap pip in the Copernican virtual environment." >&2
    exit 1
fi
python -m pip install --upgrade pip
# Install pinned dependencies.
python -m pip install -r requirements.lock
rm -rf build
python -m pip install --no-deps .
rm -rf build
exec "$SCRIPT" "$@"
