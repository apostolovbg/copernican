#!/bin/bash
# Last Updated: 2025-11-29
# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# Start the Copernican Suite on macOS.
#
# The script downloads a private Python 3.11 interpreter into '.python',
# creates a local virtual environment and re-executes itself inside that
# environment. System-wide Python installations are ignored so Python
# 3.12 never leaks into the managed bootstrap sequence.

# Abort on errors and on references to unset variables to guard against
# mistyped names.
set -eu
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"
SCRIPT_ARGS=("$@")

EXPECTED_VENV="$(pwd)/.venv"
PY_DIR="$(pwd)/.python"
TCL_LIBRARY="$(pwd)/.python/lib/tcl8.6"
TK_LIBRARY="$(pwd)/.python/lib/tk8.6"
export TCL_LIBRARY TK_LIBRARY
if [ -f "copernican_lib/VERSION" ]; then
    SUITE_VERSION="$(cat copernican_lib/VERSION)"
else
    SUITE_VERSION="unknown"
fi

STRICT=0
if [ "${COPERNICAN_STRICT_WARNINGS:-}" = "1" ]; then
    STRICT=1
fi
SUITE_INSTALLED=0
update_suite_state() {
    if python -m pip show copernican-suite >/dev/null 2>&1; then
        SUITE_INSTALLED=1
    else
        SUITE_INSTALLED=0
    fi
}

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
    echo "Managed environment removed. Re-run the launcher or recreate the env"
    echo "from the menu."
    return 0
}

rebuild_environment() {
    echo
    echo "Rebuilding the managed virtual environment..."
    rm -rf "$EXPECTED_VENV"
    unset VIRTUAL_ENV || true
    exec "$SCRIPT" "${SCRIPT_ARGS[@]}"
}

install_suite() {
    echo
    echo "Installing the Copernican Suite and pinned dependencies..."
    source .venv/bin/activate
    if ! ensure_pip; then
        echo "Unable to bootstrap pip; try running `pip install --upgrade pip` manually." >&2
        return 1
    fi
    python -m pip install --upgrade pip || return 1
    python -m pip install -r requirements.lock || return 1
    rm -rf build
    python -m pip install --no-deps . || return 1
    rm -rf build
    update_suite_state
    echo "Installation complete."
    return 0
}

uninstall_suite() {
    echo
    echo "Uninstalling the Copernican Suite from the managed environment..."
    source .venv/bin/activate
    python -m pip uninstall -y copernican-suite || true
    update_suite_state
    echo "Uninstallation complete."
    return 0
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
            echo "1) Create managed virtual environment and install deps"
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

run_test_suites() {
    echo
    echo "Running pytest..."
    COPERNICAN_STRICT_WARNINGS=$STRICT python -m pytest -q
    local pytest_status=$?
    echo
    echo "Running the unit test suite..."
    COPERNICAN_STRICT_WARNINGS=$STRICT python -m unittest discover -v
    local unittest_status=$?
    return $((pytest_status || unittest_status))
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
    "$1" - <<'PYCHECK'
import sys
sys.exit(0 if (3, 11) <= sys.version_info < (3, 12) else 1)
PYCHECK
}

# Relaunch from inside the virtual environment when already activated.
# Use parameter expansion to avoid an "unbound variable" error when
# "VIRTUAL_ENV" is unset and "set -u" is active.
# Enforce use of the repository's own virtual environment.
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$EXPECTED_VENV" ]; then
    echo "Deactivate the active virtual environment before running" >&2
    echo "start.command." >&2
    exit 1
fi
if [ "${VIRTUAL_ENV:-}" = "$EXPECTED_VENV" ]; then
    GUI_BINARY=".venv/bin/python"
    if [ -x ".venv/bin/pythonw" ]; then
        GUI_BINARY=".venv/bin/pythonw"
    fi
    update_suite_state
    while true; do
        echo
        echo "Copernican Suite ${SUITE_VERSION} Launcher:"
        echo
        echo "Choose an option or press Enter to launch the CLI"
        echo "1) Start Copernican Suite (GUI)"
        echo "2) Start Copernican Suite (CLI)"
        echo "3) Run the unit test suite"
        if [ "$STRICT" -eq 1 ]; then
            echo "4) Disable strict warning mode"
        else
            echo "4) Enable strict warning mode"
        fi
        echo "5) Environment and dependency management"
        if [ "$SUITE_INSTALLED" -eq 1 ]; then
            echo "6) Uninstall Copernican Suite"
        else
            echo "6) Install Copernican Suite"
        fi
        echo "7) Exit"
        echo
        read -r -p "Write the number of choice: " choice
        choice=${choice:-2}
        case "$choice" in
            1)
                echo "Launching the Copernican GUI inline; close the window to return."
                COPERNICAN_STRICT_WARNINGS=$STRICT COPERNICAN_DETACH_GUI=0 \
                    exec "$GUI_BINARY" copernican.py --gui
                ;;
            2)
                COPERNICAN_STRICT_WARNINGS=$STRICT \
                exec python copernican.py --cli ;;
            3)
                if ! run_test_suites; then
                    echo "One or more test suites failed; check the log above."
                fi
                ;;
            4)
                if [ "$STRICT" -eq 1 ]; then STRICT=0; else STRICT=1; fi ;;
            5)
                environment_menu ;;
            6)
                if [ "$SUITE_INSTALLED" -eq 1 ]; then
                    uninstall_suite
                else
                    install_suite
                fi
                ;;
            7)
                exit 0 ;;
            *)
                echo "Please enter a number between 1 and 7."
                ;;
        esac
    done
fi

# Always bootstrap a dedicated interpreter.
PY_BIN="$PY_DIR/bin/python3"
# Delete any interpreter that falls outside the Python 3.11 series so legacy
# downloads or stray Python 3.12 builds never survive across upgrades.
if [ -x "$PY_BIN" ] && ! python_in_311_series "$PY_BIN"; then
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
if [ -x ".venv/bin/python" ] && ! python_in_311_series ".venv/bin/python"; then
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
if ! ensure_pip; then
    echo "Unable to bootstrap pip in the Copernican virtual environment." >&2
    exit 1
fi
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
update_suite_state
exec "$SCRIPT" "${SCRIPT_ARGS[@]}"
