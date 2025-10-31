#!/bin/bash
# Last Updated: 2025-10-31
# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# Start the Copernican Suite on macOS.
#
# The launcher now opens with a management menu that can install, reinstall or
# remove the managed interpreter before the runtime starts. When the toolchain
# already exists the first menu entry launches the suite without reinstalling
# packages so contributors can relaunch quickly while still having recovery
# options when the environment becomes corrupted.

# Abort on errors and on references to unset variables to guard against
# mistyped names.
set -eu
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"
EXPECTED_VENV="$(pwd)/.venv"
PY_DIR="$(pwd)/.python"
PY_BIN="$PY_DIR/bin/python3"
PYTHON_SERIES_PROBE='import sys; exit(0 if (3, 11) <= sys.version_info < '
PYTHON_SERIES_PROBE+='(3, 12) else 1)'

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
    "$1" -c "$PYTHON_SERIES_PROBE"
}

dependencies_ready() {
    if [ ! -x "$PY_BIN" ]; then
        return 1
    fi
    if ! python_in_311_series "$PY_BIN"; then
        return 1
    fi
    if [ ! -x ".venv/bin/python" ]; then
        return 1
    fi
    if ! python_in_311_series ".venv/bin/python"; then
        return 1
    fi
    if [ ! -f ".venv/bin/activate" ]; then
        return 1
    fi
    return 0
}

print_main_menu() {
    echo "Copernican Suite"
    if dependencies_ready; then
        echo "1) Use existing environment"
        echo "2) Reinstall dependencies"
        echo "3) Uninstall dependencies"
        echo "4) Exit"
    else
        echo "1) Install dependencies"
        echo "2) Exit"
    fi
}

if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$EXPECTED_VENV" ]; then
    echo "Deactivate the active virtual environment before running" >&2
    echo "start.command." >&2
    exit 1
fi
bootstrap_python() {
    if [ -x "$PY_BIN" ] && ! python_in_311_series "$PY_BIN"; then
        rm -rf "$PY_DIR"
    fi
    if [ -x "$PY_BIN" ]; then
        return
    fi
    mkdir -p "$PY_DIR"
    local base
    local rel
    local ver
    local arch
    base="https://github.com/astral-sh/python-build-standalone/releases"
    rel="20251028"
    ver="3.11.14"
    arch="$(uname -m)"
    local plat
    plat="apple-darwin"
    local url_path
    url_path="cpython-${ver}+${rel}-${arch}-${plat}-install_only.tar.gz"
    local download_url
    download_url="$base/download/$rel/$url_path"
    if [ -z "$download_url" ]; then
        echo "Copernican Suite download URL is empty." >&2
        exit 1
    fi
    curl -fL "$download_url" | tar -xz -C "$PY_DIR" --strip-components=1
}

create_virtualenv() {
    if [ -x ".venv/bin/python" ] && \
        ! python_in_311_series ".venv/bin/python"; then
        rm -rf .venv
    fi
    if [ ! -d ".venv" ]; then
        "$PY_BIN" -m venv .venv || true
    fi
    if [ ! -f ".venv/bin/activate" ]; then
        rm -rf .venv
        "$PY_BIN" -m venv .venv || true
        if [ ! -f ".venv/bin/activate" ]; then
            echo "Virtual environment creation failed." >&2
            exit 1
        fi
    fi
}

install_dependencies() {
    echo "--- Installing managed dependencies ---"
    bootstrap_python
    create_virtualenv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    if ! ensure_pip; then
        local msg
        msg="Unable to bootstrap pip in the Copernican virtual environment."
        echo "$msg" >&2
        exit 1
    fi
    python -m pip install --upgrade pip
    python -m pip install -r requirements.lock
    rm -rf build
    python -m pip install --no-deps .
    rm -rf build
    deactivate 2>/dev/null || true
    echo "Managed dependencies installed."
}

reinstall_dependencies() {
    echo "--- Reinstalling managed dependencies ---"
    rm -rf "$PY_DIR" .venv
    install_dependencies
}

uninstall_dependencies() {
    echo "--- Removing managed dependencies ---"
    rm -rf "$PY_DIR" .venv
}

launch_runtime_menu() {
    if ! dependencies_ready; then
        echo "Managed dependencies are missing. Install them first." >&2
        exit 1
    fi
    if [ "${VIRTUAL_ENV:-}" != "$EXPECTED_VENV" ]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
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
}

launcher_main() {
    while true; do
        print_main_menu
        if dependencies_ready; then
            read -r -p "Select an option: " choice
            case "$choice" in
                1)
                    launch_runtime_menu ;;
                2)
                    reinstall_dependencies ;;
                3)
                    uninstall_dependencies ;;
                4)
                    exit 0 ;;
            esac
        else
            read -r -p "Select an option: " choice
            case "$choice" in
                1)
                    install_dependencies
                    launch_runtime_menu ;;
                2)
                    exit 0 ;;
            esac
        fi
    done
}

if [ "${COPERNICAN_LAUNCHER_TEST:-}" = "print-menu" ]; then
    print_main_menu
    exit 0
fi

if [ "${VIRTUAL_ENV:-}" = "$EXPECTED_VENV" ]; then
    launch_runtime_menu
fi

launcher_main

