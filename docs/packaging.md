# Packaging Guide

This document explains how to prepare the suite for development or packaging.

## Install Python 3.11+

### Windows
1. Download the Python 3.11 installer from https://www.python.org.
2. Enable "Add python.exe to PATH" during installation.
3. Open a new Command Prompt and run `python --version` to verify.

### macOS
1. Install Homebrew from https://brew.sh if it is not already present.
2. Run `brew install python@3.11`.
3. Verify the interpreter with `python3.11 --version`.

### Linux
1. Use your package manager. For Debian or Ubuntu run  
   `sudo apt install python3.11 python3.11-venv`.
2. Confirm the install with `python3.11 --version`.

## Bootstrap the virtual environment

Run the launcher in the project root:

- `start.bat` on Windows
- `start.command` on macOS
- `start.sh` on Linux

The script creates or reuses `.venv`, upgrades `pip` and installs packages.
Re-run it after pulling updates to refresh the environment.

## Build optional distributions

Inside `.venv` run:

```bash
pip build .
```

The command writes source archives and wheels to the `dist/` directory.

## Troubleshooting

- **No module named pip**: run `python -m ensurepip --upgrade` and relaunch
  launcher.
- **Packages not updating**: run `pip install -U pip` followed by
  `pip install -U -r requirements.txt`.
- **Permission denied**: avoid `sudo pip`; use a writable directory or the
  provided `.venv`.
- **Virtual environment missing**: ensure a launcher was used or activate with
  `source .venv/bin/activate` (`.venv\\Scripts\\activate` on Windows).
