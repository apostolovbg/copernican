#!/usr/bin/env python3
"""Run the Copernican test suites and update devcovenant/test_status.json."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_COMMANDS = [
    ["pytest"],
    [sys.executable, "-m", "unittest", "discover"],
]


def _run_command(command: list[str]) -> None:
    """Execute *command* and raise when it fails."""
    subprocess.run(command, check=True)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Run the Copernican Suite tests and record their status."
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional notes recorded alongside the test status entry.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    for command in DEFAULT_COMMANDS:
        print(f"Running: {' '.join(command)}")
        _run_command(command)

    command_str = "pytest && python -m unittest discover"
    print("Recording test status…")
    update_cmd = [
        sys.executable,
        str(repo_root / "tools" / "update_test_status.py"),
        "--command",
        command_str,
    ]
    if args.notes:
        update_cmd.extend(["--notes", args.notes])
    _run_command(update_cmd)


if __name__ == "__main__":
    main()
