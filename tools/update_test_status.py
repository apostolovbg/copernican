#!/usr/bin/env python3
"""Record the latest full test run in devcovenant/test_status.json."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
from pathlib import Path


def _utc_now() -> str:
    """Return the current UTC time in ISO-8601 form."""
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _current_sha(repo_root: Path) -> str:
    """Return the current Git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        raise SystemExit(f"Unable to read git SHA: {exc}") from exc
    return result.stdout.strip()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Update devcovenant/test_status.json after running tests."
    )
    parser.add_argument(
        "--command",
        required=True,
        help=(
            "Command used to run the suites "
            '(e.g., "pytest && python -m unittest discover").'
        ),
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional notes about the environment or deviations.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    status_path = repo_root / "devcovenant" / "test_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "last_run": _utc_now(),
        "command": args.command.strip(),
        "sha": _current_sha(repo_root),
        "notes": args.notes.strip(),
    }
    status_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Recorded test status at {payload['last_run']} "
        f"for command `{payload['command']}`."
    )


if __name__ == "__main__":
    main()
