"""Validation helpers and manifest runner for Copernican."""

from __future__ import annotations

import datetime
from pathlib import Path

from .runner import discover_manifests, run_validation_suite

VALIDATION_FILE = Path.home() / "VALIDATION.md"


def write_validation_summary(summary: str, success: bool) -> None:
    """Persist the latest validation summary to the home-folder marker."""

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    content = (
        f"## Validation run {timestamp} (success={success})\n"
        f"\n"
        f"{summary}\n"
    )
    VALIDATION_FILE.write_text(content, encoding="utf-8")


def read_validation_summary() -> str:
    """Return the current validation summary if it exists."""

    if not VALIDATION_FILE.exists():
        return ""
    return VALIDATION_FILE.read_text(encoding="utf-8")


__all__ = [
    "VALIDATION_FILE",
    "discover_manifests",
    "read_validation_summary",
    "run_validation_suite",
    "write_validation_summary",
]
