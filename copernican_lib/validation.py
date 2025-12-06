"""Shared helpers for validation status tracking."""

from __future__ import annotations

import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
VALIDATION_FILE = ROOT_DIR / "VALIDATION.md"


def write_validation_summary(summary: str, success: bool) -> None:
    """Persist the latest validation summary to VALIDATION.md."""

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    content = (
        f"## Validation run {timestamp} (success={success})\n"
        f"\n"
        f"{summary}\n"
    )
    VALIDATION_FILE.write_text(content, encoding="utf-8")


def read_validation_summary() -> str:
    """Return the contents of VALIDATION.md if it exists."""

    if not VALIDATION_FILE.exists():
        return ""
    return VALIDATION_FILE.read_text(encoding="utf-8")
