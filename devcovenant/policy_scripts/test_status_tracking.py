"""Ensure devcovenant/test_status.json is refreshed after test runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from devcovenant.base import CheckContext, PolicyCheck, Violation
from devcovenant.selectors import build_watchlists

STATUS_RELATIVE = Path("devcovenant") / "test_status.json"


def _requires_status_update(
    rel_path: Path, watched_roots: set[str], watched_files: set[str]
) -> bool:
    """Return True when *rel_path* should trigger a test status refresh."""
    if not rel_path.parts:
        return False
    if rel_path == STATUS_RELATIVE:
        return False
    first = rel_path.parts[0]
    if first in watched_roots:
        return True
    if rel_path.name in watched_files:
        return True
    return False


def _load_status(path: Path) -> dict[str, str]:
    """Return the JSON contents with basic validation."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise ValueError(f"Unable to parse {path}: {exc}") from exc

    last_run = payload.get("last_run") or ""
    try:
        datetime.fromisoformat(last_run.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "Field 'last_run' must be an ISO-8601 timestamp."
        ) from exc

    command = payload.get("command") or ""
    if not command.strip():
        raise ValueError(
            "Field 'command' must record the executed test command."
        )

    sha = (payload.get("sha") or "").strip()
    if len(sha) < 8:
        raise ValueError(
            "Field 'sha' must contain the git commit recorded for the run."
        )

    return payload


class TestStatusTrackingCheck(PolicyCheck):
    """Verify the test status file changes alongside code and is valid."""

    policy_id = "test-status-tracking"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Ensure code changes update devcovenant/test_status.json."""
        repo_root = context.repo_root
        changed_files: Iterable[Path] = context.changed_files or []
        relevant_change = False
        status_changed = False
        status_relative = Path(
            self.get_option("test_status_file", str(STATUS_RELATIVE))
        )
        watch_files, watch_dirs = build_watchlists(self, defaults={})
        watched_roots = set(watch_dirs)
        watched_files = set(Path(entry).name for entry in watch_files)

        for path in changed_files:
            try:
                rel = path.relative_to(repo_root)
            except ValueError:
                continue
            if rel == status_relative:
                status_changed = True
            if _requires_status_update(rel, watched_roots, watched_files):
                relevant_change = True

        if not relevant_change:
            return []

        status_path = repo_root / status_relative
        if not status_changed:
            return [
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=status_path,
                    line_number=1,
                    message=(
                        "Code changes require a fresh test status update. "
                        "Run `python tools/run_tests.py` "
                        "so the suite executes and "
                        "the status file is refreshed."
                    ),
                )
            ]

        try:
            _load_status(status_path)
        except ValueError as exc:
            return [
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=status_path,
                    line_number=1,
                    message=f"devcovenant/test_status.json is invalid: {exc}",
                )
            ]

        return []
