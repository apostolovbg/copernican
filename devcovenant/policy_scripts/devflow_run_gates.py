"""Session gates: pre-commit at start/end and tests after code edits."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable, List

from devcovenant.base import CheckContext, PolicyCheck, Violation

_DEFAULT_STATUS = Path("devcovenant") / "test_status.json"
_DEFAULT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}
_DEFAULT_COMMANDS = ["pytest", "python -m unittest discover"]


def _resolve_status_path(ctx: CheckContext) -> Path:
    """Return the configured test status path relative to the repository."""
    cfg = ctx.get_policy_config("devflow-run-gates")
    raw = cfg.get("test_status_file", str(_DEFAULT_STATUS))
    return Path(raw)


def _code_extensions(ctx: CheckContext) -> set[str]:
    """Return the set of extensions considered code for gating purposes."""
    cfg = ctx.get_policy_config("devflow-run-gates")
    entries = cfg.get("code_extensions", list(_DEFAULT_EXTENSIONS))
    return {entry.lower() for entry in entries}


def _required_commands(ctx: CheckContext) -> list[str]:
    """Return ordered commands that must appear in the status file."""
    cfg = ctx.get_policy_config("devflow-run-gates")
    commands = cfg.get("required_commands", list(_DEFAULT_COMMANDS))
    return [command.lower() for command in commands]


def _load_test_status(status_file: Path) -> dict | None:
    """Return the parsed test status file, or None when missing/invalid."""

    if not status_file.is_file():
        return None
    try:
        return json.loads(status_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_code_mtime(files: Iterable[Path], extensions: set[str]) -> float:
    """Return the newest modification time among code-like files."""

    latest = 0.0
    for path in files:
        if path.suffix.lower() not in extensions:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        latest = max(latest, stat.st_mtime)
    return latest


class DevflowRunGates(PolicyCheck):
    """Ensure hooks and tests run around every task."""

    @property
    def policy_id(self) -> str:
        """Return the policy identifier."""

        return "devflow-run-gates"

    def check(self, ctx: CheckContext) -> List[Violation]:
        """Validate test recency after code changes; pre-commit is active."""

        violations: List[Violation] = []
        repo_root = ctx.repo_root
        status_rel = _resolve_status_path(ctx)
        extensions = _code_extensions(ctx)
        required_commands = _required_commands(ctx)

        # The pre-commit run is happening now; nothing extra to record beyond
        # enforcing the test gate for code changes.

        code_mtime = _latest_code_mtime(ctx.changed_files, extensions)
        if code_mtime == 0.0:
            return violations

        status = _load_test_status(repo_root / status_rel)
        if not status:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=status_rel,
                    message=(
                        "Code changed but devcovenant/test_status.json is "
                        "missing; run `python tools/run_tests.py` before "
                        "replying."
                    ),
                )
            )
            return violations

        last_run = status.get("last_run_utc") or ""
        try:
            last_ts = float(status.get("last_run_epoch") or 0.0)
        except Exception:
            last_ts = 0.0

        commands: list[str] = status.get("commands") or []
        commands_lower = " ".join(commands).lower()

        missing = [
            command
            for command in required_commands
            if command not in commands_lower
        ]

        if missing:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=status_rel,
                    message=(
                        "Latest recorded test status is missing required "
                        f"commands: {', '.join(missing)}. Run "
                        "`python tools/run_tests.py` before replying."
                    ),
                )
            )
        elif last_ts < code_mtime:
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(code_mtime))
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=status_rel,
                    message=(
                        "Code changed after the last recorded test run "
                        f"({last_run or 'unknown'}); rerun "
                        "`python tools/run_tests.py` so tests post-date the "
                        f"newest code change (latest code mtime: {when}Z)."
                    ),
                )
            )

        return violations
