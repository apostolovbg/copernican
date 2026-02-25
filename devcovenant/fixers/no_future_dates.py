"""
Fixer for the no-future-dates policy.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from devcovenant.base import FixResult, PolicyFixer, Violation


class NoFutureDatesFixer(PolicyFixer):
    """Replace future timestamps with the current UTC date."""

    policy_id = "no-future-dates"

    def can_fix(self, violation: Violation) -> bool:
        """Only handle violations produced by this policy."""
        return (
            violation.policy_id == self.policy_id
            and violation.file_path is not None
            and bool(violation.context.get("match"))
        )

    def fix(self, violation: Violation) -> FixResult:
        """Rewrite the offending date in-place."""
        path = violation.file_path
        if path is None or not Path(path).exists():
            return FixResult(success=False, message="Missing file path")

        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            return FixResult(success=False, message=str(exc))

        match_text = str(violation.context.get("match"))
        today = dt.datetime.now(dt.timezone.utc).date().isoformat()

        # Prefer replacing on the recorded line to avoid touching other dates.
        if violation.line_number:
            lines = text.splitlines(keepends=True)
            line_index = min(max(violation.line_number - 1, 0), len(lines) - 1)
            line = lines[line_index]
            if match_text in line:
                lines[line_index] = line.replace(match_text, today, 1)
                new_text = "".join(lines)
            else:
                new_text = text.replace(match_text, today, 1)
        else:
            new_text = text.replace(match_text, today, 1)

        if new_text == text:
            return FixResult(
                success=False, message="Unable to locate future date"
            )

        Path(path).write_text(new_text, encoding="utf-8")
        return FixResult(
            success=True,
            message=f"Updated future date in {path}",
            files_modified=[Path(path)],
        )
