"""Autofix runtime for the last-updated policy."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from devcovenant.core.policy_contract import FixResult, PolicyFixer, Violation


class LastUpdatedFixer(PolicyFixer):
    """Keep Last Updated markers current in allowlisted managed docs."""

    policy_id = "last-updated"

    LAST_UPDATED_PATTERN = re.compile(
        r"^\s*(\*\*Last Updated:\*\*|Last Updated:|# Last Updated).*",
        re.IGNORECASE,
    )
    MANAGED_BLOCK_BEGIN = "<!-- DEVCOV:BEGIN -->"

    def can_fix(self, violation: Violation) -> bool:
        """Return True when the violation references a fixable text file."""
        if violation.file_path is not None and Path(
            violation.file_path
        ).suffix.lower() in {".yaml", ".yml"}:
            return False
        return (
            violation.policy_id == self.policy_id
            and violation.file_path is not None
        )

    def fix(self, violation: Violation) -> FixResult:
        """Insert or refresh UTC Last Updated marker in header zone."""
        if not violation.file_path:
            return FixResult(
                success=False, message="No file path provided in violation"
            )

        marker = self._format_marker()
        file_path = Path(violation.file_path)
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return FixResult(
                success=False,
                message=f"Unable to read {file_path}: {exc}",
            )

        lines = content.splitlines()
        existing_idx = self._find_marker_index(lines)
        modified = False

        if existing_idx is not None:
            if lines[existing_idx].strip() != marker:
                lines[existing_idx] = marker
                modified = True
        else:
            insert_pos = self._insert_position(lines)
            lines.insert(insert_pos, marker)
            if insert_pos < len(lines) and lines[insert_pos + 1].strip():
                lines.insert(insert_pos + 1, "")
            modified = True

        if not modified:
            return FixResult(
                success=True,
                message=(
                    f"Last Updated header already current in {file_path}"
                ),
            )

        new_content = "\n".join(lines).rstrip() + "\n"
        try:
            file_path.write_text(new_content, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return FixResult(
                success=False,
                message=f"Unable to write {file_path}: {exc}",
            )

        human_date = marker.split(":", 1)[1].strip()
        return FixResult(
            success=True,
            message=(
                "Set Last Updated header to " f"{human_date} in {file_path}"
            ),
            files_modified=[file_path],
        )

    def _format_marker(self) -> str:
        """Render Last Updated marker line with today's UTC date."""
        today = datetime.now(timezone.utc).date().isoformat()
        return f"**Last Updated:** {today}"

    def _find_marker_index(self, lines: list[str]) -> int | None:
        """Return first index containing a Last Updated marker."""
        for idx, line in enumerate(lines):
            if self.LAST_UPDATED_PATTERN.match(line):
                return idx
        return None

    def _insert_position(self, lines: list[str]) -> int:
        """Return insertion index before managed block or first body line."""
        for idx, line in enumerate(lines):
            if line.strip() == self.MANAGED_BLOCK_BEGIN:
                return idx
        for idx, line in enumerate(lines):
            if line.strip():
                return idx + 1
        return 0
