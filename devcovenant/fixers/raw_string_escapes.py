"""
Fixer for raw-string-escape violations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from devcovenant.base import FixResult, PolicyFixer, Violation
from devcovenant.policy_scripts.raw_string_escapes import _SUSPICIOUS_ESCAPE_RE


class RawStringEscapesFixer(PolicyFixer):
    """Double-escape bare backslashes inside targeted string literals."""

    policy_id = "raw-string-escapes"

    def can_fix(self, violation: Violation) -> bool:
        """Require the original token span before attempting a fix."""
        return (
            violation.policy_id == self.policy_id
            and violation.file_path is not None
            and violation.context.get("start") is not None
            and violation.context.get("end") is not None
        )

    def fix(self, violation: Violation) -> FixResult:
        """Double-escape the targeted literal's backslashes."""
        target = violation.file_path
        if target is None or not Path(target).exists():
            return FixResult(success=False, message="Missing file path")

        try:
            content = Path(target).read_text(encoding="utf-8")
        except OSError as exc:
            return FixResult(success=False, message=str(exc))

        start = self._offset(content, violation.context.get("start"))
        end = self._offset(content, violation.context.get("end"))
        if start is None or end is None or start >= end:
            return FixResult(
                success=False,
                message="Invalid token span for raw-string fix",
            )

        literal = content[start:end]
        fixed_literal = _SUSPICIOUS_ESCAPE_RE.sub(r"\\\\", literal)
        if fixed_literal == literal:
            return FixResult(
                success=False,
                message="No suspicious escapes detected to fix",
            )

        new_content = content[:start] + fixed_literal + content[end:]
        Path(target).write_text(new_content, encoding="utf-8")
        return FixResult(
            success=True,
            message=f"Escaped backslashes in {target}",
            files_modified=[Path(target)],
        )

    @staticmethod
    def _offset(content: str, token_pos: Tuple[int, int] | None) -> int | None:
        """Translate (line, column) coordinates into absolute offsets."""
        if token_pos is None:
            return None
        line, col = token_pos
        if line < 1:
            return None
        lines = content.splitlines(keepends=True)
        if line - 1 >= len(lines):
            return None
        return sum(len(lines[i]) for i in range(line - 1)) + col
