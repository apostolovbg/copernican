"""
Policy: Line Length Limit

Apply the configured `max_length` to the files selected by the unified
include/exclude metadata (suffixes, prefixes and globs).
"""

from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation
from devcovenant.selectors import SelectorSet


class LineLengthLimitCheck(PolicyCheck):
    """Check that each targeted file honors the line-length limit."""

    policy_id = "line-length-limit"
    version = "1.0.0"

    MAX_LINE_LENGTH = 79
    DEFAULT_SUFFIXES = [".py", ".md", ".rst", ".txt"]

    def _build_selector(self) -> SelectorSet:
        """Return the selector constructed from policy metadata."""
        defaults = {
            "include_suffixes": self.DEFAULT_SUFFIXES,
        }
        return SelectorSet.from_policy(self, defaults=defaults)

    def check(self, context: CheckContext) -> List[Violation]:
        """
        Check files for lines exceeding the length limit.

        Args:
            context: Check context

        Returns:
            List of violations
        """
        max_length = int(self.get_option("max_length", self.MAX_LINE_LENGTH))

        violations = []

        selector = self._build_selector()
        files_pool = context.changed_files or context.all_files or []
        files_to_check = [
            path
            for path in files_pool
            if path.is_file() and selector.matches(path, context.repo_root)
        ]

        for file_path in files_to_check:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                continue

            # Check each line
            for line_num, line in enumerate(lines, start=1):
                # Remove trailing newline for length check
                line_content = line.rstrip("\n")

                if len(line_content) > max_length:
                    # Count how many lines are too long to avoid spam
                    # Only report first 5 per file
                    file_violations = [
                        v for v in violations if v.file_path == file_path
                    ]
                    if len(file_violations) >= 5:
                        continue

                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="warning",
                            file_path=file_path,
                            line_number=line_num,
                            message=(
                                f"Line exceeds {max_length} "
                                f"characters (current: {len(line_content)})"
                            ),
                            suggestion=(
                                "Break long lines into multiple lines or "
                                "refactor for clarity"
                            ),
                            can_auto_fix=False,
                        )
                    )

        return violations
