"""
Policy: No Git Conflict Markers

Ensures no Git conflict markers exist in any file.
"""

import re
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation


class NoGitConflictMarkersCheck(PolicyCheck):
    """
    Check for Git conflict markers in files.
    """

    policy_id = "no-git-conflict-markers"
    version = "1.0.0"

    # Git conflict markers
    CONFLICT_MARKERS = [
        r"^<<<<<<<\s",
        r"^=======\s*$",
        r"^>>>>>>>\s",
    ]

    def check(self, context: CheckContext) -> List[Violation]:
        """
        Check files for Git conflict markers.

        Args:
            context: Check context

        Returns:
            List of violations
        """
        violations = []

        # Determine which files to check
        if context.changed_files:
            files_to_check = context.changed_files
        else:
            files_to_check = context.all_files

        for file_path in files_to_check:
            # Skip binary files and certain extensions
            binary_exts = [".pyc", ".png", ".jpg", ".pdf", ".gif"]
            if file_path.suffix in binary_exts:
                continue

            # Skip devcovenant's own registry
            if "devcovenant/registry.json" in str(file_path):
                continue

            # Skip test files that test for conflict markers
            if "devcovenant/tests/test_policies/test_no" in str(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                # Skip files we can't read
                continue

            # Check each line for conflict markers
            for line_num, line in enumerate(lines, start=1):
                for pattern in self.CONFLICT_MARKERS:
                    if re.match(pattern, line):
                        marker = line.strip()
                        violations.append(
                            Violation(
                                policy_id=self.policy_id,
                                severity="critical",
                                file_path=file_path,
                                line_number=line_num,
                                message=f"Git conflict marker found: {marker}",
                                suggestion=(
                                    "Resolve the merge conflict and remove "
                                    "conflict markers"
                                ),
                                can_auto_fix=False,
                            )
                        )
                        break  # Only report once per line

        return violations
