"""
Policy: Last Updated Marker Placement

Ensures Last Updated markers are only in allowlisted files.
"""

import re
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation


class LastUpdatedPlacementCheck(PolicyCheck):
    """
    Check that Last Updated markers are only in allowlisted files.
    """

    policy_id = "last-updated-placement"
    version = "1.0.0"

    # Pattern to detect Last Updated markers
    LAST_UPDATED_PATTERN = re.compile(
        r"(\*\*Last Updated:\*\*|Last Updated:|# Last Updated)", re.IGNORECASE
    )

    def check(self, context: CheckContext) -> List[Violation]:
        """
        Check for Last Updated markers in non-allowlisted files.

        Args:
            context: Check context

        Returns:
            List of violations
        """
        violations = []

        # Check all markdown and Python files
        files_to_check = []
        if context.changed_files:
            files_to_check = context.changed_files
        else:
            files_to_check = context.all_files

        allowed_entries = self.get_option("allowed_files", [])
        if isinstance(allowed_entries, str):
            allowlist = {allowed_entries.strip()}
        else:
            allowlist = {
                str(entry).strip()
                for entry in (allowed_entries or [])
                if str(entry).strip()
            }

        for file_path in files_to_check:
            # Skip non-text files
            text_extensions = [
                ".md", ".py", ".yml", ".yaml",
                ".sh", ".bat", ".command", ".cff"
            ]
            if file_path.suffix not in text_extensions:
                continue

            # Check if file is in allowlist
            relative_path = str(file_path.relative_to(context.repo_root))
            is_allowlisted = relative_path in allowlist

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Only check first 10 lines
                    lines = f.readlines()[:10]
            except Exception:
                continue

            # Check for Last Updated marker
            for line_num, line in enumerate(lines, start=1):
                if self.LAST_UPDATED_PATTERN.search(line):
                    if not is_allowlisted:
                        allowed = ", ".join(sorted(allowlist)) or "none"
                        violations.append(
                            Violation(
                                policy_id=self.policy_id,
                                severity="warning",
                                file_path=file_path,
                                line_number=line_num,
                                message=(
                                    "Last Updated marker found in "
                                    "non-allowlisted file"
                                ),
                                suggestion=(
                                    f"Remove 'Last Updated' marker from "
                                    f"this file (only allowed in: {allowed})"
                                ),
                                can_auto_fix=True,
                            )
                        )

        return violations
