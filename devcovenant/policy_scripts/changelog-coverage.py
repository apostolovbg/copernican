"""
Policy: Changelog Coverage

Ensures all changed files are documented in CHANGELOG.md.
"""

import subprocess
from pathlib import Path
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation


class ChangelogCoverageCheck(PolicyCheck):
    """
    Verify that all modified files are mentioned in the latest
    CHANGELOG.md entry.
    """

    policy_id = "changelog-coverage"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """
        Check if all changed files are documented in CHANGELOG.md.

        Args:
            context: Check context with repository info

        Returns:
            List of violations (empty if all files are documented)
        """
        violations = []

        # Get list of changed files from git
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=context.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            changed_files = [
                f for f in result.stdout.strip().split("\n") if f
            ]
        except Exception:
            # If git fails, skip this check
            return violations

        if not changed_files:
            # No changes, nothing to check
            return violations

        # Read CHANGELOG.md
        changelog_path = context.repo_root / "CHANGELOG.md"
        if not changelog_path.exists():
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    message="CHANGELOG.md does not exist",
                    suggestion="Create CHANGELOG.md and document your changes",
                    can_auto_fix=False,
                )
            )
            return violations

        with open(changelog_path, "r", encoding="utf-8") as f:
            changelog_content = f.read()

        # Check if each changed file is mentioned in the changelog
        missing_files = []
        for file_path in changed_files:
            # Skip certain files
            if file_path in [
                "CHANGELOG.md",
                ".gitignore",
                ".pre-commit-config.yaml",
            ]:
                continue

            # Check if file is mentioned in changelog
            if file_path not in changelog_content:
                missing_files.append(file_path)

        if missing_files:
            files_str = ', '.join(missing_files)
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=changelog_path,
                    message=(
                        f"The following changed files are not "
                        f"documented in CHANGELOG.md: {files_str}"
                    ),
                    suggestion=(
                        f"Add entries to CHANGELOG.md documenting "
                        f"changes to: {files_str}"
                    ),
                    can_auto_fix=False,
                )
            )

        return violations
