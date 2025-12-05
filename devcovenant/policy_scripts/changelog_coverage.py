"""
Policy: Changelog Coverage

Ensures Copernican changes land in CHANGELOG.md and RNG mini-game changes
land in rng_minigames/CHANGELOG.md.
"""

import subprocess
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation


def _latest_section(content: str) -> str:
    """Return the newest version section from a changelog."""

    marker = "## Version"
    search_start = 0
    log_marker = "## Log changes here"
    log_index = content.find(log_marker)
    if log_index != -1:
        search_start = log_index
    start = content.find(marker, search_start)
    if start == -1:
        start = content.find(marker)
        if start == -1:
            return content
    next_start = content.find("\n" + marker, start + len(marker))
    if next_start == -1:
        return content[start:]
    return content[start:next_start]


class ChangelogCoverageCheck(PolicyCheck):
    """Verify that modified files land in the appropriate changelog."""

    policy_id = "changelog-coverage"
    version = "2.2.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """
        Check if all changed files are documented in the relevant changelog.

        Args:
            context: Check context with repository info

        Returns:
            List of violations (empty if all files are documented)
        """
        violations: List[Violation] = []

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
            return violations

        if not changed_files:
            return violations

        root_changelog = context.repo_root / "CHANGELOG.md"
        rng_changelog = context.repo_root / "rng_minigames" / "CHANGELOG.md"

        skip_files = {
            "CHANGELOG.md",
            "rng_minigames/CHANGELOG.md",
            ".gitignore",
            ".pre-commit-config.yaml",
        }

        main_files: List[str] = []
        rng_files: List[str] = []

        for file_path in changed_files:
            if file_path in skip_files:
                continue
            if file_path.startswith("rng_minigames/"):
                rng_files.append(file_path)
            else:
                main_files.append(file_path)

        root_content = (
            root_changelog.read_text(encoding="utf-8")
            if (main_files or rng_files) and root_changelog.exists()
            else None
        )
        rng_content = (
            rng_changelog.read_text(encoding="utf-8")
            if (rng_files or main_files) and rng_changelog.exists()
            else None
        )
        root_section = _latest_section(root_content) if root_content else None
        rng_section = _latest_section(rng_content) if rng_content else None

        if main_files:
            if root_content is None:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        message="CHANGELOG.md does not exist",
                        suggestion=(
                            "Create CHANGELOG.md and document non-RNG changes"
                        ),
                        can_auto_fix=False,
                    )
                )
            else:
                missing = [
                    path for path in main_files if path not in root_section
                ]
                if missing:
                    files_str = ", ".join(missing)
                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="error",
                            file_path=root_changelog,
                            message=(
                                "The following files are not mentioned in "
                                f"CHANGELOG.md: {files_str}"
                            ),
                            suggestion=(
                                "Add entries to CHANGELOG.md documenting "
                                f"changes to: {files_str}"
                            ),
                            can_auto_fix=False,
                        )
                    )

        if rng_files:
            if rng_content is None:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        message=(
                            "rng_minigames/CHANGELOG.md does not exist, "
                            "but files under rng_minigames/ changed"
                        ),
                        suggestion=(
                            "Create rng_minigames/CHANGELOG.md and log the "
                            "mini-game updates"
                        ),
                        can_auto_fix=False,
                    )
                )
            else:
                missing_rng = [
                    path for path in rng_files if path not in rng_section
                ]
                if missing_rng:
                    files_str = ", ".join(missing_rng)
                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="error",
                            file_path=rng_changelog,
                            message=(
                                "The following files are not mentioned in "
                                f"rng_minigames/CHANGELOG.md: {files_str}"
                            ),
                            suggestion=(
                                "Add entries to rng_minigames/CHANGELOG.md "
                                f"documenting changes to: {files_str}"
                            ),
                            can_auto_fix=False,
                        )
                    )

        if rng_files and root_section:
            forbidden_main_mentions = [
                path for path in rng_files if path in root_section
            ]
            if forbidden_main_mentions:
                files_str = ", ".join(forbidden_main_mentions)
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=root_changelog,
                        message=(
                            "RNG mini-game files must not be logged in the "
                            f"root changelog: {files_str}"
                        ),
                        suggestion=(
                            "Remove RNG entries from CHANGELOG.md and log "
                            "them only in rng_minigames/CHANGELOG.md"
                        ),
                        can_auto_fix=False,
                    )
                )

        return violations
