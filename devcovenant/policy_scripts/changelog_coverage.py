"""
Policy: Changelog Coverage

Ensures Copernican changes land in CHANGELOG.md and RNG mini-game changes
land in rng_minigames/CHANGELOG.md.
"""

import subprocess
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation


class ChangelogCoverageCheck(PolicyCheck):
    """Verify that modified files are mentioned in the appropriate changelog."""

    policy_id = "changelog-coverage"
    version = "2.0.0"

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

        if main_files:
            if not root_changelog.exists():
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
                root_content = root_changelog.read_text(encoding="utf-8")
                missing = [
                    path for path in main_files if path not in root_content
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
            if not rng_changelog.exists():
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
                rng_content = rng_changelog.read_text(encoding="utf-8")
                missing_rng = [
                    path for path in rng_files if path not in rng_content
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

        return violations
