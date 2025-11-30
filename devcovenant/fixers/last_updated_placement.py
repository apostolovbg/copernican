"""
Fixer for Last Updated Marker Placement policy.

Automatically removes Last Updated markers from non-allowlisted files.
"""

import re

from devcovenant.base import FixResult, PolicyFixer, Violation


class LastUpdatedPlacementFixer(PolicyFixer):
    """
    Removes Last Updated markers from non-allowlisted files.
    """

    policy_id = "last-updated-placement"

    LAST_UPDATED_PATTERN = re.compile(
        r"^.*(\*\*Last Updated:\*\*|Last Updated:|# Last Updated).*$",
        re.MULTILINE,
    )

    def can_fix(self, violation: Violation) -> bool:
        """
        Check if this violation can be fixed.

        Args:
            violation: The violation to check

        Returns:
            True if this is a last-updated-placement violation
        """
        return (
            violation.policy_id == self.policy_id
            and violation.file_path is not None
        )

    def fix(self, violation: Violation) -> FixResult:
        """
        Remove the Last Updated marker from the file.

        Args:
            violation: The violation to fix

        Returns:
            FixResult indicating success/failure
        """
        if not violation.file_path:
            return FixResult(
                success=False, message="No file path provided in violation"
            )

        try:
            # Read the file
            with open(violation.file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove Last Updated lines
            original_content = content
            content = self.LAST_UPDATED_PATTERN.sub("", content)

            # Remove any resulting double blank lines
            content = re.sub(r"\n\n\n+", "\n\n", content)

            # Write back only if changed
            if content != original_content:
                with open(violation.file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                return FixResult(
                    success=True,
                    message=(
                        f"Removed Last Updated marker from "
                        f"{violation.file_path}"
                    ),
                    files_modified=[violation.file_path],
                )
            else:
                return FixResult(
                    success=True,
                    message="No changes needed",
                )

        except Exception as e:
            return FixResult(success=False, message=f"Failed to fix: {e}")
