"""
Policy: Line Length Limit

Ensures lines are under 79 characters for readability.
"""

from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation


class LineLengthLimitCheck(PolicyCheck):
    """
    Check that lines are under 79 characters.
    """

    policy_id = "line-length-limit"
    version = "1.0.0"

    MAX_LINE_LENGTH = 79

    def check(self, context: CheckContext) -> List[Violation]:
        """
        Check files for lines exceeding the length limit.

        Args:
            context: Check context

        Returns:
            List of violations
        """
        violations = []

        # Determine which files to check (only Python files)
        files_to_check = []
        if context.changed_files:
            files_to_check = [
                f for f in context.changed_files if f.suffix == ".py"
            ]
        else:
            files_to_check = [
                f for f in context.all_files if f.suffix == ".py"
            ]

        for file_path in files_to_check:
            try:
                rel_path = file_path.relative_to(context.repo_root)
            except ValueError:
                continue

            if (
                len(rel_path.parts) >= 2
                and rel_path.parts[0] == "copernican_lib"
                and rel_path.parts[1] == "vendor"
            ):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                continue

            # Check each line
            for line_num, line in enumerate(lines, start=1):
                # Remove trailing newline for length check
                line_content = line.rstrip("\n")

                if len(line_content) > self.MAX_LINE_LENGTH:
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
                                f"Line exceeds {self.MAX_LINE_LENGTH} "
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
