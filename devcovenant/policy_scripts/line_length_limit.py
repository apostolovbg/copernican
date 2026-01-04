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
    DEFAULT_SUFFIXES = [".py", ".md", ".rst", ".txt"]
    DEFAULT_SKIP_PREFIXES = ["copernican_lib/vendor", "data"]

    def check(self, context: CheckContext) -> List[Violation]:
        """
        Check files for lines exceeding the length limit.

        Args:
            context: Check context

        Returns:
            List of violations
        """
        max_length = int(self.get_option("max_length", self.MAX_LINE_LENGTH))

        suffixes = self.get_option("include_suffixes", self.DEFAULT_SUFFIXES)
        if isinstance(suffixes, str):
            suffix_list = [suffixes]
        else:
            suffix_list = list(suffixes or [])
        suffixes = [
            (suffix if suffix.startswith(".") else f".{suffix}")
            .strip()
            .lower()
            for suffix in suffix_list
            if isinstance(suffix, str) and suffix.strip()
        ]

        skip_prefixes_option = self.get_option(
            "skip_prefixes", self.DEFAULT_SKIP_PREFIXES
        )
        if isinstance(skip_prefixes_option, str):
            skip_prefixes = [skip_prefixes_option]
        else:
            skip_prefixes = list(skip_prefixes_option or [])
        skip_prefixes = [prefix.strip("/ ") for prefix in skip_prefixes]

        violations = []

        files_pool = context.changed_files or context.all_files or []
        files_to_check = [
            path for path in files_pool if path.suffix.lower() in suffixes
        ]

        for file_path in files_to_check:
            try:
                rel_path = file_path.relative_to(context.repo_root)
            except ValueError:
                continue

            rel_posix = rel_path.as_posix()
            if any(
                rel_posix == prefix or rel_posix.startswith(f"{prefix}/")
                for prefix in skip_prefixes
                if prefix
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
