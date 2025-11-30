"""DevCovenant policy: Prevent direct print() usage in library modules.

This policy ensures that library and engine code uses the managed console
output helper instead of bare print() calls, keeping diagnostics consistent
across platforms and properly routing output through dedicated utilities.
"""

import re
from pathlib import Path

from devcovenant.base import PolicyScript, Violation


PRINT_PATTERN = re.compile(r"(?<![\w.])print\s*\(")


class NoPrintInLibraryPolicy(PolicyScript):
    """Prevent direct print() usage in library modules."""

    def check(self, file_paths: list[Path]) -> list[Violation]:
        """Check for print() usage in library code."""
        violations = []

        # Define allowed files (console_output.py can use print)
        allowed = {
            self.repo_root / "copernican_lib" / "console_output.py",
        }

        for path in file_paths:
            if not path.is_file() or path.suffix != ".py":
                continue

            try:
                rel = path.relative_to(self.repo_root)
            except ValueError:
                continue

            # Only check copernican_lib/ and engines/
            if (
                not rel.parts
                or rel.parts[0] not in ("copernican_lib", "engines")
            ):
                continue

            # Skip allowed files
            if path in allowed:
                continue

            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            if PRINT_PATTERN.search(text):
                violations.append(
                    Violation(
                        policy_id="no-print-in-library",
                        path=path,
                        message=(
                            "Replace print() with "
                            "copernican_lib.console_output.write"
                        ),
                    )
                )

        return violations


def run(repo_root: Path, file_paths: list[Path]) -> list[Violation]:
    """Entry point for DevCovenant engine."""
    policy = NoPrintInLibraryPolicy(repo_root)
    return policy.check(file_paths)
