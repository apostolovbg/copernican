"""DevCovenant policy: Prevent direct print() usage in library modules.

This policy ensures that library and engine code uses the managed console
output helper instead of bare print() calls, keeping diagnostics consistent
across platforms and properly routing output through dedicated utilities.
"""

import re
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation

PRINT_PATTERN = re.compile(r"(?<![\w.])print\s*\(")


class NoPrintInLibraryCheck(PolicyCheck):
    """Prevent direct print() usage in library modules."""

    policy_id = "no-print-in-library"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Check for print() usage in library code."""
        violations = []

        # Define allowed files (console_output.py can use print)
        allowed = {
            context.repo_root / "copernican_lib" / "console_output.py",
        }

        file_paths = context.changed_files or context.all_files

        for path in file_paths:
            if not path.is_file() or path.suffix != ".py":
                continue

            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue

            # Skip bundled vendor code under copernican_lib/vendor/
            if (
                len(rel.parts) >= 2
                and rel.parts[0] == "copernican_lib"
                and rel.parts[1] == "vendor"
            ):
                continue

            # Only check copernican_lib/ and engines/
            if not rel.parts or rel.parts[0] not in (
                "copernican_lib",
                "engines",
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
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=path,
                        message=(
                            "Replace print() with "
                            "copernican_lib.console_output.write"
                        ),
                    )
                )

        return violations
