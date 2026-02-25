"""
Policy: No Print in Library

Require modules rooted in `target_roots` to use the shared helper listed in
`allowed_files` while skipping vendor paths supplied via `vendor_paths`.
"""

import re
from pathlib import Path
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation
from devcovenant.selectors import SelectorSet

PRINT_PATTERN = re.compile(r"(?<![\w.])print\s*\(")


class NoPrintInLibraryCheck(PolicyCheck):
    """Prevent direct print() usage in library modules."""

    policy_id = "no-print-in-library"
    version = "1.0.0"

    def _selector(self) -> SelectorSet:
        """Return the selector describing modules under enforcement."""
        return SelectorSet.from_policy(self)

    def check(self, context: CheckContext) -> List[Violation]:
        """Check for print() usage in library code."""
        violations = []

        allowed_option = self.get_option("allowed_files", [])
        if isinstance(allowed_option, str):
            allowed_rel = {allowed_option}
        else:
            allowed_rel = set(allowed_option or [])
        allowed = {
            (context.repo_root / Path(rel_path)).resolve()
            for rel_path in allowed_rel
        }
        selector = self._selector()
        file_paths = context.changed_files or context.all_files

        for path in file_paths:
            if not path.is_file() or path.suffix != ".py":
                continue

            if not selector.matches(path, context.repo_root):
                continue

            resolved = path.resolve()

            # Skip allowed files
            if resolved in allowed:
                continue

            try:
                text = resolved.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            if PRINT_PATTERN.search(text):
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=path,
                        message=(
                            "Replace print() with the configured "
                            "console-output helper."
                        ),
                    )
                )

        return violations
