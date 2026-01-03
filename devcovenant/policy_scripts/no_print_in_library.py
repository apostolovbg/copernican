"""DevCovenant policy: Prevent direct print() usage in library modules.

This policy ensures that library and engine code uses the managed console
output helper instead of bare print() calls, keeping diagnostics consistent
across platforms and properly routing output through dedicated utilities.
"""

import re
from pathlib import Path
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation

PRINT_PATTERN = re.compile(r"(?<![\w.])print\s*\(")
DEFAULT_ALLOWED = {"copernican_lib/console_output.py"}
DEFAULT_TARGET_ROOTS = {"copernican_lib", "engines"}
DEFAULT_VENDOR_PATHS = {"copernican_lib/vendor"}


class NoPrintInLibraryCheck(PolicyCheck):
    """Prevent direct print() usage in library modules."""

    policy_id = "no-print-in-library"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Check for print() usage in library code."""
        violations = []

        cfg = context.get_policy_config(self.policy_id)
        allowed_rel = set(cfg.get("allowed_files", list(DEFAULT_ALLOWED)))
        allowed = {
            (context.repo_root / Path(rel_path)).resolve()
            for rel_path in allowed_rel
        }
        target_roots = set(cfg.get("target_roots", list(DEFAULT_TARGET_ROOTS)))
        vendor_paths = set(cfg.get("vendor_paths", list(DEFAULT_VENDOR_PATHS)))

        file_paths = context.changed_files or context.all_files

        for path in file_paths:
            if not path.is_file() or path.suffix != ".py":
                continue

            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue

            # Skip bundled vendor code under copernican_lib/vendor/
            rel_posix = rel.as_posix()
            if any(
                rel_posix == vendor or rel_posix.startswith(f"{vendor}/")
                for vendor in vendor_paths
            ):
                continue

            # Only check copernican_lib/ and engines/
            if not rel.parts or rel.parts[0] not in target_roots:
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
                            "Replace print() with "
                            "copernican_lib.console_output.write"
                        ),
                    )
                )

        return violations
