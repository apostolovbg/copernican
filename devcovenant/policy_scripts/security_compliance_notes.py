"""Warn when security-critical files change without updating the log."""

from pathlib import Path, PurePosixPath
from typing import Iterable, List

from devcovenant.base import CheckContext, PolicyCheck, Violation

SECURITY_PATTERNS = (
    "start.sh",
    "start.command",
    "start.bat",
    "copernican_lib/security",
    "docs/security",
)

SECURITY_LOG = Path("docs/security_changes.md")


def _is_security_path(rel_path: PurePosixPath) -> bool:
    """Return True when the relative path touches a guarded area."""
    rel_str = rel_path.as_posix()
    for pattern in SECURITY_PATTERNS:
        if rel_str == pattern or rel_str.startswith(f"{pattern}/"):
            return True
    return False


def _has_security_allocation(paths: Iterable[PurePosixPath]) -> bool:
    """Return True when the security log itself is modified."""
    for path in paths:
        if path == PurePosixPath(SECURITY_LOG.as_posix()):
            return True
    return False


class SecurityComplianceNotesCheck(PolicyCheck):
    """Ensure the security change log tracks guarded edits."""

    policy_id = "security-compliance-notes"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Detect guarded files without a corresponding log update."""
        files = context.all_files or context.changed_files or []
        violations: List[Violation] = []
        security_paths: List[PurePosixPath] = []
        touched_paths: List[PurePosixPath] = []

        for path in files:
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue
            rel_posix = PurePosixPath(rel.as_posix())
            touched_paths.append(rel_posix)
            if _is_security_path(rel_posix):
                security_paths.append(rel_posix)

        if security_paths and not _has_security_allocation(touched_paths):
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=context.repo_root / SECURITY_LOG,
                    message=(
                        "Security-critical files changed without a new entry "
                        "in `docs/security_changes.md`."
                    ),
                )
            )

        return violations
