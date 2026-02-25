"""Warn when security-critical files change without updating the log."""

from pathlib import Path, PurePosixPath
from typing import Iterable, List

from devcovenant.base import CheckContext, PolicyCheck, Violation


def _is_security_path(
    rel_path: PurePosixPath, guarded_paths: List[str]
) -> bool:
    """Return True when the relative path touches a guarded area."""
    rel_str = rel_path.as_posix()
    for pattern in guarded_paths:
        if rel_str == pattern or rel_str.startswith(f"{pattern}/"):
            return True
    return False


def _has_security_allocation(
    paths: Iterable[PurePosixPath], log_path: PurePosixPath
) -> bool:
    """Return True when the security log itself is modified."""
    for path in paths:
        if path == log_path:
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
        guarded_option = self.get_option("guarded_paths", [])
        if isinstance(guarded_option, str):
            guarded_paths = [guarded_option]
        else:
            guarded_paths = list(guarded_option or [])
        log_rel = Path(self.get_option("log_path", "security_changes.md"))
        log_posix = PurePosixPath(log_rel.as_posix())

        for path in files:
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue
            rel_posix = PurePosixPath(rel.as_posix())
            touched_paths.append(rel_posix)
            if _is_security_path(rel_posix, guarded_paths):
                security_paths.append(rel_posix)

        if security_paths and not _has_security_allocation(
            touched_paths, log_posix
        ):
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=context.repo_root / log_rel,
                    message=(
                        "Security-critical files changed without a new entry "
                        f"in `{log_rel}`."
                    ),
                )
            )

        return violations
