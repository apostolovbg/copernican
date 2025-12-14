"""Remind contributors to grow documentation for user-facing changes."""

from pathlib import PurePosixPath
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation

USER_VISIBLE_DIRS = {
    "copernican_lib",
    "engines",
    "models",
    "docs",
}
USER_VISIBLE_FILES = {
    "README.md",
    "AGENTS.md",
    "copernican.py",
    "start.sh",
    "start.command",
    "start.bat",
}


def _is_user_visible(rel_path: PurePosixPath) -> bool:
    """Return True if the relative path affects end-user workflows or docs."""
    if rel_path.parts:
        if rel_path.parts[0] in USER_VISIBLE_DIRS:
            return True
    if rel_path.name in USER_VISIBLE_FILES:
        return True
    return False


class DocumentationGrowthTrackingCheck(PolicyCheck):
    """Fiducial reminder to add prose when user-visible elements change."""

    policy_id = "documentation-growth-tracking"
    version = "1.0.0"

    def check(self, context: CheckContext):
        """Emit a reminder when user-visible surfaces were touched."""
        files = context.changed_files or []
        impacted: List[PurePosixPath] = []

        for path in files:
            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue

            rel_posix = PurePosixPath(rel.as_posix())
            if _is_user_visible(rel_posix):
                impacted.append(rel_posix)

        if not impacted:
            return []

        return [
            Violation(
                policy_id=self.policy_id,
                severity="info",
                message=(
                    "User-facing code or docs changed. Please expand "
                    "README.md, AGENTS.md or the relevant docs/ entry "
                    "with a new paragraph or example so the corpus grows."
                ),
            )
        ]
