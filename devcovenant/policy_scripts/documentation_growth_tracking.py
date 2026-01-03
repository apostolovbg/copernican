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


def _is_user_visible(
    rel_path: PurePosixPath,
    dirs: List[str],
    files: List[str],
) -> bool:
    """Return True if the relative path affects end-user workflows or docs."""
    if rel_path.parts and rel_path.parts[0] in dirs:
        return True
    if rel_path.name in files:
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
        cfg = context.get_policy_config(self.policy_id)
        user_dirs = cfg.get("user_visible_dirs", list(USER_VISIBLE_DIRS))
        user_files = cfg.get("user_visible_files", list(USER_VISIBLE_FILES))

        for path in files:
            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue

            rel_posix = PurePosixPath(rel.as_posix())
            if _is_user_visible(rel_posix, user_dirs, user_files):
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
