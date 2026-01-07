"""Remind contributors to grow documentation for user-facing changes."""

from pathlib import PurePosixPath
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation


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
        user_dirs_opt = self.get_option("user_visible_dirs", [])
        user_files_opt = self.get_option("user_visible_files", [])

        def _as_list(raw) -> List[str]:
            """Return a simple list of strings parsed from metadata/config."""
            if isinstance(raw, str):
                return [raw]
            if isinstance(raw, list):
                return [str(entry) for entry in raw]
            return [str(raw)] if raw else []

        user_dirs = [
            entry.strip() for entry in _as_list(user_dirs_opt) if entry
        ]
        user_files = [
            entry.strip() for entry in _as_list(user_files_opt) if entry
        ]

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

        targets = ", ".join(sorted(user_dirs + user_files)) or "the docs set"
        return [
            Violation(
                policy_id=self.policy_id,
                severity="info",
                message=(
                    "User-facing code or docs changed. Expand the configured "
                    f"documentation set ({targets}) with fresh context before "
                    "committing."
                ),
            )
        ]
