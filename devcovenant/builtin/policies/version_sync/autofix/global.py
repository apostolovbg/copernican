"""Autofixer for version-sync violations."""

from __future__ import annotations

from pathlib import Path

from devcovenant.builtin.policies.version_sync import version_sync
from devcovenant.core.policy_contract import FixResult, PolicyFixer, Violation


class VersionSyncFixer(PolicyFixer):
    """Rewrite synchronized targets to the tracked canonical version."""

    policy_id = "version-sync"

    def can_fix(self, violation: Violation) -> bool:
        """Return True when the violation carries a target rewrite context."""
        return (
            violation.policy_id == self.policy_id
            and violation.file_path is not None
            and bool(violation.context.get("extractor_name"))
            and bool(violation.context.get("tracked_version"))
        )

    def fix(self, violation: Violation) -> FixResult:
        """Write the tracked version into the declared mismatch target."""
        if violation.file_path is None:
            return FixResult(
                success=False,
                message="No file path provided in violation.",
            )
        extractor_name = str(
            violation.context.get("extractor_name") or ""
        ).strip()
        tracked_version = str(
            violation.context.get("tracked_version") or ""
        ).strip()
        changelog_prefix = str(
            violation.context.get("changelog_prefix") or "## Version"
        ).strip()
        if not extractor_name or not tracked_version:
            return FixResult(
                success=False,
                message=(
                    "Missing extractor_name or tracked_version in "
                    "violation context."
                ),
            )

        target = Path(violation.file_path)
        try:
            changed = version_sync.write_synced_target_version(
                target,
                extractor_name=extractor_name,
                tracked_version=tracked_version,
                changelog_prefix=changelog_prefix,
            )
        except (OSError, ValueError) as error:
            return FixResult(
                success=False,
                message=str(error),
            )

        if not changed:
            return FixResult(
                success=True,
                message=(
                    f"Version-sync target is already current: "
                    f"{target.as_posix()}"
                ),
            )
        return FixResult(
            success=True,
            message=(
                "Updated version-sync target to "
                f"{tracked_version}: {target.as_posix()}"
            ),
            files_modified=[target],
        )
