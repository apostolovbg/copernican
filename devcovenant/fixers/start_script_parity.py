"""
Auto-fixer for start-script-parity violations.
"""

from __future__ import annotations

from pathlib import Path

from devcovenant.base import FixResult, PolicyFixer, Violation
from devcovenant.policy_scripts.start_script_parity import START_SCRIPTS


class StartScriptParityFixer(PolicyFixer):
    """Copy the edited launcher into the remaining start scripts."""

    policy_id = "start-script-parity"

    def can_fix(self, violation: Violation) -> bool:
        """Return True when both changed and missing launchers are known."""
        changed = violation.context.get("changed") or []
        missing = violation.context.get("missing") or []
        return (
            violation.policy_id == self.policy_id
            and violation.file_path is not None
            and bool(changed)
            and bool(missing)
        )

    def fix(self, violation: Violation) -> FixResult:
        """Copy the edited launcher's contents into the missing peers."""
        repo_root = getattr(self, "repo_root", Path.cwd())
        changed = violation.context.get("changed") or []
        missing = violation.context.get("missing") or []
        source_name = changed[0]
        source_path = repo_root / source_name
        if not source_path.exists():
            return FixResult(
                success=False,
                message=f"Source launcher {source_name} not found",
            )

        try:
            content = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            return FixResult(success=False, message=str(exc))

        modified: list[Path] = []
        for target_name in missing:
            if target_name not in START_SCRIPTS:
                continue
            target_path = repo_root / target_name
            target_path.write_text(content, encoding="utf-8")
            modified.append(target_path)

        if not modified:
            return FixResult(
                success=False, message="No launchers were updated"
            )

        return FixResult(
            success=True,
            message=(
                f"Copied {source_name} guardrails into "
                f"{', '.join(t.name for t in modified)}"
            ),
            files_modified=modified,
        )
