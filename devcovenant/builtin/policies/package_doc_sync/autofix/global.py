"""Fixer: package-doc sync."""

from __future__ import annotations

from pathlib import Path

from devcovenant.core.policy_contract import FixResult, PolicyFixer, Violation


class PackageDocSyncFixer(PolicyFixer):
    """Write the expected transformed doc text into the target path."""

    policy_id = "package-doc-sync"

    def can_fix(self, violation: Violation) -> bool:
        """Return True when the violation targets package-doc-sync."""
        return violation.policy_id == self.policy_id

    def fix(self, violation: Violation) -> FixResult:
        """Apply the expected text to the target package-facing doc."""
        expected = violation.context.get("expected_text")
        target = violation.context.get("target_path")
        if not expected or not target:
            return FixResult(
                success=False,
                message="Missing expected_text or target_path in context.",
            )
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(expected), encoding="utf-8")
        return FixResult(
            success=True,
            message=f"Synced {path} from the configured source doc.",
            files_modified=[path],
        )
