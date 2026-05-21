"""Autofixer for dependency-management violations."""

from __future__ import annotations

from pathlib import Path

from devcovenant.core.policy_contract import FixResult, PolicyFixer, Violation
from devcovenant.core.policy_runtime import run_policy_runtime_action


class DependencyManagementFixer(PolicyFixer):
    """Repair dependency-management artifacts through the policy runtime."""

    policy_id = "dependency-management"

    def can_fix(self, violation: Violation) -> bool:
        """Only run when dependency manifests are known."""
        return violation.policy_id == self.policy_id and bool(
            violation.context.get("changed_dependency_files")
        )

    def fix(self, violation: Violation) -> FixResult:
        """Run the declared dependency-management refresh action."""
        repo_root = getattr(self, "repo_root", None)
        if repo_root is None:
            return FixResult(
                success=False,
                message="Autofix repository root is unavailable.",
            )
        try:
            payload = run_policy_runtime_action(
                repo_root,
                policy_id=self.policy_id,
                action="refresh-all",
                payload={
                    "changed_dependency_files": list(
                        violation.context.get("changed_dependency_files", [])
                    )
                },
            )
        except ValueError as error:
            return FixResult(
                success=False,
                message=str(error),
            )
        refreshed_artifacts = payload.get("refreshed_artifacts", [])
        modified = [Path(str(entry)) for entry in refreshed_artifacts]
        if modified:
            file_list = ", ".join(path.as_posix() for path in modified)
            return FixResult(
                success=True,
                message=(
                    "Updated dependency-management artifacts: " f"{file_list}"
                ),
                files_modified=modified,
            )
        return FixResult(
            success=True,
            message="Dependency-management artifacts are already in sync.",
        )
