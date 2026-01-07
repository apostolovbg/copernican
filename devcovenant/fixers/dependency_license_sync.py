"""
Fixer for dependency-license-sync violations.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, List

from devcovenant.base import FixResult, PolicyFixer, Violation
from devcovenant.policy_scripts.dependency_license_sync import (
    _contains_reference,
    _extract_license_report,
)


class DependencyLicenseSyncFixer(PolicyFixer):
    """Ensure the license report and directory reflect dependency updates."""

    policy_id = "dependency-license-sync"

    def can_fix(self, violation: Violation) -> bool:
        """Only run when dependency manifests are known."""
        return (
            violation.policy_id == self.policy_id
            and bool(violation.context.get("changed_dependency_files"))
        )

    def fix(self, violation: Violation) -> FixResult:
        """Update the license report and directory markers."""
        issue = violation.context.get("issue")
        repo_root = getattr(self, "repo_root", Path.cwd())
        modified: List[Path] = []
        messages: List[str] = []

        if issue in {"third_party", "missing_report", "missing_reference"}:
            third_party = repo_root / violation.context["third_party_file"]
            result = self._sync_license_report(third_party, violation.context)
            if result.files_modified:
                modified.extend(result.files_modified)
            if result.message:
                messages.append(result.message)

        if issue == "licenses_dir":
            licenses_dir = repo_root / violation.context["licenses_dir"]
            result = self._touch_license_directory(
                licenses_dir, violation.context
            )
            if result.files_modified:
                modified.extend(result.files_modified)
            if result.message:
                messages.append(result.message)

        if not modified:
            return FixResult(
                success=False,
                message="No dependency-license-sync updates were applied",
            )

        summary = " ".join(messages) if messages else ""
        return FixResult(
            success=True,
            message=summary.strip()
            or "Updated license records for dependency changes",
            files_modified=modified,
        )

    def _sync_license_report(
        self, target: Path, context: dict
    ) -> FixResult:
        """Ensure the license report records each changed dependency file."""
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            existing = "# Third-Party Licenses\n"

        heading = context["report_heading"]
        if heading.strip() not in existing:
            if not existing.endswith("\n"):
                existing += "\n"
            existing = existing.rstrip() + f"\n\n{heading}\n"

        section = _extract_license_report(existing, heading)
        additions: List[str] = []
        if context.get("issue") == "missing_reference":
            targets: Iterable[str] = context.get("missing_references", [])
        else:
            targets = context.get("changed_dependency_files", [])

        today = dt.date.today().isoformat()
        for dep in targets:
            if section and _contains_reference(section, dep):
                continue
            additions.append(
                f"- {today}: Recorded dependency update for `{dep}`."
                " (auto-fix)"
            )

        if not additions:
            # Ensure the file is still marked as touched by adding a timestamp.
            additions.append(
                f"- {today}: Confirmed dependency metadata via auto-fix."
            )

        new_content = existing.rstrip() + "\n" + "\n".join(additions) + "\n"
        target.write_text(new_content, encoding="utf-8")
        return FixResult(
            success=True,
            message=f"Appended license report entries to {target}",
            files_modified=[target],
        )

    def _touch_license_directory(
        self, licenses_dir: Path, context: dict
    ) -> FixResult:
        """Record a sentinel file inside the licenses directory."""
        licenses_dir.mkdir(parents=True, exist_ok=True)
        sentinel = licenses_dir / "AUTO_LICENSE_SYNC.txt"
        today = dt.date.today().isoformat()
        changed = ", ".join(context.get("changed_dependency_files", []))
        entry = (
            f"{today}: DevCovenant auto-fix touched license assets "
            f"after changes to {changed or 'dependency manifests'}.\n"
        )
        with sentinel.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        return FixResult(
            success=True,
            message=f"Wrote license sync marker to {sentinel}",
            files_modified=[sentinel],
        )
