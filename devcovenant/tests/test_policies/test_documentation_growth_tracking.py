"""Tests for the documentation growth reminder policy."""

from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.documentation_growth_tracking import (
    DocumentationGrowthTrackingCheck,
)


def test_reminder_for_user_facing_file(tmp_path: Path):
    """README updates should trigger the reminder."""
    target = tmp_path / "README.md"
    target.write_text("Updated docs\n", encoding="utf-8")
    checker = DocumentationGrowthTrackingCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[target])
    violations = checker.check(context)

    assert len(violations) == 1
    assert "User-facing" in violations[0].message


def test_no_reminder_for_internal_file(tmp_path: Path):
    """Internal helpers do not trigger the reminder."""
    target = tmp_path / "tests" / "helper.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('ok')\n", encoding="utf-8")
    checker = DocumentationGrowthTrackingCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[target])
    assert checker.check(context) == []
