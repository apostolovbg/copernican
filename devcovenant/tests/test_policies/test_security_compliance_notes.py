"""Tests for the security compliance notes policy."""

from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.security_compliance_notes import (
    SecurityComplianceNotesCheck,
)


def _checker() -> SecurityComplianceNotesCheck:
    checker = SecurityComplianceNotesCheck()
    checker.set_options(
        {
            "guarded_paths": ["start.sh"],
            "log_path": "docs/security_changes.md",
        },
        {},
    )
    return checker


def _write_file(tmp_path: Path, rel: str, source: str) -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return target


def test_requires_log_update_for_guarded_changes(tmp_path: Path):
    """Security files trigger an error unless the log grows."""
    security = _write_file(tmp_path, "start.sh", "echo launch")
    _write_file(tmp_path, "docs/security_changes.md", "initial note\n")

    checker = _checker()
    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[security],
        all_files=[security],
    )
    violations = checker.check(context)

    assert violations
    assert any("security_changes.md" in v.message for v in violations)


def test_allows_guarded_changes_when_log_updated(tmp_path: Path):
    """No error when both the guarded file and log change."""
    security = _write_file(tmp_path, "start.sh", "echo launch")
    log = _write_file(tmp_path, "docs/security_changes.md", "noted\n")

    checker = _checker()
    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[security, log],
        all_files=[security, log],
    )

    assert checker.check(context) == []
