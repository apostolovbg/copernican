"""Tests for the start script parity policy."""

from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.fixers.start_script_parity import StartScriptParityFixer
from devcovenant.policy_scripts.start_script_parity import (
    START_SCRIPTS,
    StartScriptParityCheck,
)


def _create_launchers(repo_root: Path) -> None:
    for name in START_SCRIPTS:
        target = repo_root / name
        target.write_text("echo starter", encoding="utf-8")


def test_flags_unmatched_start_scripts(tmp_path: Path):
    _create_launchers(tmp_path)
    changed = [tmp_path / "start.sh"]
    context = CheckContext(repo_root=tmp_path, changed_files=changed)
    violations = StartScriptParityCheck().check(context)

    assert len(violations) == 1
    assert "start.command" in violations[0].message


def test_all_launchers_updated(tmp_path: Path):
    _create_launchers(tmp_path)
    changed = [tmp_path / name for name in START_SCRIPTS]
    context = CheckContext(repo_root=tmp_path, changed_files=changed)
    assert StartScriptParityCheck().check(context) == []


def test_auto_fix_copies_missing_launchers(tmp_path: Path):
    _create_launchers(tmp_path)
    changed = [tmp_path / "start.sh"]
    context = CheckContext(repo_root=tmp_path, changed_files=changed)
    violations = StartScriptParityCheck().check(context)
    assert violations
    fixer = StartScriptParityFixer()
    fixer.repo_root = tmp_path
    result = fixer.fix(violations[0])
    assert result.success
    for name in START_SCRIPTS:
        path = tmp_path / name
        assert path.read_text() == (tmp_path / "start.sh").read_text()
