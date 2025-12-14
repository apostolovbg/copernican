"""Tests for the read-only directories policy."""

from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.read_only_directories import (
    ReadOnlyDirectoriesCheck,
)


def _prepare_patterns(tmp_path: Path) -> None:
    """Create the patterns file under the temporary repo."""
    patterns = tmp_path / "devcovenant" / "read_only_directories.txt"
    patterns.parent.mkdir(parents=True, exist_ok=True)
    patterns.write_text("data/**\n", encoding="utf-8")


def _prepare_file(tmp_path: Path) -> Path:
    """Create a fake dataset file that will be staged."""
    target = tmp_path / "data" / "example" / "cosmo_parser_A.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("content\n", encoding="utf-8")
    return target


def _write_waiver(tmp_path: Path, relative_path: str) -> None:
    """Write a waiver entry that permits the relative path."""
    waiver = (
        tmp_path / ".devcovenant" / "waivers" / "read-only-directories.txt"
    )
    waiver.parent.mkdir(parents=True, exist_ok=True)
    waiver.write_text(f"{relative_path}\n", encoding="utf-8")


def test_blocks_read_only_change(tmp_path: Path):
    """Changes inside data/ should violate when no waiver exists."""
    _prepare_patterns(tmp_path)
    target = _prepare_file(tmp_path)

    checker = ReadOnlyDirectoriesCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[target])
    violations = checker.check(context)

    assert len(violations) == 1
    assert "read-only directories" in violations[0].message.lower()


def test_respects_waiver(tmp_path: Path):
    """Waived paths under data/ are exempt from the policy."""
    _prepare_patterns(tmp_path)
    target = _prepare_file(tmp_path)
    rel = target.relative_to(tmp_path)
    _write_waiver(tmp_path, rel.as_posix())

    checker = ReadOnlyDirectoriesCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[target])
    violations = checker.check(context)

    assert violations == []
