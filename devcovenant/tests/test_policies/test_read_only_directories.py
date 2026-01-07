"""Tests for the read-only directories policy."""

from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.read_only_directories import (
    ReadOnlyDirectoriesCheck,
)


def _prepare_file(tmp_path: Path) -> Path:
    """Create a fake dataset metadata file that should remain read-only."""
    target = tmp_path / "data" / "example" / "metadata_example.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("name: demo\n", encoding="utf-8")
    return target


def _prepare_parser(tmp_path: Path) -> Path:
    """Create a parser file that is exempt from read-only enforcement."""
    target = tmp_path / "data" / "example" / "cosmo_parser_A.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("content\n", encoding="utf-8")
    return target


def test_blocks_read_only_change(tmp_path: Path):
    """Changes inside data/ should violate when no waiver exists."""
    target = _prepare_file(tmp_path)

    checker = ReadOnlyDirectoriesCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[target])
    violations = checker.check(context)

    assert violations, "Read-only changes should fail without exemptions"


def test_exclude_globs_allow_parsers(tmp_path: Path):
    """Parsers matching the exclusion list should be editable."""
    target = _prepare_parser(tmp_path)

    checker = ReadOnlyDirectoriesCheck()
    checker.set_options(
        {
            "include_globs": ["data/**"],
            "exclude_globs": ["data/**/cosmo_parser_*.py"],
        },
        {},
    )
    context = CheckContext(repo_root=tmp_path, changed_files=[target])
    violations = checker.check(context)

    assert not violations, "Parser files must escape the read-only guard"
