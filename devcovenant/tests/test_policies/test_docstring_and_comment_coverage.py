"""Tests for the docstring and comment coverage policy."""

from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.docstring_and_comment_coverage import (
    DocstringAndCommentCoverageCheck,
)


def _create_file(tmp_path: Path, source: str) -> Path:
    target = tmp_path / "copernican_lib" / "helpers" / "example.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return target


def test_flags_missing_docstrings(tmp_path: Path):
    """Modules and functions without comments or docstrings
    trigger violations."""
    source = (
        "def foo():\n"
        "    return 42\n"
        "\n"
        "class Bar:\n"
        "    def baz(self):\n"
        "        pass\n"
    )
    target = _create_file(tmp_path, source)

    checker = DocstringAndCommentCoverageCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[target])
    violations = checker.check(context)

    assert len(violations) >= 3
    assert any("Module lacks" in v.message for v in violations)
    assert any(
        "function" in v.message.lower() and "foo" in v.message.lower()
        for v in violations
    )


def test_comments_satisfy_policy(tmp_path: Path):
    """Long comments before definitions count as documentation."""
    source = (
        "# Library helper module\n"
        "\n"
        "# Explain foo\n"
        "def foo():\n"
        "    # Internal behavior notes\n"
        "    return 1\n"
    )
    target = _create_file(tmp_path, source)

    checker = DocstringAndCommentCoverageCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[target])
    violations = checker.check(context)

    assert violations == []


def test_all_files_scanned_when_no_changes(tmp_path: Path):
    """Ensure all_files is inspected when no changed files are present."""
    source = "def foo():\n    return 5\n"
    target = _create_file(tmp_path, source)

    checker = DocstringAndCommentCoverageCheck()
    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[],
        all_files=[target],
    )
    violations = checker.check(context)

    assert any("Module lacks" in v.message for v in violations)
