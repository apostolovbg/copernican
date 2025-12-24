"""Tests for the raw string escape policy."""

from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.raw_string_escapes import RawStringEscapesCheck


def _write_module(tmp_path: Path, source: str) -> Path:
    target = tmp_path / "copernican_lib" / "helpers" / "escape_example.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return target


def test_detects_suspicious_backslash(tmp_path: Path):
    source = 'pattern = "\\s+\\."'
    target = _write_module(tmp_path, source)
    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[target],
    )

    checker = RawStringEscapesCheck()
    violations = checker.check(context)

    assert violations
    assert any("backslash" in v.message.lower() for v in violations)
    assert all(v.severity == "warning" for v in violations)


def test_allows_raw_strings(tmp_path: Path):
    source = 'regex = r"\\s+"'
    target = _write_module(tmp_path, source)
    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[target],
    )

    checker = RawStringEscapesCheck()
    assert checker.check(context) == []


def test_allows_standard_escape_sequences(tmp_path: Path):
    source = 'line = "\\n"'
    target = _write_module(tmp_path, source)
    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[target],
    )

    checker = RawStringEscapesCheck()
    assert checker.check(context) == []
