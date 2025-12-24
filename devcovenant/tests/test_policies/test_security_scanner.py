"""Tests for the security scanner policy."""

from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.security_scanner import SecurityScannerCheck


def _write_module(tmp_path: Path, name: str, source: str) -> Path:
    target = tmp_path / "copernican_lib" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return target


def test_detects_insecure_eval(tmp_path: Path):
    """`eval` usage raises a violation."""
    source = "def foo():\n    return eval('2+2')\n"
    target = _write_module(tmp_path, "helper.py", source)

    checker = SecurityScannerCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[target])
    violations = checker.check(context)

    assert violations
    assert any("eval" in v.message for v in violations)


def test_allows_safe_modules(tmp_path: Path):
    """Modules without risky patterns are ignored."""
    source = "def foo():\n    return 4\n"
    target = _write_module(tmp_path, "helper.py", source)

    checker = SecurityScannerCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[target])
    assert checker.check(context) == []


def test_ignores_tests(tmp_path: Path):
    """Test files are skipped even when they contain risky constructs."""
    target = tmp_path / "tests" / "dummy.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("exec('42')\n", encoding="utf-8")

    checker = SecurityScannerCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[target])
    assert checker.check(context) == []
