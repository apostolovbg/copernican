"""Tests for the name clarity policy."""

from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.name_clarity import NameClarityCheck


def _configured_policy() -> NameClarityCheck:
    policy = NameClarityCheck()
    policy.set_options(
        {
            "include_prefixes": ["project_lib"],
            "include_suffixes": [".py"],
            "exclude_prefixes": ["project_lib/vendor"],
        },
        {},
    )
    return policy


def _build_module(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "project_lib" / "helpers" / "naming.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_detects_placeholder_identifiers(tmp_path: Path):
    source = "def foo():\n    tmp = 1\n"
    target = _build_module(tmp_path, source)
    context = CheckContext(repo_root=tmp_path, changed_files=[target])

    violations = _configured_policy().check(context)
    assert len(violations) >= 2
    assert any("foo" in v.message for v in violations)
    assert all(v.severity == "warning" for v in violations)


def test_accepts_short_loop_counters(tmp_path: Path):
    source = "for i in range(3):\n    pass\n"
    target = _build_module(tmp_path, source)
    context = CheckContext(repo_root=tmp_path, changed_files=[target])

    assert _configured_policy().check(context) == []


def test_allows_explicit_override(tmp_path: Path):
    source = "foo = 1  # name-clarity: allow\n"
    target = _build_module(tmp_path, source)
    context = CheckContext(repo_root=tmp_path, changed_files=[target])

    assert _configured_policy().check(context) == []


def test_ignores_vendor_files(tmp_path: Path):
    path = (
        tmp_path
        / "project_lib"
        / "vendor"
        / "third_party"
        / "example"
        / "module.py"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("foo = 1\n", encoding="utf-8")
    context = CheckContext(repo_root=tmp_path, changed_files=[path])

    assert _configured_policy().check(context) == []
