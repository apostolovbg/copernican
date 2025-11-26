"""Workflow rules for DriftGuard."""

from __future__ import annotations

import subprocess
from pathlib import Path

from driftguard.rules import RuleContext
from driftguard.rules.workflows import FormatterCleanRule
from driftguard.spec import DriftConfig, DriftGuardSpec, SurfaceSpec


def _spec() -> DriftGuardSpec:
    return DriftGuardSpec(
        version=1,
        project="Tests",
        rulesets={},
        surfaces={
            "driftguard": SurfaceSpec(
                name="driftguard",
                include=["**/*.py"],
                exclude=[],
                rules=[FormatterCleanRule.name],
            )
        },
        drift=DriftConfig(),
    )


def _context(repo_root: Path) -> RuleContext:
    return RuleContext(
        repo_root=repo_root,
        spec=_spec(),
        scope="repo",
        mode="full",
    )


def test_formatter_clean_rule_flags_unformatted(tmp_path: Path) -> None:
    repo_root = tmp_path
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    source = repo_root / "example.py"
    source.write_text("def f():\n return{'a':1}\n", encoding="utf-8")

    rule = FormatterCleanRule()
    violations = rule.check(_context(repo_root))

    assert violations
    assert "Black" in violations[0].message


def test_formatter_clean_rule_accepts_formatted_code(tmp_path: Path) -> None:
    repo_root = tmp_path
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    source = repo_root / "example.py"
    source.write_text("def f():\n    return 1\n", encoding="utf-8")

    rule = FormatterCleanRule()
    violations = rule.check(_context(repo_root))

    assert violations == []


def test_formatter_clean_rule_checks_clean_repo_state(tmp_path: Path) -> None:
    repo_root = tmp_path
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    source = repo_root / "example.py"
    source.write_text("def f():\n return{'a':1}\n", encoding="utf-8")
    subprocess.run(["git", "add", "example.py"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add unformatted"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    rule = FormatterCleanRule()
    violations = rule.check(_context(repo_root))

    assert violations
    assert "Black" in violations[0].message
