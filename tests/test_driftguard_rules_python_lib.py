"""Regression coverage for Python library DriftGuard rules."""

import subprocess
from pathlib import Path

from driftguard.rules import RuleContext
from driftguard.rules.python_lib import (
    BugfixHasTestRule,
    NewModulesNeedTestsRule,
    NoPrintInLibRule,
)
from driftguard.spec import DriftConfig, DriftGuardSpec, SurfaceSpec


def _spec() -> DriftGuardSpec:
    return DriftGuardSpec(
        version=1,
        project="Tests",
        rulesets={},
        surfaces={
            "python-lib": SurfaceSpec(
                name="python-lib",
                include=[
                    "copernican_lib/**/*.py",
                    "engines/**/*.py",
                ],
                exclude=[
                    "tests/**/*.py",
                    "tools/**/*.py",
                    "driftguard/**/*.py",
                ],
                rules=[
                    "no-print",
                    "new-modules-need-tests",
                    "bugfix-has-test",
                ],
            ),
            "python-tests": SurfaceSpec(
                name="python-tests",
                include=["tests/**/*.py"],
                exclude=[],
                rules=[],
            ),
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


def _init_git_repo(repo_root: Path) -> None:
    subprocess.run(
        ["git", "init"], cwd=repo_root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "ci@example.com"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "CI"], cwd=repo_root, check=True
    )


def test_no_print_in_lib_flags_print_calls(tmp_path: Path) -> None:
    """Print calls inside python-lib surfaces should be reported."""

    lib_path = tmp_path / "copernican_lib"
    lib_path.mkdir()
    module = lib_path / "sample.py"
    module.write_text("def greet():\n    print('hi')\n", encoding="utf-8")

    rule = NoPrintInLibRule()
    context = _context(tmp_path)

    violations = rule.check(context)

    assert violations
    assert violations[0].path == module


def test_new_module_without_tests_is_hard_violation(tmp_path: Path) -> None:
    """New modules should trigger a hard failure when tests stay untouched."""

    _init_git_repo(tmp_path)
    lib_path = tmp_path / "copernican_lib"
    lib_path.mkdir()
    new_module = lib_path / "new_feature.py"
    new_module.write_text("value = 1\n", encoding="utf-8")

    rule = NewModulesNeedTestsRule()
    context = _context(tmp_path)

    violations = rule.check(context)

    assert violations
    assert violations[0].rule_name == rule.name
    assert violations[0].path == new_module


def test_new_module_with_tests_passes(tmp_path: Path) -> None:
    """Adding tests alongside a new module should satisfy the rule."""

    _init_git_repo(tmp_path)
    lib_path = tmp_path / "copernican_lib"
    tests_path = tmp_path / "tests"
    lib_path.mkdir()
    tests_path.mkdir()
    new_module = lib_path / "new_feature.py"
    new_module.write_text("value = 1\n", encoding="utf-8")
    test_file = tests_path / "test_new_feature.py"
    test_file.write_text(
        "def test_placeholder():\n    assert True\n", encoding="utf-8"
    )

    rule = NewModulesNeedTestsRule()
    context = _context(tmp_path)

    assert rule.check(context) == []


def test_bugfix_entry_without_tests_emits_warning(tmp_path: Path) -> None:
    """Bugfix changelog lines should be paired with test updates."""

    _init_git_repo(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    lib_path = tmp_path / "copernican_lib"
    tests_path = tmp_path / "tests"
    lib_path.mkdir()
    tests_path.mkdir()
    module = lib_path / "core.py"
    changelog.write_text("# Changelog\n\n## Version 0.1.0\n- seed entry\n")
    module.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "CHANGELOG.md", "copernican_lib/core.py"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    module.write_text("value = 2\n", encoding="utf-8")
    changelog.write_text(
        "\n".join(
            [
                "# Changelog",
                "",
                "## Version 0.1.0",
                "- bugfix: adjusted core logic",
            ]
        ),
        encoding="utf-8",
    )

    rule = BugfixHasTestRule()
    context = _context(tmp_path)

    violations = rule.check(context)

    assert violations
    assert violations[0].path == changelog


def test_bugfix_entry_with_tests_is_satisfied(tmp_path: Path) -> None:
    """Updating tests should satisfy bugfix expectations."""

    _init_git_repo(tmp_path)
    changelog = tmp_path / "CHANGELOG.md"
    lib_path = tmp_path / "copernican_lib"
    tests_path = tmp_path / "tests"
    lib_path.mkdir()
    tests_path.mkdir()
    module = lib_path / "core.py"
    test_file = tests_path / "test_core.py"
    changelog.write_text("# Changelog\n\n## Version 0.1.0\n- seed entry\n")
    module.write_text("value = 1\n", encoding="utf-8")
    test_file.write_text(
        "def test_seed():\n    assert True\n", encoding="utf-8"
    )
    subprocess.run(
        [
            "git",
            "add",
            "CHANGELOG.md",
            "copernican_lib/core.py",
            "tests/test_core.py",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    module.write_text("value = 2\n", encoding="utf-8")
    test_file.write_text(
        "def test_seed():\n    assert 2 == 2\n", encoding="utf-8"
    )
    changelog.write_text(
        "\n".join(
            [
                "# Changelog",
                "",
                "## Version 0.1.0",
                "- bugfix: adjusted core logic",
            ]
        ),
        encoding="utf-8",
    )

    rule = BugfixHasTestRule()
    context = _context(tmp_path)

    assert rule.check(context) == []
