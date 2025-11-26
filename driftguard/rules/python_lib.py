"""Python library rules enforced by DriftGuard."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from driftguard.rules import Rule, RuleContext, Violation
from driftguard.utils import resolve_surface_globs

_BUGFIX_PATTERN: re.Pattern[str] = re.compile(
    r"\bbug\s*fix(?:es)?\b|\bbugfix(?:es)?\b|\bfix(?:es|ed)?\b",
    re.IGNORECASE,
)


def _git_status(repo_root: Path) -> List[Tuple[str, Path]]:
    """Return parsed ``git status`` entries for the repository.

    The helper limits parsing to the porcelain format so callers can reason
    about new, modified and deleted files without handling user-facing output
    variations. A best-effort approach keeps rules resilient when Git is
    unavailable (for example in archive exports) by returning an empty list.
    """

    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []

    entries: List[Tuple[str, Path]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        status, path_text = parts
        entries.append((status, repo_root / path_text))
    return entries


def _python_lib_paths(context: RuleContext) -> List[Path]:
    """Resolve the ``python-lib`` surface to concrete files on disk."""

    if "python-lib" not in context.spec.surfaces:
        return []
    return resolve_surface_globs(context.spec, context.repo_root, "python-lib")


def _changed_tests(status_entries: Iterable[Tuple[str, Path]]) -> List[Path]:
    """Return changed or added test files from Git status output."""

    changed: List[Path] = []
    for status, path in status_entries:
        if "tests" not in path.parts:
            continue
        if status.startswith("D"):
            continue
        changed.append(path)
    return changed


def _added_python_lib_modules(
    surface_paths: Sequence[Path],
    status_entries: Sequence[Tuple[str, Path]],
) -> List[Path]:
    """Identify new Python modules on the python-lib surface.

    New modules include freshly added or untracked files. The helper aligns
    detection with the spec-driven surface resolution so ignore globs continue
    to apply even when the working tree contains broader changes.
    """

    surface_set = {path.resolve() for path in surface_paths}
    new_modules: List[Path] = []
    for status, path in status_entries:
        if path.resolve() not in surface_set:
            continue
        if path.suffix.lower() != ".py":
            continue
        if status.startswith("A") or status.startswith("??"):
            new_modules.append(path)
    return new_modules


def _python_lib_changes(
    surface_paths: Sequence[Path],
    status_entries: Sequence[Tuple[str, Path]],
) -> List[Path]:
    """Collect modified python-lib files from Git status output."""

    surface_set = {path.resolve() for path in surface_paths}
    changed: List[Path] = []
    for status, path in status_entries:
        if path.resolve() not in surface_set:
            continue
        if status.startswith("D"):
            continue
        changed.append(path)
    return changed


class NoPrintInLibRule(Rule):
    """Detect disallowed ``print`` calls in library code."""

    name = "no-print"

    def check(self, context: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        targets = _python_lib_paths(context)
        for path in targets:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, FileNotFoundError, UnicodeDecodeError):
                # Skip unreadable files so the rule remains best-effort.
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(
                    node.func, ast.Name
                ):
                    if node.func.id == "print":
                        violations.append(
                            Violation(
                                rule_name=self.name,
                                message=(
                                    "Print statements are disallowed in "
                                    "python-lib code."
                                ),
                                path=path,
                                fixable=False,
                            )
                        )
                        break
        return violations


class NewModulesNeedTestsRule(Rule):
    """Require tests alongside new modules."""

    name = "new-modules-need-tests"

    def check(self, context: RuleContext) -> List[Violation]:
        surface_paths = _python_lib_paths(context)
        status_entries = _git_status(context.repo_root)
        new_modules = _added_python_lib_modules(surface_paths, status_entries)
        if not new_modules:
            return []

        tests_changed = _changed_tests(status_entries)
        if tests_changed:
            return []

        display = ", ".join(
            sorted(
                path.relative_to(context.repo_root).as_posix()
                for path in new_modules
            )
        )
        return [
            Violation(
                rule_name=self.name,
                message=(
                    "New python-lib modules require accompanying test "
                    f"updates: {display}"
                ),
                path=new_modules[0],
            )
        ]


class BugfixHasTestRule(Rule):
    """Recommend tests for bug fixes when feasible."""

    name = "bugfix-has-test"

    def check(self, context: RuleContext) -> List[Violation]:
        surface_paths = _python_lib_paths(context)
        status_entries = _git_status(context.repo_root)
        python_lib_changes = _python_lib_changes(surface_paths, status_entries)
        if not python_lib_changes:
            return []

        changelog = context.repo_root / "CHANGELOG.md"
        changelog_changed = any(
            path.resolve() == changelog.resolve() for _, path in status_entries
        )
        if not changelog_changed:
            return []

        try:
            diff = subprocess.run(
                ["git", "diff", "--unified=0", "--", "CHANGELOG.md"],
                cwd=context.repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return []
        if diff.returncode != 0:
            return []

        bugfix_added = False
        for line in diff.stdout.splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            if _BUGFIX_PATTERN.search(line):
                bugfix_added = True
                break
        if not bugfix_added:
            return []

        tests_changed = _changed_tests(status_entries)
        if tests_changed:
            return []

        changed_display = ", ".join(
            sorted(
                path.relative_to(context.repo_root).as_posix()
                for path in python_lib_changes
            )
        )
        return [
            Violation(
                rule_name=self.name,
                message=(
                    "Bugfix entries in CHANGELOG.md should be paired with "
                    "test updates when python-lib code changes."
                    f" Changed modules: {changed_display}"
                ),
                path=changelog,
            )
        ]
