"""Python library rules enforced by DriftGuard."""

from __future__ import annotations

import ast
import re
import subprocess
import tokenize
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


def _changed_python_lib_paths(context: RuleContext) -> List[Path]:
    """Return python-lib files that are currently changed or added."""

    surface_paths = _python_lib_paths(context)
    if not surface_paths:
        return []
    surface_set = {path.resolve() for path in surface_paths}
    status_entries = _git_status(context.repo_root)
    if not status_entries:
        return surface_paths

    changed: List[Path] = []
    for status, path in status_entries:
        if path.resolve() not in surface_set:
            continue
        if status.startswith("D"):
            continue
        changed.append(path)
    return changed if changed else surface_paths


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


class CommentsExplainWhyRule(Rule):
    """Demand explanatory comments in python-lib changes."""

    name = "comments-explain-why"

    def check(self, context: RuleContext) -> List[Violation]:
        targets = _changed_python_lib_paths(context)
        violations: List[Violation] = []
        for path in targets:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, FileNotFoundError):
                continue
            has_comment = any(line.strip().startswith("#") for line in lines)
            has_why = any(
                line.strip().startswith("#")
                and ("why" in line.lower() or "because" in line.lower())
                for line in lines
            )
            if not has_comment or not has_why:
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            "Python modules must include explanatory comments "
                            "covering the rationale (why/because)."
                        ),
                        path=path,
                    )
                )
        return violations


class DocstringsExplainWhyRule(Rule):
    """Require module, class and function docstrings with rationale."""

    name = "docstrings-explain-why"

    def check(self, context: RuleContext) -> List[Violation]:
        targets = _changed_python_lib_paths(context)
        violations: List[Violation] = []
        for path in targets:
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (UnicodeDecodeError, FileNotFoundError, SyntaxError):
                continue

            module_doc = ast.get_docstring(tree)
            if not module_doc or (
                "why" not in module_doc.lower()
                and "because" not in module_doc.lower()
            ):
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            "Module docstring must describe what and why."
                        ),
                        path=path,
                    )
                )

            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    doc = ast.get_docstring(node)
                    if not doc or (
                        "why" not in doc.lower() and "because" not in doc.lower()
                    ):
                        violations.append(
                            Violation(
                                rule_name=self.name,
                                message=(
                                    "Functions and classes need docstrings "
                                    "explaining intent (why/because)."
                                ),
                                path=path,
                            )
                        )
                        break
        return violations


class NamingClearAndConciseRule(Rule):
    """Discourage ambiguous single-letter identifiers for public APIs."""

    name = "naming-clear-and-concise"

    def check(self, context: RuleContext) -> List[Violation]:
        targets = _changed_python_lib_paths(context)
        violations: List[Violation] = []
        for path in targets:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, FileNotFoundError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    name = node.name
                    if len(name) < 3 and not name.startswith("__"):
                        violations.append(
                            Violation(
                                rule_name=self.name,
                                message=(
                                    f"Identifier {name!r} should be more "
                                    "descriptive than a single letter."
                                ),
                                path=path,
                            )
                        )
                        break
        return violations


class LineLengthRule(Rule):
    """Keep Python source lines within 79 characters."""

    name = "line-length-79"

    def check(self, context: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        for path in _changed_python_lib_paths(context):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, FileNotFoundError):
                continue
            for number, line in enumerate(lines, start=1):
                if len(line) > 79:
                    violations.append(
                        Violation(
                            rule_name=self.name,
                            message=(
                                f"Line {number} exceeds 79 characters; "
                                "condense or wrap."
                            ),
                            path=path,
                        )
                    )
                    break
        return violations


class RawStringEscapingRule(Rule):
    """Encourage raw strings when backslashes appear in literals."""

    name = "raw-string-escaping"

    def check(self, context: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        for path in _changed_python_lib_paths(context):
            try:
                source = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, FileNotFoundError):
                continue
            try:
                tokens = tokenize.generate_tokens(
                    iter(source.splitlines(True)).__next__
                )
            except tokenize.TokenError:
                continue

            for token in tokens:
                if token.type != tokenize.STRING:
                    continue
                text = token.string
                quote_index = None
                for probe in ("'''", '"""', "'", '"'):
                    idx = text.find(probe)
                    if idx != -1:
                        quote_index = idx
                        break
                if quote_index is None:
                    continue
                prefix = text[:quote_index].lower()
                has_raw_prefix = "r" in prefix
                if "\\" in text and not has_raw_prefix:
                    violations.append(
                        Violation(
                            rule_name=self.name,
                            message=(
                                "Use raw strings or explicit escaping for "
                                "backslashes in string literals."
                            ),
                            path=path,
                        )
                    )
                    break
        return violations


class TestsForChangesRule(Rule):
    """Require tests alongside python-lib modifications."""

    name = "tests-for-changes"

    def check(self, context: RuleContext) -> List[Violation]:
        status_entries = _git_status(context.repo_root)
        if not status_entries:
            return []

        surface_paths = _python_lib_paths(context)
        surface_set = {path.resolve() for path in surface_paths}
        lib_changes = [
            path for _, path in status_entries if path.resolve() in surface_set
        ]
        if not lib_changes:
            return []

        tests_changed = _changed_tests(status_entries)
        if tests_changed:
            return []

        display = ", ".join(
            sorted(path.relative_to(context.repo_root).as_posix() for path in lib_changes)
        )
        return [
            Violation(
                rule_name=self.name,
                message=(
                    "Update or add tests when modifying python-lib code: "
                    f"{display}"
                ),
                path=lib_changes[0],
            )
        ]
