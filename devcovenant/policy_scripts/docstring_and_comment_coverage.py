"""Enforce docstrings/comments across files selected via policy metadata."""

import ast
import io
import tokenize
from typing import Set

from devcovenant.base import CheckContext, PolicyCheck, Violation
from devcovenant.selectors import SelectorSet


def _collect_comment_lines(source: str) -> Set[int]:
    """Return the line numbers that contain standalone comments."""
    lines: Set[int] = set()
    reader = io.StringIO(source).readline
    try:
        for token in tokenize.generate_tokens(reader):
            if token.type == tokenize.COMMENT:
                lines.add(token.start[0])
    except tokenize.TokenError:
        pass
    return lines


def _has_comment_before(
    line: int, comment_lines: Set[int], lookback: int = 3
) -> bool:
    """Check whether a comment exists in the lines immediately preceding
    the given line."""
    for offset in range(lookback + 1):
        target = line - offset
        if target <= 0:
            continue
        if target in comment_lines:
            return True
    return False


class DocstringAndCommentCoverageCheck(PolicyCheck):
    """Treat missing docstrings/comments as policy violations."""

    policy_id = "docstring-and-comment-coverage"
    version = "1.0.0"
    DEFAULT_SUFFIXES = [".py"]

    def _build_selector(self) -> SelectorSet:
        """Return the unified selector for this policy."""
        defaults = {
            "include_suffixes": self.DEFAULT_SUFFIXES,
        }
        return SelectorSet.from_policy(self, defaults=defaults)

    def check(self, context: CheckContext):
        """Detect functions, classes or modules without documentation."""
        files = context.all_files or context.changed_files or []
        violations = []
        selector = self._build_selector()

        for path in files:
            if not path.is_file():
                continue
            if not selector.matches(path, context.repo_root):
                continue

            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue

            comment_lines = _collect_comment_lines(source)

            try:
                module_node = ast.parse(source)
            except SyntaxError:
                continue

            module_doc = ast.get_docstring(module_node)
            if not module_doc and not _has_comment_before(
                1, comment_lines, lookback=5
            ):
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=path,
                        message=(
                            "Module lacks a descriptive top-level docstring "
                            "or preceding comment."
                        ),
                    )
                )

            for node in ast.walk(module_node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbol = node.name
                    symbol_type = "function"
                elif isinstance(node, ast.ClassDef):
                    symbol = node.name
                    symbol_type = "class"
                else:
                    continue

                if ast.get_docstring(node):
                    continue

                if _has_comment_before(node.lineno, comment_lines):
                    continue

                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=path,
                        message=(
                            f"{symbol_type.title()} '{symbol}' is missing "
                            "a docstring or adjacent explanatory comment."
                        ),
                    )
                )

        return violations
