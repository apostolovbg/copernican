"""Encourage documentation for modules, classes and functions."""

import ast
import io
import tokenize
from pathlib import PurePosixPath
from typing import Set

from devcovenant.base import CheckContext, PolicyCheck, Violation


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


def _should_inspect(
    rel_path: PurePosixPath,
    skip_prefixes: Set[str],
    skip_components: Set[str],
) -> bool:
    """Determine whether the file falls under the policy scope."""
    if not rel_path.parts:
        return False
    rel_posix = rel_path.as_posix()
    for prefix in skip_prefixes:
        normalized = prefix.strip("/ ")
        if not normalized:
            continue
        if rel_posix == normalized or rel_posix.startswith(f"{normalized}/"):
            return False
    for component in skip_components:
        if component and component in rel_path.parts:
            return False
    return True


class DocstringAndCommentCoverageCheck(PolicyCheck):
    """Treat missing docstrings/comments as policy violations."""

    policy_id = "docstring-and-comment-coverage"
    version = "1.0.0"
    DEFAULT_SUFFIXES = [".py"]
    DEFAULT_SKIP_PREFIXES = ["copernican_lib/vendor"]
    DEFAULT_SKIP_COMPONENTS = ["tests"]

    def check(self, context: CheckContext):
        """Detect functions, classes or modules without documentation."""
        files = context.all_files or context.changed_files or []
        violations = []

        suffixes_option = self.get_option(
            "include_suffixes", self.DEFAULT_SUFFIXES
        )
        if isinstance(suffixes_option, str):
            suffixes = [suffixes_option]
        else:
            suffixes = list(suffixes_option or [])
        suffixes = [
            (suffix if suffix.startswith(".") else f".{suffix}")
            .strip()
            .lower()
            for suffix in suffixes
            if isinstance(suffix, str) and suffix.strip()
        ]

        skip_prefixes_option = self.get_option(
            "skip_prefixes", self.DEFAULT_SKIP_PREFIXES
        )
        if isinstance(skip_prefixes_option, str):
            raw_prefixes = [skip_prefixes_option]
        else:
            raw_prefixes = list(skip_prefixes_option or [])
        skip_prefixes = {
            entry.strip("/ ") for entry in raw_prefixes if entry.strip()
        }

        skip_components_option = self.get_option(
            "skip_components", self.DEFAULT_SKIP_COMPONENTS
        )
        if isinstance(skip_components_option, str):
            raw_components = [skip_components_option]
        else:
            raw_components = list(skip_components_option or [])
        skip_components = {
            entry.strip() for entry in raw_components if entry.strip()
        }

        for path in files:
            if not path.is_file():
                continue
            if path.suffix.lower() not in suffixes:
                continue

            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue

            rel_posix = PurePosixPath(rel.as_posix())
            if not _should_inspect(rel_posix, skip_prefixes, skip_components):
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
