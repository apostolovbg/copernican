"""Shared assertion-signal helpers for tests-coverage policy."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Pattern

import devcovenant.core.repository_paths as yaml_cache_service

_DEFAULT_FIXTURE_MARKER = r"\bDEVCOV_FIXTURE_OK:\s*(?P<reason>\S.*)"
_DEFAULT_ASSERTION_PATTERN = r"\bassert\b"
_DEFAULT_TAUTOLOGY_PATTERNS = (
    r"^\s*assert\s*\(\s*true\s*\)\s*;?\s*$",
    r"^\s*assert\s+true\s*;?\s*$",
    r"^\s*assert!\s*\(\s*true\s*\)\s*;?\s*$",
    r"^\s*assert\w*\s*\(\s*([A-Za-z_][\w]*)\s*,\s*\1\s*\)\s*;?\s*$",
    r"^\s*assert\s+([A-Za-z_][\w]*)\s*==\s*\1\s*;?\s*$",
)
_COMMENT_PREFIXES = ("#", "//", "--", ";", "/*", "*", "<!--")


@dataclass(frozen=True)
class AssertionSignalAnalysis:
    """Result bundle for one test-file assertion-signal scan."""

    has_assertion_signal: bool
    covered_symbols: tuple[str, ...]


def _compile_patterns(patterns: Iterable[str]) -> tuple[Pattern[str], ...]:
    """Compile configured regex patterns, skipping invalid entries."""
    compiled: list[Pattern[str]] = []
    for raw in patterns:
        token = str(raw or "").strip()
        if not token:
            continue
        try:
            compiled.append(re.compile(token, re.IGNORECASE))
        except re.error:
            continue
    return tuple(compiled)


def _normalize_symbols(
    symbol_names: Iterable[str], *, min_length: int
) -> tuple[str, ...]:
    """Return deduplicated lowercase symbols for assertion matching."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in symbol_names:
        token = str(raw or "").strip().lower()
        if not token or len(token) < min_length:
            continue
        if token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return tuple(normalized)


def _compile_symbol_patterns(
    symbols: Iterable[str],
) -> dict[str, Pattern[str]]:
    """Compile per-symbol word-boundary matchers."""
    compiled: dict[str, Pattern[str]] = {}
    for symbol in symbols:
        token = str(symbol or "").strip().lower()
        if not token:
            continue
        escaped = re.escape(token)
        compiled[token] = re.compile(
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
        )
    return compiled


def _is_fixture_marker_comment(
    line: str, marker_pattern: Pattern[str]
) -> bool:
    """Return True when *line* declares a valid fixture-escape marker."""
    stripped = line.strip()
    if not stripped:
        return False
    if not any(stripped.startswith(prefix) for prefix in _COMMENT_PREFIXES):
        return False
    match = marker_pattern.search(stripped)
    if match is None:
        return False
    group = match.groupdict().get("reason")
    if group is None:
        return True
    return bool(group.strip())


def _fixture_escaped_assert_lines(
    lines: list[str], marker_pattern: Pattern[str]
) -> set[int]:
    """Return one-based assertion line numbers escaped by fixture marker."""
    escaped: set[int] = set()
    for index, line in enumerate(lines):
        if _is_fixture_marker_comment(line, marker_pattern):
            escaped.add(index + 2)
    return escaped


def _same_expression(left: ast.AST, right: ast.AST) -> bool:
    """Return True when two AST expressions are structurally equal."""
    return ast.dump(left, include_attributes=False) == ast.dump(
        right,
        include_attributes=False,
    )


def _is_python_tautology(node: ast.AST) -> bool:
    """Return True when a Python assertion node is tautological."""
    if isinstance(node, ast.Assert):
        test = node.test
        if isinstance(test, ast.Constant) and test.value is True:
            return True
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and _same_expression(test.left, test.comparators[0])
        ):
            return True
        return False

    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr.lower()
    elif isinstance(func, ast.Name):
        name = func.id.lower()
    else:
        return False

    if name == "asserttrue" and node.args:
        value = node.args[0]
        return isinstance(value, ast.Constant) and value.value is True

    tautology_names = {
        "assertequal",
        "assertequals",
        "assertsequenceequal",
        "assertlistequal",
        "asserttupleequal",
        "assertsetequal",
        "assertdictequal",
        "assertmultilineequal",
        "assertis",
    }
    if name in tautology_names and len(node.args) >= 2:
        return _same_expression(node.args[0], node.args[1])
    return False


def _line_matches_any(
    lines: list[str], line_number: int, patterns: tuple[Pattern[str], ...]
) -> bool:
    """Return True when the specific line matches any regex pattern."""
    if not patterns:
        return False
    if line_number <= 0 or line_number > len(lines):
        return False
    line = lines[line_number - 1]
    return any(pattern.search(line) for pattern in patterns)


def _collect_symbol_hits(
    lines: list[str],
    *,
    line_number: int,
    symbol_patterns: dict[str, Pattern[str]],
    symbol_window: int,
) -> set[str]:
    """Collect covered symbols around one assertion line."""
    if not symbol_patterns:
        return set()
    start = max(1, line_number - symbol_window)
    end = min(len(lines), line_number + symbol_window)
    window = "\n".join(lines[start - 1 : end]).lower()
    covered: set[str] = set()
    for symbol, pattern in symbol_patterns.items():
        if pattern.search(window):
            covered.add(symbol)
    return covered


def _analyze_python(
    text: str,
    *,
    tree: ast.AST,
    marker_pattern: Pattern[str],
    tautology_patterns: tuple[Pattern[str], ...],
    symbol_patterns: dict[str, Pattern[str]],
    symbol_window: int,
) -> AssertionSignalAnalysis:
    """Analyze assertion signal from Python test source."""
    lines = text.splitlines()
    escaped_lines = _fixture_escaped_assert_lines(lines, marker_pattern)
    covered_symbols: set[str] = set()

    class _AssertionVisitor(ast.NodeVisitor):
        """Collect assertion signal without a full generic ast.walk loop."""

        def __init__(self) -> None:
            """Initialize the visitor-level assertion flag."""
            self.has_assertion = False

        def visit_Assert(self, node: ast.Assert) -> None:
            """Record direct `assert` statements before visiting children."""
            self._record_if_signal(node)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            """Record assertion-style helper calls before visiting children."""
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr.lower()
            elif isinstance(func, ast.Name):
                name = func.id.lower()
            else:
                self.generic_visit(node)
                return
            if name.startswith("assert"):
                self._record_if_signal(node)
            self.generic_visit(node)

        def _record_if_signal(self, node: ast.AST) -> None:
            """Persist one assertion hit when it is not tautological noise."""
            line_no = getattr(node, "lineno", 0)
            tautology = _is_python_tautology(node) or _line_matches_any(
                lines,
                line_no,
                tautology_patterns,
            )
            if tautology and line_no not in escaped_lines:
                return
            self.has_assertion = True
            covered_symbols.update(
                _collect_symbol_hits(
                    lines,
                    line_number=line_no,
                    symbol_patterns=symbol_patterns,
                    symbol_window=symbol_window,
                )
            )

    visitor = _AssertionVisitor()
    visitor.visit(tree)

    return AssertionSignalAnalysis(
        has_assertion_signal=visitor.has_assertion,
        covered_symbols=tuple(sorted(covered_symbols)),
    )


def _analyze_generic(
    text: str,
    *,
    assertion_patterns: tuple[Pattern[str], ...],
    tautology_patterns: tuple[Pattern[str], ...],
    marker_pattern: Pattern[str],
    symbol_patterns: dict[str, Pattern[str]],
    symbol_window: int,
) -> AssertionSignalAnalysis:
    """Analyze assertion signal for non-Python test source."""
    lines = text.splitlines()
    escaped_lines = _fixture_escaped_assert_lines(lines, marker_pattern)
    covered_symbols: set[str] = set()
    has_assertion = False

    for index, raw_line in enumerate(lines, start=1):
        if not any(pattern.search(raw_line) for pattern in assertion_patterns):
            continue
        tautology = any(
            pattern.search(raw_line) for pattern in tautology_patterns
        )
        if tautology and index not in escaped_lines:
            continue
        has_assertion = True
        covered_symbols.update(
            _collect_symbol_hits(
                lines,
                line_number=index,
                symbol_patterns=symbol_patterns,
                symbol_window=symbol_window,
            )
        )

    return AssertionSignalAnalysis(
        has_assertion_signal=has_assertion,
        covered_symbols=tuple(sorted(covered_symbols)),
    )


def analyze_assertion_signal(
    path: Path,
    *,
    language: str = "",
    assertion_patterns: Iterable[str] = tuple(),
    tautology_patterns: Iterable[str] = tuple(),
    fixture_marker_pattern: str = _DEFAULT_FIXTURE_MARKER,
    symbol_names: Iterable[str] = tuple(),
    symbol_window: int = 2,
    symbol_name_min_length: int = 1,
) -> AssertionSignalAnalysis:
    """Return assertion signal and covered symbols for one test file."""
    try:
        text = yaml_cache_service.read_text(path)
    except (OSError, UnicodeDecodeError):
        return AssertionSignalAnalysis(False, tuple())

    marker = fixture_marker_pattern.strip() or _DEFAULT_FIXTURE_MARKER
    try:
        marker_pattern = re.compile(marker, re.IGNORECASE)
    except re.error:
        marker_pattern = re.compile(_DEFAULT_FIXTURE_MARKER, re.IGNORECASE)

    compiled_assertions = _compile_patterns(assertion_patterns)
    if not compiled_assertions:
        compiled_assertions = _compile_patterns((_DEFAULT_ASSERTION_PATTERN,))

    compiled_tautologies = _compile_patterns(tautology_patterns)
    if not compiled_tautologies:
        compiled_tautologies = _compile_patterns(_DEFAULT_TAUTOLOGY_PATTERNS)

    symbols = _normalize_symbols(
        symbol_names,
        min_length=max(symbol_name_min_length, 1),
    )
    symbol_patterns = _compile_symbol_patterns(symbols)
    window = max(symbol_window, 0)
    normalized_language = str(language or "").strip().lower()
    if normalized_language == "python" or path.suffix.lower() == ".py":
        tree = yaml_cache_service.parse_python_ast(path)
        if tree is None:
            return AssertionSignalAnalysis(False, tuple())
        return _analyze_python(
            text,
            tree=tree,
            marker_pattern=marker_pattern,
            tautology_patterns=compiled_tautologies,
            symbol_patterns=symbol_patterns,
            symbol_window=window,
        )

    return _analyze_generic(
        text,
        assertion_patterns=compiled_assertions,
        tautology_patterns=compiled_tautologies,
        marker_pattern=marker_pattern,
        symbol_patterns=symbol_patterns,
        symbol_window=window,
    )


def has_assertion_signal(path: Path, **kwargs: object) -> bool:
    """Return True when one test file carries usable assertion signal."""
    return analyze_assertion_signal(path, **kwargs).has_assertion_signal
