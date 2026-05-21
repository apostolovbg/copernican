"""Policy: detect raw error anti-patterns in Python source files."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Sequence

import devcovenant.core.repository_paths as yaml_cache_service
from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)
from devcovenant.core.selectors import SelectorSet

_VALID_SEVERITIES = {"critical", "error", "warning", "info"}
_DEFAULT_WAIVER_MARKERS = ("DEVCOV_ALLOW_BROAD_ONCE",)
_DEFAULT_WAIVER_BETWEEN = ("DEVCOV_BROAD_BEGIN=>DEVCOV_BROAD_END",)


def _coerce_bool(value: object, *, default: bool) -> bool:
    """Return a bool parsed from metadata/config values."""
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "1", "yes", "on", "enabled"}:
        return True
    if token in {"false", "0", "no", "off", "disabled"}:
        return False
    return default


def _option_tokens(raw_value: object) -> list[str]:
    """Normalize one metadata option into a list of non-empty tokens."""
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [
            token.strip()
            for token in raw_value.replace("\n", ",").split(",")
            if token.strip()
        ]
    if isinstance(raw_value, (list, tuple, set)):
        tokens: list[str] = []
        for entry in raw_value:
            token = str(entry or "").strip()
            if token:
                tokens.append(token)
        return tokens
    token = str(raw_value).strip()
    return [token] if token else []


def _between_pairs(raw_value: object) -> list[tuple[str, str]]:
    """Normalize `left=>right` waiver region tokens."""
    pairs: list[tuple[str, str]] = []
    for token in _option_tokens(raw_value):
        if "=>" not in token:
            continue
        left, right = token.split("=>", 1)
        left_token = left.strip()
        right_token = right.strip()
        if left_token and right_token:
            pairs.append((left_token, right_token))
    return pairs


def _contains_any_marker(
    line_content: str,
    markers: Sequence[str],
) -> bool:
    """Return True when any marker appears in one line."""
    return any(marker and marker in line_content for marker in markers)


def _waiver_regions(
    lines: Sequence[str],
    between_pairs: Sequence[tuple[str, str]],
) -> list[tuple[int, int]]:
    """Return inclusive line ranges where broad-handler waivers are active."""
    regions: list[tuple[int, int]] = []
    for left, right in between_pairs:
        start_line = 0
        for line_number, line_content in enumerate(lines, start=1):
            if not start_line and left in line_content:
                start_line = line_number
            if start_line and right in line_content:
                regions.append((start_line, line_number))
                start_line = 0
    return regions


def _line_in_regions(
    line_number: int,
    regions: Sequence[tuple[int, int]],
) -> bool:
    """Return True when line number falls within one configured region."""
    for start_line, end_line in regions:
        if start_line <= line_number <= end_line:
            return True
    return False


def _line_has_comment_waiver(
    lines: Sequence[str],
    line_number: int,
    markers: Sequence[str],
) -> bool:
    """Return True when waiver marker exists on same/prior comment line."""
    if line_number < 1 or line_number > len(lines):
        return False
    if _contains_any_marker(lines[line_number - 1], markers):
        return True
    previous_index = line_number - 2
    while previous_index >= 0 and not lines[previous_index].strip():
        previous_index -= 1
    if previous_index < 0:
        return False
    previous_line = lines[previous_index].lstrip()
    if not previous_line.startswith("#"):
        return False
    return _contains_any_marker(previous_line, markers)


def _is_exception_name(node: ast.AST | None) -> bool:
    """Return True when node represents Exception/BaseException."""
    if isinstance(node, ast.Name):
        return node.id in {"Exception", "BaseException"}
    return False


def _is_broad_exception_handler(node: ast.ExceptHandler) -> bool:
    """Return True when except handler targets Exception/BaseException."""
    if node.type is None:
        return False
    if _is_exception_name(node.type):
        return True
    if isinstance(node.type, ast.Tuple):
        return any(_is_exception_name(item) for item in node.type.elts)
    return False


def _is_generic_exception_raise(node: ast.Raise) -> bool:
    """Return True when raise targets Exception/BaseException constructors."""
    target = node.exc
    if target is None:
        return False
    if _is_exception_name(target):
        return True
    if isinstance(target, ast.Call):
        return _is_exception_name(target.func)
    return False


def _is_silent_pass_handler(node: ast.ExceptHandler) -> bool:
    """Return True for handlers that only contain `pass`."""
    return len(node.body) == 1 and isinstance(node.body[0], ast.Pass)


class _RawErrorVisitor(ast.NodeVisitor):
    """Visit only exception constructs relevant to this policy."""

    def __init__(
        self,
        *,
        path: Path,
        severity: str,
        policy_id: str,
        lines: Sequence[str],
        waiver_markers: Sequence[str],
        waiver_regions: Sequence[tuple[int, int]],
        forbid_bare_except: bool,
        forbid_raise_exception: bool,
        forbid_broad_exception_handlers: bool,
        forbid_silent_exception_pass: bool,
    ) -> None:
        """Store runtime options used while collecting violations."""
        self.path = path
        self.severity = severity
        self.policy_id = policy_id
        self.lines = lines
        self.waiver_markers = waiver_markers
        self.waiver_regions = waiver_regions
        self.forbid_bare_except = forbid_bare_except
        self.forbid_raise_exception = forbid_raise_exception
        self.forbid_broad_exception_handlers = forbid_broad_exception_handlers
        self.forbid_silent_exception_pass = forbid_silent_exception_pass
        self.violations: list[Violation] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Validate one except-handler node."""
        if self.forbid_bare_except and node.type is None:
            self.violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity=self.severity,
                    file_path=self.path,
                    line_number=node.lineno,
                    message=(
                        "Bare `except:` is not allowed; catch specific "
                        "exception types."
                    ),
                    suggestion=(
                        "Catch explicit exception classes and raise/report "
                        "explicit failures."
                    ),
                )
            )
        is_broad_handler = _is_broad_exception_handler(node)
        is_silent_handler = _is_silent_pass_handler(node)
        if (
            self.forbid_silent_exception_pass
            and is_broad_handler
            and is_silent_handler
        ):
            self.violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity=self.severity,
                    file_path=self.path,
                    line_number=node.lineno,
                    message="Silent `except Exception: pass` hides failures.",
                    suggestion=(
                        "Handle explicitly with error context or narrow the "
                        "exception to expected cases."
                    ),
                )
            )
        if (
            self.forbid_broad_exception_handlers
            and is_broad_handler
            and not is_silent_handler
        ):
            line_number = int(getattr(node, "lineno", 0) or 0)
            has_comment_waiver = _line_has_comment_waiver(
                self.lines,
                line_number,
                self.waiver_markers,
            )
            in_waiver_region = _line_in_regions(
                line_number,
                self.waiver_regions,
            )
            if not (has_comment_waiver or in_waiver_region):
                self.violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity=self.severity,
                        file_path=self.path,
                        line_number=line_number,
                        message=(
                            "Broad `except Exception` handlers are not "
                            "allowed."
                        ),
                        suggestion=(
                            "Catch specific exception classes; use a waiver "
                            "marker only at an explicit boundary with "
                            "rationale."
                        ),
                    )
                )
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        """Validate one raise node for generic exceptions."""
        if self.forbid_raise_exception and _is_generic_exception_raise(node):
            self.violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity=self.severity,
                    file_path=self.path,
                    line_number=node.lineno,
                    message=("Generic `raise Exception(...)` is not allowed."),
                    suggestion=(
                        "Raise a specific exception type with explicit "
                        "failure context."
                    ),
                )
            )
        self.generic_visit(node)


class NoRawErrorsCheck(PolicyCheck):
    """Block raw error anti-patterns that hide explicit failure intent."""

    policy_id = "no-raw-errors"
    version = "1.0.0"

    def _severity(self) -> str:
        """Return normalized policy severity."""
        raw = str(self.get_option("severity", "error")).strip().lower()
        if raw in _VALID_SEVERITIES:
            return raw
        return "error"

    def _selector(self) -> SelectorSet:
        """Return merged selector metadata for this policy."""
        defaults = {"include_suffixes": [".py"], "include_globs": ["*.py"]}
        return SelectorSet.from_policy(self, defaults=defaults)

    def check(self, context: CheckContext) -> List[Violation]:
        """Scan in-scope Python files for raw error anti-patterns."""
        files = context.all_files or context.changed_files or []
        selector = self._selector()
        severity = self._severity()
        forbid_bare_except = _coerce_bool(
            self.get_option("forbid_bare_except", True),
            default=True,
        )
        forbid_raise_exception = _coerce_bool(
            self.get_option("forbid_raise_exception", True),
            default=True,
        )
        forbid_broad_exception_handlers = _coerce_bool(
            self.get_option("forbid_broad_exception_handlers", True),
            default=True,
        )
        forbid_silent_exception_pass = _coerce_bool(
            self.get_option("forbid_silent_exception_pass", True),
            default=True,
        )
        waiver_markers = _option_tokens(
            self.get_option(
                "broad_exception_waiver_markers",
                list(_DEFAULT_WAIVER_MARKERS),
            )
        )
        if not waiver_markers:
            waiver_markers = list(_DEFAULT_WAIVER_MARKERS)
        waiver_between = _between_pairs(
            self.get_option(
                "broad_exception_waiver_between",
                list(_DEFAULT_WAIVER_BETWEEN),
            )
        )
        if not waiver_between:
            waiver_between = _between_pairs(_DEFAULT_WAIVER_BETWEEN)

        violations: List[Violation] = []
        for path in files:
            candidate = Path(path)
            if candidate.suffix.lower() != ".py":
                continue
            if not selector.matches(candidate, context.repo_root):
                continue
            try:
                source = yaml_cache_service.read_text(candidate)
            except OSError as error:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity=severity,
                        file_path=candidate,
                        message=(
                            "Unable to read Python source while validating "
                            f"raw errors: {error}"
                        ),
                        suggestion=(
                            "Restore readable UTF-8 source and rerun checks."
                        ),
                    )
                )
                continue
            tree = yaml_cache_service.parse_python_ast(candidate)
            if tree is None:
                continue
            lines = source.splitlines()
            waiver_regions = _waiver_regions(lines, waiver_between)
            visitor = _RawErrorVisitor(
                path=candidate,
                severity=severity,
                policy_id=self.policy_id,
                lines=lines,
                waiver_markers=waiver_markers,
                waiver_regions=waiver_regions,
                forbid_bare_except=forbid_bare_except,
                forbid_raise_exception=forbid_raise_exception,
                forbid_broad_exception_handlers=(
                    forbid_broad_exception_handlers
                ),
                forbid_silent_exception_pass=forbid_silent_exception_pass,
            )
            visitor.visit(tree)
            violations.extend(visitor.violations)

        return violations
