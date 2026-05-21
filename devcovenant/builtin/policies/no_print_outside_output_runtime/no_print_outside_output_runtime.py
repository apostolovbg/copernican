"""Enforce metadata-driven direct-output sink boundaries."""

from __future__ import annotations

import ast
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, List

import devcovenant.core.repository_paths as yaml_cache_service
from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)
from devcovenant.core.selectors import SelectorSet

_VALID_SEVERITIES = {"critical", "error", "warning", "info"}

_JS_TS_DECLARATION_PATTERNS = (
    re.compile(r"^\s*function\s+([A-Za-z_$][\w$]*)\s*\("),
    re.compile(
        r"^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*="
        r"\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|"
        r"[A-Za-z_$][\w$]*\s*=>)"
    ),
)
_GO_DECLARATION_PATTERNS = (
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\("),
)
_RUST_DECLARATION_PATTERNS = (
    re.compile(
        r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
    ),
)


@dataclass(frozen=True)
class SinkHit:
    """One detected sink invocation."""

    line_number: int
    kind: str
    target: str
    symbol: str = ""


class _PythonSinkVisitor(ast.NodeVisitor):
    """Visit Python AST nodes and collect configured sink calls."""

    def __init__(
        self,
        *,
        call_targets: set[str],
        attr_targets: set[str],
    ) -> None:
        """Store sink targets and initialize result containers."""
        self.call_targets = call_targets
        self.attr_targets = attr_targets
        self.symbol_stack: list[str] = []
        self.hits: list[SinkHit] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track symbol scope for function definitions."""
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track symbol scope for async function definitions."""
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """Record call and attribute sinks configured for Python."""
        dotted = _dotted_name(node.func)
        leaf = _leaf_name(node.func)
        symbol = self.symbol_stack[-1] if self.symbol_stack else ""

        for target in sorted(self.call_targets):
            if _python_call_matches(target=target, dotted=dotted, leaf=leaf):
                self.hits.append(
                    SinkHit(
                        line_number=node.lineno,
                        kind="call",
                        target=target,
                        symbol=symbol,
                    )
                )

        for target in sorted(self.attr_targets):
            if dotted == target:
                self.hits.append(
                    SinkHit(
                        line_number=node.lineno,
                        kind="attr",
                        target=target,
                        symbol=symbol,
                    )
                )

        self.generic_visit(node)


def _normalize_list(raw: object | None) -> list[str]:
    """Return non-empty string entries from metadata/config values."""
    if raw is None:
        return []
    if isinstance(raw, str):
        candidates = [entry.strip() for entry in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        candidates = [str(entry).strip() for entry in raw]
    else:
        candidates = [str(raw).strip()]
    return [entry for entry in candidates if entry]


def _parse_lang_targets(
    entries: Iterable[str], *, metadata_key: str
) -> dict[str, set[str]]:
    """Parse `language=>target` entries into a mapping."""
    mapping: dict[str, set[str]] = {}
    for entry in entries:
        if "=>" not in entry:
            raise ValueError(
                f"{metadata_key} entries must use `language=>target` format."
            )
        language_raw, target_raw = entry.split("=>", 1)
        language = language_raw.strip().lower()
        target = target_raw.strip()
        if not language or not target:
            raise ValueError(
                f"{metadata_key} entries must include language and target."
            )
        mapping.setdefault(language, set()).add(target)
    return mapping


def _dotted_name(node: ast.AST) -> str:
    """Return dotted name for call targets when available."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return ""


def _leaf_name(node: ast.AST) -> str:
    """Return leaf callable name for a call target."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _python_call_matches(*, target: str, dotted: str, leaf: str) -> bool:
    """Return True when one Python call target matches."""
    if not target:
        return False
    if "." in target:
        return dotted == target
    return target in {dotted, leaf}


def _targets_for_language(
    mapping: dict[str, set[str]], language: str
) -> set[str]:
    """Return targets defined for one language plus wildcard defaults."""
    values = set(mapping.get(language, set()))
    values.update(mapping.get("*", set()))
    return values


def _matches_glob(path: Path, repo_root: Path, patterns: list[str]) -> bool:
    """Return True when path matches one configured glob."""
    if not patterns:
        return False
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        rel = path
    rel_posix = PurePosixPath(rel.as_posix()).as_posix()
    return any(fnmatch.fnmatch(rel_posix, pattern) for pattern in patterns)


def _line_has_waiver(
    lines: list[str], line_number: int, waiver_comment: str
) -> bool:
    """Return True when the sink line carries a waiver marker."""
    if not waiver_comment:
        return False
    if line_number <= 0 or line_number > len(lines):
        return False
    return waiver_comment in lines[line_number - 1]


def _scan_python(
    path: Path,
    source: str,
    *,
    call_targets: set[str],
    attr_targets: set[str],
) -> list[SinkHit]:
    """Detect sinks in Python source via AST traversal."""
    tree = yaml_cache_service.parse_python_ast(path)
    if tree is None:
        return []
    visitor = _PythonSinkVisitor(
        call_targets=call_targets,
        attr_targets=attr_targets,
    )
    visitor.visit(tree)
    return visitor.hits


def _declaration_patterns(language: str) -> tuple[re.Pattern[str], ...]:
    """Return declaration patterns used for symbol tracking."""
    if language in {"javascript", "typescript"}:
        return _JS_TS_DECLARATION_PATTERNS
    if language == "go":
        return _GO_DECLARATION_PATTERNS
    if language == "rust":
        return _RUST_DECLARATION_PATTERNS
    return tuple()


def _line_invokes_target(
    *,
    line: str,
    target: str,
    macro: bool,
) -> bool:
    """Return True when the line invokes one sink target."""
    escaped = re.escape(target)
    if macro:
        pattern = rf"(?<![A-Za-z0-9_$]){escaped}\s*!\s*\("
    else:
        pattern = rf"(?<![A-Za-z0-9_$]){escaped}\s*\("
    return bool(re.search(pattern, line))


def _scan_text(
    source: str,
    *,
    language: str,
    call_targets: set[str],
    attr_targets: set[str],
    macro_targets: set[str],
) -> list[SinkHit]:
    """Detect sinks in non-Python source via textual matching."""
    patterns = _declaration_patterns(language)
    hits: list[SinkHit] = []
    current_symbol = ""
    for index, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("//")
        ):
            continue

        for declaration in patterns:
            matched = declaration.search(line)
            if matched:
                current_symbol = matched.group(1)

        for target in sorted(call_targets):
            if _line_invokes_target(line=line, target=target, macro=False):
                hits.append(
                    SinkHit(
                        line_number=index,
                        kind="call",
                        target=target,
                        symbol=current_symbol,
                    )
                )
        for target in sorted(attr_targets):
            if _line_invokes_target(line=line, target=target, macro=False):
                hits.append(
                    SinkHit(
                        line_number=index,
                        kind="attr",
                        target=target,
                        symbol=current_symbol,
                    )
                )
        for target in sorted(macro_targets):
            if _line_invokes_target(line=line, target=target, macro=True):
                hits.append(
                    SinkHit(
                        line_number=index,
                        kind="macro",
                        target=target,
                        symbol=current_symbol,
                    )
                )
    return hits


def _dedupe_hits(hits: list[SinkHit]) -> list[SinkHit]:
    """Deduplicate sink hits while preserving deterministic order."""
    unique: list[SinkHit] = []
    seen: set[tuple[int, str, str, str]] = set()
    for hit in sorted(
        hits,
        key=lambda item: (
            item.line_number,
            item.kind,
            item.target,
            item.symbol,
        ),
    ):
        key = (hit.line_number, hit.kind, hit.target, hit.symbol)
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique


class NoPrintOutsideOutputRuntimeCheck(PolicyCheck):
    """Metadata-driven sink governance across configured languages."""

    policy_id = "no-print-outside-output-runtime"
    version = "2.0.0"

    def _severity(self) -> str:
        """Return normalized severity for generated violations."""
        token = str(self.get_option("severity", "error")).strip().lower()
        if token in _VALID_SEVERITIES:
            return token
        return "error"

    def _selector(self) -> SelectorSet:
        """Build the policy selector from merged metadata."""
        return SelectorSet.from_policy(self)

    @staticmethod
    def _selector_is_configured(selector: SelectorSet) -> bool:
        """Return True when include constraints were configured."""
        return any(
            (
                selector.include_suffixes,
                selector.include_prefixes,
                selector.include_globs,
                selector.force_include_globs,
            )
        )

    def _scan_language_sinks(
        self,
        *,
        path: Path,
        source: str,
        language: str,
        call_targets: set[str],
        attr_targets: set[str],
        macro_targets: set[str],
    ) -> list[SinkHit]:
        """Return sink hits for one language source unit."""
        if language == "python":
            return _dedupe_hits(
                _scan_python(
                    path,
                    source,
                    call_targets=call_targets,
                    attr_targets=attr_targets,
                )
            )
        return _dedupe_hits(
            _scan_text(
                source,
                language=language,
                call_targets=call_targets,
                attr_targets=attr_targets,
                macro_targets=macro_targets,
            )
        )

    def check(self, context: CheckContext) -> List[Violation]:
        """Flag direct-output sinks outside metadata-defined boundaries."""
        runtime = context.translator_runtime
        if runtime is None:
            return []

        selector = self._selector()
        if not self._selector_is_configured(selector):
            return []

        try:
            call_map = _parse_lang_targets(
                _normalize_list(self.get_option("sink_call_targets", [])),
                metadata_key="sink_call_targets",
            )
            attr_map = _parse_lang_targets(
                _normalize_list(self.get_option("sink_attr_targets", [])),
                metadata_key="sink_attr_targets",
            )
            macro_map = _parse_lang_targets(
                _normalize_list(self.get_option("sink_macro_targets", [])),
                metadata_key="sink_macro_targets",
            )
            allowed_symbol_map = _parse_lang_targets(
                _normalize_list(self.get_option("allowed_symbol_targets", [])),
                metadata_key="allowed_symbol_targets",
            )
        except ValueError as error:
            return [
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    message=f"Invalid metadata: {error}",
                )
            ]

        allowed_file_globs = _normalize_list(
            self.get_option("allowed_file_globs", [])
        )
        waiver_comment = str(
            self.get_option("allow_waiver_comment", "")
        ).strip()
        severity = self._severity()
        files = context.all_files or context.changed_files or []
        violations: list[Violation] = []

        for path in sorted(files):
            if not path.is_file() or not selector.matches(
                path,
                context.repo_root,
            ):
                continue

            resolution = runtime.resolve(
                path=path,
                policy_id=self.policy_id,
                context=context,
            )
            if not resolution.is_resolved:
                violations.extend(resolution.violations)
                continue

            declaration = resolution.declaration
            if declaration is None:
                continue
            language = str(declaration.translator_id or "").strip().lower()
            if not language:
                continue

            call_targets = _targets_for_language(call_map, language)
            attr_targets = _targets_for_language(attr_map, language)
            macro_targets = _targets_for_language(macro_map, language)
            if not any((call_targets, attr_targets, macro_targets)):
                continue

            source = yaml_cache_service.read_text(path)
            sink_hits = self._scan_language_sinks(
                path=path,
                source=source,
                language=language,
                call_targets=call_targets,
                attr_targets=attr_targets,
                macro_targets=macro_targets,
            )
            if not sink_hits:
                continue

            source_lines = source.splitlines()
            allow_entire_file = _matches_glob(
                path,
                context.repo_root,
                allowed_file_globs,
            )
            allowed_symbols = _targets_for_language(
                allowed_symbol_map, language
            )

            for hit in sink_hits:
                if allow_entire_file:
                    continue
                if _line_has_waiver(
                    source_lines, hit.line_number, waiver_comment
                ):
                    continue
                if hit.symbol and hit.symbol in allowed_symbols:
                    continue
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity=severity,
                        file_path=path,
                        line_number=hit.line_number,
                        message=(
                            f"Direct output sink `{hit.target}` is not "
                            f"allowed for language `{language}`."
                        ),
                        suggestion=(
                            "Route user-visible output through the configured "
                            "output runtime boundary."
                        ),
                    )
                )

        return violations
