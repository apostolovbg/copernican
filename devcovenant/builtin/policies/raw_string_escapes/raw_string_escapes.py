"""Warn when string literals contain suspicious backslash escapes.

Python literals are scanned through `tokenize` for precise spans. Other
languages are scanned through metadata-driven literal patterns.
"""

from __future__ import annotations

import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Pattern

from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)
from devcovenant.core.selectors import SelectorSet

_VALID_SEVERITIES = {"critical", "error", "warning", "info"}
_PYTHON_PREFIX_RE = re.compile(r"(?P<prefix>[rubfRUBF]*)(?P<quote>['\"]{1,3})")
_SUSPICIOUS_ESCAPE_RE = re.compile(r"\\(?![\\'\"abfnrtv0-7xuUN])")
_DEFAULT_INCLUDE_SUFFIXES = [
    ".py",
    ".pyi",
    ".pyw",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".cs",
    ".kt",
    ".swift",
    ".php",
    ".rb",
]
_DEFAULT_LANGUAGE_SUFFIXES = (
    "python=>.py",
    "python=>.pyi",
    "python=>.pyw",
    "javascript=>.js",
    "javascript=>.jsx",
    "typescript=>.ts",
    "typescript=>.tsx",
    "go=>.go",
    "rust=>.rs",
    "java=>.java",
    "csharp=>.cs",
    "kotlin=>.kt",
    "swift=>.swift",
    "php=>.php",
    "ruby=>.rb",
)
_DEFAULT_LITERAL_PATTERNS = (
    r'*=>"(?:\\.|[^"\\])*"',
    r"*=>'(?:\\.|[^'\\])*'",
    r"*=>`(?:\\.|[^`\\])*`",
)
_DEFAULT_RAW_LITERAL_PATTERNS = (
    r'python=>^[rubfRUBF]*r["\']{1,3}',
    r'csharp=>^@"',
    r'rust=>^r#*"',
)
_DEFAULT_SUSPICIOUS_ESCAPE_PATTERNS = (r"*=>\\(?![\\'\"abfnrtv0-7xuUN])",)
_PYTHON_SUFFIXES = {".py", ".pyi", ".pyw"}
_RAW_PREFIX_CHARS = frozenset("rRuUbBfF@#")


@dataclass(frozen=True)
class _LiteralHit:
    """One suspicious literal hit."""

    line_number: int
    start_offset: int
    end_offset: int
    text: str
    can_auto_fix: bool = False
    context: dict[str, object] = field(default_factory=dict)


def _normalize_tokens(raw: object) -> list[str]:
    """Normalize list-like metadata values into non-empty tokens."""
    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = [entry.strip() for entry in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        tokens = [str(entry).strip() for entry in raw]
    else:
        tokens = [str(raw).strip()]
    return [token for token in tokens if token]


def _parse_lang_values(raw: object) -> dict[str, list[str]]:
    """Parse `language=>value` metadata tokens."""
    mapping: dict[str, list[str]] = {}
    for token in _normalize_tokens(raw):
        if "=>" not in token:
            mapping.setdefault("*", []).append(token)
            continue
        language_raw, value_raw = token.split("=>", 1)
        language = language_raw.strip().lower() or "*"
        value = value_raw.strip()
        if not value:
            continue
        mapping.setdefault(language, []).append(value)
    return mapping


def _parse_language_suffixes(raw: object) -> dict[str, set[str]]:
    """Parse `language=>.suffix` metadata into suffix lookup map."""
    mapping: dict[str, set[str]] = {}
    for token in _normalize_tokens(raw):
        if "=>" not in token:
            continue
        language_raw, suffix_raw = token.split("=>", 1)
        language = language_raw.strip().lower()
        suffix = suffix_raw.strip().lower()
        if not language or not suffix:
            continue
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        mapping.setdefault(language, set()).add(suffix)
    return mapping


def _values_for_language(
    mapping: dict[str, tuple[Pattern[str], ...]], language: str
) -> tuple[Pattern[str], ...]:
    """Return compiled patterns for one language plus wildcard defaults."""
    language_token = str(language or "").strip().lower()
    values: list[Pattern[str]] = []
    values.extend(mapping.get("*", tuple()))
    values.extend(mapping.get(language_token, tuple()))
    return tuple(values)


def _compile_patterns(patterns: Iterable[str]) -> tuple[Pattern[str], ...]:
    """Compile regex patterns, skipping invalid entries."""
    compiled: list[Pattern[str]] = []
    for raw_pattern in patterns:
        pattern = str(raw_pattern or "").strip()
        if not pattern:
            continue
        try:
            compiled.append(re.compile(pattern))
        except re.error:
            continue
    return tuple(compiled)


def _compile_pattern_map(raw: object) -> dict[str, tuple[Pattern[str], ...]]:
    """Compile a language-keyed regex map from metadata tokens."""
    compiled: dict[str, tuple[Pattern[str], ...]] = {}
    for language, patterns in _parse_lang_values(raw).items():
        entries = _compile_patterns(patterns)
        if entries:
            compiled[language] = entries
    return compiled


def _contains_match(text: str, patterns: tuple[Pattern[str], ...]) -> bool:
    """Return True when any compiled regex matches text."""
    return any(pattern.search(text) for pattern in patterns)


def _line_number_from_offset(source: str, offset: int) -> int:
    """Convert one zero-based offset into one-based line number."""
    return source.count("\n", 0, max(offset, 0)) + 1


def _compiled_pattern_key(
    patterns: tuple[Pattern[str], ...],
) -> tuple[str, ...]:
    """Return one stable cache key for compiled regex patterns."""
    return tuple(pattern.pattern for pattern in patterns)


def _path_cache_key(path: Path, *, repo_root: Path) -> tuple[object, ...]:
    """Return one run-scoped cache key for a file path."""
    try:
        relative = path.relative_to(repo_root).as_posix()
    except ValueError:
        relative = path.as_posix()
    try:
        stat = path.stat()
    except OSError:
        return (relative, None, None)
    return (relative, stat.st_mtime_ns, stat.st_size)


def _is_python_raw_literal(
    token_value: str, raw_patterns: tuple[Pattern[str], ...]
) -> bool:
    """Return True when one Python literal uses raw syntax."""
    if raw_patterns and _contains_match(token_value, raw_patterns):
        return True
    match = _PYTHON_PREFIX_RE.match(token_value)
    if match is None:
        return False
    return "r" in match.group("prefix").lower()


def _scan_python_literals(
    path: Path,
    *,
    suspicious_patterns: tuple[Pattern[str], ...],
    raw_patterns: tuple[Pattern[str], ...],
) -> list[_LiteralHit]:
    """Scan Python string tokens and return suspicious literal hits."""
    hits: list[_LiteralHit] = []
    try:
        with path.open(encoding="utf-8") as handle:
            tokens = tokenize.generate_tokens(handle.readline)
            for token in tokens:
                if token.type != tokenize.STRING:
                    continue
                token_text = token.string
                if _is_python_raw_literal(token_text, raw_patterns):
                    continue
                if not _contains_match(token_text, suspicious_patterns):
                    continue
                hits.append(
                    _LiteralHit(
                        line_number=token.start[0],
                        start_offset=0,
                        end_offset=0,
                        text=token_text,
                        can_auto_fix=True,
                        context={
                            "start": token.start,
                            "end": token.end,
                        },
                    )
                )
    except (OSError, tokenize.TokenError):
        return []
    return hits


def _scan_generic_literals(
    source: str,
    *,
    literal_patterns: tuple[Pattern[str], ...],
    raw_patterns: tuple[Pattern[str], ...],
    suspicious_patterns: tuple[Pattern[str], ...],
) -> list[_LiteralHit]:
    """Scan non-Python source through configured literal patterns."""
    if not literal_patterns or not suspicious_patterns:
        return []

    hits: list[_LiteralHit] = []
    for literal_pattern in literal_patterns:
        for match in literal_pattern.finditer(source):
            literal_text = match.group(0)
            literal_with_prefix = literal_text
            prefix_start = match.start()
            while prefix_start > 0:
                token = source[prefix_start - 1]
                if token not in _RAW_PREFIX_CHARS:
                    break
                prefix_start -= 1
            if prefix_start != match.start():
                literal_with_prefix = source[prefix_start : match.end()]
            if raw_patterns and _contains_match(
                literal_with_prefix, raw_patterns
            ):
                continue
            if not _contains_match(literal_text, suspicious_patterns):
                continue
            start_offset = match.start()
            hits.append(
                _LiteralHit(
                    line_number=_line_number_from_offset(source, start_offset),
                    start_offset=start_offset,
                    end_offset=match.end(),
                    text=literal_text,
                )
            )
    return hits


def _dedupe_hits(hits: list[_LiteralHit]) -> list[_LiteralHit]:
    """Deduplicate literal hits with deterministic ordering."""
    unique: list[_LiteralHit] = []
    seen: set[tuple[int, int, int, bool]] = set()
    for hit in sorted(
        hits,
        key=lambda entry: (
            entry.line_number,
            entry.start_offset,
            entry.end_offset,
            entry.can_auto_fix,
        ),
    ):
        key = (
            hit.line_number,
            hit.start_offset,
            hit.end_offset,
            hit.can_auto_fix,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique


def _resolve_language_for_path(
    *,
    path: Path,
    context: CheckContext,
    policy_id: str,
    suffix_map: dict[str, set[str]],
) -> str:
    """Resolve language from translator runtime or suffix metadata."""
    runtime = context.translator_runtime
    resolver = getattr(runtime, "resolve", None)
    if resolver is not None:
        try:
            resolution = resolver(
                path=path,
                policy_id=policy_id,
                context=context,
            )
        # DEVCOV_ALLOW_BROAD_ONCE translator resolution boundary.
        except Exception:
            resolution = None
        if resolution is not None and getattr(
            resolution, "is_resolved", False
        ):
            declaration = getattr(resolution, "declaration", None)
            if declaration is not None:
                language = (
                    str(getattr(declaration, "translator_id", ""))
                    .strip()
                    .lower()
                )
                if language:
                    return language

    suffix = path.suffix.lower()
    for language, suffixes in suffix_map.items():
        if suffix in suffixes:
            return language
    return ""


class RawStringEscapesCheck(PolicyCheck):
    """Warn when literals contain suspicious backslash escapes."""

    policy_id = "raw-string-escapes"
    version = "2.0.0"

    def _severity(self) -> str:
        """Return configured severity with deterministic fallback."""
        raw = str(self.get_option("severity", "warning")).strip().lower()
        if raw in _VALID_SEVERITIES:
            return raw
        return "warning"

    def _selector(self) -> SelectorSet:
        """Return selector describing files in scope."""
        defaults = {"include_suffixes": _DEFAULT_INCLUDE_SUFFIXES}
        return SelectorSet.from_policy(self, defaults=defaults)

    def check(self, context: CheckContext) -> List[Violation]:
        """Inspect literals for unknown backslash escapes."""
        files = context.all_files or context.changed_files or []
        violations: List[Violation] = []
        selector = self._selector()
        severity = self._severity()
        suffix_map = _parse_language_suffixes(
            self.get_option(
                "language_suffixes", list(_DEFAULT_LANGUAGE_SUFFIXES)
            )
        )
        literal_map = _compile_pattern_map(
            self.get_option(
                "literal_patterns",
                list(_DEFAULT_LITERAL_PATTERNS),
            )
        )
        raw_literal_map = _compile_pattern_map(
            self.get_option(
                "raw_literal_patterns",
                list(_DEFAULT_RAW_LITERAL_PATTERNS),
            )
        )
        suspicious_map = _compile_pattern_map(
            self.get_option(
                "suspicious_escape_patterns",
                list(_DEFAULT_SUSPICIOUS_ESCAPE_PATTERNS),
            )
        )

        for path in files:
            if not path.is_file():
                continue
            if not selector.matches(path, context.repo_root):
                continue

            language = _resolve_language_for_path(
                path=path,
                context=context,
                policy_id=self.policy_id,
                suffix_map=suffix_map,
            )
            suspicious_patterns = _values_for_language(
                suspicious_map, language
            )
            if not suspicious_patterns:
                suspicious_patterns = (_SUSPICIOUS_ESCAPE_RE,)
            raw_patterns = _values_for_language(raw_literal_map, language)
            cache_bucket = context.runtime_cache_bucket("raw_string_escapes")
            literal_patterns = _values_for_language(literal_map, language)
            cache_key = (
                _path_cache_key(path, repo_root=context.repo_root),
                language,
                _compiled_pattern_key(literal_patterns),
                _compiled_pattern_key(raw_patterns),
                _compiled_pattern_key(suspicious_patterns),
            )
            cached_hits = cache_bucket.get(cache_key)
            if isinstance(cached_hits, tuple):
                hits = list(cached_hits)
            elif isinstance(cached_hits, list):
                hits = list(cached_hits)
            else:
                if (
                    language == "python"
                    or path.suffix.lower() in _PYTHON_SUFFIXES
                ):
                    hits = _scan_python_literals(
                        path,
                        suspicious_patterns=suspicious_patterns,
                        raw_patterns=raw_patterns,
                    )
                else:
                    try:
                        source = path.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
                    hits = _scan_generic_literals(
                        source,
                        literal_patterns=literal_patterns,
                        raw_patterns=raw_patterns,
                        suspicious_patterns=suspicious_patterns,
                    )
                hits = _dedupe_hits(hits)
                cache_bucket[cache_key] = tuple(hits)

            language_label = language or "unknown"
            for hit in hits:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity=severity,
                        file_path=path,
                        line_number=hit.line_number,
                        message=(
                            "String literal has a suspicious backslash "
                            f"escape in {language_label} source. Use "
                            "raw/verbatim syntax or double-escape the "
                            "backslash."
                        ),
                        can_auto_fix=hit.can_auto_fix,
                        context=hit.context,
                    )
                )

        return violations
