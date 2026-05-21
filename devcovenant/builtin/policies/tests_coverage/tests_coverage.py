"""Validate tests-coverage from related test files directly."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import devcovenant.core.repository_paths as yaml_cache_service
from devcovenant.builtin.policies.modules_need_tests import (
    modules_need_tests as modules_runtime,
)
from devcovenant.builtin.policies.tests_coverage.assertion_signal import (
    analyze_assertion_signal,
)
from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)
from devcovenant.core.selectors import SelectorSet

_VALID_SEVERITIES = {"critical", "error", "warning", "info"}
_DEFAULT_SYMBOL_KINDS = ("function", "class")
_DEFAULT_FIXTURE_PATTERN = r"\bDEVCOV_FIXTURE_OK:\s*(?P<reason>\S.*)"
_DEFAULT_ASSERTION_PATTERNS = (r"*=>\bassert\b",)
_DEFAULT_TAUTOLOGY_PATTERNS = (
    r"*=>^\s*assert\s*\(\s*true\s*\)\s*;?\s*$",
    r"*=>^\s*assert\s+true\s*;?\s*$",
    r"rust=>^\s*assert!\s*\(\s*true\s*\)\s*;?\s*$",
)


def _configured_severity(policy: PolicyCheck) -> str:
    """Return normalized severity from policy metadata/config."""
    raw = str(policy.get_option("severity", "error")).strip().lower()
    if raw in _VALID_SEVERITIES:
        return raw
    return "error"


def _normalize_tokens(raw: object) -> list[str]:
    """Normalize list-like metadata values into non-empty strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [entry.strip() for entry in raw.split(",")]
    elif isinstance(raw, list):
        values = [str(entry).strip() for entry in raw]
    else:
        values = [str(raw).strip()]
    return [entry for entry in values if entry]


def _parse_lang_values(raw: object) -> dict[str, list[str]]:
    """Parse `language=>value` metadata into language-keyed lists."""
    mapping: dict[str, list[str]] = {}
    for token in _normalize_tokens(raw):
        if "=>" not in token:
            mapping.setdefault("*", []).append(token)
            continue
        language, value = token.split("=>", 1)
        language_token = language.strip().lower() or "*"
        value_token = value.strip()
        if not value_token:
            continue
        mapping.setdefault(language_token, []).append(value_token)
    return mapping


def _values_for_language(
    mapping: dict[str, list[str]], language: str
) -> tuple[str, ...]:
    """Resolve configured values for a language with wildcard defaults."""
    language_token = str(language or "").strip().lower()
    values: list[str] = []
    values.extend(mapping.get("*", []))
    values.extend(mapping.get(language_token, []))
    return tuple(values)


def _configured_bool(policy: PolicyCheck, key: str, default: bool) -> bool:
    """Return one policy option as a deterministic boolean."""
    raw = policy.get_option(key, default)
    if isinstance(raw, bool):
        return raw
    token = str(raw).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


def _configured_int(policy: PolicyCheck, key: str, default: int) -> int:
    """Return one policy option coerced to int with a configured default."""
    raw = policy.get_option(key, default)
    if isinstance(raw, int):
        return raw
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _tracked_symbols(
    names: Iterable[str], *, min_length: int
) -> tuple[str, ...]:
    """Normalize and dedupe symbol names tracked for fidelity checks."""
    symbols: list[str] = []
    seen: set[str] = set()
    for raw in names:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        if token.startswith("_"):
            continue
        if len(token) < min_length:
            continue
        if token in seen:
            continue
        seen.add(token)
        symbols.append(token)
    return tuple(symbols)


class TestsCoverageCheck(PolicyCheck):
    """Validate structural assertion coverage for related tests."""

    policy_id = "tests-coverage"
    version = "1.0.0"
    __test__ = False

    @staticmethod
    def _changed_modules_scope(context: CheckContext) -> set[Path]:
        """Return changed module scope for symbol-fidelity enforcement."""
        state = context.change_state
        if (
            state.stage
            and state.stage != "start"
            and state.session_valid
            and state.session_paths
        ):
            return set(state.session_paths)
        return set(context.changed_files)

    def check(self, context: CheckContext) -> List[Violation]:
        """Require assertion signals in related tests."""
        if context.change_state.stage == "start":
            return []

        runtime = context.translator_runtime
        if runtime is None:
            return []

        severity = _configured_severity(self)
        selector = SelectorSet.from_policy(self)
        tests_dirs = modules_runtime._test_roots(self)
        test_files = modules_runtime._list_existing_tests(
            context.repo_root,
            tests_dirs,
            context=context,
        )
        indexed_tests = modules_runtime._index_tests(
            test_files,
            repo_root=context.repo_root,
            context=context,
        )
        repo_files = modules_runtime._collect_repo_files(
            context.repo_root,
            context=context,
        )

        modules = [
            path
            for path in sorted(repo_files)
            if modules_runtime._is_module_candidate(
                path,
                selector=selector,
                repo_root=context.repo_root,
                tests_dirs=tests_dirs,
            )
        ]
        assertion_map = _parse_lang_values(
            self.get_option(
                "assertion_signal_patterns",
                list(_DEFAULT_ASSERTION_PATTERNS),
            )
        )
        tautology_map = _parse_lang_values(
            self.get_option(
                "tautology_patterns", list(_DEFAULT_TAUTOLOGY_PATTERNS)
            )
        )
        fixture_pattern = str(
            self.get_option("fixture_marker_pattern", _DEFAULT_FIXTURE_PATTERN)
        ).strip()
        tracked_kinds = {
            token.lower()
            for token in _normalize_tokens(
                self.get_option("symbol_kinds", list(_DEFAULT_SYMBOL_KINDS))
            )
        }
        if not tracked_kinds:
            tracked_kinds = set(_DEFAULT_SYMBOL_KINDS)
        symbol_min_length = max(
            _configured_int(self, "symbol_name_min_length", 3),
            1,
        )
        symbol_window = max(
            _configured_int(self, "symbol_assertion_window", 2),
            0,
        )
        enforce_symbol_fidelity = _configured_bool(
            self, "enforce_symbol_fidelity", True
        )
        changed_scope = self._changed_modules_scope(context)

        violations: List[Violation] = []
        for module in modules:
            resolution = runtime.resolve(
                path=module,
                policy_id=self.policy_id,
                context=context,
            )
            if not resolution.is_resolved:
                violations.extend(resolution.violations)
                continue

            source = yaml_cache_service.read_text(module)
            unit = runtime.translate(
                resolution,
                path=module,
                source=source,
                context=context,
            )
            if unit is None:
                continue

            related_tests = modules_runtime._related_tests(
                module=module,
                unit_templates=unit.test_name_templates,
                indexed_tests=indexed_tests,
            )

            module_rel = module.relative_to(context.repo_root).as_posix()
            if not related_tests:
                continue
            related_sorted = sorted(related_tests)
            tracked_symbols = _tracked_symbols(
                (
                    fact.name
                    for fact in unit.symbol_doc_facts
                    if fact.kind.lower() in tracked_kinds
                ),
                min_length=symbol_min_length,
            )
            covered_symbols: set[str] = set()
            has_assertion = False
            for test_path in related_sorted:
                analysis = analyze_assertion_signal(
                    test_path,
                    language=unit.language,
                    assertion_patterns=_values_for_language(
                        assertion_map, unit.language
                    ),
                    tautology_patterns=_values_for_language(
                        tautology_map, unit.language
                    ),
                    fixture_marker_pattern=fixture_pattern,
                    symbol_names=tracked_symbols,
                    symbol_window=symbol_window,
                    symbol_name_min_length=symbol_min_length,
                )
                if not analysis.has_assertion_signal:
                    continue
                has_assertion = True
                covered_symbols.update(analysis.covered_symbols)

            if has_assertion:
                module_changed = module in changed_scope
                if (
                    not enforce_symbol_fidelity
                    or not module_changed
                    or not tracked_symbols
                ):
                    continue
                missing = sorted(
                    symbol
                    for symbol in tracked_symbols
                    if symbol not in covered_symbols
                )
                if not missing:
                    continue
                related_rel = ", ".join(
                    path.relative_to(context.repo_root).as_posix()
                    for path in related_sorted
                )
                missing_text = ", ".join(missing)
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity=severity,
                        file_path=module,
                        message=(
                            "Related tests missing symbol-level assertion "
                            f"coverage for module {module_rel}. Missing: "
                            f"{missing_text}. Related tests: {related_rel}"
                        ),
                    )
                )
                continue

            related_rel = ", ".join(
                path.relative_to(context.repo_root).as_posix()
                for path in related_sorted
            )
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity=severity,
                    file_path=module,
                    message=(
                        "Related tests missing assertion coverage "
                        f"signals for module {module_rel}: {related_rel}"
                    ),
                )
            )

        return violations
