"""Language-agnostic clarity checks using shared translator LanguageUnit."""

from __future__ import annotations

from typing import List

from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)
from devcovenant.core.selectors import SelectorSet
from devcovenant.core.translator import IdentifierFact, LanguageUnit

ALLOW_NAME_CLARITY = "name-clarity: allow"
_PLACEHOLDER_NAMES = {
    "foo",
    "bar",
    "baz",
    "tmp",
    "temp",
    "var",
    "data",
    "val",
    "obj",
}
_SHORT_NAME_ALLOW = {
    "i",
    "j",
    "k",
    "x",
    "y",
    "z",
    "d",
    "e",
    "f",
    "v",
    "re",
    "os",
    "io",
    "dt",
    "_dt",
}


def flag_name_clarity_identifiers(
    unit: LanguageUnit,
) -> tuple[IdentifierFact, ...]:
    """Return identifier facts that violate name-clarity heuristics."""
    lines = unit.source.splitlines()
    violations: list[IdentifierFact] = []
    seen: set[tuple[str, int]] = set()
    for fact in unit.identifier_facts:
        cleaned = fact.name.lstrip("_")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in _SHORT_NAME_ALLOW:
            continue
        if len(cleaned) >= 3 and lowered not in _PLACEHOLDER_NAMES:
            continue
        if 1 <= fact.line_number <= len(lines):
            if ALLOW_NAME_CLARITY in lines[fact.line_number - 1]:
                continue
        key = (fact.name, fact.line_number)
        if key in seen:
            continue
        seen.add(key)
        violations.append(fact)
    return tuple(violations)


class NameClarityCheck(PolicyCheck):
    """Warn when placeholder or overly short identifiers are introduced."""

    policy_id = "name-clarity"
    version = "1.3.0"
    DEFAULT_SUFFIXES = [
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
    ]

    def _selector(self) -> SelectorSet:
        """Return selector describing files enforced by the policy."""
        return SelectorSet.from_policy(
            self, defaults={"include_suffixes": self.DEFAULT_SUFFIXES}
        )

    def check(self, context: CheckContext) -> List[Violation]:
        """Run the check across all matching files using translated units."""
        files = context.all_files or context.changed_files or []
        violations: List[Violation] = []
        selector = self._selector()
        runtime = context.translator_runtime
        if runtime is None:
            return violations

        for path in files:
            if not path.is_file() or not selector.matches(
                path, context.repo_root
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
            source = path.read_text(encoding="utf-8")
            unit = runtime.translate(
                resolution,
                path=path,
                source=source,
                context=context,
            )
            if unit is None:
                continue
            for fact in flag_name_clarity_identifiers(unit):
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="warning",
                        file_path=path,
                        line_number=fact.line_number,
                        message=(
                            f"Identifier '{fact.name}' is overly generic "
                            "or too short; choose a more descriptive name."
                        ),
                    )
                )

        return violations
