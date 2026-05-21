"""Ruby translator for DevCovenant LanguageUnit generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from devcovenant.core.translator import (
    IdentifierFact,
    LanguageUnit,
    RiskFact,
    SymbolDocFact,
    TranslatorDeclaration,
    can_handle_declared_extensions,
)

ALLOW_SECURITY = "security-scanner: allow"
TEST_TEMPLATES = ("{stem}_test.rb", "test_{stem}.rb", "{stem}_spec.rb")
SYMBOL_PATTERNS = (
    ("class", re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_:]*)")),
    ("module", re.compile(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_:]*)")),
    ("function", re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_!?=]*)")),
    (
        "variable",
        re.compile(r"\b([a-z_][a-zA-Z0-9_]*)\s*="),
    ),
)
RISK_PATTERNS = (
    # security-scanner: allow (pattern literals for policy translation)
    (re.compile(r"\beval\s*\("), "Avoid eval()."),
    (re.compile(r"\b(?:system|exec)\s*\("), "Review process execution."),
    (re.compile(r"\bIO\.popen\s*\("), "Review IO.popen usage."),
)


def can_handle(
    *, path: Path, declaration: TranslatorDeclaration, **kwargs: Any
) -> bool:
    """Return True when declared extensions include the file suffix."""
    return can_handle_declared_extensions(
        path=path, declaration=declaration, **kwargs
    )


def _has_nearby_comment(line_number: int, lines: list[str]) -> bool:
    """Return True when a nearby Ruby comment marker is present."""
    start = max(1, line_number - 3)
    for current in range(start, line_number + 1):
        if current > len(lines):
            continue
        if lines[current - 1].strip().startswith("#"):
            return True
    return False


def translate(
    *, path: Path, source: str, declaration: TranslatorDeclaration, **_: Any
) -> LanguageUnit:
    """Translate Ruby source into a policy-agnostic LanguageUnit."""
    lines = source.splitlines()
    module_documented = any(line.strip().startswith("#") for line in lines[:5])

    identifiers: list[IdentifierFact] = []
    symbol_docs: list[SymbolDocFact] = []
    for line_number, line in enumerate(lines, start=1):
        for kind, pattern in SYMBOL_PATTERNS:
            for match in pattern.finditer(line):
                name = match.group(1)
                identifiers.append(IdentifierFact(name, line_number, kind))
                if kind in {"class", "module", "function"}:
                    symbol_docs.append(
                        SymbolDocFact(
                            kind,
                            name,
                            line_number,
                            _has_nearby_comment(line_number, lines),
                        )
                    )

    risks: list[RiskFact] = []
    for pattern, message in RISK_PATTERNS:
        for line_number, line in enumerate(lines, start=1):
            if ALLOW_SECURITY in line:
                continue
            if pattern.search(line):
                risks.append(RiskFact("error", line_number, message))

    return LanguageUnit(
        translator_id=declaration.translator_id,
        profile_name=declaration.profile_name,
        language="ruby",
        path=str(path),
        suffix=path.suffix.lower(),
        source=source,
        module_documented=module_documented,
        identifier_facts=tuple(identifiers),
        symbol_doc_facts=tuple(symbol_docs),
        risk_facts=tuple(risks),
        test_name_templates=TEST_TEMPLATES,
    )
