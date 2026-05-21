"""TypeScript translator for DevCovenant LanguageUnit generation."""

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
TEST_TEMPLATES = (
    "{stem}.test.ts",
    "{stem}.spec.ts",
    "test_{stem}.ts",
    "{stem}.test.tsx",
    "{stem}.spec.tsx",
)
SYMBOL_PATTERNS = (
    ("function", re.compile(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("class", re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)")),
    (
        "variable",
        re.compile(r"\b(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    ),
)
RISK_PATTERNS = (
    # security-scanner: allow (pattern literals for policy translation)
    (re.compile(r"\beval\s*\("), "Avoid eval()."),
    (re.compile(r"\bnew\s+Function\s*\("), "Avoid Function constructor."),
    (re.compile(r"child_process\.exec"), "Avoid child_process.exec."),
)


def can_handle(
    *, path: Path, declaration: TranslatorDeclaration, **kwargs: Any
) -> bool:
    """Return True when declared extensions include the file suffix."""
    return can_handle_declared_extensions(
        path=path, declaration=declaration, **kwargs
    )


def _has_nearby_comment(line_number: int, lines: list[str]) -> bool:
    """Return True when a nearby TS comment marker is present."""
    start = max(1, line_number - 3)
    for current in range(start, line_number + 1):
        if current > len(lines):
            continue
        text = lines[current - 1].strip()
        if text.startswith(("//", "/*", "/**", "*")):
            return True
    return False


def translate(
    *, path: Path, source: str, declaration: TranslatorDeclaration, **_: Any
) -> LanguageUnit:
    """Translate TypeScript source into a policy-agnostic LanguageUnit."""
    lines = source.splitlines()
    module_documented = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        module_documented = stripped.startswith(("//", "/*", "/**", "*"))
        break

    identifiers: list[IdentifierFact] = []
    symbol_docs: list[SymbolDocFact] = []
    for line_number, line in enumerate(lines, start=1):
        for kind, pattern in SYMBOL_PATTERNS:
            for match in pattern.finditer(line):
                name = match.group(1)
                identifiers.append(IdentifierFact(name, line_number, kind))
                if kind in {"function", "class"}:
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
        language="typescript",
        path=str(path),
        suffix=path.suffix.lower(),
        source=source,
        module_documented=module_documented,
        identifier_facts=tuple(identifiers),
        symbol_doc_facts=tuple(symbol_docs),
        risk_facts=tuple(risks),
        test_name_templates=TEST_TEMPLATES,
    )
