"""Java translator for DevCovenant LanguageUnit generation."""

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

TEST_TEMPLATES = ("{stem}Test.java", "Test{stem}.java")
SYMBOL_PATTERNS = (
    ("class", re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)")),
    (
        "method",
        re.compile(
            r"(?:public|private|protected|static|final|\s)+"
            r"[\w<>,\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
        ),
    ),
    (
        "variable",
        re.compile(
            r"\b(?:int|long|double|float|boolean|char|byte|short|"
            r"String|var)\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)\b"
        ),
    ),
)
RISK_PATTERNS = (
    # security-scanner: allow (pattern literals for policy translation)
    (
        re.compile(  # security-scanner: allow
            r"\bRuntime\.getRuntime\s*\(\)\.exec\s*\("
        ),
        "Avoid runtime command execution.",
    ),
    (
        re.compile(r"\bProcessBuilder\s*\("),
        "Review ProcessBuilder command execution.",
    ),
)


def can_handle(
    *, path: Path, declaration: TranslatorDeclaration, **kwargs: Any
) -> bool:
    """Return True when declared extensions include the file suffix."""
    return can_handle_declared_extensions(
        path=path, declaration=declaration, **kwargs
    )


def _has_nearby_comment(line_number: int, lines: list[str]) -> bool:
    """Return True when a nearby Java comment marker is present."""
    start = max(1, line_number - 3)
    for current in range(start, line_number + 1):
        if current > len(lines):
            continue
        text = lines[current - 1].strip()
        if text.startswith(("//", "/*", "*", "/**")):
            return True
    return False


def translate(
    *, path: Path, source: str, declaration: TranslatorDeclaration, **_: Any
) -> LanguageUnit:
    """Translate Java source into a policy-agnostic LanguageUnit."""
    lines = source.splitlines()
    module_documented = any(
        line.strip().startswith(("//", "/*", "/**", "*")) for line in lines[:5]
    )

    identifiers: list[IdentifierFact] = []
    symbol_docs: list[SymbolDocFact] = []
    for line_number, line in enumerate(lines, start=1):
        for kind, pattern in SYMBOL_PATTERNS:
            for match in pattern.finditer(line):
                name = match.group(1)
                identifiers.append(IdentifierFact(name, line_number, kind))
                if kind in {"class", "method"}:
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
            if pattern.search(line):
                risks.append(RiskFact("error", line_number, message))

    return LanguageUnit(
        translator_id=declaration.translator_id,
        profile_name=declaration.profile_name,
        language="java",
        path=str(path),
        suffix=path.suffix.lower(),
        source=source,
        module_documented=module_documented,
        identifier_facts=tuple(identifiers),
        symbol_doc_facts=tuple(symbol_docs),
        risk_facts=tuple(risks),
        test_name_templates=TEST_TEMPLATES,
    )
