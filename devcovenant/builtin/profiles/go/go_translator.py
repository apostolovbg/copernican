"""Go translator for DevCovenant LanguageUnit generation."""

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

TEST_TEMPLATES = ("{stem}_test.go",)
SYMBOL_PATTERNS = (
    ("function", re.compile(r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)")),
    (
        "variable",
        re.compile(r"\bvar\s+([A-Za-z_][A-Za-z0-9_]*)"),
    ),
)
RISK_PATTERNS = (
    (re.compile(r"\bexec\.Command\s*\("), "Avoid os/exec.Command()."),
    (re.compile(r'"os/exec"'), "Review os/exec usage."),
    (re.compile(r"\bos\.Exec\s*\("), "Avoid dynamic os.Exec usage."),
)


def can_handle(
    *, path: Path, declaration: TranslatorDeclaration, **kwargs: Any
) -> bool:
    """Return True when declared extensions include the file suffix."""
    return can_handle_declared_extensions(
        path=path, declaration=declaration, **kwargs
    )


def _has_nearby_comment(line_number: int, lines: list[str]) -> bool:
    """Return True when a nearby Go comment marker is present."""
    start = max(1, line_number - 3)
    for current in range(start, line_number + 1):
        if current > len(lines):
            continue
        if lines[current - 1].strip().startswith("//"):
            return True
    return False


def translate(
    *, path: Path, source: str, declaration: TranslatorDeclaration, **_: Any
) -> LanguageUnit:
    """Translate Go source into a policy-agnostic LanguageUnit."""
    lines = source.splitlines()
    module_documented = any(
        line.strip().startswith("//") for line in lines[:5]
    )

    identifiers: list[IdentifierFact] = []
    symbol_docs: list[SymbolDocFact] = []
    for line_number, line in enumerate(lines, start=1):
        for kind, pattern in SYMBOL_PATTERNS:
            for match in pattern.finditer(line):
                name = match.group(1)
                identifiers.append(IdentifierFact(name, line_number, kind))
                if kind == "function":
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
        language="go",
        path=str(path),
        suffix=path.suffix.lower(),
        source=source,
        module_documented=module_documented,
        identifier_facts=tuple(identifiers),
        symbol_doc_facts=tuple(symbol_docs),
        risk_facts=tuple(risks),
        test_name_templates=TEST_TEMPLATES,
    )
