"""Python translator for DevCovenant LanguageUnit generation."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import devcovenant.core.repository_paths as yaml_cache_service
from devcovenant.core.translator import (
    IdentifierFact,
    LanguageUnit,
    RiskFact,
    SymbolDocFact,
    TranslatorDeclaration,
    can_handle_declared_extensions,
)

ALLOW_SECURITY = "security-scanner: allow"
TEST_TEMPLATES = ("test_{stem}.py", "{stem}_test.py")
RISK_PATTERNS = (
    # security-scanner: allow (pattern literals for policy translation)
    (re.compile(r"\beval\s*\("), "Avoid eval()."),
    (re.compile(r"\bexec\s*\("), "Avoid exec()."),
    (
        re.compile(r"\bpickle\.loads\s*\("),  # security-scanner: allow
        "Avoid untrusted pickle.loads().",
    ),
    (
        re.compile(r"\bsubprocess\.run\s*\([^)]*shell\s*=\s*True"),
        "Avoid shell=True in subprocess.run().",
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
    """Return True when a nearby comment marker is present."""
    start = max(1, line_number - 3)
    for current in range(start, line_number + 1):
        if current > len(lines):
            continue
        if lines[current - 1].strip().startswith("#"):
            return True
    return False


def _path_signature(path: Path) -> tuple[object, ...]:
    """Return one run-scoped path signature for lightweight caches."""
    normalized = path.as_posix()
    try:
        stat = path.stat()
    except OSError:
        return (normalized, None, None)
    return (normalized, stat.st_mtime_ns, stat.st_size)


def _parse_python_module(path: Path, source: str) -> ast.AST | None:
    """Parse Python source using the shared AST cache when possible."""
    if path.exists():
        return yaml_cache_service.parse_python_ast(path)
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError:
        return None


class _PythonFactsVisitor(ast.NodeVisitor):
    """Collect identifier and documentation facts in one tree walk."""

    def __init__(self, *, lines: list[str]) -> None:
        """Store source lines and initialize fact containers."""
        self.lines = lines
        self.identifiers: list[IdentifierFact] = []
        self.symbol_docs: list[SymbolDocFact] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collect function identifiers and documentation facts."""
        self._visit_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Collect async-function identifiers and documentation facts."""
        self._visit_function(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Collect class identifiers and documentation facts."""
        self.identifiers.append(
            IdentifierFact(node.name, node.lineno, "class")
        )
        documented = bool(ast.get_docstring(node)) or _has_nearby_comment(
            node.lineno,
            self.lines,
        )
        self.symbol_docs.append(
            SymbolDocFact("class", node.name, node.lineno, documented)
        )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Collect generic identifier facts."""
        self.identifiers.append(
            IdentifierFact(node.id, getattr(node, "lineno", 1), "identifier")
        )

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Collect function facts for sync and async nodes."""
        self.identifiers.append(
            IdentifierFact(node.name, node.lineno, "function")
        )
        documented = bool(ast.get_docstring(node)) or _has_nearby_comment(
            node.lineno,
            self.lines,
        )
        self.symbol_docs.append(
            SymbolDocFact("function", node.name, node.lineno, documented)
        )
        args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        for arg in args:
            self.identifiers.append(
                IdentifierFact(
                    arg.arg,
                    getattr(arg, "lineno", node.lineno),
                    "argument",
                )
            )
        if node.args.vararg:
            self.identifiers.append(
                IdentifierFact(node.args.vararg.arg, node.lineno, "argument")
            )
        if node.args.kwarg:
            self.identifiers.append(
                IdentifierFact(node.args.kwarg.arg, node.lineno, "argument")
            )


class _PythonDocumentationFactsVisitor(ast.NodeVisitor):
    """Collect only documentation facts for lightweight policy paths."""

    def __init__(self, *, lines: list[str]) -> None:
        """Store source lines and initialize symbol-doc facts."""
        self.lines = lines
        self.symbol_docs: list[SymbolDocFact] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collect function documentation facts."""
        self._record_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Collect async-function documentation facts."""
        self._record_function(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Collect class documentation facts."""
        documented = bool(ast.get_docstring(node)) or _has_nearby_comment(
            node.lineno,
            self.lines,
        )
        self.symbol_docs.append(
            SymbolDocFact("class", node.name, node.lineno, documented)
        )
        self.generic_visit(node)

    def _record_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Collect one sync or async function documentation fact."""
        documented = bool(ast.get_docstring(node)) or _has_nearby_comment(
            node.lineno,
            self.lines,
        )
        self.symbol_docs.append(
            SymbolDocFact("function", node.name, node.lineno, documented)
        )


def translate(
    *, path: Path, source: str, declaration: TranslatorDeclaration, **_: Any
) -> LanguageUnit:
    """Translate Python source into a policy-agnostic LanguageUnit."""
    lines = source.splitlines()
    risks: list[RiskFact] = []
    module = _parse_python_module(path, source)
    if module is None:
        return LanguageUnit(
            translator_id=declaration.translator_id,
            profile_name=declaration.profile_name,
            language="python",
            path=str(path),
            suffix=path.suffix.lower(),
            source=source,
            module_documented=False,
            identifier_facts=tuple(),
            symbol_doc_facts=tuple(),
            risk_facts=tuple(),
            test_name_templates=TEST_TEMPLATES,
        )

    module_documented = bool(ast.get_docstring(module)) or any(
        line.strip().startswith("#") for line in lines[:5]
    )
    visitor = _PythonFactsVisitor(lines=lines)
    visitor.visit(module)

    for pattern, message in RISK_PATTERNS:
        for match in pattern.finditer(source):
            line_number = source.count("\n", 0, match.start()) + 1
            window = lines[max(0, line_number - 3) : line_number]
            if any(ALLOW_SECURITY in text for text in window):
                continue
            risks.append(RiskFact("error", line_number, message))

    return LanguageUnit(
        translator_id=declaration.translator_id,
        profile_name=declaration.profile_name,
        language="python",
        path=str(path),
        suffix=path.suffix.lower(),
        source=source,
        module_documented=module_documented,
        identifier_facts=tuple(visitor.identifiers),
        symbol_doc_facts=tuple(visitor.symbol_docs),
        risk_facts=tuple(risks),
        test_name_templates=TEST_TEMPLATES,
    )


def translate_minimal(
    *,
    path: Path,
    source: str,
    declaration: TranslatorDeclaration,
    context: Any | None = None,
    **_: Any,
) -> LanguageUnit:
    """Translate Python source without identifier/risk collection."""
    cache_bucket_getter = getattr(context, "runtime_cache_bucket", None)
    cache_bucket = (
        cache_bucket_getter("python_translator_minimal")
        if callable(cache_bucket_getter)
        else None
    )
    cache_key = (
        _path_signature(path),
        declaration.translator_id,
        declaration.profile_name,
    )
    if isinstance(cache_bucket, dict) and cache_key in cache_bucket:
        cached = cache_bucket[cache_key]
        if isinstance(cached, LanguageUnit):
            return cached

    lines = source.splitlines()
    module = _parse_python_module(path, source)
    if module is None:
        unit = LanguageUnit(
            translator_id=declaration.translator_id,
            profile_name=declaration.profile_name,
            language="python",
            path=str(path),
            suffix=path.suffix.lower(),
            source=source,
            module_documented=False,
            identifier_facts=tuple(),
            symbol_doc_facts=tuple(),
            risk_facts=tuple(),
            test_name_templates=TEST_TEMPLATES,
        )
    else:
        module_documented = bool(ast.get_docstring(module)) or any(
            line.strip().startswith("#") for line in lines[:5]
        )
        visitor = _PythonDocumentationFactsVisitor(lines=lines)
        visitor.visit(module)
        unit = LanguageUnit(
            translator_id=declaration.translator_id,
            profile_name=declaration.profile_name,
            language="python",
            path=str(path),
            suffix=path.suffix.lower(),
            source=source,
            module_documented=module_documented,
            identifier_facts=tuple(),
            symbol_doc_facts=tuple(visitor.symbol_docs),
            risk_facts=tuple(),
            test_name_templates=TEST_TEMPLATES,
        )

    if isinstance(cache_bucket, dict):
        cache_bucket[cache_key] = unit
    return unit
