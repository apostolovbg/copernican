"""Warn when placeholder or overly short identifiers appear."""

import ast
from typing import List, Sequence

from devcovenant.base import CheckContext, PolicyCheck, Violation

BLACKLIST = {
    "foo",
    "bar",
    "baz",
    "tmp",
    "temp",
    "var",
    "data",
    "val",
    "value",
    "obj",
    "item",
}
SHORT_ARG_ALLOW = {"i", "j", "k", "x", "y", "z"}
MIN_LENGTH = 3
ALLOW_COMMENT = "name-clarity: allow"


class _NameClarityVisitor(ast.NodeVisitor):
    """Collect identifiers that violate clarity rules."""

    def __init__(self, lines: Sequence[str]):
        self.lines = lines
        self.violations: List[tuple[str, int]] = []

    def _clean_name(self, name: str) -> str:
        return name.lstrip("_")

    def _should_flag(self, name: str) -> bool:
        if not name:
            return False

        cleaned = self._clean_name(name)
        if not cleaned:
            return False

        cleaned_lower = cleaned.lower()
        if cleaned_lower in BLACKLIST:
            return True
        if (
            len(cleaned) < MIN_LENGTH
            and cleaned_lower not in SHORT_ARG_ALLOW
        ):
            return True
        return False

    def _has_allow_comment(self, lineno: int) -> bool:
        if not (1 <= lineno <= len(self.lines)):
            return False
        return ALLOW_COMMENT in self.lines[lineno - 1]

    def _record(self, name: str, lineno: int) -> None:
        if self._has_allow_comment(lineno):
            return
        self.violations.append((name, lineno))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._should_flag(node.name):
            self._record(node.name, node.lineno)
        self._visit_arguments(node.args)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._should_flag(node.name):
            self._record(node.name, node.lineno)
        self._visit_arguments(node.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._should_flag(node.name):
            self._record(node.name, node.lineno)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._visit_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.target:
            self._visit_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_target(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_target(node.target)
        self.generic_visit(node)

    def _visit_arguments(self, args: ast.arguments) -> None:
        for arg in (
            args.posonlyargs
            + args.args
            + args.kwonlyargs
            + ([] if not args.vararg else [args.vararg])
            + ([] if not args.kwarg else [args.kwarg])
        ):
            if arg.arg and self._should_flag(arg.arg):
                lineno = getattr(arg, "lineno", 0) or 0
                self._record(arg.arg, lineno)

    def _visit_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name) and self._should_flag(target.id):
            self._record(target.id, target.lineno)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._visit_target(element)


class NameClarityCheck(PolicyCheck):
    """Warn when placeholder or overly short identifiers are introduced."""

    policy_id = "name-clarity"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        files = context.all_files or context.changed_files or []
        violations: List[Violation] = []

        for path in files:
            if path.suffix != ".py":
                continue
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue
            if "tests" in rel.parts:
                continue

            text = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue

            lines = text.splitlines()
            visitor = _NameClarityVisitor(lines)
            visitor.visit(tree)

            for name, lineno in visitor.violations:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="info",
                        file_path=path,
                        line_number=lineno,
                        message=(
                            f"Identifier '{name}' is overly generic or too short; "
                            "choose a more descriptive name."
                        ),
                    )
                )

        return violations
