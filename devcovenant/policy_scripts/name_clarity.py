"""Warn when placeholder or overly short identifiers appear in scope."""

import ast
from typing import List, Sequence

from devcovenant.base import CheckContext, PolicyCheck, Violation
from devcovenant.selectors import SelectorSet

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
        """Store the source lines for allow-comment detection."""
        self.lines = lines
        self.violations: List[tuple[str, int]] = []

    def _clean_name(self, name: str) -> str:
        """Strip leading underscores from the identifier."""
        return name.lstrip("_")

    def _should_flag(self, name: str) -> bool:
        """Determine whether the identifier violates the clarity policy."""
        if not name:
            return False

        cleaned = self._clean_name(name)
        if not cleaned:
            return False

        cleaned_lower = cleaned.lower()
        if cleaned_lower in BLACKLIST:
            return True
        if len(cleaned) < MIN_LENGTH and cleaned_lower not in SHORT_ARG_ALLOW:
            return True
        return False

    def _has_allow_comment(self, lineno: int) -> bool:
        """Return True if the line includes an allow comment."""
        if not (1 <= lineno <= len(self.lines)):
            return False
        return ALLOW_COMMENT in self.lines[lineno - 1]

    def _record(self, name: str, lineno: int) -> None:
        """Record the violation unless it has an allow comment."""
        if self._has_allow_comment(lineno):
            return
        self.violations.append((name, lineno))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function definitions for placeholder names."""
        if self._should_flag(node.name):
            self._record(node.name, node.lineno)
        self._visit_arguments(node.args)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Apply the same checks to async functions."""
        if self._should_flag(node.name):
            self._record(node.name, node.lineno)
        self._visit_arguments(node.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Flag classes using short or generic names."""
        if self._should_flag(node.name):
            self._record(node.name, node.lineno)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Inspect assignments for terse identifiers."""
        for target in node.targets:
            self._visit_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Inspect annotated assignments for clarity."""
        if node.target:
            self._visit_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Check loop targets for short identifiers."""
        self._visit_target(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Check async loop targets for clarity."""
        self._visit_target(node.target)
        self.generic_visit(node)

    def _visit_arguments(self, args: ast.arguments) -> None:
        """Visit function arguments to apply the clarity check."""
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
        """Recursively examine assignment targets for bad names."""
        if isinstance(target, ast.Name) and self._should_flag(target.id):
            self._record(target.id, target.lineno)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._visit_target(element)


class NameClarityCheck(PolicyCheck):
    """Warn when placeholder or overly short identifiers are introduced."""

    policy_id = "name-clarity"
    version = "1.1.0"
    DEFAULT_SUFFIXES = [".py"]

    def _selector(self) -> SelectorSet:
        """Return selector describing files enforced by the policy."""
        return SelectorSet.from_policy(
            self, defaults={"include_suffixes": self.DEFAULT_SUFFIXES}
        )

    def check(self, context: CheckContext) -> List[Violation]:
        """Run the check across all matching Python files."""
        files = context.all_files or context.changed_files or []
        violations: List[Violation] = []
        selector = self._selector()

        for path in files:
            if not path.is_file():
                continue
            if not selector.matches(path, context.repo_root):
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
                        severity="warning",
                        file_path=path,
                        line_number=lineno,
                        message=(
                            f"Identifier '{name}' is overly generic "
                            "or too short; choose a more descriptive name."
                        ),
                    )
                )

        return violations
