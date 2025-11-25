# Last Updated: 2025-11-25
"""Core orchestration logic for the DriftGuard policy engine.

The :class:`PolicyEngine` coordinates rule execution using only the filesystem
and the loaded specification. It deliberately avoids Copernican-specific
imports so the module can be reused when the engine is spun off.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from driftguard import utils
from driftguard.rules import (
    Metric,
    Rule,
    RuleContext,
    Violation,
    get_all_rules,
)
from driftguard.spec import DriftGuardSpec


ALLOWED_SCOPES = ("repo", "staged")
ALLOWED_MODES = ("fast", "full")


@dataclass
class PolicyEngine:
    """Run DriftGuard rules and aggregate their results."""

    spec: DriftGuardSpec
    repo_root: Path
    rules: Sequence[Rule] | None = None

    def __post_init__(self) -> None:
        if self.rules is None:
            self.rules = tuple(get_all_rules(self.spec))
        self.repo_root = self.repo_root.resolve()

    def _context(self, scope: str, mode: str) -> RuleContext:
        return RuleContext(
            repo_root=self.repo_root,
            spec=self.spec,
            scope=scope,
            mode=mode,
        )

    def _run_checks(
        self, rules: Iterable[Rule], context: RuleContext
    ) -> Tuple[List[Violation], List[Metric]]:
        violations: List[Violation] = []
        metrics: List[Metric] = []
        for rule in rules:
            if not rule.supports_scope(context.scope):
                continue
            if not rule.supports_mode(context.mode):
                continue
            rule_violations, rule_metrics = rule.check(context)
            violations.extend(rule_violations)
            metrics.extend(rule_metrics)
        return violations, metrics

    def check(
        self, scope: str = "repo", mode: str = "fast"
    ) -> Tuple[List[Violation], List[Metric]]:
        """Run rules and return violations plus drift metrics."""

        normalized_scope = utils.ensure_scope(scope, ALLOWED_SCOPES)
        normalized_mode = utils.ensure_mode(mode, ALLOWED_MODES)
        context = self._context(normalized_scope, normalized_mode)
        rules = self.rules or ()
        return self._run_checks(rules, context)

    def fix(self, scope: str = "staged", safe_only: bool = True) -> List[str]:
        """Apply safe auto-fixes for the given scope."""

        normalized_scope = utils.ensure_scope(scope, ALLOWED_SCOPES)
        utils.ensure_mode("fast", ALLOWED_MODES)
        context = self._context(normalized_scope, "fast")
        messages: List[str] = []
        for rule in self.rules or ():
            if not rule.supports_scope(context.scope):
                continue
            if not rule.can_fix:
                continue
            if safe_only and not rule.safe_fix:
                continue
            messages.extend(rule.fix(context, safe_only=safe_only))
        return messages
