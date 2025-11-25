"""Drift metric rules for DriftGuard."""

from __future__ import annotations

from typing import List

from driftguard.rules import Metric, Rule, RuleContext, Violation


class TodoCountRule(Rule):
    """Track TODO occurrences for drift reporting."""

    name = "todo-count"

    def check(self, context: RuleContext) -> List[Violation]:
        _ = context
        return []

    def fix(self, context: RuleContext) -> List[Violation]:
        _ = context
        return []

    def metrics(self, context: RuleContext) -> List[Metric]:
        _ = context
        return []


class TestCouplingRule(Rule):
    """Measure coupling between new modules and added tests."""

    name = "test-coupling"

    def check(self, context: RuleContext) -> List[Violation]:
        _ = context
        return []

    def metrics(self, context: RuleContext) -> List[Metric]:
        _ = context
        return []


class DocAgeRule(Rule):
    """Report documentation age drift metrics."""

    name = "doc-age"

    def check(self, context: RuleContext) -> List[Violation]:
        _ = context
        return []

    def metrics(self, context: RuleContext) -> List[Metric]:
        _ = context
        return []
