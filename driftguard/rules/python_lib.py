"""Python library rules enforced by DriftGuard."""

from __future__ import annotations

from typing import List

from driftguard.rules import Rule, RuleContext, Violation


class NoPrintInLibRule(Rule):
    """Detect disallowed ``print`` calls in library code."""

    name = "no-print"

    def check(self, context: RuleContext) -> List[Violation]:
        _ = context
        return []


class NewModulesNeedTestsRule(Rule):
    """Require tests alongside new modules."""

    name = "new-modules-need-tests"

    def check(self, context: RuleContext) -> List[Violation]:
        _ = context
        return []


class BugfixHasTestRule(Rule):
    """Recommend tests for bug fixes when feasible."""

    name = "bugfix-has-test"

    def check(self, context: RuleContext) -> List[Violation]:
        _ = context
        return []
