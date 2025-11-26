"""Policy engine orchestrating rule evaluation and drift metric collection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from driftguard.rules import Metric, RuleContext, get_all_rules
from driftguard.spec import DriftGuardSpec


class PolicyEngine:
    """Evaluate DriftGuard rules against repository surfaces.

    The engine maintains minimal state for now and defers all rule lookup to
    the registry returned by :func:`driftguard.rules.get_all_rules`. Future
    revisions will wire in rule execution, reporting and autofix support.
    """

    def __init__(self, spec: DriftGuardSpec, repo_root: Optional[Path | str]):
        self.repo_root = (
            Path(repo_root) if repo_root is not None else Path.cwd()
        )
        self.spec = spec
        self.rules = get_all_rules(spec)

    def _build_context(self, scope: str, mode: str) -> RuleContext:
        """Create a :class:`RuleContext` for the requested scope and mode."""

        return RuleContext(
            repo_root=self.repo_root, spec=self.spec, scope=scope, mode=mode
        )

    def check(
        self, scope: str = "repo", mode: str = "full"
    ) -> Dict[str, List]:
        """Run policy checks and return violations and metrics."""

        context = self._build_context(scope=scope, mode=mode)
        violations: List = []
        metrics: List[Metric] = []
        for rule in self.rules:
            violations.extend(rule.check(context))
            metrics.extend(rule.metrics(context))
        return {"violations": violations, "metrics": metrics}

    def fix(
        self, scope: str = "repo", mode: str = "full", safe_only: bool = False
    ) -> Dict[str, List]:
        """Run auto-fixable rules and return resulting results."""

        context = self._build_context(scope=scope, mode=mode)
        violations: List = []
        metrics: List[Metric] = []
        for rule in self.rules:
            violations.extend(rule.fix(context, safe_only=safe_only))
            metrics.extend(rule.metrics(context))
        return {"violations": violations, "metrics": metrics}

    def metrics(self, scope: str = "repo", mode: str = "full") -> List[Metric]:
        """Collect drift metrics without running enforcement rules."""

        context = self._build_context(scope=scope, mode=mode)
        metrics: List[Metric] = []
        for rule in self.rules:
            metrics.extend(rule.metrics(context))
        return metrics
