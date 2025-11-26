# Last Updated: 2025-11-26
"""Policy engine orchestrating rule evaluation and drift metric collection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from driftguard.logging_utils import get_logger
from driftguard.rules import Metric, RuleContext, get_all_rules
from driftguard.spec import DriftGuardSpec

logger = get_logger()


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
        logger.info(
            "PolicyEngine initialised with %d rules for project %s",
            len(self.rules),
            spec.project,
        )

    def _build_context(self, scope: str, mode: str) -> RuleContext:
        """Create a :class:`RuleContext` for the requested scope and mode."""

        return RuleContext(
            repo_root=self.repo_root, spec=self.spec, scope=scope, mode=mode
        )

    def check(
        self, scope: str = "repo", mode: str = "full"
    ) -> Dict[str, List]:
        """Run policy checks and return violations and metrics."""

        logger.info(
            "Running DriftGuard check for scope=%s mode=%s", scope, mode
        )
        context = self._build_context(scope=scope, mode=mode)
        violations: List = []
        metrics: List[Metric] = []
        for rule in self.rules:
            logger.debug("Evaluating rule %s", rule.__class__.__name__)
            violations.extend(rule.check(context))
            metrics.extend(rule.metrics(context))
        logger.info(
            "DriftGuard check completed with %d violations and %d metrics",
            len(violations),
            len(metrics),
        )
        return {"violations": violations, "metrics": metrics}

    def fix(
        self, scope: str = "repo", mode: str = "full", safe_only: bool = False
    ) -> Dict[str, List]:
        """Run auto-fixable rules and return resulting results."""

        logger.info(
            "Running DriftGuard fix for scope=%s mode=%s safe_only=%s",
            scope,
            mode,
            safe_only,
        )
        context = self._build_context(scope=scope, mode=mode)
        violations: List = []
        metrics: List[Metric] = []
        for rule in self.rules:
            logger.debug("Applying fixes for rule %s", rule.__class__.__name__)
            violations.extend(rule.fix(context, safe_only=safe_only))
            metrics.extend(rule.metrics(context))
        logger.info(
            "DriftGuard fix completed with %d violations and %d metrics",
            len(violations),
            len(metrics),
        )
        return {"violations": violations, "metrics": metrics}

    def metrics(self, scope: str = "repo", mode: str = "full") -> List[Metric]:
        """Collect drift metrics without running enforcement rules."""

        logger.info(
            "Collecting DriftGuard metrics for scope=%s mode=%s", scope, mode
        )
        context = self._build_context(scope=scope, mode=mode)
        metrics: List[Metric] = []
        for rule in self.rules:
            logger.debug(
                "Gathering metrics from rule %s", rule.__class__.__name__
            )
            metrics.extend(rule.metrics(context))
        logger.info("Collected %d total metrics", len(metrics))
        return metrics
