# Last Updated: 2025-11-25
"""Tests for the DriftGuard PolicyEngine orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import pytest

from driftguard.core import PolicyEngine
from driftguard.rules import Metric, Rule, RuleContext, Violation
from driftguard.spec import DriftGuardSpec, RuleSetSpec, RuleSurface


class DummyRule(Rule):
    rule_id = "dummy-rule"

    def __init__(
        self,
        *,
        can_fix: bool = False,
        safe_fix: bool = True,
        run_modes: Sequence[str] | None = None,
    ) -> None:
        self.can_fix = can_fix
        self.safe_fix = safe_fix
        if run_modes is not None:
            self.run_modes = tuple(run_modes)

    def check(self, context: RuleContext):
        metric = Metric(name=f"metric-{context.mode}", value=1)
        violation = Violation(
            rule_id=self.rule_id, message=f"run-{context.scope}"
        )
        return [violation], [metric]

    def fix(self, context: RuleContext, safe_only: bool) -> List[str]:
        return [f"fixed-{context.scope}-{safe_only}"]


@pytest.fixture()
def sample_spec() -> DriftGuardSpec:
    surface = RuleSurface(
        name="repo", include=["**/*"], exclude=[], rules=None
    )
    return DriftGuardSpec(
        version=1,
        project="Test",
        rulesets={"core": RuleSetSpec(name="core", severity="hard")},
        rules={},
        surfaces={"repo": surface},
    )


def test_check_runs_rules(sample_spec: DriftGuardSpec, tmp_path: Path):
    rule = DummyRule()
    engine = PolicyEngine(spec=sample_spec, repo_root=tmp_path, rules=[rule])
    violations, metrics = engine.check(scope="repo", mode="fast")
    assert len(violations) == 1
    assert violations[0].rule_id == "dummy-rule"
    assert len(metrics) == 1
    assert metrics[0].name == "metric-fast"


def test_check_skips_unmatched_mode(
    sample_spec: DriftGuardSpec, tmp_path: Path
):
    rule = DummyRule(run_modes=["full"])
    engine = PolicyEngine(spec=sample_spec, repo_root=tmp_path, rules=[rule])
    violations, metrics = engine.check(scope="repo", mode="fast")
    assert violations == []
    assert metrics == []


def test_fix_respects_safe_only(sample_spec: DriftGuardSpec, tmp_path: Path):
    safe_rule = DummyRule(can_fix=True, safe_fix=True)
    unsafe_rule = DummyRule(can_fix=True, safe_fix=False)
    engine = PolicyEngine(
        spec=sample_spec, repo_root=tmp_path, rules=[safe_rule, unsafe_rule]
    )
    messages = engine.fix(scope="staged", safe_only=True)
    assert messages == ["fixed-staged-True"]


def test_fix_runs_all_when_not_safe_only(
    sample_spec: DriftGuardSpec, tmp_path: Path
):
    safe_rule = DummyRule(can_fix=True, safe_fix=True)
    unsafe_rule = DummyRule(can_fix=True, safe_fix=False)
    engine = PolicyEngine(
        spec=sample_spec, repo_root=tmp_path, rules=[safe_rule, unsafe_rule]
    )
    messages = engine.fix(scope="staged", safe_only=False)
    assert messages == ["fixed-staged-False", "fixed-staged-False"]
