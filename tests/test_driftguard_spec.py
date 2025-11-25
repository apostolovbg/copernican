# Last Updated: 2025-11-25
"""Tests for the DriftGuard spec loader and rule factory."""
from __future__ import annotations

from pathlib import Path

import pytest

from driftguard.rules import Rule, get_all_rules
from driftguard.spec import (
    DriftGuardSpec,
    RuleConfig,
    RuleSetSpec,
    RuleSurface,
    load_spec,
)


class DummyAlignedRule(Rule):
    """Rule whose scopes and modes mirror the spec declaration."""

    rule_id = "dummy-aligned"
    run_modes = ("fast",)
    scopes = ("repo",)

    def __init__(self, marker: str = "") -> None:
        self.marker = marker


class DummyMismatchedRule(Rule):
    """Rule whose modes and scopes intentionally diverge."""

    rule_id = "dummy-mismatch"
    run_modes = ("full",)
    scopes = ("staged",)


@pytest.fixture()
def aligned_spec() -> DriftGuardSpec:
    """Build a spec that matches the DummyAlignedRule declaration."""

    ruleset = RuleSetSpec(name="core", severity="hard")
    rule_config = RuleConfig(
        rule_id="dummy-aligned",
        impl="tests.test_driftguard_spec:DummyAlignedRule",
        ruleset="core",
        run_modes=("fast",),
        scopes=("repo",),
        options={"marker": "configured"},
    )
    surface = RuleSurface(
        name="repo",
        include=["**/*"],
        exclude=[],
        rules=["dummy-aligned"],
    )
    return DriftGuardSpec(
        version=1,
        project="Test",
        rulesets={"core": ruleset},
        rules={"dummy-aligned": rule_config},
        surfaces={"repo": surface},
    )


def test_load_spec_reads_from_repo_root(tmp_path: Path) -> None:
    """Ensure load_spec resolves the repo root before parsing YAML."""

    spec_yaml = tmp_path / "driftguard.yml"
    spec_yaml.write_text(
        """
        version: 2
        project: Demo
        rulesets:
          quality:
            severity: warn
            modes: [fast]
            scopes: [repo]
        rules:
          sample-rule:
            impl: tests.test_driftguard_spec:DummyAlignedRule
            ruleset: quality
            modes: [fast]
            scopes: [repo]
            options:
              marker: yaml-loaded
        surfaces:
          repo:
            include:
              - "src/**"
            exclude:
              - "tests/**"
            rules:
              - sample-rule
        """,
        encoding="utf-8",
    )
    spec = load_spec(tmp_path)
    assert spec.version == 2
    assert spec.project == "Demo"
    assert "quality" in spec.rulesets
    assert "sample-rule" in spec.rules
    surface = spec.surfaces["repo"]
    assert surface.include == ["src/**"]
    assert surface.exclude == ["tests/**"]
    assert surface.rules == ["sample-rule"]


def test_get_all_rules_builds_and_validates(
    aligned_spec: DriftGuardSpec,
) -> None:
    """Rules instantiated from the spec inherit severity and options."""

    rules = list(get_all_rules(aligned_spec))
    assert len(rules) == 1
    rule = rules[0]
    assert rule.rule_id == "dummy-aligned"
    assert rule.severity == "hard"
    assert isinstance(rule, DummyAlignedRule)
    assert rule.marker == "configured"


def test_get_all_rules_rejects_scope_or_mode_mismatch() -> None:
    """Detect when rule implementations diverge from the spec declaration."""

    ruleset = RuleSetSpec(name="core", severity="hard")
    rule_config = RuleConfig(
        rule_id="dummy-mismatch",
        impl="tests.test_driftguard_spec:DummyMismatchedRule",
        ruleset="core",
        run_modes=("fast",),
        scopes=("repo",),
    )
    surface = RuleSurface(
        name="repo", include=["**/*"], exclude=[], rules=["dummy-mismatch"]
    )
    spec = DriftGuardSpec(
        version=1,
        project="Test",
        rulesets={"core": ruleset},
        rules={"dummy-mismatch": rule_config},
        surfaces={"repo": surface},
    )
    with pytest.raises(ValueError):
        list(get_all_rules(spec))
