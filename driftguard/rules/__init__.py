# Last Updated: 2025-11-25
"""Base rule definitions for DriftGuard.

The rule interfaces are intentionally small so downstream projects can plug in
custom implementations without depending on Copernican internals. Rule loading
is spec-driven to keep the engine predictable and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from driftguard.spec import DriftGuardSpec, RuleConfig


@dataclass
class RuleContext:
    """Execution context passed to rules."""

    repo_root: Path
    spec: DriftGuardSpec
    scope: str
    mode: str


@dataclass
class Violation:
    """A failed rule condition."""

    rule_id: str
    message: str
    severity: str = "hard"
    path: Path | None = None


@dataclass
class Metric:
    """A drift metric reported by a rule."""

    name: str
    value: float | int
    details: str = ""


class Rule:
    """Base class for DriftGuard rules."""

    rule_id: str = "generic"
    severity: str = "hard"
    run_modes: Sequence[str] = ("fast", "full")
    scopes: Sequence[str] = ("repo", "staged")
    can_fix: bool = False
    safe_fix: bool = True

    def supports_mode(self, mode: str) -> bool:
        return mode in {item.lower() for item in self.run_modes}

    def supports_scope(self, scope: str) -> bool:
        return scope in {item.lower() for item in self.scopes}

    def check(
        self, context: RuleContext
    ) -> Tuple[List[Violation], List[Metric]]:
        return [], []

    def fix(self, context: RuleContext, safe_only: bool) -> List[str]:
        return []


def get_all_rules(spec: DriftGuardSpec) -> Iterable[Rule]:
    """Return an iterable of rule instances for the provided spec.

    Rules are instantiated based on the configured surfaces and validated to
    ensure they advertise the scopes and modes declared by the YAML spec. This
    prevents rules from silently skipping a configured surface or running
    against an unintended mode.
    """

    def _load_rule(rule_config: RuleConfig) -> Rule:
        if not rule_config.impl:
            raise ValueError(
                "Rule "
                f"'{rule_config.rule_id}' is missing an implementation path."
            )
        if ":" in rule_config.impl:
            module_path, class_name = rule_config.impl.split(":", 1)
        else:
            module_path, class_name = rule_config.impl.rsplit(".", 1)
        module = import_module(module_path)
        rule_cls = getattr(module, class_name)
        rule_instance = rule_cls(**rule_config.options)
        if not isinstance(rule_instance, Rule):
            raise TypeError(
                f"Loaded rule '{rule_config.rule_id}' is not a Rule subclass."
            )
        if getattr(rule_instance, "rule_id", None) in {None, ""}:
            rule_instance.rule_id = rule_config.rule_id
        rule_instance.severity = spec.rulesets[rule_config.ruleset].severity
        return rule_instance

    def _normalized(values: Iterable[str]) -> set[str]:
        return {value.lower() for value in values}

    def _validate_alignment(rule: Rule, rule_config: RuleConfig) -> None:
        expected_modes = _normalized(rule_config.run_modes)
        expected_scopes = _normalized(rule_config.scopes)
        actual_modes = _normalized(rule.run_modes)
        actual_scopes = _normalized(rule.scopes)
        if expected_modes != actual_modes:
            declared = sorted(expected_modes)
            reported = sorted(actual_modes)
            raise ValueError(
                "Rule "
                f"'{rule_config.rule_id}' declares modes {declared} but "
                f"implementation reports {reported}."
            )
        if expected_scopes != actual_scopes:
            declared = sorted(expected_scopes)
            reported = sorted(actual_scopes)
            raise ValueError(
                "Rule "
                f"'{rule_config.rule_id}' declares scopes {declared} but "
                f"implementation reports {reported}."
            )

    configured_rule_ids: set[str] = set()
    for surface in spec.surfaces.values():
        if surface.rules is None:
            configured_rule_ids.update(spec.rules.keys())
        else:
            configured_rule_ids.update(surface.rules)

    rules: Dict[str, Rule] = {}
    for rule_id in sorted(configured_rule_ids):
        if rule_id not in spec.rules:
            raise ValueError(
                "Surface references unknown rule "
                f"'{rule_id}' not present in the spec rules section."
            )
        rule_config = spec.rules[rule_id]
        rule_instance = _load_rule(rule_config)
        _validate_alignment(rule_instance, rule_config)
        rules[rule_id] = rule_instance

    return list(rules.values())
