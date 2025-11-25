# Last Updated: 2025-11-25
"""Specification loader for DriftGuard policy definitions.

The policy format is intentionally lightweight and YAML-based so it can be
updated alongside repository metadata without recompiling any code. Only the
filesystem and the spec contents are consulted; no Copernican-specific modules
are imported here so the implementation can be lifted into a standalone
package later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import yaml

from driftguard.utils import resolve_repo_root

DEFAULT_SCOPES = ("repo", "staged")
DEFAULT_MODES = ("fast", "full")


@dataclass(frozen=True)
class RuleSetSpec:
    """Describe a ruleset with its default severity and coverage modes."""

    name: str
    severity: str
    modes: Iterable[str] = DEFAULT_MODES
    scopes: Iterable[str] = DEFAULT_SCOPES


@dataclass(frozen=True)
class RuleConfig:
    """Detailed configuration for a single rule implementation."""

    rule_id: str
    impl: str
    ruleset: str
    run_modes: Iterable[str] = DEFAULT_MODES
    scopes: Iterable[str] = DEFAULT_SCOPES
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleSurface:
    """Describe a set of files or directories targeted by rules."""

    name: str
    include: List[str]
    exclude: List[str] = field(default_factory=list)
    rules: Optional[List[str]] = None


@dataclass(frozen=True)
class DriftGuardSpec:
    """Container for the policy specification."""

    version: int
    project: str
    rulesets: Mapping[str, RuleSetSpec]
    rules: Mapping[str, RuleConfig]
    surfaces: Mapping[str, RuleSurface]


def _load_yaml(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_rulesets(
    raw_rulesets: Mapping[str, object]
) -> Dict[str, RuleSetSpec]:
    """Parse ruleset definitions while preserving default coverage flags."""

    parsed: Dict[str, RuleSetSpec] = {}
    for name, raw_ruleset in raw_rulesets.items():
        severity = (
            str(raw_ruleset) if not isinstance(raw_ruleset, dict) else None
        )
        if isinstance(raw_ruleset, dict):
            severity = str(raw_ruleset.get("severity", "hard"))
            modes = tuple(raw_ruleset.get("modes", DEFAULT_MODES))
            scopes = tuple(raw_ruleset.get("scopes", DEFAULT_SCOPES))
        else:
            modes = DEFAULT_MODES
            scopes = DEFAULT_SCOPES
        parsed[name] = RuleSetSpec(
            name=name,
            severity=severity or "hard",
            modes=modes,
            scopes=scopes,
        )
    return parsed


def _parse_rules(raw_rules: Mapping[str, object]) -> Dict[str, RuleConfig]:
    """Parse rule entries into structured configurations."""

    parsed: Dict[str, RuleConfig] = {}
    for rule_id, raw_rule in raw_rules.items():
        rule_data = raw_rule or {}
        impl = str(rule_data.get("impl", ""))
        ruleset = str(rule_data.get("ruleset", ""))
        options = rule_data.get("options", {})
        run_modes = tuple(rule_data.get("modes", DEFAULT_MODES))
        scopes = tuple(rule_data.get("scopes", DEFAULT_SCOPES))
        parsed[rule_id] = RuleConfig(
            rule_id=rule_id,
            impl=impl,
            ruleset=ruleset,
            run_modes=run_modes,
            scopes=scopes,
            options=options,
        )
    return parsed


def _parse_surfaces(
    raw_surfaces: Mapping[str, object]
) -> Dict[str, RuleSurface]:
    """Parse surface definitions and normalise optional rule lists."""

    parsed: Dict[str, RuleSurface] = {}
    for name, raw_surface in raw_surfaces.items():
        surface_data = raw_surface or {}
        include = list(surface_data.get("include", []))
        exclude = list(surface_data.get("exclude", []))
        rules = surface_data.get("rules")
        parsed[name] = RuleSurface(
            name=name,
            include=include,
            exclude=exclude,
            rules=list(rules) if rules is not None else None,
        )
    return parsed


def _validate_rulesets(
    rules: Mapping[str, RuleConfig], rulesets: Mapping[str, RuleSetSpec]
) -> None:
    """Ensure every rule points at a declared ruleset."""

    for rule in rules.values():
        if rule.ruleset not in rulesets:
            raise ValueError(
                f"Rule '{rule.rule_id}' references unknown ruleset "
                f"'{rule.ruleset}'."
            )


def _validate_surfaces(
    surfaces: Mapping[str, RuleSurface], rules: Mapping[str, RuleConfig]
) -> None:
    """Ensure surface definitions and rule scopes remain aligned."""

    if not surfaces:
        raise ValueError(
            "At least one surface must be defined in driftguard.yml."
        )
    defined_scopes = set(surfaces.keys())
    for rule in rules.values():
        invalid_scopes = set(map(str.lower, rule.scopes)) - set(
            map(str.lower, defined_scopes)
        )
        if invalid_scopes:
            raise ValueError(
                f"Rule '{rule.rule_id}' declares unsupported scopes: "
                f"{sorted(invalid_scopes)}."
            )
    for surface in surfaces.values():
        if surface.rules is None:
            continue
        for rule_id in surface.rules:
            if rule_id not in rules:
                raise ValueError(
                    f"Surface '{surface.name}' references unknown rule "
                    f"'{rule_id}'."
                )
            rule = rules[rule_id]
            scoped_rules = {scope.lower() for scope in rule.scopes}
            if surface.name.lower() not in scoped_rules:
                raise ValueError(
                    "Rule "
                    f"'{rule_id}' does not declare scope '{surface.name}' "
                    "listed by the surface."
                )


def _load_spec_data(repo_root: Path) -> MutableMapping[str, object]:
    """Read the specification YAML file from the resolved repository root."""

    spec_path = repo_root / "driftguard.yml"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"Unable to find driftguard.yml at {spec_path.as_posix()}"
        )
    return _load_yaml(spec_path)


def load_spec(repo_root: Optional[Path | str] = None) -> DriftGuardSpec:
    """Load and parse the DriftGuard YAML specification.

    Parameters
    ----------
    repo_root:
        Optional repository root path. When omitted, the root is resolved
        relative to the DriftGuard package location so the helper works from
        a checked-out repository without additional configuration.

    Returns
    -------
    DriftGuardSpec
        Parsed specification instance ready for engine construction.
    """

    resolved_root = resolve_repo_root(repo_root)
    data = _load_spec_data(resolved_root)
    rulesets = _parse_rulesets(data.get("rulesets", {}))
    rules = _parse_rules(data.get("rules", {}))
    surfaces = _parse_surfaces(data.get("surfaces", {}))
    _validate_rulesets(rules, rulesets)
    _validate_surfaces(surfaces, rules)
    return DriftGuardSpec(
        version=int(data.get("version", 1)),
        project=str(data.get("project", "")),
        rulesets=rulesets,
        rules=rules,
        surfaces=surfaces,
    )
