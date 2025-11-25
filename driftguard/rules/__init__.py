# Last Updated: 2025-11-25
"""Rule and metric scaffolding for DriftGuard.

Concrete rule implementations live in sibling modules. This file hosts shared
interfaces so the engine can discover rule instances without tightly coupling
to rule modules or Copernican-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from driftguard.spec import DriftGuardSpec


@dataclass
class RuleContext:
    """Execution context shared across rules."""

    repo_root: Path
    spec: DriftGuardSpec
    scope: str
    mode: str


@dataclass
class Violation:
    """Describe a policy violation detected by a rule."""

    rule_name: str
    message: str
    path: Path | None = None
    fixable: bool = False


@dataclass
class Metric:
    """Describe a drift metric emitted by a rule."""

    name: str
    value: Any
    path: Path | None = None
    threshold: float | None = None


class Rule:
    """Base class for DriftGuard rules.

    Rule subclasses should override :meth:`check` and optionally :meth:`fix`.
    TODO: add structured severities and consistent autofix reporting.
    """

    name: str = ""

    def check(self, context: RuleContext) -> List[Violation]:
        """Inspect the repository and return any policy violations."""

        _ = context
        return []

    def fix(
        self, context: RuleContext, safe_only: bool = False
    ) -> List[Violation]:
        """Attempt to auto-fix violations and return the resulting state."""

        _ = context
        _ = safe_only
        return []


def get_all_rules(spec: DriftGuardSpec) -> List[Rule]:
    """Instantiate all rules referenced in ``spec``.

    The factory inspects the loaded :class:`~driftguard.spec.DriftGuardSpec`
    and instantiates each rule at most once. Rules are keyed by their
    canonical ``name`` attribute so the mapping remains declarative and easy
    to extend as new rules arrive.
    """

    from driftguard.rules.metadata import (  # Import locally to avoid cycles.
        ChangelogRule,
        CitationYamlRule,
        LastUpdatedDocsRule,
        NoFutureDatesRule,
        VersionSyncRule,
    )
    from driftguard.rules.python_lib import (
        BugfixHasTestRule,
        NewModulesNeedTestsRule,
        NoPrintInLibRule,
    )

    registry: Dict[str, Rule] = {
        LastUpdatedDocsRule.name: LastUpdatedDocsRule(),
        VersionSyncRule.name: VersionSyncRule(),
        NoFutureDatesRule.name: NoFutureDatesRule(),
        CitationYamlRule.name: CitationYamlRule(),
        ChangelogRule.name: ChangelogRule(),
        NoPrintInLibRule.name: NoPrintInLibRule(),
        NewModulesNeedTestsRule.name: NewModulesNeedTestsRule(),
        BugfixHasTestRule.name: BugfixHasTestRule(),
    }

    requested: List[str] = []
    for surface in spec.surfaces.values():
        requested.extend(surface.rules)

    unique_rules: List[Rule] = []
    for name in sorted(set(requested)):
        rule = registry.get(name)
        if rule is not None:
            unique_rules.append(rule)
    return unique_rules
