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

    def metrics(self, context: RuleContext) -> List[Metric]:
        """Collect drift metrics emitted by the rule, if any."""

        _ = context
        return []


def get_all_rules(spec: DriftGuardSpec) -> List[Rule]:
    """Instantiate all rules referenced in ``spec``.

    The factory inspects the loaded :class:`~driftguard.spec.DriftGuardSpec`
    and instantiates each rule at most once. Rules are keyed by their
    canonical ``name`` attribute so the mapping remains declarative and easy
    to extend as new rules arrive.
    """

    from driftguard.rules.drift import (
        DocAgeRule,
        TestCouplingRule,
        TodoCountRule,
    )
    from driftguard.rules.metadata import (  # Import locally to avoid cycles.
        ChangelogDiffCoverageRule,
        ChangelogRule,
        CitationYamlRule,
        DocumentationRefreshRule,
        HumanEditPreservationRule,
        LastUpdatedDocsRule,
        ManagedVenvOnlyRule,
        NoFutureDatesRule,
        SecurityComplianceRule,
        SemverBumpRequiredRule,
        StartLauncherParityRule,
        TimestampValidationRule,
        VersionSyncRule,
    )
    from driftguard.rules.python_lib import (
        BugfixHasTestRule,
        CommentsExplainWhyRule,
        DocstringsExplainWhyRule,
        LineLengthRule,
        NamingClearAndConciseRule,
        NewModulesNeedTestsRule,
        NoPrintInLibRule,
        RawStringEscapingRule,
        TestsForChangesRule,
    )
    from driftguard.rules.workflows import (
        DependencyLicenseAuditRule,
        DependencyRefreshRule,
        DriftGuardPrecommitRequiredRule,
        FormatterCleanRule,
        FullTestSuiteInCIRule,
    )

    registry: Dict[str, Rule] = {
        LastUpdatedDocsRule.name: LastUpdatedDocsRule(),
        TimestampValidationRule.name: TimestampValidationRule(),
        HumanEditPreservationRule.name: HumanEditPreservationRule(),
        DocumentationRefreshRule.name: DocumentationRefreshRule(),
        ChangelogDiffCoverageRule.name: ChangelogDiffCoverageRule(),
        VersionSyncRule.name: VersionSyncRule(),
        SemverBumpRequiredRule.name: SemverBumpRequiredRule(),
        NoFutureDatesRule.name: NoFutureDatesRule(),
        CitationYamlRule.name: CitationYamlRule(),
        ChangelogRule.name: ChangelogRule(),
        StartLauncherParityRule.name: StartLauncherParityRule(),
        ManagedVenvOnlyRule.name: ManagedVenvOnlyRule(),
        SecurityComplianceRule.name: SecurityComplianceRule(),
        NoPrintInLibRule.name: NoPrintInLibRule(),
        CommentsExplainWhyRule.name: CommentsExplainWhyRule(),
        DocstringsExplainWhyRule.name: DocstringsExplainWhyRule(),
        NamingClearAndConciseRule.name: NamingClearAndConciseRule(),
        LineLengthRule.name: LineLengthRule(),
        RawStringEscapingRule.name: RawStringEscapingRule(),
        NewModulesNeedTestsRule.name: NewModulesNeedTestsRule(),
        BugfixHasTestRule.name: BugfixHasTestRule(),
        TestsForChangesRule.name: TestsForChangesRule(),
        TodoCountRule.name: TodoCountRule(),
        TestCouplingRule.name: TestCouplingRule(),
        DocAgeRule.name: DocAgeRule(),
        FullTestSuiteInCIRule.name: FullTestSuiteInCIRule(),
        DriftGuardPrecommitRequiredRule.name: (
            DriftGuardPrecommitRequiredRule()
        ),
        DependencyLicenseAuditRule.name: DependencyLicenseAuditRule(),
        DependencyRefreshRule.name: DependencyRefreshRule(),
        FormatterCleanRule.name: FormatterCleanRule(),
    }

    requested: List[str] = []
    for surface in spec.surfaces.values():
        requested.extend(surface.rules)
    requested.extend(spec.drift.metrics.keys())

    unique_rules: List[Rule] = []
    for name in sorted(set(requested)):
        rule = registry.get(name)
        if rule is not None:
            unique_rules.append(rule)
    return unique_rules
