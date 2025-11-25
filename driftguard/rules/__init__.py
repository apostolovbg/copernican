"""Rule and metric scaffolding for DriftGuard.

Concrete rule implementations live in sibling modules. This file hosts shared
interfaces so the engine can discover rule instances without tightly coupling
to rule modules or Copernican-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

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

    def fix(self, context: RuleContext) -> List[Violation]:
        """Attempt to auto-fix violations and return the resulting state."""

        _ = context
        return []


def get_all_rules(spec: DriftGuardSpec) -> List[Rule]:
    """Instantiate all rules referenced in ``spec``.

    TODO: map rule names to concrete classes once implementations exist.
    """

    _ = spec
    return []
