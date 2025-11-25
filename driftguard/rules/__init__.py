# Last Updated: 2025-11-25
"""Base rule definitions for DriftGuard.

The rule interfaces are intentionally small so downstream projects can plug in
custom implementations without depending on Copernican internals.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from driftguard.spec import DriftGuardSpec


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
    """Return an iterable of rule instances for the provided spec."""

    _ = spec
    return []
