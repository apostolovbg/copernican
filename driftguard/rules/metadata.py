# Last Updated: 2025-11-25
"""Metadata-related DriftGuard rules.

The rule bodies remain TODO placeholders until the policy spec is finalised.
"""

from __future__ import annotations

from typing import List

from driftguard.rules import Rule, RuleContext, Violation


class LastUpdatedDocsRule(Rule):
    """Enforce ``Last Updated`` headers on documentation surfaces."""

    name = "last-updated-header"

    def check(self, context: RuleContext) -> List[Violation]:
        _ = context
        return []


class VersionSyncRule(Rule):
    """Keep version markers aligned across required files."""

    name = "version-sync"

    def check(self, context: RuleContext) -> List[Violation]:
        _ = context
        return []


class NoFutureDatesRule(Rule):
    """Guard against future-dated metadata entries."""

    name = "no-future-dates"

    def check(self, context: RuleContext) -> List[Violation]:
        _ = context
        return []


class CitationYamlRule(Rule):
    """Validate citation metadata against the policy spec."""

    name = "citation-yaml"

    def check(self, context: RuleContext) -> List[Violation]:
        _ = context
        return []


class ChangelogRule(Rule):
    """Ensure changelog entries accompany code and policy changes."""

    name = "changelog-entry"

    def check(self, context: RuleContext) -> List[Violation]:
        _ = context
        return []
