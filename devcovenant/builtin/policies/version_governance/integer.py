"""Integer-version adapter for the version-governance policy."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .version_governance import (
        VersionGovernanceCheck,
        VersionReleaseContext,
    )


class IntegerScheme:
    """Handle simple integer-version parsing and ordering."""

    name = "integer"

    def preflight(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
        version_path: Path,
    ) -> list:
        """Integer versioning has no extra runtime prerequisites."""
        del check, repo_root, version_path
        return []

    def version_pattern(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str:
        """Return the changelog header regex fragment for integer versions."""
        del check, repo_root
        return r"\d+"

    def parse_version(
        self,
        value: str,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> int:
        """Parse one integer version token into a numeric value."""
        del check, repo_root
        token = str(value or "").strip()
        if not re.fullmatch(r"\d+", token):
            raise ValueError(f"`{token}` is not a valid integer version")
        return int(token)

    def compare_versions(self, left: int, right: int) -> int:
        """Compare two parsed integer versions."""
        if left < right:
            return -1
        if left > right:
            return 1
        return 0

    def canonicalize_version(
        self,
        parsed: int,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str:
        """Return the canonical integer spelling with no padding."""
        del check, repo_root
        return str(parsed)

    def validate_progression(
        self,
        check: "VersionGovernanceCheck",
        release: "VersionReleaseContext",
    ) -> List:
        """Integer versioning adds no extra progression rules."""
        return []

    def validate_release(
        self,
        check: "VersionGovernanceCheck",
        release: "VersionReleaseContext",
    ) -> List:
        """Integer versioning does not impose extra release-scope rules."""
        return []
