"""CalVer adapter for the version-governance policy."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .version_governance import (
        VersionGovernanceCheck,
        VersionReleaseContext,
    )

_DEFAULT_CALVER_PATTERN = r"\d{4}\.\d{1,2}(?:\.\d{1,2})?"


class CalverScheme:
    """Handle calendar-version parsing and forward ordering."""

    name = "calver"

    def preflight(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
        version_path: Path,
    ) -> list:
        """CalVer has no extra runtime prerequisites."""
        del check, repo_root, version_path
        return []

    def version_pattern(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str:
        """Return the changelog header regex fragment for CalVer."""
        del repo_root
        pattern = str(check.get_option("calver_pattern", "")).strip()
        return pattern or _DEFAULT_CALVER_PATTERN

    def parse_version(
        self,
        value: str,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> tuple[int, ...]:
        """Parse one CalVer token into comparable numeric segments."""
        del repo_root
        token = str(value or "").strip()
        pattern = self.version_pattern(check, Path("."))
        if not re.fullmatch(pattern, token):
            raise ValueError(f"`{token}` is not a valid calver version")
        digits = tuple(int(part) for part in re.findall(r"\d+", token))
        if not digits:
            raise ValueError(f"`{token}` does not contain comparable digits")
        return digits

    def compare_versions(
        self, left: tuple[int, ...], right: tuple[int, ...]
    ) -> int:
        """Compare two parsed CalVer values."""
        if left < right:
            return -1
        if left > right:
            return 1
        return 0

    def canonicalize_version(
        self,
        parsed: tuple[int, ...],
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str | None:
        """CalVer keeps the repository's chosen formatting, including
        padding."""
        del parsed, check, repo_root
        return None

    def validate_progression(
        self,
        check: "VersionGovernanceCheck",
        release: "VersionReleaseContext",
    ) -> List:
        """CalVer adds no extra progression rules beyond forward ordering."""
        return []

    def validate_release(
        self,
        check: "VersionGovernanceCheck",
        release: "VersionReleaseContext",
    ) -> List:
        """CalVer does not impose extra release-scope rules."""
        return []
