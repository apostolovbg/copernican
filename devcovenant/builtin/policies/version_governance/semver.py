"""SemVer adapter for the version-governance policy."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from devcovenant.core.policy_contract import Violation

if TYPE_CHECKING:
    from .version_governance import (
        VersionGovernanceCheck,
        VersionReleaseContext,
    )

try:
    from semver import VersionInfo
except ImportError:  # pragma: no cover - dependency not installed
    VersionInfo = None  # type: ignore[assignment]

_SEMVER_TAG_RE = re.compile(r"\[semver:(major|minor|patch)\]", re.IGNORECASE)
_LEVELS = {"patch": 0, "minor": 1, "major": 2}
_LEVEL_NAMES = {value: key for key, value in _LEVELS.items()}


class SemverScheme:
    """Handle SemVer parsing, comparison, and scope-tag validation."""

    name = "semver"

    def preflight(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
        version_path: Path,
    ) -> List[Violation]:
        """Validate that the SemVer runtime dependency is available."""
        del repo_root
        if VersionInfo is not None:
            return []
        return [
            Violation(
                policy_id=check.policy_id,
                severity="error",
                file_path=version_path,
                message=(
                    "SemVer runtime dependency missing; install `semver` so "
                    "version-governance can parse semver strings."
                ),
            )
        ]

    def version_pattern(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str:
        """Return the changelog header regex fragment for SemVer."""
        del check, repo_root
        return r"\d+\.\d+\.\d+"

    def parse_version(
        self,
        value: str,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> VersionInfo:
        """Parse one SemVer token into a comparable VersionInfo."""
        del check, repo_root
        if VersionInfo is None:
            raise ValueError("SemVer runtime dependency missing")
        return VersionInfo.parse(str(value or "").strip())

    def compare_versions(self, left: VersionInfo, right: VersionInfo) -> int:
        """Compare two parsed SemVer values."""
        if left < right:
            return -1
        if left > right:
            return 1
        return 0

    def canonicalize_version(
        self,
        parsed: VersionInfo,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str:
        """Return the canonical SemVer spelling for one parsed version."""
        del check, repo_root
        return str(parsed)

    def validate_progression(
        self,
        check: "VersionGovernanceCheck",
        release: "VersionReleaseContext",
    ) -> List[Violation]:
        """Apply SemVer-specific bump progression validation."""
        del check
        actual_level = self._compute_bump_level(
            release.previous_parsed,
            release.current_parsed,
        )
        if actual_level is None:
            return [
                Violation(
                    policy_id=release.policy_id,
                    severity="error",
                    file_path=release.version_path,
                    message=(
                        "Version bump must update one SemVer component "
                        "rather than skipping backwards or repeating a "
                        "stored value."
                    ),
                )
            ]
        return []

    def validate_release(
        self,
        check: "VersionGovernanceCheck",
        release: "VersionReleaseContext",
    ) -> List[Violation]:
        """Apply SemVer-specific scope-marker validation."""
        if not check._bool_option("semver_scope_tags_required"):
            return []

        required_level, marker_levels = self._determine_required_level(
            release.latest_block,
        )
        if required_level is None:
            return [
                Violation(
                    policy_id=release.policy_id,
                    severity="error",
                    file_path=release.changelog_path,
                    message=(
                        "Add at least one `[semver:patch|minor|major]` tag "
                        "to the latest changelog entry."
                    ),
                )
            ]

        if (
            release.changelog_path in release.changed_files
            and release.version_path not in release.changed_files
            and marker_levels
        ):
            return [
                Violation(
                    policy_id=release.policy_id,
                    severity="error",
                    file_path=release.version_path,
                    message=(
                        "CHANGELOG declares a release scope but "
                        f"{release.version_label} was not updated; bump the "
                        "version file alongside the changelog entry."
                    ),
                )
            ]

        unique_levels = set(marker_levels)
        if len(unique_levels) > 1:
            return [
                Violation(
                    policy_id=release.policy_id,
                    severity="error",
                    file_path=release.changelog_path,
                    message=(
                        "Latest changelog entry mixes multiple SemVer "
                        "scopes; use a single explicit level per release."
                    ),
                )
            ]

        actual_level = self._compute_bump_level(
            release.previous_parsed,
            release.current_parsed,
        )
        if actual_level is None:
            return [
                Violation(
                    policy_id=release.policy_id,
                    severity="error",
                    file_path=release.version_path,
                    message=(
                        "Version bump must update one SemVer component "
                        "rather than skipping backwards or repeating a "
                        "stored value."
                    ),
                )
            ]

        if actual_level == required_level:
            return []

        required_name = _LEVEL_NAMES[required_level]
        actual_name = _LEVEL_NAMES[actual_level]
        return [
            Violation(
                policy_id=release.policy_id,
                severity="error",
                file_path=release.version_path,
                message=(
                    "Changelog tags demand a "
                    f"{required_name} bump but {release.version_label} is "
                    f"recorded as a {actual_name} change."
                ),
            )
        ]

    def _determine_required_level(
        self,
        latest_block: str,
    ) -> Tuple[Optional[int], List[int]]:
        """Return the SemVer level required by the latest changelog block."""
        markers = _SEMVER_TAG_RE.findall(latest_block)
        if not markers:
            return None, []
        levels = [_LEVELS[marker.lower()] for marker in markers]
        return max(levels), levels

    def _compute_bump_level(
        self,
        previous: VersionInfo,
        current: VersionInfo,
    ) -> Optional[int]:
        """Compute which SemVer component changed between two versions."""
        if current.major > previous.major:
            return _LEVELS["major"]
        if current.major < previous.major:
            return None

        if current.minor > previous.minor:
            return _LEVELS["minor"]
        if current.minor < previous.minor:
            return None

        if current.patch > previous.patch:
            return _LEVELS["patch"]
        return None
