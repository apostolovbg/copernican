"""PEP 440 adapter for the version-governance policy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from .version_governance import (
        VersionGovernanceCheck,
        VersionReleaseContext,
    )


class Pep440Scheme:
    """Handle PEP 440 parsing and ordering for Python package versions."""

    name = "pep440"

    def preflight(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
        version_path: Path,
    ) -> list:
        """PEP 440 parsing uses the packaged `packaging` dependency."""
        del check, repo_root, version_path
        return []

    def version_pattern(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str:
        """Return a permissive token pattern for PEP 440 changelog headers."""
        del check, repo_root
        return r"[A-Za-z0-9!+._-]+"

    def parse_version(
        self,
        value: str,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> Version:
        """Parse one PEP 440 version string into a comparable Version."""
        del check, repo_root
        token = str(value or "").strip()
        try:
            return Version(token)
        except InvalidVersion as exc:
            raise ValueError(
                f"`{token}` is not a valid pep440 version"
            ) from exc

    def compare_versions(self, left: Version, right: Version) -> int:
        """Compare two parsed PEP 440 versions."""
        if (
            left.release == right.release
            and right.is_devrelease
            and not left.is_devrelease
        ):
            return -1
        if left < right:
            return -1
        if left > right:
            return 1
        return 0

    def canonicalize_version(
        self,
        parsed: Version,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str:
        """Return the canonical normalized PEP 440 spelling."""
        del check, repo_root
        return str(parsed)

    def validate_progression(
        self,
        check: "VersionGovernanceCheck",
        release: "VersionReleaseContext",
    ) -> List:
        """Apply PEP 440 marker governance for the recorded version."""
        violations = []
        current = release.current_parsed
        if current.is_prerelease and not self._bool_option(
            check,
            "pep440_allow_prereleases",
            default=True,
        ):
            violations.append(
                self._violation(
                    release,
                    "PEP 440 prerelease markers are disabled for this "
                    f"repository; {release.version_label} cannot use "
                    f"`{release.current_version}`.",
                )
            )
        if current.is_devrelease and not self._bool_option(
            check,
            "pep440_allow_dev_releases",
            default=True,
        ):
            violations.append(
                self._violation(
                    release,
                    "PEP 440 development-release markers are disabled for "
                    f"this repository; {release.version_label} cannot use "
                    f"`{release.current_version}`.",
                )
            )
        if current.is_postrelease and not self._bool_option(
            check,
            "pep440_allow_post_releases",
            default=True,
        ):
            violations.append(
                self._violation(
                    release,
                    "PEP 440 post-release markers are disabled for this "
                    f"repository; {release.version_label} cannot use "
                    f"`{release.current_version}`.",
                )
            )
        return violations

    def validate_release(
        self,
        check: "VersionGovernanceCheck",
        release: "VersionReleaseContext",
    ) -> List:
        """PEP 440 imposes no extra changelog-scope rules by default."""
        return []

    def _bool_option(
        self,
        check: "VersionGovernanceCheck",
        key: str,
        *,
        default: bool,
    ) -> bool:
        """Read one boolean option with an explicit scheme-level default."""
        raw = check.get_option(key, default)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _violation(
        self,
        release: "VersionReleaseContext",
        message: str,
    ):
        """Build one policy violation for a release-marker rule."""
        from devcovenant.core.policy_contract import Violation

        return Violation(
            policy_id=release.policy_id,
            severity="error",
            file_path=release.version_path,
            message=message,
        )
