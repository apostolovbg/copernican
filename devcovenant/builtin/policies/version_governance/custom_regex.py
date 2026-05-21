"""Custom regex adapter for the version-governance policy."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List

from devcovenant.core.policy_contract import Violation

if TYPE_CHECKING:
    from .version_governance import (
        VersionGovernanceCheck,
        VersionReleaseContext,
    )


class CustomRegexScheme:
    """Validate arbitrary version formats through one configured regex."""

    name = "custom_regex"

    def preflight(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
        version_path: Path,
    ) -> List[Violation]:
        """Require a valid regex pattern and disable ordered bump checks."""
        del repo_root
        pattern = str(check.get_option("custom_regex_pattern", "")).strip()
        if not pattern:
            return [
                Violation(
                    policy_id=check.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        "Set `custom_regex_pattern` when using the "
                        "`custom_regex` version-governance scheme."
                    ),
                )
            ]
        try:
            re.compile(pattern)
        except re.error as exc:
            return [
                Violation(
                    policy_id=check.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        "Configured `custom_regex_pattern` is not a valid "
                        f"regular expression: {exc}"
                    ),
                )
            ]
        if check._bool_option("enforce_bumping"):
            return [
                Violation(
                    policy_id=check.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        "`custom_regex` validates format only; disable "
                        "`enforce_bumping` or use `custom_adapter` for "
                        "ordered version progression."
                    ),
                )
            ]
        return []

    def version_pattern(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str:
        """Return the configured changelog-header regex fragment."""
        del repo_root
        return (
            str(check.get_option("custom_regex_pattern", "")).strip() or r".+"
        )

    def parse_version(
        self,
        value: str,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str:
        """Accept one version string only when it matches the regex fully."""
        del repo_root
        token = str(value or "").strip()
        pattern = self.version_pattern(check, Path("."))
        if not re.fullmatch(pattern, token):
            raise ValueError(
                f"`{token}` does not match `custom_regex_pattern`"
            )
        return token

    def compare_versions(self, left: str, right: str) -> int:
        """Reject ordered comparison because regex-only mode is format-only."""
        del left, right
        raise ValueError(
            "`custom_regex` does not define version ordering; disable "
            "`enforce_bumping` or use `custom_adapter`."
        )

    def canonicalize_version(
        self,
        parsed: str,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str | None:
        """Regex-only mode does not define a canonical spelling."""
        del parsed, check, repo_root
        return None

    def validate_progression(
        self,
        check: "VersionGovernanceCheck",
        release: "VersionReleaseContext",
    ) -> List:
        """Regex-only mode adds no progression rules."""
        del check, release
        return []

    def validate_release(
        self,
        check: "VersionGovernanceCheck",
        release: "VersionReleaseContext",
    ) -> List:
        """Custom regex mode does not add release-scope rules."""
        del check, release
        return []
