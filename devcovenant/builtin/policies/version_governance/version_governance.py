"""Validate repository version format and scheme-specific bump rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Protocol

import devcovenant.core.policy_metadata as metadata_runtime
from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)
from devcovenant.core.policy_registry import load_policy_descriptor

_DEFAULT_CHANGELOG_HEADER_PREFIX = "## Version"
_LOG_MARKER = "## Log changes here"
_MANAGED_BEGIN = "<!-- DEVCOV:BEGIN -->"
_MANAGED_END = "<!-- DEVCOV:END -->"


@dataclass(frozen=True)
class VersionReleaseContext:
    """Carry shared release-state inputs into one scheme adapter."""

    repo_root: Path
    policy_id: str
    version_label: str
    version_path: Path
    changelog_path: Path
    changed_files: list[Path]
    latest_block: str
    current_version: str
    current_parsed: Any
    previous_version: str
    previous_parsed: Any


class VersionScheme(Protocol):
    """Common interface for one version-governance scheme adapter."""

    name: str

    def preflight(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
        version_path: Path,
    ) -> List[Violation]:
        """Validate scheme runtime prerequisites before parsing versions."""

    def version_pattern(
        self,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str:
        """Return one regex fragment for changelog version headers."""

    def parse_version(
        self,
        value: str,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> Any:
        """Parse one version string into a comparable scheme value."""

    def compare_versions(self, left: Any, right: Any) -> int:
        """Compare two parsed version values for ordering."""

    def canonicalize_version(
        self,
        parsed: Any,
        check: "VersionGovernanceCheck",
        repo_root: Path,
    ) -> str | None:
        """Return one canonical string form when the scheme defines one."""

    def validate_progression(
        self,
        check: "VersionGovernanceCheck",
        release: VersionReleaseContext,
    ) -> List[Violation]:
        """Return scheme-specific progression governance violations."""

    def validate_release(
        self,
        check: "VersionGovernanceCheck",
        release: VersionReleaseContext,
    ) -> List[Violation]:
        """Return scheme-specific release governance violations."""


def _scheme_registry() -> dict[str, VersionScheme]:
    """Return the builtin scheme adapter registry."""
    from .calver import CalverScheme
    from .custom_adapter import CustomAdapterScheme
    from .custom_regex import CustomRegexScheme
    from .integer import IntegerScheme
    from .pep440 import Pep440Scheme
    from .semver import SemverScheme

    schemes: list[VersionScheme] = [
        SemverScheme(),
        CalverScheme(),
        IntegerScheme(),
        Pep440Scheme(),
        CustomRegexScheme(),
        CustomAdapterScheme(),
    ]
    return {scheme.name: scheme for scheme in schemes}


def resolve_named_scheme(scheme_name: str) -> VersionScheme:
    """Return one named version-governance scheme adapter."""
    token = str(scheme_name or "").strip()
    if not token:
        raise ValueError("Version-governance scheme name cannot be empty.")
    scheme = _scheme_registry().get(token)
    if scheme is None:
        raise ValueError(
            "Unsupported version-governance scheme "
            f"`{token}` configured for this repository."
        )
    return scheme


def resolve_runtime_check(
    repo_root: Path,
    *,
    config_payload: Mapping[str, Any] | None = None,
) -> "VersionGovernanceCheck":
    """Return a version-governance checker resolved for one repo runtime."""
    repo_root = Path(repo_root).resolve()
    descriptor = load_policy_descriptor(repo_root, "version-governance")
    if descriptor is None:
        raise ValueError("Missing `version-governance` policy descriptor.")

    payload = dict(config_payload or {})
    metadata_context = metadata_runtime.build_metadata_context_from_payload(
        repo_root,
        payload,
    )
    current_order, current_values = (
        metadata_runtime.descriptor_metadata_order_values(descriptor)
    )
    bundle = metadata_runtime.resolve_policy_metadata_bundle(
        "version-governance",
        current_order,
        current_values,
        descriptor,
        metadata_context,
    )
    checker = VersionGovernanceCheck()
    config_context = CheckContext(repo_root=repo_root, config=payload)
    checker.set_options(
        bundle.decode_options(),
        config_context.get_policy_config("version-governance"),
    )
    return checker


def resolve_runtime_scheme(
    repo_root: Path,
    *,
    config_payload: Mapping[str, Any] | None = None,
) -> tuple[str, VersionScheme, "VersionGovernanceCheck"]:
    """Return the active version scheme adapter for one repo runtime."""
    checker = resolve_runtime_check(
        repo_root,
        config_payload=config_payload,
    )
    scheme_name = checker._scheme_name()
    if not scheme_name:
        raise ValueError(
            "Configure `version-governance.scheme` explicitly before "
            "using version-governance or version-sync."
        )
    scheme = resolve_named_scheme(scheme_name)
    return scheme_name, scheme, checker


class VersionGovernanceCheck(PolicyCheck):
    """Check version format and optional scheme-specific bump progression."""

    policy_id = "version-governance"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Validate version-file and changelog version governance rules."""
        violations: List[Violation] = []
        repo_root = context.repo_root
        version_rel = Path(self.get_option("version_file", "VERSION"))
        changelog_rel = Path(self.get_option("changelog_file", "CHANGELOG.md"))
        version_path = repo_root / version_rel
        changelog_path = repo_root / changelog_rel
        version_label = version_rel.as_posix()
        scheme_name = self._scheme_name()
        ignored_prefixes = self._ignored_prefixes()
        enforce_bumping = self._bool_option("enforce_bumping")
        require_canonical = self._bool_option("canonical_versions_required")

        if not scheme_name:
            return [
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        "Configure `version-governance.scheme` explicitly "
                        "before enabling version-governance."
                    ),
                )
            ]

        if not version_path.exists():
            return violations

        changed_files = context.changed_files or []
        if not changed_files:
            return violations

        should_check = (
            version_path in changed_files or changelog_path in changed_files
        )
        if not should_check:
            return violations

        if not self._has_relevant_changes(
            changed_files,
            repo_root,
            version_path,
            changelog_path,
            ignored_prefixes,
        ):
            return violations

        registry = _scheme_registry()
        scheme = registry.get(scheme_name)
        if scheme is None:
            supported = ", ".join(sorted(registry))
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        "Unsupported version-governance scheme "
                        f"`{scheme_name}`. Supported schemes: {supported}."
                    ),
                )
            )
            return violations

        violations.extend(scheme.preflight(self, repo_root, version_path))
        if violations:
            return violations

        try:
            current_version = version_path.read_text(encoding="utf-8").strip()
            current_parsed = scheme.parse_version(
                current_version,
                self,
                repo_root,
            )
        except (OSError, ValueError) as exc:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        f"Cannot read or parse {version_label} as "
                        f"{scheme_name}: {exc}"
                    ),
                )
            )
            return violations

        if require_canonical:
            canonical = scheme.canonicalize_version(
                current_parsed,
                self,
                repo_root,
            )
            if canonical and canonical != current_version:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=version_path,
                        message=(
                            f"{version_label} must use the canonical "
                            f"`{scheme_name}` spelling `{canonical}` rather "
                            f"than `{current_version}`."
                        ),
                    )
                )
                return violations

        try:
            changelog_text = changelog_path.read_text(encoding="utf-8")
        except OSError as exc:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=changelog_path,
                    message=f"Cannot read {changelog_rel.name}: {exc}",
                )
            )
            return violations

        latest_block, versions = self._extract_version_block(
            changelog_text,
            self._version_header_re(scheme, repo_root),
        )
        if not versions:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=changelog_path,
                    message=(
                        f"{changelog_rel.name} does not contain any "
                        "version headers."
                    ),
                )
            )
            return violations

        latest_recorded = versions[0]
        if latest_recorded != current_version:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=changelog_path,
                    message=(
                        "Top changelog entry "
                        f"({latest_recorded}) does not match {version_label} "
                        f"({current_version})."
                    ),
                )
            )
            return violations

        previous_version = versions[1] if len(versions) > 1 else None
        if previous_version is None:
            return violations

        try:
            previous_parsed = scheme.parse_version(
                previous_version,
                self,
                repo_root,
            )
        except ValueError:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=changelog_path,
                    message=(
                        "Previous changelog version "
                        f"`{previous_version}` is not valid for scheme "
                        f"`{scheme_name}`."
                    ),
                )
            )
            return violations

        if not enforce_bumping:
            return violations

        try:
            comparison = scheme.compare_versions(
                previous_parsed,
                current_parsed,
            )
        except ValueError as exc:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        "Cannot compare version progression under scheme "
                        f"`{scheme_name}`: {exc}"
                    ),
                )
            )
            return violations

        if comparison >= 0:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        f"Version {current_version} must be greater than "
                        f"{previous_version} under scheme `{scheme_name}`."
                    ),
                )
            )
            return violations

        release = VersionReleaseContext(
            repo_root=repo_root,
            policy_id=self.policy_id,
            version_label=version_label,
            version_path=version_path,
            changelog_path=changelog_path,
            changed_files=changed_files,
            latest_block=latest_block or "",
            current_version=current_version,
            current_parsed=current_parsed,
            previous_version=previous_version,
            previous_parsed=previous_parsed,
        )
        try:
            violations.extend(scheme.validate_progression(self, release))
            if violations:
                return violations
            violations.extend(scheme.validate_release(self, release))
            return violations
        except ValueError as exc:
            return [
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=changelog_path,
                    message=(
                        "Cannot validate scheme-specific release rules under "
                        f"`{scheme_name}`: {exc}"
                    ),
                )
            ]

    def _scheme_name(self) -> str:
        """Return normalized scheme token for version governance."""
        raw = str(self.get_option("scheme", "")).strip().lower()
        if raw == "int":
            return "integer"
        return raw

    def _bool_option(self, key: str) -> bool:
        """Return one metadata option normalized as a boolean flag."""
        raw = self.get_option(key, False)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _ignored_prefixes(self) -> tuple[str, ...]:
        """Return ignored repository-root prefixes for relevance filtering."""
        prefixes_option = self.get_option("ignored_prefixes", [])
        if isinstance(prefixes_option, str):
            return (prefixes_option,) if prefixes_option else ()
        return tuple(prefixes_option or ())

    def _version_header_re(
        self,
        scheme: VersionScheme,
        repo_root: Path,
    ) -> re.Pattern[str]:
        """Return regex for changelog version headers under one scheme."""
        prefix = str(
            self.get_option(
                "changelog_header_prefix",
                _DEFAULT_CHANGELOG_HEADER_PREFIX,
            )
        ).strip()
        if not prefix:
            prefix = _DEFAULT_CHANGELOG_HEADER_PREFIX
        pattern = scheme.version_pattern(self, repo_root)
        return re.compile(
            rf"^{re.escape(prefix)}\s+(?P<version>{pattern})\s*$",
            re.MULTILINE,
        )

    def _has_relevant_changes(
        self,
        changed_files: List[Path],
        repo_root: Path,
        version_path: Path,
        changelog_path: Path,
        ignored_prefixes: tuple[str, ...],
    ) -> bool:
        """Return True when files outside the ignored prefixes changed."""
        for path in changed_files:
            if path == changelog_path:
                return True
            if path == version_path:
                continue
            try:
                rel = path.relative_to(repo_root)
            except ValueError:
                rel = path
            parts = rel.parts
            if parts and parts[0] in ignored_prefixes:
                continue
            return True
        return False

    def _extract_version_block(
        self,
        changelog_text: str,
        version_header_re: re.Pattern[str],
    ) -> tuple[Optional[str], List[str]]:
        """Return the latest version block and the ordered version list."""
        search_text = self._searchable_changelog(changelog_text)
        matches = list(version_header_re.finditer(search_text))
        versions = [match.group("version") for match in matches]
        if not matches:
            return None, versions

        latest = matches[0]
        start = latest.start()
        next_start = (
            matches[1].start() if len(matches) > 1 else len(search_text)
        )
        block = search_text[start:next_start]
        return block, versions

    def _searchable_changelog(self, changelog_text: str) -> str:
        """Return changelog text outside managed blocks and fenced examples."""
        start = changelog_text.find(_LOG_MARKER)
        if start >= 0:
            content = changelog_text[start:]
        else:
            content = changelog_text

        kept_lines: list[str] = []
        in_managed = False
        in_fence = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped == _MANAGED_BEGIN:
                in_managed = True
                continue
            if stripped == _MANAGED_END:
                in_managed = False
                continue
            if in_managed:
                continue
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines)
