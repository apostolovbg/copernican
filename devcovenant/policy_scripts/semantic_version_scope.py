"""Ensure `version_file` bumps align with the SemVer tags in
`changelog_file`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from devcovenant.base import CheckContext, PolicyCheck, Violation

try:
    from semver import VersionInfo
except ImportError:  # pragma: no cover - dependency not installed
    VersionInfo = None  # type: ignore[assignment]


_SEMVER_TAG_RE = re.compile(r"\[semver:(major|minor|patch)\]", re.IGNORECASE)
_VERSION_HEADER_RE = re.compile(
    r"^##\s+Version\s+(\d+\.\d+\.\d+)", re.MULTILINE
)
_LEVELS = {"patch": 0, "minor": 1, "major": 2}
_LEVEL_NAMES = {value: key for key, value in _LEVELS.items()}


class SemanticVersionScopeCheck(PolicyCheck):
    """Check that version bumps honor the declared SemVer scope."""

    policy_id = "semantic-version-scope"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Validate the latest changelog tags against the version bump."""
        violations: List[Violation] = []
        repo_root = context.repo_root
        version_rel = Path(self.get_option("version_file", "VERSION"))
        changelog_rel = Path(self.get_option("changelog_file", "CHANGELOG.md"))
        override_rel = Path(
            self.get_option(
                "override_file",
                ".devcovenant/waivers/semantic-version-scope.txt",
            )
        )
        prefixes_option = self.get_option("ignored_prefixes", [])
        if isinstance(prefixes_option, str):
            ignored_prefixes = (prefixes_option,)
        else:
            ignored_prefixes = tuple(prefixes_option or ())
        version_path = repo_root / version_rel
        changelog_path = repo_root / changelog_rel
        version_label = version_rel.as_posix()
        override_label = override_rel.as_posix()

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

        if VersionInfo is None:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        "SemVer runtime dependency missing; install `semver` "
                        "so semantic-version-scope can parse version strings."
                    ),
                )
            )
            return violations

        try:
            current_version = version_path.read_text(encoding="utf-8").strip()
            current_semver = VersionInfo.parse(current_version)
        except (OSError, ValueError) as exc:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=f"Cannot read or parse VERSION file: {exc}",
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
                    message=f"Cannot read CHANGELOG.md: {exc}",
                )
            )
            return violations

        latest_block, versions = self._extract_version_block(changelog_text)
        if not versions:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=changelog_path,
                    message=(
                        "CHANGELOG.md does not contain any version headers."
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
                        f"({latest_recorded}) does not match "
                        f"{version_label} ({current_version})."
                    ),
                )
            )
            return violations

        previous_version = versions[1] if len(versions) > 1 else None
        if previous_version is None:
            # Nothing to compare against (initial release).
            return violations

        try:
            previous_semver = VersionInfo.parse(previous_version)
        except ValueError:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=changelog_path,
                    message=(
                        "Previous changelog version "
                        f"'{previous_version}' is not valid SemVer."
                    ),
                )
            )
            return violations

        if current_semver <= previous_semver:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        f"Version {current_semver} must be greater than "
                        f"{previous_semver}."
                    ),
                )
            )
            return violations

        required_level, marker_levels = self._determine_required_level(
            repo_root, latest_block or "", override_rel
        )
        if required_level is None:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=changelog_path,
                    message=(
                        "Add at least one `[semver:patch|minor|major]` tag to "
                        "the latest changelog entry or specify "
                        f"`override=<level>` in `{override_label}`."
                    ),
                )
            )
            return violations

        if (
            changelog_path in changed_files
            and version_path not in changed_files
            and marker_levels
        ):
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        "CHANGELOG declares a release scope but "
                        f"{version_label} was not updated; bump the "
                        "version file alongside the changelog entry."
                    ),
                )
            )
            return violations

        unique_levels = set(marker_levels)
        if len(unique_levels) > 1:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=changelog_path,
                    message=(
                        "Latest changelog entry mixes multiple SemVer scopes; "
                        "use a single explicit level per release."
                    ),
                )
            )
            return violations

        actual_level = self._compute_bump_level(
            previous_semver, current_semver
        )
        if actual_level is None:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        "Version bump must update one SemVer component "
                        "rather than skipping backwards or repeating a stored "
                        "value."
                    ),
                )
            )
            return violations

        if actual_level != required_level:
            required_name = _LEVEL_NAMES[required_level]
            actual_name = _LEVEL_NAMES[actual_level]
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=version_path,
                    message=(
                        "Changelog tags demand a "
                        f"{required_name} bump but {version_label} is "
                        f"recorded as a {actual_name} change."
                    ),
                )
            )

        return violations

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
        self, changelog_text: str
    ) -> tuple[Optional[str], List[str]]:
        """Return the latest version block and the ordered version list."""
        matches = list(_VERSION_HEADER_RE.finditer(changelog_text))
        versions = [match.group(1) for match in matches]
        if not matches:
            return None, versions

        latest = matches[0]
        start = latest.start()
        next_start = (
            matches[1].start() if len(matches) > 1 else len(changelog_text)
        )
        block = changelog_text[start:next_start]
        return block, versions

    def _determine_required_level(
        self, repo_root: Path, latest_block: str, override_rel: Path
    ) -> Tuple[Optional[int], List[int]]:
        """Return the SemVer level required for this release."""
        override_path = repo_root / override_rel
        override_level = self._read_override(override_path)
        if override_level is not None:
            return override_level, [override_level]

        markers = _SEMVER_TAG_RE.findall(latest_block)
        if not markers:
            return None, []
        levels = [_LEVELS[marker.lower()] for marker in markers]
        return max(levels), levels

    def _read_override(self, override_path: Path) -> Optional[int]:
        """Parse the optional override file."""
        if not override_path.exists():
            return None
        try:
            text = override_path.read_text(encoding="utf-8")
            for raw_line in text.splitlines():
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" in stripped:
                    stripped = stripped.split("=", 1)[1].strip()
                level = _LEVELS.get(stripped.lower())
                if level is not None:
                    return level
        except OSError:
            return None
        return None

    def _compute_bump_level(
        self, previous: VersionInfo, current: VersionInfo
    ) -> Optional[int]:
        """Compute the SemVer component that changed."""
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
