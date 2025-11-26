"""Metadata-related DriftGuard rules.

The rules mirror Copernican's existing metadata policy while using the policy
YAML to determine which files to inspect. They deliberately favour clarity over
brevity so contributors can trace every decision back to the published policy
text.
"""

from __future__ import annotations

import datetime
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import yaml
from yaml import YAMLError

from driftguard.rules import Rule, RuleContext, Violation
from driftguard.utils import resolve_surface_globs

_LAST_UPDATED_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(
        r"^\*\*Last Updated:\*\*\s*(\d{4}-\d{2}-\d{2})\s*$",
        re.MULTILINE,
    ),
    re.compile(r"^# Last Updated:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE),
)


def _utc_today() -> datetime.date:
    """Return today's date in Coordinated Universal Time (UTC)."""

    return datetime.datetime.now(datetime.timezone.utc).date()


def _read_text(path: Path) -> str:
    """Read file contents using UTF-8 with consistent error handling."""

    return path.read_text(encoding="utf-8")


def _iter_surface_paths(
    context: RuleContext, surfaces: Iterable[str]
) -> List[Path]:
    """Resolve the requested surfaces to concrete paths in the repository."""

    paths: List[Path] = []
    for surface in surfaces:
        if surface not in context.spec.surfaces:
            continue
        paths.extend(
            resolve_surface_globs(context.spec, context.repo_root, surface)
        )
    return paths


def _git_status(repo_root: Path) -> List[Tuple[str, Path]]:
    """Return parsed git status entries for the working tree."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []

    entries: List[Tuple[str, Path]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        status, path_text = parts
        entries.append((status, repo_root / path_text))
    return entries


def _git_diff_names(repo_root: Path) -> List[Path]:
    """Return file paths mentioned in staged or working tree diffs."""

    names: List[Path] = []
    diff_commands = (
        ["git", "diff", "--name-only"],
        ["git", "diff", "--name-only", "--cached"],
    )
    for args in diff_commands:
        try:
            diff = subprocess.run(
                args,
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            continue
        if diff.returncode != 0:
            continue
        for line in diff.stdout.splitlines():
            if line.strip():
                names.append((repo_root / line.strip()).resolve())
    return list(dict.fromkeys(names))


def _header_in_first_three(
    text: str,
) -> tuple[int | None, datetime.date | None]:
    """Return header index and parsed date when present in the top lines."""

    header_lines = text.splitlines()[:3]
    for idx, line in enumerate(header_lines):
        for pattern in _LAST_UPDATED_PATTERNS:
            match = pattern.match(line)
            if match:
                return idx, datetime.date.fromisoformat(match.group(1))
    return None, None


def _find_all_last_updated(text: str) -> List[datetime.date]:
    """Collect all Last Updated stamps within a file."""

    dates: List[datetime.date] = []
    for pattern in _LAST_UPDATED_PATTERNS:
        for match in pattern.finditer(text):
            dates.append(datetime.date.fromisoformat(match.group(1)))
    return dates


def _render_header(path: Path, today: datetime.date) -> str:
    """Render a ``Last Updated`` header matching the file type."""

    if path.suffix.lower() == ".md":
        return f"**Last Updated:** {today.isoformat()}"
    return f"# Last Updated: {today.isoformat()}"


def _write_header(path: Path, text: str, today: datetime.date) -> None:
    """Refresh the ``Last Updated`` header near the top of the file."""

    lines = text.splitlines()
    target_header = _render_header(path, today)
    shebang = lines[0] if lines and lines[0].startswith("#!") else None

    existing_index = None
    for idx, line in enumerate(lines):
        for pattern in _LAST_UPDATED_PATTERNS:
            if pattern.match(line):
                existing_index = idx
                break
        if existing_index is not None:
            break

    if existing_index is not None:
        lines.pop(existing_index)

    insert_at = 1 if shebang is not None else 0
    if shebang is not None:
        lines = [shebang] + lines[1:]
    lines.insert(insert_at, target_header)
    updated = "\n".join(lines)
    if not updated.endswith("\n"):
        updated += "\n"
    path.write_text(updated, encoding="utf-8")


def _iso_date_strings(text: str) -> List[datetime.date]:
    """Extract ISO-8601 dates embedded anywhere in the text."""

    matches = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    return [datetime.date.fromisoformat(value) for value in matches]


class LastUpdatedDocsRule(Rule):
    """Enforce ``Last Updated`` headers on documentation surfaces."""

    name = "last-updated-header"

    def check(self, context: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        today = _utc_today()
        targets = _iter_surface_paths(context, ("docs", "interfaces"))
        for path in targets:
            text = _read_text(path)
            header_index, header_date = _header_in_first_three(text)
            header_dates = _find_all_last_updated(text)
            if header_index is None:
                message = (
                    "Missing Last Updated header in the first three lines."
                )
                if header_dates:
                    message = (
                        "Last Updated must appear within the first "
                        "three lines."
                    )
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message=message,
                        path=path,
                        fixable=True,
                    )
                )
                continue
            if header_date is None:
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message="Unreadable Last Updated header.",
                        path=path,
                        fixable=True,
                    )
                )
                continue
            if header_date > today:
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            "Last Updated cannot be in the future relative to "
                            f"today ({today.isoformat()})."
                        ),
                        path=path,
                    )
                )
        return violations

    def fix(
        self, context: RuleContext, safe_only: bool = False
    ) -> List[Violation]:
        today = _utc_today()
        targets = _iter_surface_paths(context, ("docs", "interfaces"))
        for path in targets:
            text = _read_text(path)
            header_index, header_date = _header_in_first_three(text)
            header_dates = _find_all_last_updated(text)
            requires_fix = header_index is None or header_date is None
            requires_fix = requires_fix or (
                header_date is not None and header_date > today
            )
            requires_fix = requires_fix or (
                header_index is None and bool(header_dates)
            )
            if not requires_fix:
                continue
            _write_header(path, text, today)
        return self.check(context)


class VersionSyncRule(Rule):
    """Keep version markers aligned across required files."""

    name = "version-sync"

    def check(self, context: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        version_file = context.repo_root / "copernican_lib" / "VERSION"
        if not version_file.exists():
            return [
                Violation(
                    rule_name=self.name,
                    message="Missing copernican_lib/VERSION file.",
                    path=version_file,
                )
            ]

        tracked_version = _read_text(version_file).strip()
        readme = context.repo_root / "README.md"
        readme_match = None
        if readme.exists():
            readme_match = re.search(
                r"\*\*Version:\*\*\s*(?P<version>\S+)",
                _read_text(readme),
            )
        if readme_match is None:
            violations.append(
                Violation(
                    rule_name=self.name,
                    message="README.md is missing the Version header.",
                    path=readme,
                )
            )
        elif readme_match.group("version") != tracked_version:
            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "README.md version does not match copernican_lib/"
                        "VERSION."
                    ),
                    path=readme,
                )
            )

        citation = context.repo_root / "CITATION.cff"
        if citation.exists():
            citation_versions = re.findall(
                r"^\s*version:\s*\"?(\d+\.\d+\.\d+)\"?",
                _read_text(citation),
                flags=re.MULTILINE,
            )
            if not citation_versions:
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message="CITATION.cff must declare project versions.",
                        path=citation,
                    )
                )
            elif any(
                version != tracked_version for version in citation_versions
            ):
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            "CITATION.cff versions are out of sync with "
                            "copernican_lib/VERSION."
                        ),
                        path=citation,
                    )
                )
        else:
            violations.append(
                Violation(
                    rule_name=self.name,
                    message="Missing CITATION.cff for version validation.",
                    path=citation,
                )
            )
        return violations


class NoFutureDatesRule(Rule):
    """Guard against future-dated metadata entries."""

    name = "no-future-dates"

    def check(self, context: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        today = _utc_today()
        targets = _iter_surface_paths(
            context, ("docs", "interfaces", "metadata")
        )
        for path in targets:
            text = _read_text(path)
            for stamp in _iso_date_strings(text):
                if stamp > today:
                    violations.append(
                        Violation(
                            rule_name=self.name,
                            message=(
                                f"Found future date {stamp.isoformat()} in "
                                "metadata."
                            ),
                            path=path,
                        )
                    )
        return violations


class CitationYamlRule(Rule):
    """Validate citation metadata against the policy spec."""

    name = "citation-yaml"

    def check(self, context: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        citation = context.repo_root / "CITATION.cff"
        version_file = context.repo_root / "copernican_lib" / "VERSION"
        tracked_version = (
            _read_text(version_file).strip() if version_file.exists() else None
        )

        if not citation.exists():
            return [
                Violation(
                    rule_name=self.name,
                    message="CITATION.cff is required for citation metadata.",
                    path=citation,
                )
            ]
        try:
            citation_data = yaml.safe_load(_read_text(citation))
        except YAMLError:
            return [
                Violation(
                    rule_name=self.name,
                    message="CITATION.cff could not be parsed as YAML.",
                    path=citation,
                )
            ]
        if not isinstance(citation_data, dict):
            return [
                Violation(
                    rule_name=self.name,
                    message="CITATION.cff must contain a mapping.",
                    path=citation,
                )
            ]
        required_fields = {
            "cff-version",
            "title",
            "version",
            "date-released",
            "authors",
            "preferred-citation",
        }
        missing = [
            field for field in required_fields if field not in citation_data
        ]
        if missing:
            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "CITATION.cff is missing required fields: "
                        + ", ".join(sorted(missing))
                    ),
                    path=citation,
                )
            )
            return violations

        authors = citation_data.get("authors")
        if not isinstance(authors, list) or not authors:
            violations.append(
                Violation(
                    rule_name=self.name,
                    message="CITATION.cff must list at least one author.",
                    path=citation,
                )
            )

        if tracked_version and citation_data.get("version") != tracked_version:
            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "CITATION.cff version must align with copernican_lib/"
                        "VERSION."
                    ),
                    path=citation,
                )
            )

        _, header_date = _header_in_first_three(_read_text(citation))
        if header_date is not None:
            released = citation_data.get("date-released")
            if released and released != header_date.isoformat():
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            "CITATION.cff date-released should match its Last "
                            "Updated header."
                        ),
                        path=citation,
                    )
                )
        return violations


class ChangelogRule(Rule):
    """Ensure changelog entries accompany code and policy changes."""

    name = "changelog-entry"

    def check(self, context: RuleContext) -> List[Violation]:
        changelog = context.repo_root / "CHANGELOG.md"
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=context.repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return []
        if status.returncode != 0:
            return []

        changed_paths: List[Path] = []
        for line in status.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            changed_paths.append(context.repo_root / parts[1])

        if not changed_paths:
            return []
        non_changelog = [
            path
            for path in changed_paths
            if path.resolve() != changelog.resolve()
        ]
        if non_changelog and changelog not in changed_paths:
            display = ", ".join(
                sorted(
                    path.relative_to(context.repo_root).as_posix()
                    for path in non_changelog
                )
            )
            return [
                Violation(
                    rule_name=self.name,
                    message=(
                        "CHANGELOG.md must be updated when other files "
                        f"change: {display}"
                    ),
                    path=changelog,
                )
            ]
        return []


class ChangelogDiffCoverageRule(Rule):
    """Require the latest changelog entry to mention touched files."""

    name = "changelog-diff-coverage"

    def check(self, context: RuleContext) -> List[Violation]:
        repo_root = context.repo_root
        changelog = repo_root / "CHANGELOG.md"
        if not changelog.exists():
            return []

        changed = [
            path
            for path in _git_diff_names(repo_root)
            if path.is_file()
            and path.resolve() != changelog.resolve()
            and repo_root in path.parents
        ]
        if not changed:
            return []

        newest_entry: list[str] = []
        lines = changelog.read_text(encoding="utf-8").splitlines()
        in_latest = False
        for line in lines:
            if line.startswith("## Version"):
                if in_latest:
                    break
                in_latest = True
                continue
            if in_latest:
                newest_entry.append(line)

        newest_text = "\n".join(newest_entry).lower()
        missing: List[str] = []
        for path in changed:
            relative = path.relative_to(repo_root).as_posix()
            parts = relative.split("/")
            identifiers = {relative.lower(), parts[0].lower()}
            identifiers.add(Path(relative).name.lower())
            if not any(
                identifier in newest_text for identifier in identifiers
            ):
                missing.append(relative)

        if not missing:
            return []

        return [
            Violation(
                rule_name=self.name,
                message=(
                    "Latest CHANGELOG.md entry must list touched files or "
                    "subsystems: " + ", ".join(sorted(missing))
                ),
                path=changelog,
            )
        ]


class DocumentationRefreshRule(Rule):
    """Ensure documentation updates accompany behavioural changes."""

    name = "documentation-refresh-required"

    def check(self, context: RuleContext) -> List[Violation]:
        repo_root = context.repo_root
        status_entries = _git_status(repo_root)
        if not status_entries:
            return []

        doc_surfaces = _iter_surface_paths(context, ("docs",))
        doc_paths = {path.resolve() for path in doc_surfaces}

        changed_docs = [
            path
            for _, path in status_entries
            if path.resolve() in doc_paths and not path.name.startswith(".")
        ]
        if changed_docs:
            return []

        changed_non_docs = [
            path
            for _, path in status_entries
            if path.resolve() not in doc_paths
            and path.suffix
            and not path.name.startswith(".")
            and (repo_root / "CHANGELOG.md").resolve() != path.resolve()
        ]
        if not changed_non_docs:
            return []

        display = ", ".join(
            sorted(
                path.relative_to(repo_root).as_posix()
                for path in changed_non_docs
            )
        )
        return [
            Violation(
                rule_name=self.name,
                message=(
                    "Documentation must be refreshed alongside code changes: "
                    f"{display}"
                ),
                path=changed_non_docs[0],
            )
        ]


class PolicySyncRule(Rule):
    """Keep DriftGuard policy text, YAML and enforcement in lockstep."""

    name = "driftguard-policy-sync"
    _DOC = Path("DRIFTGUARD.md")
    _POLICY = Path("driftguard/repo_policy.yml")
    _CODE_ROOT = Path("driftguard")

    def check(self, context: RuleContext) -> List[Violation]:
        status_entries = _git_status(context.repo_root)
        if not status_entries:
            return []

        repo_root = context.repo_root
        changed = {path.resolve() for _, path in status_entries}
        doc_path = (repo_root / self._DOC).resolve()
        policy_path = (repo_root / self._POLICY).resolve()
        code_root = (repo_root / self._CODE_ROOT).resolve()

        doc_changed = doc_path in changed
        policy_changed = policy_path in changed
        enforcement_changed = any(
            path.suffix == ".py" and code_root in path.parents
            for path in changed
        )

        if not (doc_changed or policy_changed or enforcement_changed):
            return []

        missing: list[str] = []
        if not doc_changed:
            missing.append(self._DOC.as_posix())
        if not policy_changed:
            missing.append(self._POLICY.as_posix())
        if not enforcement_changed:
            missing.append("driftguard/**/*.py")

        if not missing:
            return []

        focus_path = doc_path if doc_changed else policy_path
        if not (doc_changed or policy_changed):
            focus_path = repo_root / self._CODE_ROOT

        message = (
            "DriftGuard policy updates must move together: also update "
            + ", ".join(sorted(missing))
        )

        return [
            Violation(
                rule_name=self.name,
                message=message,
                path=focus_path,
            )
        ]


class TimestampValidationRule(Rule):
    """Prevent chronological drift across tracked timestamps."""

    name = "timestamp-validation"

    def check(self, context: RuleContext) -> List[Violation]:
        repo_root = context.repo_root
        violations: List[Violation] = []
        today = _utc_today()

        changelog = repo_root / "CHANGELOG.md"
        if changelog.exists():
            dates: List[datetime.date] = []
            for line in _read_text(changelog).splitlines():
                match = re.match(r"^-\s*(\d{4}-\d{2}-\d{2})", line.strip())
                if match:
                    dates.append(datetime.date.fromisoformat(match.group(1)))
            if dates:
                newest = dates[0]
                for index, value in enumerate(dates[1:], start=1):
                    if value > dates[index - 1]:
                        violations.append(
                            Violation(
                                rule_name=self.name,
                                message=(
                                    "CHANGELOG.md dates must remain in "
                                    "reverse chronological order."
                                ),
                                path=changelog,
                            )
                        )
                        break
                if newest > today:
                    violations.append(
                        Violation(
                            rule_name=self.name,
                            message=(
                                "CHANGELOG.md must not contain future-dated "
                                "entries."
                            ),
                            path=changelog,
                        )
                    )

        tracked = _iter_surface_paths(context, ("docs", "interfaces"))
        for path in tracked:
            header_index, header_date = _header_in_first_three(
                _read_text(path)
            )
            if header_index is None or header_date is None:
                continue
            try:
                previous = subprocess.run(
                    ["git", "show", f"HEAD:{path.as_posix()}"],
                    cwd=repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
            prior_header = _header_in_first_three(previous.stdout)[1]
            if prior_header and header_date < prior_header:
                if prior_header > today and header_date <= today:
                    continue
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            "Last Updated must not move backward; preserve "
                            "human chronology."
                        ),
                        path=path,
                    )
                )
        return violations


class HumanEditPreservationRule(Rule):
    """Confirm metadata retains or advances human-authored timestamps."""

    name = "human-edit-preservation"

    def check(self, context: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        for path in _iter_surface_paths(
            context, ("docs", "metadata", "interfaces")
        ):
            header_index, header_date = _header_in_first_three(
                _read_text(path)
            )
            if header_index is None or header_date is None:
                continue
            try:
                previous = subprocess.run(
                    ["git", "show", f"HEAD:{path.as_posix()}"],
                    cwd=context.repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
            prior_header = _header_in_first_three(previous.stdout)[1]
            if prior_header and header_date < prior_header:
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            "Human-authored Last Updated stamps must not be "
                            "reversed or overwritten with older dates."
                        ),
                        path=path,
                    )
                )
        return violations


class SemverBumpRequiredRule(Rule):
    """Require semantic version bumps when core code changes."""

    name = "semver-bump-required"

    def check(self, context: RuleContext) -> List[Violation]:
        repo_root = context.repo_root
        status_entries = _git_status(repo_root)
        if not status_entries:
            return []

        version_file = repo_root / "copernican_lib" / "VERSION"
        version_changed = any(
            path.resolve() == version_file.resolve()
            for _, path in status_entries
        )

        surfaces = ["python-lib", "interfaces"]
        relevant_paths: List[Path] = []
        for surface in surfaces:
            if surface not in context.spec.surfaces:
                continue
            relevant_paths.extend(
                resolve_surface_globs(context.spec, repo_root, surface)
            )
        flat_relevant = {path.resolve() for path in relevant_paths}

        code_changes = [
            path
            for _, path in status_entries
            if (
                path.resolve() in flat_relevant
                and not path.name.startswith(".")
            )
        ]
        if code_changes and not version_changed:
            display = ", ".join(
                sorted(
                    path.relative_to(repo_root).as_posix()
                    for path in code_changes
                )
            )
            return [
                Violation(
                    rule_name=self.name,
                    message=(
                        "Bump copernican_lib/VERSION per SemVer when code "
                        f"changes occur: {display}"
                    ),
                    path=version_file,
                )
            ]
        return []


class StartLauncherParityRule(Rule):
    """Ensure start scripts are updated together."""

    name = "start-launcher-parity"
    _STARTERS = (
        Path("start.sh"),
        Path("start.bat"),
        Path("start.command"),
    )

    def check(self, context: RuleContext) -> List[Violation]:
        status_entries = _git_status(context.repo_root)
        if not status_entries:
            return []

        starter_names = {starter.name for starter in self._STARTERS}
        changed = {
            path.resolve()
            for _, path in status_entries
            if path.name in starter_names
        }
        if not changed:
            return []

        expected = {
            (context.repo_root / starter).resolve()
            for starter in self._STARTERS
        }
        if changed == expected:
            return []

        missing = expected - changed
        return [
            Violation(
                rule_name=self.name,
                message=(
                    (
                        "When one start script changes, review and touch the "
                        "others: "
                    )
                    + ", ".join(
                        sorted(
                            path.relative_to(context.repo_root).as_posix()
                            for path in missing
                        )
                    )
                ),
                path=next(iter(changed)),
            )
        ]


class ManagedVenvOnlyRule(Rule):
    """Confirm launchers enforce the managed virtual environment."""

    name = "managed-venv-only"

    def check(self, context: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        starters = (
            context.repo_root / name
            for name in ("start.sh", "start.bat", "start.command")
        )
        for starter in starters:
            if not starter.exists():
                continue
            text = starter.read_text(encoding="utf-8").lower()
            if ".venv" not in text:
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            "Managed virtual environment must be enforced in "
                            f"{starter.name}."
                        ),
                        path=starter,
                    )
                )
        return violations


class SecurityComplianceRule(Rule):
    """Check launchers keep pinned dependency installation flows."""

    name = "security-compliance"

    def check(self, context: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        keywords = ("requirements.lock", "pip install --no-deps")
        starters = (
            context.repo_root / name
            for name in ("start.sh", "start.bat", "start.command")
        )
        for starter in starters:
            if not starter.exists():
                continue
            text = starter.read_text(encoding="utf-8").lower()
            if not all(keyword in text for keyword in keywords):
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            (
                                f"{starter.name} must install from "
                                "requirements.lock and use --no-deps for "
                                "project installs."
                            )
                        ),
                        path=starter,
                    )
                )
        return violations
