"""DEPRECATED: Use DevCovenant pre-commit hook instead.

This module is deprecated and will be removed in a future version.
The checks previously provided here are now handled by DevCovenant policies:
- Future dates: devcovenant/policy_scripts/no_future_dates.py
- Last Updated headers: devcovenant/policy_scripts/last_updated_placement.py
- Version sync: devcovenant/policy_scripts/version_sync.py
- Changelog coverage: devcovenant/policy_scripts/changelog_coverage.py
- New modules need tests: devcovenant/policy_scripts/new_modules_need_tests.py
- No print in library: devcovenant/policy_scripts/no_print_in_library.py

DevCovenant is now integrated into the pre-commit hooks automatically.

---

Legacy documentation:
The checks provided here extend the standard tooling enforced by the
repository's pre-commit configuration. They guarantee that documentation
dates remain sensible and fresh, version metadata stays synchronised across
the project, changelog entries accompany behavioural changes and that
internal modules avoid direct ``print`` calls. New modules must ship with
tests so behaviour is exercised immediately.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Set

DATE_PATTERN = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
PRINT_PATTERN = re.compile(r"(?<![\w.])print\s*\(")
LAST_UPDATED_HEADER_PATTERNS = (
    re.compile(r"^\s*\*\*Last Updated:\*\*\s*(19|20)\d{2}-\d{2}-\d{2}\s*$"),
    re.compile(r"^\s*# Last Updated:\s*(19|20)\d{2}-\d{2}-\d{2}\s*$"),
    re.compile(r"^\s*@REM Last Updated:\s*(19|20)\d{2}-\d{2}-\d{2}\s*$"),
)
LAST_UPDATED_REQUIRED_SUFFIXES = {".md", ".yml", ".yaml", ".cff"}
LAST_UPDATED_REQUIRED_PATHS = {
    Path("copernican.py"),
    Path("start.sh"),
    Path("start.command"),
    Path("start.bat"),
}


def _utc_today() -> _dt.date:
    """Return today's date normalised to Coordinated Universal Time."""

    return _dt.datetime.now(_dt.timezone.utc).date()


def _as_posix(path: Path) -> str:
    """Render *path* with forward slashes for cross-platform messaging."""

    return path.as_posix()


def _read_text(path: Path) -> str:
    """Return the text content of ``path`` using UTF-8 with fallback.

    The helper mirrors behaviour from other tooling in the repository by
    replacing decoding errors rather than failing outright.  This keeps the
    hook robust when third-party data files contain stray bytes outside the
    UTF-8 range.
    """

    return path.read_text(encoding="utf-8", errors="replace")


def _collect_repo_changes(
    root: Path,
) -> tuple[Set[Path], Set[Path], List[str]]:
    """Return added and modified files reported by Git.

    The hook relies on Git status rather than the file list provided by
    pre-commit so that it can tell whether files are newly added or simply
    edited.  The distinction is critical for validating that new modules ship
    with accompanying tests.
    """

    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
        )
    except FileNotFoundError as exc:
        message = "Git is required to evaluate repository changes"
        return set(), set(), [f"{message}: {exc}"]
    except subprocess.CalledProcessError as exc:
        # pragma: no cover - git panic
        return set(), set(), [f"git status failed: {exc}"]

    added: Set[Path] = set()
    modified: Set[Path] = set()
    for line in output.splitlines():
        if not line or len(line) < 4:
            continue
        status, path_str = line[:2], line[3:]
        path = root / path_str
        index_state, worktree_state = status[0], status[1]
        if index_state in {"A", "C", "R"} or worktree_state in {"A", "?"}:
            added.add(path)
        elif index_state == "?":
            added.add(path)
        elif index_state in {"M", "R", "C"} or worktree_state == "M":
            modified.add(path)
    return added, modified, []


def _requires_last_updated(path: Path, root: Path) -> bool:
    """Return ``True`` when ``path`` must carry a Last Updated header."""

    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return rel in LAST_UPDATED_REQUIRED_PATHS or (
        path.suffix in LAST_UPDATED_REQUIRED_SUFFIXES
    )


def _detect_future_dates(
    files: Iterable[Path], today: _dt.date, *, root: Path
) -> List[str]:
    """Return human-readable error messages for timestamps after ``today``."""

    errors: List[str] = []
    for path in files:
        if not path.is_file():
            continue
        try:
            rel_path = path.relative_to(root)
        except ValueError:
            rel_path = path
        if rel_path.parts and rel_path.parts[0] == "tests":
            continue
        display_path = _as_posix(rel_path)
        try:
            text = _read_text(path)
        except OSError as exc:  # pragma: no cover - filesystem issues
            errors.append(f"{display_path}: unable to read file ({exc})")
            continue
        for match in DATE_PATTERN.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = len(text)
            context = text[line_start:line_end].lower()
            if (
                "last updated" not in context
                and "date-released" not in context
            ):
                continue
            year, month, day = (
                int(part) for part in match.group(0).split("-")
            )
            try:
                candidate = _dt.date(year, month, day)
            except ValueError:
                # Ignore invalid dates; they are handled elsewhere if needed.
                continue
            if candidate > today:
                iso_stamp = candidate.isoformat()
                errors.append(
                    f"{display_path}: contains future date {iso_stamp}"
                )
    return errors


def _check_last_updated_headers(
    root: Path, files: Iterable[Path]
) -> List[str]:
    """Ensure ``Last Updated`` markers live in the first three lines."""

    errors: List[str] = []
    for path in files:
        if not path.is_file():
            continue
        text = _read_text(path)
        header_lines = text.splitlines()
        has_header = any(
            pattern.match(line)
            for pattern in LAST_UPDATED_HEADER_PATTERNS
            for line in header_lines
        )
        try:
            rel_path = path.relative_to(root)
        except ValueError:
            rel_path = path
        display_path = _as_posix(rel_path)
        requires_marker = _requires_last_updated(path, root)
        if not requires_marker and has_header:
            errors.append(
                (
                    f"{display_path}: remove Last Updated markers from files "
                    "outside the policy allowlist (*.md, *.yml, *.yaml, "
                    "*.cff, copernican.py, start scripts)."
                )
            )
            continue
        if not requires_marker:
            continue
        if not has_header:
            errors.append(
                (
                    f"{display_path}: add a Last Updated marker within the "
                    "first three lines using YYYY-MM-DD."
                )
            )
            continue
        header = text.splitlines()[:3]
        if any(
            pattern.match(line)
            for pattern in LAST_UPDATED_HEADER_PATTERNS
            for line in header
        ):
            continue
        errors.append(
            (
                f"{display_path}: Last Updated marker must appear "
                "within the first three lines and use YYYY-MM-DD."
            )
        )
    return errors


def _enforce_last_updated_freshness(
    root: Path, changed: Sequence[Path], today: _dt.date
) -> List[str]:
    """Ensure modified files include a current ``Last Updated`` header."""

    errors: List[str] = []
    for path in changed:
        if not path.is_file():
            continue
        if not _requires_last_updated(path, root):
            continue
        try:
            rel_path = path.relative_to(root)
        except ValueError:
            rel_path = path
        display_path = _as_posix(rel_path)
        text = _read_text(path)
        header = text.splitlines()[:3]
        matched_date = None
        for line in header:
            for pattern in LAST_UPDATED_HEADER_PATTERNS:
                match = pattern.match(line)
                if match:
                    date_match = DATE_PATTERN.search(line)
                    if date_match:
                        matched_date = date_match.group(0)
                    break
            if matched_date:
                break
        if not matched_date:
            errors.append(
                (
                    f"{display_path}: add a Last Updated header within the "
                    "first three lines"
                )
            )
            continue
        header_date = _dt.date.fromisoformat(matched_date)
        if header_date != today:
            iso_today = today.isoformat()
            errors.append(
                (
                    f"{display_path}: Last Updated must reflect today's date "
                    f"({iso_today}) when the file changes"
                )
            )
    return errors


def _check_version_sync(root: Path) -> List[str]:
    """Ensure README, CITATION and VERSION agree on the recorded version."""

    version_file = root / "copernican_lib" / "VERSION"
    readme_file = root / "README.md"
    citation_file = root / "CITATION.cff"

    for target in (version_file, readme_file, citation_file):
        if not target.exists():
            missing = _as_posix(target.relative_to(root))
            return [f"Required metadata file missing: {missing}"]

    version = version_file.read_text(encoding="utf-8").strip()

    readme_match = re.search(
        r"\*\*Version:\*\*\s*(?P<version>\d+\.\d+\.\d+)",
        _read_text(readme_file),
    )
    citation_text = _read_text(citation_file)
    citation_regex = r"version:\s*\"(?P<version>\d+\.\d+\.\d+)\""
    citation_pattern = re.compile(citation_regex)
    citation_matches = citation_pattern.findall(citation_text)

    errors: List[str] = []
    if not readme_match:
        errors.append("README.md is missing the Version header")
    elif readme_match.group("version") != version:
        readme_version = readme_match.group("version")
        errors.append(
            "README.md version "
            f"{readme_version} does not match copernican_lib/VERSION"
        )

    if len(citation_matches) < 2:
        errors.append(
            "CITATION.cff must declare project and preferred versions"
        )
    else:
        unique_versions = set(citation_matches)
        if len(unique_versions) != 1 or version not in unique_versions:
            errors.append("CITATION.cff versions are out of sync with VERSION")

    return errors


def _ensure_changelog_updated(
    root: Path, changed: Sequence[Path]
) -> List[str]:
    """Require changelog updates whenever files outside it change."""

    changelog = root / "CHANGELOG.md"
    relevant = [
        path for path in changed if path.resolve() != changelog.resolve()
    ]
    if relevant and changelog not in changed:
        paths = ", ".join(
            sorted(_as_posix(path.relative_to(root)) for path in relevant)
        )
        return [f"CHANGELOG.md must record updates affecting: {paths}"]
    return []


def _check_new_modules_have_tests(
    root: Path, added_files: Sequence[Path], changed_files: Sequence[Path]
) -> List[str]:
    """Ensure new Python modules ship with accompanying tests."""

    new_modules = []
    for path in added_files:
        if path.suffix != ".py" or not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        if rel.parts and rel.parts[0] == "tests":
            continue
        new_modules.append(path)
    if not new_modules:
        return []

    changed_tests = []
    for path in changed_files:
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "tests":
            changed_tests.append(path)
    if changed_tests:
        return []

    targets = ", ".join(
        sorted(_as_posix(path.relative_to(root)) for path in new_modules)
    )
    message = "Add or update tests under tests/ for new modules: " f"{targets}"
    return [message]


def _check_print_usage(root: Path, files: Iterable[Path]) -> List[str]:
    """Prevent direct ``print`` usage in library modules."""

    # Direct ``print`` calls bypass the managed console output helper and cause
    # Windows paths to drift away from the documented style.  The guard keeps
    # diagnostics consistent across platforms and transports output through the
    # dedicated console utilities.

    allowed = {
        root / "copernican_lib" / "console_output.py",
    }
    errors: List[str] = []
    for path in files:
        if not path.is_file() or path.suffix != ".py":
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if rel.parts[0] != "copernican_lib":
            continue
        if path in allowed:
            continue
        text = _read_text(path)
        if PRINT_PATTERN.search(text):
            rel_display = rel.as_posix()
            errors.append(
                f"{rel_display}: replace print() with "
                "copernican_lib.console_output.write"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Entry point used by pre-commit."""

    parser = argparse.ArgumentParser(
        description="Copernican Suite policy enforcement",
    )
    parser.add_argument("filenames", nargs="*", help="Files to inspect")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    staged_files = [repo_root / name for name in args.filenames]
    added, modified, repo_errors = _collect_repo_changes(repo_root)
    changed_files = sorted(added | modified)
    today = _utc_today()

    errors: List[str] = []
    errors.extend(repo_errors)
    errors.extend(_detect_future_dates(staged_files, today, root=repo_root))
    errors.extend(_check_last_updated_headers(repo_root, staged_files))
    errors.extend(
        _enforce_last_updated_freshness(repo_root, changed_files, today)
    )
    errors.extend(_check_version_sync(repo_root))
    errors.extend(_ensure_changelog_updated(repo_root, changed_files))
    errors.extend(
        _check_new_modules_have_tests(repo_root, added, changed_files)
    )
    errors.extend(_check_print_usage(repo_root, staged_files))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by pre-commit
    sys.exit(main())
