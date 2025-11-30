"""DEPRECATED: Use DevCovenant policies instead.

This module is deprecated and will be removed in a future version.
The checks previously provided here are now handled by DevCovenant policies:
- Version sync: devcovenant/policy_scripts/version_sync.py
- Future dates: devcovenant/policy_scripts/no_future_dates.py
- Last Updated placement: devcovenant/policy_scripts/last_updated_placement.py

Please run: python devcovenant_check.py

---

Legacy documentation:
The Copernican Suite keeps the canonical release number in
``copernican_lib/VERSION`` and mirrors the value across the README and the
``CITATION.cff`` file. Documentation pages carry ``Last Updated`` timestamps on
the allowlisted surfaces (Markdown, YAML, `CITATION.cff`, the start scripts and
`copernican.py`) that must never extend into the future. This module offers a
small collection of helpers so the test-suite can guard those expectations and
warn contributors when the tracked metadata drifts out of sync.
"""

from __future__ import annotations

import datetime as _dt
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

REQUIRED_LAST_UPDATED_SUFFIXES = {".md", ".yml", ".yaml", ".cff"}
REQUIRED_LAST_UPDATED_PATHS = {
    Path("copernican.py"),
    Path("start.sh"),
    Path("start.command"),
    Path("start.bat"),
}

_LAST_UPDATED_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(
        r"^\*\*Last Updated:\*\*\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE
    ),
    re.compile(r"^# Last Updated:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE),
    re.compile(r"^@REM Last Updated:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE),
)


def _utc_today() -> _dt.date:
    """Return today's date in Coordinated Universal Time (UTC)."""

    return _dt.datetime.now(_dt.timezone.utc).date()


def _repo_root(base_path: Path | None = None) -> Path:
    """Return the repository root, defaulting to the project directory."""

    if base_path is not None:
        return base_path
    return Path(__file__).resolve().parents[1]


def _version_file(path: Path) -> str:
    """Load the tracked project version string."""

    return path.read_text(encoding="utf-8").strip()


def _posix_relative(path: Path, *, root: Path) -> str:
    """Return a repository-relative POSIX path for stable diagnostics."""

    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return relative.as_posix()


def extract_last_updated_dates(text: str) -> List[_dt.date]:
    """Find every ``Last Updated`` timestamp embedded in *text*.

    The repository uses either Markdown bold headers or line comments in
    ``CITATION.cff``.  The helper returns a list of discovered dates while
    preserving their source order.
    """

    dates: List[_dt.date] = []
    for pattern in _LAST_UPDATED_PATTERNS:
        for match in pattern.finditer(text):
            dates.append(_dt.date.fromisoformat(match.group(1)))
    return dates


def _header_contains_last_updated(text: str) -> bool:
    """Return ``True`` if a ``Last Updated`` marker appears in the header."""

    header_lines = text.splitlines()[:3]
    header_text = "\n".join(header_lines)
    for pattern in _LAST_UPDATED_PATTERNS:
        if pattern.search(header_text):
            return True
    return False


def _extract_readme_version(text: str) -> str | None:
    """Return the release string tracked inside the README header."""

    match = re.search(
        r"^\*\*Version:\*\*\s*(?P<version>\S+)", text, flags=re.MULTILINE
    )
    if match:
        return match.group("version")
    return None


def _extract_citation_versions(text: str) -> List[str]:
    """Capture every version string declared inside ``CITATION.cff``."""

    pattern = re.compile(
        r"^\s*version:\s*\"?(?P<version>\d+\.\d+\.\d+)\"?",
        flags=re.MULTILINE,
    )
    return [match.group("version") for match in pattern.finditer(text)]


def requires_last_updated(path: Path, root: Path) -> bool:
    """Return ``True`` when *path* must expose a ``Last Updated`` header."""

    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return rel in REQUIRED_LAST_UPDATED_PATHS or path.suffix in (
        REQUIRED_LAST_UPDATED_SUFFIXES
    )


def _last_updated_targets(base_path: Path) -> Iterable[Path]:
    """Yield files that must expose ``Last Updated`` markers."""

    seen: set[Path] = set()
    for rel in REQUIRED_LAST_UPDATED_PATHS:
        target = base_path / rel
        if target.exists():
            seen.add(target)

    for suffix in sorted(REQUIRED_LAST_UPDATED_SUFFIXES):
        for path in base_path.rglob(f"*{suffix}"):
            if any(
                part.startswith(".git")
                or part == "__pycache__"
                or part.startswith(".pytest_cache")
                for part in path.parts
            ):
                continue
            seen.add(path)

    yield from sorted(seen)


def validate_metadata(
    base_path: Path | None = None, today: _dt.date | None = None
) -> List[str]:
    """Validate release metadata and return a list of error messages."""

    root = _repo_root(base_path)
    current_date = today or _utc_today()
    errors: List[str] = []

    version_file = root / "copernican_lib" / "VERSION"
    tracked_version = _version_file(version_file)

    readme = root / "README.md"
    readme_version = _extract_readme_version(
        readme.read_text(encoding="utf-8")
    )
    if readme_version != tracked_version:
        errors.append(
            f"README.md records version '{readme_version}' but"
            f" copernican_lib/VERSION declares '{tracked_version}'."
        )

    citation = root / "CITATION.cff"
    if citation.exists():
        citation_text = citation.read_text(encoding="utf-8")
        citation_versions = _extract_citation_versions(citation_text)
        for version in citation_versions:
            if version != tracked_version:
                errors.append(
                    "CITATION.cff version fields must match the tracked "
                    f"release '{tracked_version}' but found '{version}'."
                )

    for target in _last_updated_targets(root):
        text = target.read_text(encoding="utf-8")
        display_name = _posix_relative(target, root=root)
        dates = extract_last_updated_dates(text)
        if not dates:
            errors.append(f"{display_name} is missing a Last Updated marker.")
            continue
        if not _header_contains_last_updated(text):
            errors.append(
                (
                    f"{display_name} must place a Last Updated marker "
                    "within the first three lines."
                )
            )
        for stamp in dates:
            if stamp > current_date:
                errors.append(
                    f"{display_name} carries future timestamp"
                    f" {stamp.isoformat()} (today is"
                    f" {current_date.isoformat()})."
                )

    return errors


def main() -> int:
    """Entry point for command-line usage."""

    errors = validate_metadata()
    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        return 1
    print("Metadata check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
