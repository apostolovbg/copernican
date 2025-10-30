"""Custom pre-commit validations specific to the Copernican Suite.

The checks provided here extend the standard tooling enforced by the
repository's pre-commit configuration.  They guarantee that documentation
dates remain sensible, version metadata stays synchronised across the project
and that internal modules avoid direct ``print`` calls.  The latter keeps the
logging infrastructure consistent and ensures end users receive mirrored
output through the managed console helpers.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path
from typing import Iterable, List


DATE_PATTERN = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
PRINT_PATTERN = re.compile(r'(?<![\w.])print\s*\(')


def _read_text(path: Path) -> str:
    """Return the text content of ``path`` using UTF-8 with fallback.

    The helper mirrors behaviour from other tooling in the repository by
    replacing decoding errors rather than failing outright.  This keeps the
    hook robust when third-party data files contain stray bytes outside the
    UTF-8 range.
    """

    return path.read_text(encoding="utf-8", errors="replace")


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
        try:
            text = _read_text(path)
        except OSError as exc:  # pragma: no cover - filesystem issues
            errors.append(f"{path}: unable to read file ({exc})")
            continue
        for match in DATE_PATTERN.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = len(text)
            context = text[line_start:line_end].lower()
            if "last updated" not in context and "date-released" not in context:
                continue
            year, month, day = (int(part) for part in match.group(0).split("-"))
            try:
                candidate = _dt.date(year, month, day)
            except ValueError:
                # Ignore invalid dates; they are handled elsewhere if needed.
                continue
            if candidate > today:
                errors.append(
                    f"{path}: contains future date {candidate.isoformat()}"
                )
    return errors


def _check_version_sync(root: Path) -> List[str]:
    """Ensure README, CITATION and VERSION agree on the recorded version."""

    version_file = root / "copernican_lib" / "VERSION"
    readme_file = root / "README.md"
    citation_file = root / "CITATION.cff"

    for target in (version_file, readme_file, citation_file):
        if not target.exists():
            return [f"Required metadata file missing: {target}"]

    version = version_file.read_text(encoding="utf-8").strip()

    readme_match = re.search(
        r"\*\*Version:\*\*\s*(?P<version>\d+\.\d+\.\d+)",
        _read_text(readme_file),
    )
    citation_text = _read_text(citation_file)
    citation_matches = re.findall(r"version:\s*\"(?P<version>\d+\.\d+\.\d+)\"", citation_text)

    errors: List[str] = []
    if not readme_match:
        errors.append("README.md is missing the Version header")
    elif readme_match.group("version") != version:
        errors.append(
            "README.md version {readme} does not match copernican_lib/VERSION"
            .format(readme=readme_match.group("version"))
        )

    if len(citation_matches) < 2:
        errors.append("CITATION.cff must declare project and preferred versions")
    else:
        unique_versions = set(citation_matches)
        if len(unique_versions) != 1 or version not in unique_versions:
            errors.append("CITATION.cff versions are out of sync with VERSION")

    return errors


def _check_print_usage(root: Path, files: Iterable[Path]) -> List[str]:
    """Prevent direct ``print`` usage in library modules outside console I/O."""

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
            errors.append(
                f"{rel}: replace print() with copernican_lib.console_output.write"
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
    today = _dt.date.today()

    errors: List[str] = []
    errors.extend(_detect_future_dates(staged_files, today, root=repo_root))
    errors.extend(_check_version_sync(repo_root))
    errors.extend(_check_print_usage(repo_root, staged_files))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by pre-commit
    sys.exit(main())
