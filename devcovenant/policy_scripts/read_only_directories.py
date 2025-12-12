"""DevCovenant policy: Prevent edits to read-only data directories."""

import fnmatch
from pathlib import Path, PurePosixPath
from typing import Iterable, List

from devcovenant.base import CheckContext, PolicyCheck, Violation

PATTERNS_FILE = Path("devcovenant") / "read_only_directories.txt"
WAIVER_FILE = Path(".devcovenant") / "waivers" / "read-only-directories.txt"


def _load_pattern_file(path: Path) -> List[str]:
    """Read newline-delimited patterns from the given file."""
    if not path.is_file():
        return []

    patterns: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        patterns.append(clean.replace("\\", "/"))
    return patterns


def _matches_any(rel: PurePosixPath, patterns: Iterable[str]) -> bool:
    """Return True if the relative path matches any of the patterns."""
    rel_posix = rel.as_posix()
    for pattern in patterns:
        normalized = pattern.lstrip("/")
        if fnmatch.fnmatch(rel_posix, normalized):
            return True
    return False


class ReadOnlyDirectoriesCheck(PolicyCheck):
    """Blocks changes inside read-only directories unless a waiver exists."""

    policy_id = "read-only-directories"
    version = "1.0.0"

    def check(self, context: CheckContext):
        """Ensure modified files stay outside read-only globs unless waived."""
        files = context.changed_files or []
        if not files:
            return []

        patterns = _load_pattern_file(context.repo_root / PATTERNS_FILE)
        if not patterns:
            return []

        waiver_patterns = _load_pattern_file(context.repo_root / WAIVER_FILE)
        violations = []

        for path in files:
            try:
                rel_path = path.relative_to(context.repo_root)
            except ValueError:
                continue

            rel_norm = PurePosixPath(rel_path.as_posix())
            if not _matches_any(rel_norm, patterns):
                continue

            if _matches_any(rel_norm, waiver_patterns):
                continue

            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=path,
                    message=(
                        "Read-only directories (see "
                        "`devcovenant/read_only_directories.txt`) were modified "
                        "without a waiver."
                    ),
                    suggestion=(
                        "Create `.devcovenant/waivers/read-only-directories.txt` "
                        "with the allowed relative paths or remove the change to "
                        "stay compliant."
                    ),
                )
            )

        return violations
