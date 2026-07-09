"""Test helpers for repo files in sync-managed working trees."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from copernican.lib import file_io

_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2, 0.4)
_IGNORED_BASENAMES = {
    ".git",
    ".venv",
    ".python",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "build",
    "dist",
    "htmlcov",
}


def read_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read one repository file through the retrying production helper."""

    return file_io.read_text(path, encoding=encoding)


def stage_repo_snapshot(
    repo_root: str | Path,
    destination_root: str | Path,
) -> Path:
    """Copy the repository outside the source tree.

    The staged copy lands under ``destination_root``.
    """

    source_root = Path(repo_root)
    target_root = Path(destination_root) / "repo"

    def _ignore(current_dir: str, names: list[str]) -> list[str]:
        current_path = Path(current_dir)
        ignored: list[str] = []
        for name in names:
            candidate = current_path / name
            relative = candidate.relative_to(source_root)
            if name in _IGNORED_BASENAMES:
                ignored.append(name)
                continue
            if relative.parts[:2] == ("devcovenant", "logs"):
                ignored.append(name)
                continue
            if relative.parts[:3] == ("copernican", "models", "cache"):
                ignored.append(name)
        return ignored

    last_attempt_index = len(_RETRY_DELAYS_SECONDS)
    for attempt_index in range(last_attempt_index + 1):
        try:
            if target_root.exists():
                shutil.rmtree(target_root)
            shutil.copytree(source_root, target_root, ignore=_ignore)
            return target_root
        except OSError as exc:
            if (
                not file_io.is_transient_file_timeout(exc)
                or attempt_index == last_attempt_index
            ):
                raise
            time.sleep(_RETRY_DELAYS_SECONDS[attempt_index])
    raise RuntimeError("unreachable snapshot retry loop exit")
