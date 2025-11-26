from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_FILES = {Path("DRIFTGUARD.md"), Path("driftguard/repo_policy.yml")}
COMPANION_PATTERNS = ("driftguard/**/*.py", "tests/test_driftguard_*.py")


def _git_available() -> bool:
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    return True


def _have_previous_commit() -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    try:
        return int(result.stdout.strip()) > 1
    except ValueError:
        return False


def _changed_files() -> set[Path]:
    if not (_git_available() and _have_previous_commit()):
        return set()

    diff = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        Path(line.strip()) for line in diff.stdout.splitlines() if line.strip()
    }


def _matches_any(path: Path, patterns: Iterable[str]) -> bool:
    return any(path.match(pattern) for pattern in patterns)


def test_policy_files_require_repo_policy_and_enforcement_updates() -> None:
    """Policy documents must stay in lockstep with enforcement logic."""

    changed = _changed_files()
    if not changed:
        pytest.skip(
            "No prior commit to diff against; skipping policy sync check."
        )

    if not (POLICY_FILES & changed):
        return

    missing_policy = POLICY_FILES - changed
    assert not missing_policy, (
        "Changes to DRIFTGUARD.md or driftguard/repo_policy.yml must update "
        "both files."
    )

    companion_updates = [
        path for path in changed if _matches_any(path, COMPANION_PATTERNS)
    ]
    assert companion_updates, (
        "DriftGuard policy changes must be paired with enforcement "
        "code or tests."
    )
