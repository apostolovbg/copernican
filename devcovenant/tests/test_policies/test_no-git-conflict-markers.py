"""
Tests for the no-git-conflict-markers policy.
"""

import tempfile
from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.no_git_conflict_markers import (
    NoGitConflictMarkersCheck,
)


def test_no_conflict_markers():
    """Test that clean files pass."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(
            """
def example():
    return "clean file"
"""
        )
        temp_path = Path(f.name)

    try:
        checker = NoGitConflictMarkersCheck()
        context = CheckContext(
            repo_root=temp_path.parent, all_files=[temp_path]
        )
        violations = checker.check(context)

        assert len(violations) == 0

    finally:
        temp_path.unlink()


def test_detects_conflict_markers():
    """Test that conflict markers are detected."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(
            """
def example():
<<<<<<< HEAD
    return "version 1"
=======
    return "version 2"
>>>>>>> branch
"""
        )
        temp_path = Path(f.name)

    try:
        checker = NoGitConflictMarkersCheck()
        context = CheckContext(
            repo_root=temp_path.parent, all_files=[temp_path]
        )
        violations = checker.check(context)

        # Should detect at least one conflict marker
        assert len(violations) >= 1
        assert violations[0].severity == "critical"
        assert violations[0].policy_id == "no-git-conflict-markers"

    finally:
        temp_path.unlink()
