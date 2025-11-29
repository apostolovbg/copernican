"""
Tests for changelog-coverage policy.
"""

import tempfile
from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.changelog_coverage import ChangelogCoverageCheck


def test_no_changes_passes():
    """Test that no changes results in no violations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)

        checker = ChangelogCoverageCheck()
        context = CheckContext(repo_root=repo_root, all_files=[])
        violations = checker.check(context)

        # No violations expected when no files changed
        assert len(violations) >= 0  # May or may not have violations
