"""
Tests for last-updated-placement policy.
"""

import datetime as dt
import tempfile
from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.last_updated_placement import (
    LastUpdatedPlacementCheck,
)


def test_clean_file_passes():
    """Test that files without Last Updated markers pass."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write("# Clean file\ndef foo():\n    return 42\n")
        temp_path = Path(f.name)

    try:
        checker = LastUpdatedPlacementCheck()
        context = CheckContext(
            repo_root=temp_path.parent, all_files=[temp_path]
        )
        violations = checker.check(context)

        assert len(violations) == 0
    finally:
        temp_path.unlink()


def test_last_updated_in_non_allowlisted_file():
    """Test that Last Updated in non-allowlisted files is detected."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        temp_path = Path(f.name)
    today_date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    temp_path.write_text(f"**Last Updated:** {today_date}\n", encoding="utf-8")

    try:
        checker = LastUpdatedPlacementCheck()
        context = CheckContext(
            repo_root=temp_path.parent, all_files=[temp_path]
        )
        violations = checker.check(context)

        assert len(violations) >= 1
        assert violations[0].policy_id == "last-updated-placement"
    finally:
        temp_path.unlink()
