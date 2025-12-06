"""
Tests for line-length-limit policy.
"""

import shutil
import tempfile
from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.line_length_limit import LineLengthLimitCheck


def test_short_lines_pass():
    """Test that short lines pass."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write("# Short line\ndef foo():\n    return 42\n")
        temp_path = Path(f.name)

    try:
        checker = LineLengthLimitCheck()
        context = CheckContext(
            repo_root=temp_path.parent, all_files=[temp_path]
        )
        violations = checker.check(context)

        assert len(violations) == 0
    finally:
        temp_path.unlink()


def test_long_lines_detected():
    """Test that long lines are detected."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        # Create a line longer than 79 characters
        long_line = "# " + "x" * 80 + "\n"
        f.write(long_line)
        temp_path = Path(f.name)

    try:
        checker = LineLengthLimitCheck()
        context = CheckContext(
            repo_root=temp_path.parent, all_files=[temp_path]
        )
        violations = checker.check(context)

        assert len(violations) >= 1
        assert violations[0].policy_id == "line-length-limit"
    finally:
        temp_path.unlink()


def test_vendor_files_ignored():
    """Vendor files should be skipped even when lines are long."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        vendor_file = temp_dir / "copernican_lib" / "vendor" / "bundle.py"
        vendor_file.parent.mkdir(parents=True, exist_ok=True)
        vendor_file.write_text("# " + "x" * 200 + "\n")

        checker = LineLengthLimitCheck()
        context = CheckContext(repo_root=temp_dir, all_files=[vendor_file])
        violations = checker.check(context)

        assert len(violations) == 0
    finally:
        shutil.rmtree(temp_dir)
