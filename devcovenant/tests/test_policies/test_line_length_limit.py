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
        vendor_file = temp_dir / "project_lib" / "vendor" / "bundle.py"
        vendor_file.parent.mkdir(parents=True, exist_ok=True)
        vendor_file.write_text("# " + "x" * 200 + "\n")

        checker = LineLengthLimitCheck()
        checker.set_options(
            {
                "include_suffixes": [".py"],
                "exclude_prefixes": ["project_lib/vendor"],
            },
            {},
        )
        context = CheckContext(repo_root=temp_dir, all_files=[vendor_file])
        violations = checker.check(context)

        assert len(violations) == 0
    finally:
        shutil.rmtree(temp_dir)


def test_markdown_checked_by_default():
    """Markdown documentation should now be subject to the same limit."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False
    ) as handle:
        handle.write("# " + "x" * 90 + "\n")
        temp_path = Path(handle.name)

    try:
        checker = LineLengthLimitCheck()
        context = CheckContext(
            repo_root=temp_path.parent, all_files=[temp_path]
        )
        violations = checker.check(context)
        assert violations, "Markdown lines over 79 chars must violate policy"
    finally:
        temp_path.unlink(missing_ok=True)


def test_configurable_suffixes_and_threshold():
    """Custom suffix and limit should be honoured via configuration."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        notes = temp_dir / "notes.txt"
        notes.write_text("line:" + "a" * 20 + "\n", encoding="utf-8")

        checker = LineLengthLimitCheck()
        context = CheckContext(
            repo_root=temp_dir,
            all_files=[notes],
            config={
                "policies": {
                    "line-length-limit": {
                        "max_length": 10,
                        "include_suffixes": [".txt"],
                        "exclude_prefixes": [],
                    }
                }
            },
        )
        checker.set_options(
            {},
            context.get_policy_config("line-length-limit"),
        )
        violations = checker.check(context)
        assert violations, "Custom suffix + limit should trigger a violation"
    finally:
        shutil.rmtree(temp_dir)


def test_custom_skip_prefix():
    """Custom skip prefixes should exempt directories when configured."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        target = temp_dir / "docs" / "generated" / "file.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# " + "x" * 120 + "\n", encoding="utf-8")

        checker = LineLengthLimitCheck()
        context = CheckContext(
            repo_root=temp_dir,
            all_files=[target],
            config={
                "policies": {
                    "line-length-limit": {
                        "include_suffixes": [".md"],
                        "exclude_prefixes": ["docs/generated"],
                    }
                }
            },
        )
        checker.set_options(
            {},
            context.get_policy_config("line-length-limit"),
        )
        violations = checker.check(context)
        assert not violations, "Custom skip prefix should suppress violations"
    finally:
        shutil.rmtree(temp_dir)


def test_metadata_options_drive_scope(tmp_path: Path) -> None:
    """Policy-def metadata should configure suffixes and thresholds."""
    notes = tmp_path / "notes.txt"
    notes.write_text("header " + "x" * 20 + "\n", encoding="utf-8")

    checker = LineLengthLimitCheck()
    checker.set_options(
        {
            "include_suffixes": [".py"],
            "exclude_prefixes": ["data"],
            "force_include_globs": ["data/**/cosmo_parser_demo.py"],
        },
        {},
    )
    checker.set_options(
        {
            "include_suffixes": [".txt"],
            "max_length": 10,
            "exclude_prefixes": [],
        },
        {},
    )
    context = CheckContext(
        repo_root=tmp_path,
        all_files=[notes],
    )
    violations = checker.check(context)
    assert violations, "Metadata-driven suffixes should trigger violations"


def test_force_include_globs_override_skip(tmp_path: Path) -> None:
    """Files under skipped prefixes can be re-included via glob patterns."""
    target = tmp_path / "data" / "set" / "cosmo_parser_demo.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# " + "y" * 200 + "\n", encoding="utf-8")

    checker = LineLengthLimitCheck()
    context = CheckContext(
        repo_root=tmp_path,
        all_files=[target],
    )
    violations = checker.check(context)
    assert (
        violations
    ), "force_include_globs should re-check parsers under read-only trees"
