"""Tests for no_print_in_library policy."""

import tempfile
import unittest
from pathlib import Path

from devcovenant.policy_scripts.no_print_in_library import (
    NoPrintInLibraryPolicy,
)


class TestNoPrintInLibraryPolicy(unittest.TestCase):
    """Test suite for NoPrintInLibraryPolicy."""

    def test_detects_print_in_library(self):
        """Policy should detect print() usage in library code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            lib_dir = repo_root / "copernican_lib"
            lib_dir.mkdir()
            test_file = lib_dir / "module.py"
            test_file.write_text('def foo():\n    print("hello")\n')

            policy = NoPrintInLibraryPolicy(repo_root)
            violations = policy.check([test_file])

            self.assertEqual(len(violations), 1)
            self.assertIn("print", violations[0].message.lower())

    def test_allows_print_in_tests(self):
        """Policy should allow print() in test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            test_dir = repo_root / "tests"
            test_dir.mkdir()
            test_file = test_dir / "test_module.py"
            test_file.write_text('def test_foo():\n    print("hello")\n')

            policy = NoPrintInLibraryPolicy(repo_root)
            violations = policy.check([test_file])

            self.assertEqual(len(violations), 0)

    def test_allows_print_in_console_output(self):
        """Policy should allow print() in console_output.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            lib_dir = repo_root / "copernican_lib"
            lib_dir.mkdir()
            console_file = lib_dir / "console_output.py"
            console_file.write_text('def write(msg):\n    print(msg)\n')

            policy = NoPrintInLibraryPolicy(repo_root)
            violations = policy.check([console_file])

            self.assertEqual(len(violations), 0)
