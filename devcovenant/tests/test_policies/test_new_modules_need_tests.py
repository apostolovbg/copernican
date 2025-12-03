"""Tests for new_modules_need_tests policy."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.new_modules_need_tests import (
    NewModulesNeedTestsCheck,
)


class TestNewModulesNeedTestsPolicy(unittest.TestCase):
    """Test suite for NewModulesNeedTestsCheck."""

    @patch("subprocess.check_output")
    def test_detects_new_module_without_tests(self, mock_subprocess):
        """Policy should detect new modules without test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            # Create the new module file
            lib_dir = repo_root / "copernican_lib"
            lib_dir.mkdir()
            new_module = lib_dir / "new_module.py"
            new_module.write_text("def foo(): pass\n")

            # Simulate git status showing new module
            mock_subprocess.return_value = "A  copernican_lib/new_module.py\n"

            context = CheckContext(repo_root=repo_root)
            policy = NewModulesNeedTestsCheck()
            violations = policy.check(context)

            self.assertEqual(len(violations), 1)
            self.assertIn("test", violations[0].message.lower())

    @patch("subprocess.check_output")
    def test_allows_new_module_with_tests(self, mock_subprocess):
        """Policy should pass when new modules have tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            # Simulate git status showing new module and test
            mock_subprocess.return_value = (
                "A  copernican_lib/new_module.py\n"
                "M  tests/test_new_module.py\n"
            )

            context = CheckContext(repo_root=repo_root)
            policy = NewModulesNeedTestsCheck()
            violations = policy.check(context)

            self.assertEqual(len(violations), 0)

    @patch("subprocess.check_output")
    def test_detects_removed_module_without_tests(self, mock_subprocess):
        """Policy should flag removed modules when no tests change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            mock_subprocess.return_value = " D copernican_lib/old_module.py\n"

            context = CheckContext(repo_root=repo_root)
            policy = NewModulesNeedTestsCheck()
            violations = policy.check(context)

            self.assertEqual(len(violations), 1)
            self.assertIn("removing modules", violations[0].message)

    @patch("subprocess.check_output")
    def test_allows_removed_module_with_tests(self, mock_subprocess):
        """Policy should allow module removals when tests are updated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            mock_subprocess.return_value = (
                " D copernican_lib/old_module.py\n"
                "M  tests/test_old_module.py\n"
            )

            context = CheckContext(repo_root=repo_root)
            policy = NewModulesNeedTestsCheck()
            violations = policy.check(context)

            self.assertEqual(len(violations), 0)
