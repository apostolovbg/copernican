"""Tests for new_modules_need_tests policy."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from devcovenant.policy_scripts.new_modules_need_tests import (
    NewModulesNeedTestsPolicy,
)


class TestNewModulesNeedTestsPolicy(unittest.TestCase):
    """Test suite for NewModulesNeedTestsPolicy."""

    @patch("subprocess.check_output")
    def test_detects_new_module_without_tests(self, mock_subprocess):
        """Policy should detect new modules without test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            # Simulate git status showing new module
            mock_subprocess.return_value = "A  copernican_lib/new_module.py\n"

            policy = NewModulesNeedTestsPolicy(repo_root)
            violations = policy.check([])

            self.assertEqual(len(violations), 1)
            self.assertIn("test", violations[0].message.lower())

    @patch("subprocess.check_output")
    def test_allows_new_module_with_tests(self, mock_subprocess):
        """Policy should pass when new modules have tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            # Simulate git status showing new module and test
            mock_subprocess.return_value = (
                "A  copernican_lib/new_module.py\nM  tests/test_new_module.py\n"
            )

            policy = NewModulesNeedTestsPolicy(repo_root)
            violations = policy.check([])

            self.assertEqual(len(violations), 0)
