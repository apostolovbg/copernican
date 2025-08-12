"""Tests for command-line helpers in ``copernican.py``."""

import importlib
import sys
import unittest
from unittest import mock

with mock.patch("sys.version_info", (3, 12, 0)):
    copernican = importlib.import_module("copernican")


class RunTestsFlagTestCase(unittest.TestCase):
    """Verify that ``--run-tests`` runs ``python -m unittest`` discovery."""

    @mock.patch("subprocess.run")
    def test_run_startup_tests_invokes_unittest_discover(self, run_mock):
        """Ensure the helper spawns the expected discovery command."""
        run_mock.return_value.returncode = 0
        result = copernican.run_startup_tests()
        self.assertTrue(result)
        run_mock.assert_called_once()
        cmd = run_mock.call_args[0][0]
        self.assertEqual(cmd[:3], [sys.executable, "-m", "unittest"])
        self.assertEqual(cmd[3], "discover")
        self.assertIn("-v", cmd)


if __name__ == "__main__":
    unittest.main()
