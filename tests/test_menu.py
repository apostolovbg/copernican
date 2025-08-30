# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Tests for menu interaction helpers in ``copernican.py``."""

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

with mock.patch("sys.version_info", (3, 12, 0)):
    with mock.patch.dict(
        os.environ,
        {"VIRTUAL_ENV": str(Path(__file__).resolve().parents[1] / ".venv")},
    ):
        copernican = importlib.import_module("copernican")
import copernican_lib.data_loaders


class MenuRunTestsTestCase(unittest.TestCase):
    """Verify the menu invokes ``python -m unittest`` discovery."""

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


class SelectSourceDisplayTestCase(unittest.TestCase):
    """Ensure selection prompts show names and return identifiers."""

    @mock.patch("copernican_lib.data_loaders.console.ask", return_value="1")
    def test_select_source_shows_name(self, _ask_mock):
        registry = {
            "dummy_id": {
                "dataset_name": "Dummy Dataset",
                "description": "demo",
                "data_dir": None,
                "function": lambda *_: None,
            }
        }
        captured = []
        with mock.patch(
            "copernican_lib.data_loaders.console.write",
            lambda msg: captured.append(msg),
        ):
            result = copernican_lib.data_loaders._select_source(
                registry, "SNe"
            )
        self.assertEqual(result, "dummy_id")
        out = "".join(captured)
        self.assertIn("Dummy Dataset", out)
        self.assertNotIn("dummy_id", out)


class DependencyPromptTestCase(unittest.TestCase):
    """Test dependency installer confirmation and CI override."""

    @mock.patch("copernican.Path")
    def test_installs_after_confirmation(self, path_mock):
        path_mock.return_value.resolve.return_value.name = ".venv"
        with (
            mock.patch(
                "copernican._gather_required_packages", return_value=["demo"]
            ),
            mock.patch("importlib.util.find_spec", return_value=None),
            mock.patch("copernican.console.ask", return_value="y") as ask_mock,
            mock.patch("subprocess.run") as run_mock,
            mock.patch("importlib.import_module"),
        ):
            copernican.check_dependencies()
            ask_mock.assert_called_once()
            run_mock.assert_called_once_with(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--require-hashes",
                    "-r",
                    "requirements.lock",
                ],
                check=True,
            )

    @mock.patch("copernican.Path")
    def test_auto_confirm_skips_prompt(self, path_mock):
        path_mock.return_value.resolve.return_value.name = ".venv"
        with (
            mock.patch(
                "copernican._gather_required_packages", return_value=["demo"]
            ),
            mock.patch("importlib.util.find_spec", return_value=None),
            mock.patch("copernican.console.ask") as ask_mock,
            mock.patch("subprocess.run") as run_mock,
            mock.patch("importlib.import_module"),
        ):
            copernican.check_dependencies(auto_confirm=True)
            ask_mock.assert_not_called()
            run_mock.assert_called_once_with(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--require-hashes",
                    "-r",
                    "requirements.lock",
                ],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
