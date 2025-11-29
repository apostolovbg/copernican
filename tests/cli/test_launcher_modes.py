# Last Updated: 2025-11-29
"""Tests for the launcher shim and GUI/CLI mode selection."""

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class LaunchArgParsingTestCase(unittest.TestCase):
    """Ensure launcher arguments map cleanly onto orchestration modes."""

    def setUp(self) -> None:
        self.venv_path = str(Path(__file__).resolve().parents[2] / ".venv")
        self.env_patch = mock.patch.dict(
            os.environ,
            {"VIRTUAL_ENV": self.venv_path},
            clear=True,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.copernican = importlib.reload(
            importlib.import_module("copernican")
        )

    def tearDown(self) -> None:
        if hasattr(self.copernican, "_legacy_stage_menu_override"):
            self.copernican._legacy_stage_menu_override = False

    def test_default_mode_prefers_cli(self) -> None:
        args = self.copernican._parse_launch_args([])
        self.assertEqual(
            args.mode, self.copernican.orchestration.LaunchMode.CLI
        )
        self.assertFalse(args.legacy_stage_menu)

    def test_gui_flag_switches_mode(self) -> None:
        args = self.copernican._parse_launch_args(["--gui"])
        self.assertEqual(
            args.mode, self.copernican.orchestration.LaunchMode.GUI
        )

    def test_env_flag_enables_legacy_menu(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "VIRTUAL_ENV": self.venv_path,
                "COPERNICAN_ENABLE_STAGED_MENU": "1",
            },
            clear=True,
        ):
            args = self.copernican._parse_launch_args([])
        self.assertTrue(args.legacy_stage_menu)

    def test_override_flag_enables_legacy_menu(self) -> None:
        args = self.copernican._parse_launch_args(
            ["--enable-legacy-stage-menu"]
        )
        self.assertTrue(args.legacy_stage_menu)

    def test_legacy_stage_menu_enabled_respects_override(self) -> None:
        self.copernican._legacy_stage_menu_override = False
        with mock.patch.dict(
            os.environ,
            {"VIRTUAL_ENV": self.venv_path},
            clear=True,
        ):
            self.assertFalse(self.copernican.legacy_stage_menu_enabled())
        self.copernican._legacy_stage_menu_override = True
        self.assertTrue(self.copernican.legacy_stage_menu_enabled())

    def test_no_gui_flag_forces_cli(self) -> None:
        args = self.copernican._parse_launch_args(["--no-gui"])
        self.assertEqual(
            args.mode, self.copernican.orchestration.LaunchMode.CLI
        )

    def test_manifest_and_output_dir_paths_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "run" / "manifest.yml"
            output_dir = Path(tmpdir) / "output"
            args = self.copernican._parse_launch_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )
        self.assertEqual(args.manifest_path, manifest_path.resolve())
        self.assertEqual(args.output_dir, output_dir.resolve())

    def test_detach_flag_respects_environment_override(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"VIRTUAL_ENV": self.venv_path, "COPERNICAN_DETACH_GUI": "0"},
            clear=True,
        ):
            args = self.copernican._parse_launch_args(["--gui"])
        self.assertFalse(args.detach_gui)
