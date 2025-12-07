"""Tests for the launcher shim and GUI/CLI mode selection."""

import importlib
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock


class LaunchArgParsingTestCase(unittest.TestCase):
    """Ensure launcher arguments map cleanly onto orchestration modes."""

    def setUp(self) -> None:
        self.venv_path = str(Path(__file__).resolve().parents[2] / ".venv")
        self.settings_path = (
            Path(tempfile.gettempdir())
            / f"copernican_settings_test_{uuid.uuid4().hex}.yml"
        )
        self.env_patch = mock.patch.dict(
            os.environ,
            {
                "VIRTUAL_ENV": self.venv_path,
                "COPERNICAN_SETTINGS_PATH": str(self.settings_path),
            },
            clear=True,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.copernican = importlib.reload(
            importlib.import_module("copernican")
        )

    def tearDown(self) -> None:
        if self.settings_path.exists():
            self.settings_path.unlink(missing_ok=True)

    def test_default_mode_prefers_cli(self) -> None:
        args = self.copernican._parse_launch_args([])
        self.assertEqual(
            args.mode, self.copernican.orchestration.LaunchMode.CLI
        )

    def test_gui_flag_switches_mode(self) -> None:
        args = self.copernican._parse_launch_args(["--gui"])
        self.assertEqual(
            args.mode, self.copernican.orchestration.LaunchMode.GUI
        )

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
