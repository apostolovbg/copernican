# Last Updated: 2025-11-24
"""Tests for the launcher shim and GUI/CLI mode selection."""

import importlib
import os
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
        mode, legacy = self.copernican._parse_launch_args([])
        self.assertEqual(mode, self.copernican.orchestration.LaunchMode.CLI)
        self.assertFalse(legacy)

    def test_gui_flag_switches_mode(self) -> None:
        mode, _ = self.copernican._parse_launch_args(["--gui"])
        self.assertEqual(mode, self.copernican.orchestration.LaunchMode.GUI)

    def test_env_flag_enables_legacy_menu(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "VIRTUAL_ENV": self.venv_path,
                "COPERNICAN_ENABLE_STAGED_MENU": "1",
            },
            clear=True,
        ):
            _, legacy = self.copernican._parse_launch_args([])
        self.assertTrue(legacy)

    def test_override_flag_enables_legacy_menu(self) -> None:
        _, legacy = self.copernican._parse_launch_args(
            [
                "--enable-legacy-stage-menu",
            ]
        )
        self.assertTrue(legacy)

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
