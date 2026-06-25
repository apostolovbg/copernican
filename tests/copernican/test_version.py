"""Tests for the package version helper."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest import TestCase

import copernican.lib.plotter as plotter
import copernican.lib.run_manifest as run_manifest
import copernican.version as version_module


class VersionTests(TestCase):
    """Verify the version helper reads the tracked file and falls back."""

    def setUp(self) -> None:
        """Store the original environment and module getter."""

        self.original_env = os.environ.get("COPERNICAN_VERSION")
        self.original_getter = getattr(version_module, "get_version", None)

    def tearDown(self) -> None:
        """Restore environment and version helper state."""

        if self.original_env is None:
            os.environ.pop("COPERNICAN_VERSION", None)
        else:
            os.environ["COPERNICAN_VERSION"] = self.original_env
        if self.original_getter is not None:
            version_module.get_version = self.original_getter
        elif hasattr(version_module, "get_version"):
            delattr(version_module, "get_version")
        importlib.reload(plotter)
        importlib.reload(run_manifest)

    def test_version_file_is_used(self) -> None:
        """The helper reads the tracked package VERSION file."""

        version_path = Path(version_module.__file__).with_name("VERSION")
        expected = version_path.read_text(encoding="utf-8").strip()
        self.assertTrue(expected)
        self.assertEqual(version_module.get_version(), expected)

    def test_environment_variable_is_ignored(self) -> None:
        """The helper ignores COPERNICAN_VERSION and reads file data."""

        os.environ["COPERNICAN_VERSION"] = "1.2.3-custom"
        version_path = Path(version_module.__file__).with_name("VERSION")
        expected = version_path.read_text(encoding="utf-8").strip()
        self.assertEqual(version_module.get_version(), expected)

    def test_missing_getter_falls_back_to_unknown(self) -> None:
        """Consumers get a stable fallback when the getter is absent."""

        if hasattr(version_module, "get_version"):
            delattr(version_module, "get_version")
        importlib.reload(plotter)
        importlib.reload(run_manifest)
        self.assertEqual(plotter.COPERNICAN_VERSION, "0+unknown")
        self.assertEqual(run_manifest._copernican_version(), "0+unknown")
