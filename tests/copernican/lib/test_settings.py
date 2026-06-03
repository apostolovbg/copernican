"""Smoke tests for copernican.lib.settings."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from copernican.lib import settings as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.settings")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the settings surface to the coverage policy."""

    def test_public_symbols_are_present(self) -> None:
        self.assertTrue(hasattr(module, "get_settings_path"))
        self.assertTrue(hasattr(module, "load_settings"))
        self.assertTrue(hasattr(module, "save_settings"))
        self.assertTrue(hasattr(module, "get_settings"))

    def test_settings_path_lives_under_global_settings(self) -> None:
        expected = (
            Path(module.__file__).resolve().parent
            / "global_settings"
            / "copernican_settings.yml"
        )
        self.assertEqual(module.get_settings_path(), expected)

    def test_missing_settings_file_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing-settings.yml"
            with mock.patch.object(
                module,
                "get_settings_path",
                return_value=path,
            ):
                self.assertEqual(
                    module.load_settings(), module.DEFAULT_SETTINGS
                )
            self.assertFalse(path.exists())

    def test_save_settings_writes_the_packaged_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.yml"
            with mock.patch.object(
                module,
                "get_settings_path",
                return_value=path,
            ):
                module.save_settings(module.DEFAULT_SETTINGS)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
